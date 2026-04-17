"""Single-drone position tracking environment for Crazyflie Brushless."""
from __future__ import annotations

import math
from pathlib import Path
import torch
from tensordict import TensorDict

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationData
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import TiledCamera
from isaaclab.sensors.camera.utils import save_images_to_file
from isaaclab.utils import math as math_utils
import isaacsim.core.utils.prims as prim_utils

from source.isaac_pursuit_evasion.assets.crazyflie_brushless import (
    CrazyflieBrushlessPursuer,
    fpv_camera_cfg,
    fpv_camera_center_line,
)
from source.isaac_pursuit_evasion.controllers.crazy_controller import DEFAULT_GAINS, build_crazyflie_pid
from source.isaac_pursuit_evasion.controllers.rl_controllers import (
    CrazyflieRLBodyRatesWrapper,
    CrazyflieRLVelocityWrapper,
)
from source.isaac_pursuit_evasion.dynamics.propellers import Drone_cfg, Propellers

from .pos_tracking_env_cfg import PosTrackingEnvCfg

DONE_REASON_LABELS = {
    0: "running",
    1: "success",
    2: "crash",
    3: "out_of_bounds",
    4: "timeout",
    5: "invalid_state",
}


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class PosTrackingEnv(DirectRLEnv):
    """Single-drone position (+optional yaw) tracking environment."""

    cfg: PosTrackingEnvCfg
    DONE_REASON_MAP = DONE_REASON_LABELS

    def __init__(self, cfg: PosTrackingEnvCfg, **kwargs) -> None:
        cfg.observation_space = self._compute_obs_dim(cfg)
        cfg.state_space = cfg.observation_space

        drone_name = cfg.drone_name.lower()
        if drone_name not in ("crazyflie_brushless", "cf_brushless"):
            raise ValueError(f"Unsupported drone_name '{cfg.drone_name}'. Only crazyflie_brushless is supported.")

        if cfg.robot is None:
            cfg.robot = CrazyflieBrushlessPursuer.replace(prim_path="/World/envs/env_.*/Robot")

        if cfg.enable_cameras:
            cam_cfg = fpv_camera_cfg(tiled=True)
            self._camera_cfg = self._camera_cfg_with_resolution_limit(cam_cfg, cfg)
            self._cam_origin, self._cam_line = fpv_camera_center_line(length=5.0, device=cfg.sim.device)
        else:
            self._camera_cfg = None
            self._cam_origin = None
            self._cam_line = None

        super().__init__(cfg, **kwargs)

        self._camera = self.scene.sensors.get("robot_camera") if self.cfg.enable_cameras else None
        self._camera_save_stride = 1
        if self._camera_cfg is not None:
            update_period = float(getattr(self._camera_cfg, "update_period", 0.0))
            dt = float(self.sim.cfg.dt)
            if update_period > 0.0 and dt > 0.0:
                self._camera_save_stride = max(1, int(round(update_period / dt)))

        self._body_id = self._robot.find_bodies("body")[0]

        self._drone_cfg = Drone_cfg(cfg.drone_name, device=self.device)
        masses = self._robot.root_physx_view.get_masses()[0].to(self.device)
        mass_total = masses.sum()
        inertia_body = self._robot.root_physx_view.get_inertias()[0, self._body_id, :].view(3, 3).to(self.device)
        self._mass_total = mass_total
        self._inertia_body = inertia_body
        self._drone_cfg.set_physical_params(mass_total, inertia_body)

        self._propellers = Propellers(self.num_envs, self._drone_cfg, self.sim.cfg.dt, use=True, device=self.device)
        self._prop_joint_ids = self._find_prop_joints(self._robot)

        self._pid_params = {
            "sim_rate_hz": float(self.cfg.sim_frequency),
            "pid_loop_rate_hz": float(self.cfg.pid_loop_rate_hz),
            "pid_posvel_loop_rate_hz": float(self.cfg.pid_posvel_loop_rate_hz),
        }
        self._action_wrapper = self._build_action_wrapper()
        self._use_position_controller = bool(self.cfg.use_position_controller)
        self._baseline_controller = None
        if self._use_position_controller:
            self._baseline_controller = build_crazyflie_pid(
                num_envs=self.num_envs,
                drone_cfg=self._drone_cfg,
                dt=self.sim.cfg.dt,
                device=self.device,
                pid_params=self._pid_params,
            )

        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._action_diff = torch.zeros_like(self._actions)
        self._commands = torch.zeros(self.num_envs, 4, device=self.device)
        self._wrench = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros_like(self._thrust)

        self._step_dt = self.sim.cfg.dt * self.cfg.decimation

        self._arena_min = torch.tensor(self.cfg.arena_min, device=self.device, dtype=torch.float32)
        self._arena_max = torch.tensor(self.cfg.arena_max, device=self.device, dtype=torch.float32)
        margin = torch.tensor(self.cfg.arena_margin, device=self.device, dtype=torch.float32)
        self._arena_min_safe = self._arena_min + margin
        self._arena_max_safe = self._arena_max - margin
        self._arena_min_safe[2] = torch.maximum(
            self._arena_min[2] + margin,
            torch.tensor(self.cfg.collision_altitude, device=self.device),
        )

        self._pillar_radius = float(self.cfg.pillar_radius)
        self._pillar_collision_radius = float(self.cfg.pillar_radius + self.cfg.drone_collision_radius)
        self._pillar_top_z = float(self.cfg.arena_min[2] + self.cfg.pillar_height)
        if len(self.cfg.pillar_positions_xy) > 0:
            self._pillar_positions_xy = torch.tensor(self.cfg.pillar_positions_xy, device=self.device, dtype=torch.float32)
        else:
            self._pillar_positions_xy = torch.zeros((0, 2), device=self.device, dtype=torch.float32)

        self._ref_pos_min = torch.tensor(self.cfg.ref_pos_min, device=self.device, dtype=torch.float32)
        self._ref_pos_max = torch.tensor(self.cfg.ref_pos_max, device=self.device, dtype=torch.float32)
        self._ref_yaw_min = float(self.cfg.ref_yaw_range[0])
        self._ref_yaw_max = float(self.cfg.ref_yaw_range[1])

        self._reference_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._reference_yaw = torch.zeros(self.num_envs, 1, device=self.device)
        self._reference_timer = torch.zeros(self.num_envs, device=self.device)

        self._success_hold_steps = max(1, int(round(self.cfg.success_hold_time_s / self._step_dt)))
        self._success_counter = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._last_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._last_rewards = torch.zeros(self.num_envs, device=self.device)
        self._last_reward_components: dict[str, torch.Tensor] = {}
        self._last_done_reason = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        self._body_x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)

        self._ref_markers: VisualizationMarkers | None = None
        self._setup_visualizers()

        self._dr_cfg = self.cfg.domain_randomization
        self._init_domain_randomization()

    # ---------------------------------------------------------------------
    # IsaacLab interface
    # ---------------------------------------------------------------------

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        if self.cfg.enable_cameras and self._camera_cfg is not None:
            cam_cfg = self._camera_cfg
            cam_cfg.prim_path = f"{self.scene.env_regex_ns}/Robot/body/fpv_camera"
            self.scene.sensors["robot_camera"] = TiledCamera(cam_cfg)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        if self.cfg.enable_walls:
            self._spawn_arena_walls()
        if self.cfg.enable_pillars:
            self._spawn_arena_pillars()

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self._set_camera_view()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._maybe_update_reference()
        if self._use_position_controller:
            self._actions.zero_()
            self._action_diff.zero_()
            self._prev_actions.zero_()
            self._commands.zero_()
        else:
            self._actions = actions.clamp(-1.0, 1.0)
            self._action_diff = self._actions - self._prev_actions
            self._prev_actions = self._actions.clone()

            td = TensorDict(
                {
                    "rl_action": self._actions,
                    "root_state": self._robot.data.root_state_w,
                    "body_rate": self._robot.data.root_ang_vel_b,
                },
                batch_size=[self.num_envs],
                device=self.device,
            )
            self._commands = self._action_wrapper.command(td)

        self._update_visualizers()

    def _apply_action(self) -> None:
        if self._use_position_controller and self._baseline_controller is not None:
            env_origins = self._terrain.env_origins
            target_pos = self._reference_pos + env_origins
            target_yaw = self._reference_yaw if self.cfg.flag_yaw_tracking else None
            thrust, moment = self._baseline_controller(
                root_state=self._robot.data.root_state_w,
                target_pos=target_pos,
                target_yaw=target_yaw,
                command_level="position",
            )
            self._wrench = torch.cat((thrust, moment), dim=-1)
        else:
            self._wrench = self._action_wrapper.wrench_from_command(self._robot.data.root_state_w, self._commands)
        omega_ref = self._propellers.compute_motor_speeds_from_wrench(self._wrench)
        self._propellers.compute_omega(omega_ref)
        vel_body = self._robot.data.root_lin_vel_b
        state_stub = torch.zeros(self.num_envs, 6, device=self.device)
        state_stub[:, 3:6] = vel_body
        self._thrust, self._moment = self._propellers.compute_force_and_torque(state_stub)
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
        self._update_prop_visuals()

    def _get_observations(self) -> dict:
        env_origins = self._terrain.env_origins
        pos_local = self._robot.data.root_pos_w - env_origins
        vel_world = self._robot.data.root_lin_vel_w
        ang_vel = self._robot.data.root_ang_vel_b
        quat = self._robot.data.root_quat_w

        pos_error = self._reference_pos - pos_local
        dist = torch.norm(pos_error, dim=-1, keepdim=True)

        parts = [pos_error, dist, vel_world, ang_vel, quat]

        if self.cfg.flag_yaw_tracking:
            yaw_err, yaw_align = self._yaw_features(quat)
            parts.append(yaw_align)
            parts.append(yaw_err)
        if self.cfg.flag_action_smoothness_penalty:
            parts.append(self._prev_actions)

        obs = torch.cat(parts, dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        env_origins = self._terrain.env_origins
        pos_local = self._robot.data.root_pos_w - env_origins
        pos_error = torch.norm(self._reference_pos - pos_local, dim=-1)

        pos_reward = self.cfg.reward_pos * torch.exp(-self.cfg.reward_pos_scale * pos_error)
        rewards = pos_reward.clone()
        components: dict[str, torch.Tensor] = {
            "pos": pos_reward,
        }

        if self.cfg.flag_yaw_tracking:
            yaw_err, yaw_align = self._yaw_features(self._robot.data.root_quat_w)
            yaw_reward = self.cfg.reward_yaw * yaw_align.squeeze(-1)
            rewards += yaw_reward
            components["yaw"] = yaw_reward
        else:
            components["yaw"] = torch.zeros_like(rewards)

        body_rates = self._robot.data.root_ang_vel_b
        body_rate_pen = self.cfg.reward_body_rates * torch.norm(body_rates, dim=-1)
        rewards -= body_rate_pen
        components["body_rates"] = -body_rate_pen

        if self.cfg.flag_penalize_linvel:
            lin_vel = self._robot.data.root_lin_vel_w
            lin_vel_pen = self.cfg.reward_lin_vel * torch.norm(lin_vel, dim=-1)
            rewards -= lin_vel_pen
            components["lin_vel"] = -lin_vel_pen
        else:
            components["lin_vel"] = torch.zeros_like(rewards)

        if self.cfg.flag_action_smoothness_penalty:
            smooth_pen = self.cfg.reward_action_smoothness * torch.norm(self._action_diff, dim=-1)
            rewards -= smooth_pen
            components["action_smoothness"] = -smooth_pen
        else:
            components["action_smoothness"] = torch.zeros_like(rewards)

        crash = self._crash_mask(pos_local)
        out_of_bounds = self._out_of_bounds_mask(pos_local)
        if crash.any():
            crash_pen = torch.zeros_like(rewards)
            crash_pen[crash] = self.cfg.reward_crash
            rewards -= crash_pen
            components["crash"] = -crash_pen
        else:
            components["crash"] = torch.zeros_like(rewards)

        if out_of_bounds.any():
            bounds_pen = torch.zeros_like(rewards)
            bounds_pen[out_of_bounds] = self.cfg.reward_out_of_bounds
            rewards -= bounds_pen
            components["out_of_bounds"] = -bounds_pen
        else:
            components["out_of_bounds"] = torch.zeros_like(rewards)

        pillar_collision = self._pillar_collision_mask(pos_local)
        if pillar_collision.any():
            pillar_pen = torch.zeros_like(rewards)
            pillar_pen[pillar_collision] = self.cfg.reward_pillar_collision
            rewards -= pillar_pen
            components["pillar_collision"] = -pillar_pen
        else:
            components["pillar_collision"] = torch.zeros_like(rewards)

        self._last_rewards = rewards
        self._last_reward_components = components
        self._maybe_save_camera_images()
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        timeout = self.episode_length_buf >= self.max_episode_length
        env_origins = self._terrain.env_origins
        pos_local = self._robot.data.root_pos_w - env_origins
        crash = self._crash_mask(pos_local)
        out_of_bounds = self._out_of_bounds_mask(pos_local)
        invalid = ~torch.isfinite(self._robot.data.root_state_w).all(dim=-1)

        success = self._update_success_flags(pos_local)
        self._last_success = success

        terminated = crash | out_of_bounds | invalid
        if self.cfg.terminate_on_success:
            terminated = terminated | success

        truncated = timeout & (~terminated)

        done_reason = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        done_reason[success] = 1
        done_reason[crash] = 2
        done_reason[out_of_bounds] = 3
        done_reason[timeout] = 4
        done_reason[invalid] = 5
        self._last_done_reason = done_reason

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._propellers.reset(env_ids)
        if self._action_wrapper is not None:
            self._action_wrapper.reset(env_ids)
        if self._baseline_controller is not None:
            self._baseline_controller.reset(env_ids)

        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._action_diff[env_ids] = 0.0
        self._success_counter[env_ids] = 0
        self._last_success[env_ids] = False
        self._reference_timer[env_ids] = 0.0

        self._resample_reference(env_ids)
        self._apply_domain_randomization(env_ids)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _compute_obs_dim(cfg: PosTrackingEnvCfg) -> int:
        dim = 3 + 1 + 3 + 3 + 4
        if cfg.flag_yaw_tracking:
            dim += 2
        if cfg.flag_action_smoothness_penalty:
            dim += cfg.action_space
        return dim

    def _build_action_wrapper(self):
        def passthrough(td: TensorDict) -> torch.Tensor:
            return td.get("rl_action")

        dt = self.sim.cfg.dt * self.cfg.decimation
        if self.cfg.control_mode == "RL_velocity":
            return CrazyflieRLVelocityWrapper(
                num_envs=self.num_envs,
                drone_cfg=self._drone_cfg,
                policy=passthrough,
                dt=dt,
                pid_dt=self.sim.cfg.dt,
                device=self.device,
                action_key="rl_action",
                root_state_key="root_state",
                vel_scale=torch.tensor(self.cfg.vel_scale, device=self.device, dtype=torch.float32),
                yaw_rate_scale=self.cfg.yaw_rate_scale,
                pid_params=self._pid_params,
            )
        if self.cfg.control_mode == "RL_rates":
            return CrazyflieRLBodyRatesWrapper(
                num_envs=self.num_envs,
                drone_cfg=self._drone_cfg,
                policy=passthrough,
                dt=dt,
                pid_dt=self.sim.cfg.dt,
                device=self.device,
                action_key="rl_action",
                root_state_key="root_state",
                thrust_scale=self.cfg.thrust_to_weight,
                pid_params=self._pid_params,
            )
        raise ValueError(f"Unsupported control_mode '{self.cfg.control_mode}'.")

    def _yaw_features(self, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        yaw = math_utils.euler_xyz_from_quat(quat)[2].unsqueeze(-1)
        yaw_ref = self._reference_yaw
        yaw_err = _wrap_angle(yaw_ref - yaw)

        desired = torch.stack(
            [torch.cos(yaw_ref.squeeze(-1)), torch.sin(yaw_ref.squeeze(-1)), torch.zeros_like(yaw_ref.squeeze(-1))],
            dim=-1,
        )
        body_x = self._body_x_axis.view(1, 3).expand(quat.shape[0], -1)
        forward = math_utils.quat_apply(quat, body_x)
        forward_xy = forward.clone()
        forward_xy[:, 2] = 0.0
        desired_xy = desired.clone()
        desired_xy[:, 2] = 0.0
        forward_xy = forward_xy / torch.norm(forward_xy, dim=-1, keepdim=True).clamp_min(1e-6)
        desired_xy = desired_xy / torch.norm(desired_xy, dim=-1, keepdim=True).clamp_min(1e-6)
        yaw_align = torch.sum(forward_xy * desired_xy, dim=-1, keepdim=True)
        return yaw_err, yaw_align

    def _crash_mask(self, pos_local: torch.Tensor) -> torch.Tensor:
        return pos_local[:, 2] < self.cfg.collision_altitude

    def _out_of_bounds_mask(self, pos_local: torch.Tensor) -> torch.Tensor:
        below = pos_local < self._arena_min_safe
        above = pos_local > self._arena_max_safe
        return torch.any(below | above, dim=-1)

    def _pillar_collision_mask(self, pos_local: torch.Tensor) -> torch.Tensor:
        if (not self.cfg.enable_pillars) or self._pillar_positions_xy.shape[0] == 0:
            return torch.zeros(pos_local.shape[0], dtype=torch.bool, device=self.device)

        xy = pos_local[:, :2].unsqueeze(1)
        pillar_xy = self._pillar_positions_xy.unsqueeze(0)
        dxy = torch.norm(xy - pillar_xy, dim=-1)
        inside_radius = dxy <= self._pillar_collision_radius
        inside_height = (pos_local[:, 2] >= self.cfg.arena_min[2]) & (pos_local[:, 2] <= self._pillar_top_z)
        return torch.any(inside_radius, dim=1) & inside_height

    def _update_success_flags(self, pos_local: torch.Tensor) -> torch.Tensor:
        pos_error = torch.norm(self._reference_pos - pos_local, dim=-1)
        within = pos_error < self.cfg.pos_tolerance
        if self.cfg.flag_yaw_tracking:
            yaw_err, _ = self._yaw_features(self._robot.data.root_quat_w)
            within = within & (torch.abs(yaw_err.squeeze(-1)) < self.cfg.yaw_tolerance)

        self._success_counter = torch.where(within, self._success_counter + 1, torch.zeros_like(self._success_counter))
        return self._success_counter >= self._success_hold_steps

    def _maybe_update_reference(self) -> None:
        if self.cfg.ref_update_interval_s <= 0.0:
            return
        self._reference_timer += self._step_dt
        update_mask = self._reference_timer >= self.cfg.ref_update_interval_s
        if update_mask.any():
            env_ids = torch.nonzero(update_mask).squeeze(-1)
            self._reference_timer[env_ids] = 0.0
            self._resample_reference(env_ids)

    def _resample_reference(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        low = self._ref_pos_min
        high = self._ref_pos_max
        sample = torch.rand(env_ids.shape[0], 3, device=self.device)
        pos = low + (high - low) * sample
        if self.cfg.enable_pillars and self._pillar_positions_xy.shape[0] > 0:
            # Keep references clear of obstacle cylinders so the objective is always feasible.
            safety_radius = self._pillar_collision_radius + 0.2
            for _ in range(8):
                dxy = torch.cdist(pos[:, :2], self._pillar_positions_xy)
                colliding = torch.any(dxy <= safety_radius, dim=1)
                if not colliding.any():
                    break
                count = int(colliding.sum().item())
                resample = torch.rand(count, 3, device=self.device)
                pos[colliding] = low + (high - low) * resample
        yaw = torch.empty(env_ids.shape[0], 1, device=self.device).uniform_(self._ref_yaw_min, self._ref_yaw_max)
        self._reference_pos[env_ids] = pos
        self._reference_yaw[env_ids] = yaw

    # ---------------------------------------------------------------------
    # Domain randomization
    # ---------------------------------------------------------------------

    def _init_domain_randomization(self) -> None:
        self._dr_default_masses = self._robot.root_physx_view.get_masses().clone().cpu()
        self._dr_default_inertias = self._robot.root_physx_view.get_inertias().clone().cpu()
        self._dr_masses = self._dr_default_masses.clone()
        self._dr_inertias = self._dr_default_inertias.clone()

        self._dr_nominal_mass = float(self._mass_total)
        self._dr_nominal_inertia = torch.diagonal(self._inertia_body).clone().to(self.device)
        self._dr_nominal_k_eta = float(self._propellers.k_eta[0, 0])
        self._dr_nominal_k_m = float(self._propellers.k_m[0, 0])
        self._dr_nominal_tau = float(self._propellers.tau_m[0, 0])
        self._dr_nominal_k_aero_xy = float(self._propellers.K_aero[0, 0])
        self._dr_nominal_k_aero_z = float(self._propellers.K_aero[0, 2])

        self._dr_rate_kp_nominal = torch.as_tensor(DEFAULT_GAINS["rate"]["kp"], device=self.device, dtype=torch.float32)
        self._dr_rate_ki_nominal = torch.as_tensor(DEFAULT_GAINS["rate"]["ki"], device=self.device, dtype=torch.float32)
        self._dr_rate_kd_nominal = torch.as_tensor(DEFAULT_GAINS["rate"]["kd"], device=self.device, dtype=torch.float32)

        self._dr_mass = torch.full((self.num_envs,), self._dr_nominal_mass, device=self.device)
        self._dr_inertia = self._dr_nominal_inertia.view(1, 3).repeat(self.num_envs, 1)
        self._dr_k_eta = torch.full((self.num_envs,), self._dr_nominal_k_eta, device=self.device)
        self._dr_k_m = torch.full((self.num_envs,), self._dr_nominal_k_m, device=self.device)
        self._dr_tau = torch.full((self.num_envs,), self._dr_nominal_tau, device=self.device)
        self._dr_k_aero_xy = torch.full((self.num_envs,), self._dr_nominal_k_aero_xy, device=self.device)
        self._dr_k_aero_z = torch.full((self.num_envs,), self._dr_nominal_k_aero_z, device=self.device)
        self._dr_rate_kp = self._dr_rate_kp_nominal.view(1, 3).repeat(self.num_envs, 1)
        self._dr_rate_ki = self._dr_rate_ki_nominal.view(1, 3).repeat(self.num_envs, 1)
        self._dr_rate_kd = self._dr_rate_kd_nominal.view(1, 3).repeat(self.num_envs, 1)

        if self._dr_cfg.enable:
            self._apply_domain_randomization(torch.arange(self.num_envs, device=self.device))

    def _apply_domain_randomization(self, env_ids: torch.Tensor) -> None:
        if not self._dr_cfg.enable:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        n = env_ids.shape[0]
        scale_min = float(self._dr_cfg.scale_min)
        scale_max = float(self._dr_cfg.scale_max)

        def _sample_scale(enabled: bool) -> torch.Tensor:
            if not enabled:
                return torch.ones(n, device=self.device)
            return torch.empty(n, device=self.device).uniform_(scale_min, scale_max)

        mass_scale = _sample_scale(self._dr_cfg.randomize_mass)
        inertia_scale = _sample_scale(self._dr_cfg.randomize_inertia)
        mass = mass_scale * self._dr_nominal_mass
        inertia = inertia_scale.view(-1, 1) * self._dr_nominal_inertia.view(1, 3)

        self._apply_mass_inertia(env_ids, mass_scale, inertia_scale)
        self._dr_mass[env_ids] = mass
        self._dr_inertia[env_ids] = inertia

        k_eta_scale = _sample_scale(self._dr_cfg.randomize_k_eta)
        k_m_scale = _sample_scale(self._dr_cfg.randomize_k_m)
        tau_scale = _sample_scale(self._dr_cfg.randomize_tau)

        k_eta = k_eta_scale * self._dr_nominal_k_eta
        k_m = k_m_scale * self._dr_nominal_k_m
        tau_m = tau_scale * self._dr_nominal_tau

        if self._dr_cfg.randomize_k_aero:
            k_aero_xy = torch.empty(n, device=self.device).uniform_(
                self._dr_nominal_k_aero_xy * self._dr_cfg.k_aero_xy_min_scale,
                self._dr_nominal_k_aero_xy * self._dr_cfg.k_aero_xy_max_scale,
            )
            k_aero_z = torch.empty(n, device=self.device).uniform_(
                self._dr_nominal_k_aero_z * self._dr_cfg.k_aero_z_min_scale,
                self._dr_nominal_k_aero_z * self._dr_cfg.k_aero_z_max_scale,
            )
        else:
            k_aero_xy = torch.full((n,), self._dr_nominal_k_aero_xy, device=self.device)
            k_aero_z = torch.full((n,), self._dr_nominal_k_aero_z, device=self.device)

        if self._dr_cfg.randomize_rate_gains:
            kp_rp = torch.empty(n, device=self.device).uniform_(
                self._dr_cfg.rate_kp_min_scale, self._dr_cfg.rate_kp_max_scale
            ) * self._dr_rate_kp_nominal[0]
            kp_y = torch.empty(n, device=self.device).uniform_(
                self._dr_cfg.rate_kp_min_scale, self._dr_cfg.rate_kp_max_scale
            ) * self._dr_rate_kp_nominal[2]

            ki_rp = torch.empty(n, device=self.device).uniform_(
                self._dr_cfg.rate_ki_min_scale, self._dr_cfg.rate_ki_max_scale
            ) * self._dr_rate_ki_nominal[0]
            ki_y = torch.empty(n, device=self.device).uniform_(
                self._dr_cfg.rate_ki_min_scale, self._dr_cfg.rate_ki_max_scale
            ) * self._dr_rate_ki_nominal[2]

            kd_rp = torch.empty(n, device=self.device).uniform_(
                self._dr_cfg.rate_kd_min_scale, self._dr_cfg.rate_kd_max_scale
            ) * self._dr_rate_kd_nominal[0]
            kd_y = torch.empty(n, device=self.device).uniform_(
                self._dr_cfg.rate_kd_min_scale, self._dr_cfg.rate_kd_max_scale
            ) * self._dr_rate_kd_nominal[2]
        else:
            kp_rp = self._dr_rate_kp_nominal[0].expand(n)
            kp_y = self._dr_rate_kp_nominal[2].expand(n)
            ki_rp = self._dr_rate_ki_nominal[0].expand(n)
            ki_y = self._dr_rate_ki_nominal[2].expand(n)
            kd_rp = self._dr_rate_kd_nominal[0].expand(n)
            kd_y = self._dr_rate_kd_nominal[2].expand(n)

        rate_kp = torch.stack([kp_rp, kp_rp, kp_y], dim=1)
        rate_ki = torch.stack([ki_rp, ki_rp, ki_y], dim=1)
        rate_kd = torch.stack([kd_rp, kd_rp, kd_y], dim=1)

        self._dr_k_eta[env_ids] = k_eta
        self._dr_k_m[env_ids] = k_m
        self._dr_tau[env_ids] = tau_m
        self._dr_k_aero_xy[env_ids] = k_aero_xy
        self._dr_k_aero_z[env_ids] = k_aero_z
        self._dr_rate_kp[env_ids] = rate_kp
        self._dr_rate_ki[env_ids] = rate_ki
        self._dr_rate_kd[env_ids] = rate_kd

        k_aero = torch.stack([k_aero_xy, k_aero_xy, k_aero_z], dim=1)
        self._propellers.set_params(env_ids, k_eta=k_eta, k_m=k_m, tau_m=tau_m, k_aero=k_aero)
        self._action_wrapper.pid.set_rate_gains(rate_kp=rate_kp, rate_ki=rate_ki, rate_kd=rate_kd, env_ids=env_ids)
        if self._baseline_controller is not None:
            self._baseline_controller.set_rate_gains(
                rate_kp=rate_kp,
                rate_ki=rate_ki,
                rate_kd=rate_kd,
                env_ids=env_ids,
            )

    def _apply_mass_inertia(
        self,
        env_ids: torch.Tensor,
        mass_scale: torch.Tensor,
        inertia_scale: torch.Tensor,
    ) -> None:
        env_ids_cpu = env_ids.to(device="cpu", dtype=torch.int)
        mass_scale_cpu = mass_scale.detach().to("cpu").view(-1, 1)
        inertia_scale_cpu = inertia_scale.detach().to("cpu").view(-1, 1, 1)
        self._dr_masses[env_ids_cpu] = self._dr_default_masses[env_ids_cpu] * mass_scale_cpu
        self._dr_inertias[env_ids_cpu] = self._dr_default_inertias[env_ids_cpu] * inertia_scale_cpu
        self._robot.root_physx_view.set_masses(self._dr_masses, env_ids_cpu)
        self._robot.root_physx_view.set_inertias(self._dr_inertias, env_ids_cpu)

    # ---------------------------------------------------------------------
    # Visualization helpers
    # ---------------------------------------------------------------------

    def _setup_visualizers(self) -> None:
        if not (self.cfg.debug_vis and self.cfg.debug_visualizer):
            return

        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        ref_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/PosTracking/Reference",
            markers={
                "frame": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.35, 0.35, 0.35),
                ),
                "sphere": sim_utils.SphereCfg(
                    radius=0.06,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.1)),
                ),
            },
        )
        self._ref_markers = VisualizationMarkers(ref_cfg)

    def _update_visualizers(self) -> None:
        if not (self.cfg.debug_vis and self.cfg.debug_visualizer and self._ref_markers is not None):
            return

        env_origins = self._terrain.env_origins
        ref_world = self._reference_pos + env_origins
        yaw = self._reference_yaw.squeeze(-1)
        orientations = math_utils.quat_from_euler_xyz(
            torch.zeros_like(yaw),
            torch.zeros_like(yaw),
            yaw,
        )
        scales = torch.ones(self.num_envs, 3, device=self.device)
        self._ref_markers.visualize(translations=ref_world, orientations=orientations, scales=scales)

    def _camera_cfg_with_resolution_limit(self, cam_cfg, cfg: PosTrackingEnvCfg):
        if cfg is None:
            return cam_cfg
        num_views = cfg.scene.num_envs
        cols = math.ceil(math.sqrt(num_views))
        rows = math.ceil(num_views / cols)
        tile_w = cam_cfg.width * cols
        tile_h = cam_cfg.height * rows
        total_px = tile_w * tile_h
        max_px = max(1.0, float(cfg.camera_tiled_max_megapixels)) * 1e6
        if total_px > max_px:
            scale = math.sqrt(max_px / total_px)
            new_w = max(16, int(cam_cfg.width * scale))
            new_h = max(16, int(cam_cfg.height * scale))
            if new_w != cam_cfg.width or new_h != cam_cfg.height:
                cam_cfg.width = new_w
                cam_cfg.height = new_h
        return cam_cfg

    def _set_camera_view(self) -> None:
        if not self.cfg.debug_vis:
            return
        try:
            from isaacsim.core.utils.viewports import set_camera_view
        except Exception:
            return

        env_origins = getattr(self._terrain, "env_origins", None)
        if env_origins is None or env_origins.numel() == 0:
            return

        env_min, _ = torch.min(env_origins, dim=0)
        env_max, _ = torch.max(env_origins, dim=0)
        arena_min = torch.tensor(self.cfg.arena_min, device=self.device, dtype=torch.float32)
        arena_max = torch.tensor(self.cfg.arena_max, device=self.device, dtype=torch.float32)
        world_min = env_min + arena_min
        world_max = env_max + arena_max

        center = 0.5 * (world_min + world_max)
        extent = world_max - world_min
        top_height = max(float(extent[0]), float(extent[1])) * 2.2 + float(extent[2])

        eye = (float(center[0]), float(center[1]), float(center[2]) + top_height)
        target = (float(center[0]), float(center[1]), float(center[2]))
        set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")

    def _maybe_save_camera_images(self) -> None:
        if not (self.cfg.enable_cameras and self.cfg.save_camera_images and self._camera is not None):
            return
        if self._camera_save_stride > 1:
            if int(self.common_step_counter) % self._camera_save_stride != 0:
                return
        images = self._camera.data.output.get("rgb", None)
        if images is None:
            return
        img = images.detach().clone()
        if img.dim() == 5 and img.shape[1] == 1:
            img = img.squeeze(1)
        if img.dim() == 4 and img.shape[1] in (1, 3, 4):
            img = img.permute(0, 2, 3, 1)
        img = img.to(torch.float32)
        if img.max() > 1.0:
            img = img / 255.0
        img = img.clamp(0.0, 1.0)
        out_root = Path(self.cfg.camera_image_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        step_idx = int(self.common_step_counter)
        for env_id in range(img.shape[0]):
            env_dir = out_root / f"env_{env_id:03d}"
            env_dir.mkdir(parents=True, exist_ok=True)
            frame = img[env_id : env_id + 1].cpu()
            save_images_to_file(frame, str(env_dir / f"step_{step_idx:06d}.png"))

    def _spawn_arena_walls(self) -> None:
        span_x = (self.cfg.arena_max[0] - self.cfg.arena_min[0]) + 2 * self.cfg.wall_extra_margin
        span_y = (self.cfg.arena_max[1] - self.cfg.arena_min[1]) + 2 * self.cfg.wall_extra_margin
        height = self.cfg.arena_max[2] - self.cfg.arena_min[2]
        thickness = self.cfg.wall_thickness
        half_thickness = 0.5 * thickness

        center_z_local = 0.5 * (self.cfg.arena_max[2] + self.cfg.arena_min[2])

        x_pos_local = self.cfg.arena_max[0] + self.cfg.wall_extra_margin + half_thickness
        x_neg_local = self.cfg.arena_min[0] - self.cfg.wall_extra_margin - half_thickness
        y_pos_local = self.cfg.arena_max[1] + self.cfg.wall_extra_margin + half_thickness
        y_neg_local = self.cfg.arena_min[1] - self.cfg.wall_extra_margin - half_thickness

        material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.04, 0.04, 0.04))

        wall_x_cfg = sim_utils.CuboidCfg(
            size=(thickness, span_y, height),
            visual_material=material,
            copy_from_source=False,
        )
        wall_y_cfg = sim_utils.CuboidCfg(
            size=(span_x, thickness, height),
            visual_material=material,
            copy_from_source=False,
        )

        for env_id in range(self.scene.cfg.num_envs):
            base = f"/World/envs/env_{env_id}/Walls"
            placements = [
                (f"{base}/WallXPos", (x_pos_local, 0.0, center_z_local)),
                (f"{base}/WallXNeg", (x_neg_local, 0.0, center_z_local)),
                (f"{base}/WallYPos", (0.0, y_pos_local, center_z_local)),
                (f"{base}/WallYNeg", (0.0, y_neg_local, center_z_local)),
            ]

            for path, translation in placements:
                if prim_utils.is_prim_path_valid(path):
                    continue
                cfg = wall_x_cfg if "WallX" in path else wall_y_cfg
                cfg.func(path, cfg, translation=translation)

    def _spawn_arena_pillars(self) -> None:
        if len(self.cfg.pillar_positions_xy) == 0:
            return

        center_z_local = self.cfg.arena_min[2] + 0.5 * self.cfg.pillar_height
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18))
        pillar_cfg = sim_utils.CylinderCfg(
            radius=self.cfg.pillar_radius,
            height=self.cfg.pillar_height,
            visual_material=material,
            copy_from_source=False,
        )

        for env_id in range(self.scene.cfg.num_envs):
            base = f"/World/envs/env_{env_id}/Pillars"
            for pillar_id, (x_local, y_local) in enumerate(self.cfg.pillar_positions_xy):
                path = f"{base}/Pillar{pillar_id}"
                if prim_utils.is_prim_path_valid(path):
                    continue
                pillar_cfg.func(path, pillar_cfg, translation=(x_local, y_local, center_z_local))

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------

    def _find_prop_joints(self, drone: ArticulationData | Articulation) -> list[int]:
        import re

        joint_ids, joint_names = drone.find_joints(["revolute_prop_.*"], preserve_order=True)
        if not joint_ids:
            joint_ids, joint_names = drone.find_joints(".*prop.*", preserve_order=True)
        if not joint_ids:
            return []
        indexed = []
        for joint_id, joint_name in zip(joint_ids, joint_names):
            match = re.search(r"(\d+)$", joint_name)
            if match:
                indexed.append((int(match.group(1)), joint_id))
        if indexed:
            indexed.sort(key=lambda item: item[0])
            joint_ids = [item[1] for item in indexed]
        return joint_ids

    def _update_prop_visuals(self) -> None:
        if not self._prop_joint_ids:
            return
        count = min(len(self._prop_joint_ids), self._propellers.omega.shape[1])
        vis = self._propellers.omega[:, :count].clone()
        if count > 1:
            vis[:, 0::2] *= -1.0
        self._robot.write_joint_velocity_to_sim(vis, joint_ids=self._prop_joint_ids[:count])

    # ---------------------------------------------------------------------
    # Public helpers for benchmarks
    # ---------------------------------------------------------------------

    def get_last_rewards(self) -> torch.Tensor:
        return self._last_rewards

    def get_last_reward_components(self) -> dict[str, torch.Tensor]:
        return self._last_reward_components

    def get_last_done_reasons(self) -> torch.Tensor:
        return self._last_done_reason

    def get_reference_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._reference_pos.clone(), self._reference_yaw.clone()
