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
from isaaclab.sensors.ray_caster import MultiMeshRayCaster, MultiMeshRayCasterCfg, patterns
from isaaclab.sim.views import XformPrimView
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
from source.isaac_pursuit_evasion.dgppo.utils import (
    compute_pos_tracking_safety_costs,
    extract_graph_states_from_flat_obs,
)
from source.isaac_pursuit_evasion.dynamics.propellers import Drone_cfg, Propellers

from .pos_tracking_env_cfg import PosTrackingEnvCfg
from .scenario_pool import load_scenario_pools

EPISODE_STATUS_LABELS = {
    0: "running",
    1: "success",
    2: "low_altitude",
    3: "boundary",
    4: "timeout",
    5: "invalid_state",
    6: "pillar_collision",
}
DONE_REASON_LABELS = EPISODE_STATUS_LABELS
REASON_RUNNING = 0
REASON_SUCCESS = 1
REASON_ALTITUDE = 2
REASON_BOUNDARY = 3
REASON_TIMEOUT = 4
REASON_INVALID = 5
REASON_PILLAR = 6


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class PosTrackingEnv(DirectRLEnv):
    """Single-drone position and yaw tracking environment."""

    cfg: PosTrackingEnvCfg
    EPISODE_STATUS_MAP = EPISODE_STATUS_LABELS
    DONE_REASON_MAP = EPISODE_STATUS_MAP

    def __init__(self, cfg: PosTrackingEnvCfg, **kwargs) -> None:
        valid_obstacle_modes = {"pillars", "ray_caster", "none"}
        if cfg.obstacle_observation_mode not in valid_obstacle_modes:
            raise ValueError(
                f"Unsupported obstacle_observation_mode '{cfg.obstacle_observation_mode}'. "
                f"Expected one of {sorted(valid_obstacle_modes)}."
            )
        valid_ray_modes = {"ray_ordered_hits", "top_k_hits"}
        if cfg.ray_caster_observation_mode not in valid_ray_modes:
            raise ValueError(
                f"Unsupported ray_caster_observation_mode '{cfg.ray_caster_observation_mode}'. "
                f"Expected one of {sorted(valid_ray_modes)}."
            )
        valid_ray_data = {"xy", "distances"}
        if cfg.ray_caster_observation_data not in valid_ray_data:
            raise ValueError(
                f"Unsupported ray_caster_observation_data '{cfg.ray_caster_observation_data}'. "
                f"Expected one of {sorted(valid_ray_data)}."
            )
        if cfg.enable_pursuit_evasion_curriculum:
            cfg.enable_walls = True
            cfg.enable_pillars = True
            cfg.ref_update_interval_s = 0.0
        cfg.enable_ray_caster = bool(cfg.enable_ray_caster or cfg.obstacle_observation_mode == "ray_caster")
        self._prepare_initial_obstacle_slots(cfg)
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

        self._ray_caster_cfg = self._make_ray_caster_cfg(cfg) if cfg.enable_ray_caster else None

        super().__init__(cfg, **kwargs)

        self._camera = self.scene.sensors.get("robot_camera") if self.cfg.enable_cameras else None
        self._ray_caster = self.scene.sensors.get("lidar_ray_caster") if self.cfg.enable_ray_caster else None
        self._render_obstacles = bool(self.sim.has_gui() or self.cfg.enable_cameras)
        self._pursuit_enabled = bool(self.cfg.enable_pursuit_evasion_curriculum)
        self._max_static_obstacles = self._configured_static_obstacle_slots(self.cfg)
        self._max_dynamic_obstacles = self._configured_dynamic_obstacle_slots(self.cfg)
        self._spawned_static_obstacles = self._configured_spawned_static_obstacles(self.cfg)
        self._spawned_dynamic_obstacles = self._configured_spawned_dynamic_obstacles(self.cfg)
        self._static_obstacle_view: XformPrimView | None = None
        self._dynamic_obstacle_view: XformPrimView | None = None
        self._static_obstacle_view_indices: list[list[int]] = []
        self._dynamic_obstacle_view_indices: list[list[int]] = []
        self._static_obstacle_view_index = torch.full(
            (self.num_envs, self._max_static_obstacles), -1, device=self.device, dtype=torch.long
        )
        self._dynamic_obstacle_view_index = torch.full(
            (self.num_envs, self._max_dynamic_obstacles), -1, device=self.device, dtype=torch.long
        )
        self._setup_obstacle_views()
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
        self._outer_limit_min = self._arena_min.clone()
        self._outer_limit_max = self._arena_max.clone()
        altitude_margin = float(self.cfg.altitude_outer_margin)
        self._outer_limit_min[2] -= altitude_margin
        self._outer_limit_max[2] += altitude_margin
        if self.cfg.enable_walls:
            wall = float(self.cfg.wall_extra_margin + self.cfg.wall_thickness)
            self._outer_limit_min[:2] -= wall
            self._outer_limit_max[:2] += wall

        self._pillar_radius = float(self.cfg.pillar_radius)
        self._pillar_collision_radius = float(self.cfg.pillar_radius + self.cfg.drone_collision_radius)
        self._pillar_top_z = float(self.cfg.arena_min[2] + self.cfg.pillar_height)
        if len(self.cfg.pillar_positions_xy) > 0:
            self._pillar_positions_xy = torch.tensor(
                self.cfg.pillar_positions_xy, device=self.device, dtype=torch.float32
            )
        else:
            self._pillar_positions_xy = torch.zeros((0, 2), device=self.device, dtype=torch.float32)
        self._num_pillars = self._max_static_obstacles if self._pursuit_enabled else int(self._pillar_positions_xy.shape[0])
        self._reference_pillar_clearance = float(self._pillar_collision_radius + self.cfg.reference_obstacle_clearance)

        self._static_obstacle_positions_xy = torch.zeros(
            self.num_envs, self._max_static_obstacles, 2, device=self.device
        )
        self._static_obstacle_active = torch.zeros(
            self.num_envs, self._max_static_obstacles, dtype=torch.bool, device=self.device
        )
        if self._pillar_positions_xy.shape[0] > 0 and self._max_static_obstacles > 0:
            n = min(int(self._pillar_positions_xy.shape[0]), self._max_static_obstacles)
            self._static_obstacle_positions_xy[:, :n] = self._pillar_positions_xy[:n].unsqueeze(0)
            self._static_obstacle_active[:, :n] = True

        self._path_steps = int(self.max_episode_length) + 1
        waypoint_dt = max(self._step_dt, float(self.cfg.pursuit_path_waypoint_dt))
        self._path_waypoint_stride = max(1, int(round(waypoint_dt / max(self._step_dt, 1e-6))))
        last_path_step = max(0, self._path_steps - 1)
        self._path_waypoint_count = max(2, int(math.ceil(last_path_step / self._path_waypoint_stride)) + 1)
        self._path_waypoint_steps = torch.arange(
            self._path_waypoint_count, device=self.device, dtype=torch.long
        ) * self._path_waypoint_stride
        self._path_waypoint_steps[-1] = last_path_step
        self._evader_waypoints = torch.zeros(self.num_envs, self._path_waypoint_count, 3, device=self.device)
        self._current_evader_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._evader_path_type = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._pursuer_start_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._scenario_phase = torch.ones(self.num_envs, dtype=torch.int64, device=self.device)
        self._scenario_fallback = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._scenario_fallback_count = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._dynamic_obstacle_waypoints = torch.zeros(
            self.num_envs, self._path_waypoint_count, self._max_dynamic_obstacles, 3, device=self.device
        )
        self._dynamic_obstacle_positions = torch.zeros(
            self.num_envs, self._max_dynamic_obstacles, 3, device=self.device
        )
        self._dynamic_obstacle_active = torch.zeros(
            self.num_envs, self._max_dynamic_obstacles, dtype=torch.bool, device=self.device
        )
        self._dynamic_collision_radius = float(
            self.cfg.pursuit_dynamic_obstacle_radius + self.cfg.drone_collision_radius
        )

        self._graph_state_dim = self._compute_graph_state_dim(self.cfg)
        self._num_obstacle_obs = self._compute_obstacle_obs_count(self.cfg)
        self._obstacle_obs_dim = self._compute_obstacle_obs_dim(self.cfg)

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
        self._last_success_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._last_rewards = torch.zeros(self.num_envs, device=self.device)
        self._last_reward_components: dict[str, torch.Tensor] = {}
        self._last_body_rates = torch.zeros(self.num_envs, 3, device=self.device)
        self._dgppo_reward_aux = {
            "body_rates": torch.zeros_like(self._last_body_rates),
            "action_diff": torch.zeros_like(self._actions),
        }
        self._last_episode_status = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._episode_status_counts = torch.zeros(len(EPISODE_STATUS_LABELS), dtype=torch.int64, device=self.device)
        self._last_step_snapshot: dict[str, torch.Tensor] = {}

        self._last_position_error = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self._body_x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)

        self._ref_markers: VisualizationMarkers | None = None
        self._setup_visualizers()

        self._dr_cfg = self.cfg.domain_randomization
        self._init_domain_randomization()

        self._scenario_pools_cpu: dict[int, dict[str, torch.Tensor]] = {}
        self._scenario_pools: dict[int, dict[str, torch.Tensor]] = {}
        if self._pursuit_enabled:
            self._scenario_pools_cpu = load_scenario_pools(
                self.cfg,
                self._path_waypoint_steps,
                self._step_dt,
            )

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

        if self.cfg.enable_ray_caster and self._ray_caster_cfg is not None:
            self._ray_caster_cfg.prim_path = f"{self.scene.env_regex_ns}/Robot/body"
            self.scene.sensors["lidar_ray_caster"] = MultiMeshRayCaster(self._ray_caster_cfg)

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
        if self.cfg.enable_pursuit_evasion_curriculum:
            self._spawn_dynamic_obstacles()

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

        self._last_body_rates = self._robot.data.root_ang_vel_b.detach().clone()
        self._dgppo_reward_aux = {
            "body_rates": self._last_body_rates,
            "action_diff": self._action_diff.detach().clone(),
        }

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

        # Agent state: pos (3) + vel (3) + body-to-world rotation matrix (9)
        agent_pos = self._robot.data.root_pos_w - env_origins  # (E, 3)
        agent_vel = self._robot.data.root_lin_vel_w             # (E, 3)

        quat = self._robot.data.root_quat_w
        rot_matrix = math_utils.matrix_from_quat(quat)
        agent_rot_body_to_world = rot_matrix.reshape(quat.shape[0], -1)

        agent_state_flat = torch.cat([agent_pos, agent_vel, agent_rot_body_to_world], dim=-1)

        # Goal state: pos (3) + scaled closing velocity (3)
        goal_pos = self._reference_pos                          # (E, 3)
        goal_vel = self._current_evader_vel
        velocity_scale = agent_vel.new_tensor((15.0, 15.0, 5.0))
        closing_velocity = (goal_vel - agent_vel) / velocity_scale

        # Obstacle state, flattened.
        if self.cfg.obstacle_observation_mode == "none":
            obstacle_flat = agent_pos.new_empty(self.num_envs, 0)
        elif self.cfg.obstacle_observation_mode == "ray_caster":
            if self.cfg.ray_caster_observation_data == "distances":
                obstacle_flat = self._get_ray_obstacle_distances(env_origins, agent_pos).reshape(self.num_envs, -1)
            else:
                obstacle_flat = self._get_ray_obstacle_points_xy(env_origins, agent_pos).reshape(self.num_envs, -1)
        elif self._num_pillars > 0:
            positions = self._static_obstacle_positions_xy
            if self._static_obstacle_active.numel() > 0:
                miss = self._ray_miss_xy(agent_pos).unsqueeze(1).expand_as(positions)
                positions = torch.where(self._static_obstacle_active.unsqueeze(-1), positions, miss)
            obstacle_flat = positions.reshape(self.num_envs, -1)  # (E, O*2)
        else:
            obstacle_flat = agent_pos.new_empty(self.num_envs, 0)

        # Layout: [agent_state(S), goal_pos(3), closing_vel(3), obstacle_data]
        obs = torch.cat([agent_state_flat, goal_pos, closing_velocity, obstacle_flat], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        env_origins = self._terrain.env_origins
        pos_local = self._robot.data.root_pos_w - env_origins

        # Position rewards
        pos_error_now = torch.norm(self._reference_pos - pos_local, dim=-1)
        pos_error_old = self._last_position_error
        progress_reward = self.cfg.reward_pos * (pos_error_old - pos_error_now)
        
        approach_reward = self.cfg.reward_approach * (pos_error_now < self.cfg.pos_tolerance).float()
        rewards = progress_reward + approach_reward
        components: dict[str, torch.Tensor] = {
            "pos": progress_reward + approach_reward,
        }

        # Body rates penalties
        ang_vel_b = self._robot.data.root_ang_vel_b
        roll_pitch = torch.sum(ang_vel_b[:, :2] ** 2, dim=-1)
        yaw = ang_vel_b[:, 2] ** 2

        body_roll_pitch_penalty = -self.cfg.reward_body_rates_roll_pitch * roll_pitch
        body_yaw_penalty = -self.cfg.reward_body_rates_yaw * yaw

        rewards += body_roll_pitch_penalty + body_yaw_penalty
        components["body_rates"] = body_roll_pitch_penalty + body_yaw_penalty

        # SIMPLER VERSION:
        # body_rates = self._robot.data.root_ang_vel_b
        # body_rate_pen = self.cfg.reward_body_rates * torch.norm(body_rates, dim=-1)
        # rewards -= body_rate_pen
        # components["body_rates"] = -body_rate_pen

        if self.cfg.flag_yaw_tracking:
            yaw_err, yaw_align = self._yaw_features(self._robot.data.root_quat_w)
            yaw_reward = self.cfg.reward_yaw * yaw_align.squeeze(-1)
            rewards += yaw_reward
            components["yaw"] = yaw_reward
        else:
            components["yaw"] = torch.zeros_like(rewards)

        if self.cfg.flag_penalize_linvel:
            lin_vel = self._robot.data.root_lin_vel_w
            lin_vel_pen = self.cfg.reward_lin_vel * torch.norm(lin_vel, dim=-1)
            rewards -= lin_vel_pen
            components["lin_vel"] = -lin_vel_pen
        else:
            components["lin_vel"] = torch.zeros_like(rewards)

        if self.cfg.flag_action_smoothness_penalty:
            action_delta_sq = self._action_diff**2
            action_delta_rpy = torch.sum(action_delta_sq[:, :3], dim=-1)
            action_delta_thrust = action_delta_sq[:, 3]
            smooth_weight_default = float(getattr(self.cfg, "reward_action_smoothness", 0.0))
            smooth_weight_rpy_cfg = getattr(self.cfg, "reward_action_smoothness_rpy", None)
            smooth_weight_thrust_cfg = getattr(self.cfg, "reward_action_smoothness_thrust", None)
            smooth_weight_rpy = smooth_weight_default if smooth_weight_rpy_cfg is None else float(smooth_weight_rpy_cfg)
            smooth_weight_thrust = (
                smooth_weight_default if smooth_weight_thrust_cfg is None else float(smooth_weight_thrust_cfg)
            )
            action_smoothness_penalty = (
                smooth_weight_rpy * action_delta_rpy
                + smooth_weight_thrust * action_delta_thrust
            )

            rewards -= action_smoothness_penalty
            components["action_smoothness"] = -action_smoothness_penalty
        else:
            components["action_smoothness"] = torch.zeros_like(rewards)

        altitude_limit, xy_limit = self._arena_limit_masks(pos_local)
        pillar_collision = self._pillar_collision_mask(pos_local)
        altitude_pen = self._masked_penalty(rewards, altitude_limit, self.cfg.penalty_altitude_limit)
        xy_boundary_pen = self._masked_penalty(rewards, xy_limit, self.cfg.penalty_xy_boundary)
        pillar_collision_pen = self._masked_penalty(rewards, pillar_collision, self.cfg.penalty_pillar_collision)
        rewards -= altitude_pen + xy_boundary_pen + pillar_collision_pen

        components["altitude_limit"] = -altitude_pen
        components["xy_boundary"] = -xy_boundary_pen
        components["pillar_collision"] = -pillar_collision_pen

        success_reward = float(self.cfg.reward_success) * self._last_success_event.to(rewards.dtype)
        timeout_penalty = float(self.cfg.penalty_timeout) * self.reset_time_outs.to(rewards.dtype)
        rewards += success_reward - timeout_penalty
        components["success"] = success_reward
        components["timeout"] = -timeout_penalty

        self._last_rewards = rewards
        self._last_reward_components = components
        self._last_position_error = pos_error_now.detach().clone()
        self._maybe_save_camera_images()
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        timeout = self.episode_length_buf >= self.max_episode_length
        pos_local = self._clip_agent_root_state_in_sim()

        altitude_limit, xy_limit = self._arena_limit_masks(pos_local)
        pillar_collision = self._pillar_collision_mask(pos_local)
        outer_boundary = self._outer_boundary_mask(pos_local)
        invalid = ~torch.isfinite(self._robot.data.root_state_w).all(dim=-1)

        success = self._update_success_flags(pos_local)
        self._last_success = success
        self._last_success_event = success & (self._success_counter == self._success_hold_steps)

        safety_violation = altitude_limit | xy_limit | pillar_collision
        terminated = invalid.clone()
        if self.cfg.terminate_on_safety_violation:
            terminated |= safety_violation
        if self.cfg.terminate_on_out_of_boundaries:
            terminated |= outer_boundary
        if self.cfg.terminate_on_success:
            terminated |= success

        truncated = timeout & (~terminated)
        self._dgppo_reward_aux["success_event"] = self._last_success_event.detach().clone()
        self._dgppo_reward_aux["timeout"] = truncated.detach().clone()

        episode_status = torch.full((self.num_envs,), REASON_RUNNING, dtype=torch.int32, device=self.device)
        if self.cfg.terminate_on_success:
            episode_status[terminated & success] = REASON_SUCCESS
        if self.cfg.terminate_on_safety_violation:
            episode_status[terminated & altitude_limit] = REASON_ALTITUDE
            episode_status[terminated & xy_limit] = REASON_BOUNDARY
            episode_status[terminated & pillar_collision] = REASON_PILLAR
        if self.cfg.terminate_on_out_of_boundaries:
            episode_status[terminated & outer_boundary] = REASON_BOUNDARY
        episode_status[truncated] = REASON_TIMEOUT
        episode_status[terminated & invalid] = REASON_INVALID
        self._last_episode_status = episode_status
        done_status = episode_status[episode_status != REASON_RUNNING].to(torch.int64)
        if done_status.numel() > 0:
            self._episode_status_counts += torch.bincount(
                done_status,
                minlength=self._episode_status_counts.numel(),
            )
        self._last_step_snapshot = {
            "pos_local": pos_local.detach().clone(),
            "vel_world": self._robot.data.root_lin_vel_w.detach().clone(),
            "quat": self._robot.data.root_quat_w.detach().clone(),
            "ang_vel": self._robot.data.root_ang_vel_b.detach().clone(),
            "ref_pos": self._reference_pos.detach().clone(),
            "ref_yaw": self._reference_yaw.detach().clone(),
            "evader_vel": self._current_evader_vel.detach().clone(),
            "scenario_phase": self._scenario_phase.detach().clone(),
            "scenario_fallback": self._scenario_fallback.detach().clone(),
            "scenario_fallback_count": self._scenario_fallback_count.detach().clone(),
            "altitude_limit": altitude_limit.detach().clone(),
            "xy_limit": xy_limit.detach().clone(),
            "pillar_collision": pillar_collision.detach().clone(),
        }

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]

        if self._pursuit_enabled:
            self._resample_pursuit_scenarios(env_ids)
            default_root_state[:, :3] = self._pursuer_start_pos[env_ids] + self._terrain.env_origins[env_ids]
            default_root_state[:, 3:7] = self._spawn_yaw_quat(env_ids)
        else:
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._clip_agent_root_state_in_sim(env_ids)

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
        self._last_success_event[env_ids] = False
        self._reference_timer[env_ids] = 0.0
        
        pos_local = self._robot.data.root_pos_w[env_ids] - self._terrain.env_origins[env_ids]
        pos_error = torch.norm(
            self._reference_pos[env_ids] - pos_local,
            dim=-1,
        )
        self._last_position_error[env_ids] = pos_error

        if self._pursuit_enabled:
            self._update_pursuit_episode_motion(env_ids)
        else:
            self._resample_reference(env_ids)
        self._apply_domain_randomization(env_ids)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------


    # Graph observation helpers (used by DGPPOAgent._build_graph)
    @property
    def num_agents(self) -> int:
        return 1

    @property
    def n_constraints(self) -> int:
        return 2

    @property
    def cost_components(self) -> tuple[str, ...]:
        return ("vertical_bounds", "ray_obstacle")

    @property
    def graph_obs_layout(self) -> dict:
        """Return the layout of the flat policy-obs vector for graph construction."""
        if self._num_obstacle_obs > 0 and self._obstacle_obs_dim != 2:
            raise RuntimeError(
                "Graph observations need obstacle xy positions. "
                "Use ray_caster_observation_data='xy' for DG-PPO graph runs."
            )
        agent_end = self._graph_state_dim * self.num_agents
        goal_state_dim = self._compute_goal_state_dim(self.cfg)
        goal_end = agent_end + goal_state_dim * self.num_agents
        obstacles_end = goal_end + self._num_obstacle_obs * 2

        return {
            "state_dim"   : self._graph_state_dim,
            "goal_state_dim": goal_state_dim,
            "n_agents"    : self.num_agents,
            "n_obstacles" : self._num_obstacle_obs,
            "agent_end"   : agent_end,
            "goal_end"    : goal_end,
            "obstacles_end"     : obstacles_end,
        }

    def compute_dgppo_costs_from_observation(self, observations: torch.Tensor | dict) -> torch.Tensor:
        """Return DG-PPO safety costs from the same graph observation stored in rollout memory."""
        agent_state, _goal_state, obs_state = self._dgppo_graph_states_from_observation(observations)
        return compute_pos_tracking_safety_costs(
            agent_state=agent_state,
            obs_state=obs_state,
            safe_arena_min=self._arena_min_safe,
            safe_arena_max=self._arena_max_safe,
            obstacle_collision_distance=float(self.cfg.drone_collision_radius),
        )

    def compute_dgppo_reward_from_observation_action(
        self,
        observations: torch.Tensor | dict,
        actions: torch.Tensor,
        reward_aux: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return the DG-PPO reward from rollout data plus snapshotted reward auxiliaries."""
        agent_state, goal_state, _obs_state = self._dgppo_graph_states_from_observation(observations)
        n_envs = agent_state.shape[0]
        action = torch.as_tensor(actions, device=self.device, dtype=agent_state.dtype)
        action = action.reshape(n_envs, -1)
        if action.shape[0] != n_envs:
            raise ValueError(f"DG-PPO reward got {action.shape[0]} action rows for {n_envs} observations.")

        pos_local = agent_state[:, 0, :3]
        goal_pos = goal_state[:, 0, :3]
        pos_error = torch.linalg.vector_norm(goal_pos - pos_local, dim=-1)
        pos_error_squared = pos_error**2
        pos_reward = self.cfg.reward_pos * torch.exp(-self.cfg.reward_pos_scale * pos_error_squared)
        rewards = pos_reward

        if self.cfg.flag_yaw_tracking:
            if agent_state.shape[-1] < 15:
                raise RuntimeError("DG-PPO yaw reward requires a flattened rotation matrix in the graph observation.")
            yaw_cos = agent_state[:, 0, 6]
            yaw_sin = agent_state[:, 0, 9]
            yaw_ref = self._reference_yaw.squeeze(-1).to(device=self.device, dtype=agent_state.dtype)
            yaw_align = yaw_cos * torch.cos(yaw_ref) + yaw_sin * torch.sin(yaw_ref)
            rewards = rewards + float(self.cfg.reward_yaw) * yaw_align

        if self.cfg.flag_penalize_linvel:
            lin_vel = agent_state[:, 0, 3:6]
            rewards = rewards - float(self.cfg.reward_lin_vel) * torch.linalg.vector_norm(lin_vel, dim=-1)

        body_rate_roll_pitch_weight = float(self.cfg.reward_body_rates_roll_pitch)
        body_rate_yaw_weight = float(self.cfg.reward_body_rates_yaw)
        if body_rate_roll_pitch_weight != 0.0 or body_rate_yaw_weight != 0.0:
            if reward_aux is None or "body_rates" not in reward_aux:
                raise RuntimeError("DG-PPO reward recompute requires pre-step 'body_rates' auxiliary data.")
            body_rates = reward_aux["body_rates"].to(device=self.device, dtype=agent_state.dtype).reshape(n_envs, 3)
            roll_pitch = torch.sum(body_rates[:, :2] ** 2, dim=-1)
            yaw = body_rates[:, 2] ** 2
            rewards = rewards - body_rate_roll_pitch_weight * roll_pitch - body_rate_yaw_weight * yaw

        if self.cfg.flag_action_smoothness_penalty:
            if reward_aux is None or "action_diff" not in reward_aux:
                raise RuntimeError("DG-PPO reward recompute requires pre-step 'action_diff' auxiliary data.")
            action_diff = reward_aux["action_diff"].to(device=self.device, dtype=agent_state.dtype).reshape(n_envs, -1)
            if action_diff.shape[1] < 4:
                raise ValueError(f"DG-PPO action_diff needs at least 4 action components, got {action_diff.shape[1]}.")
            action_delta_sq = action_diff**2
            action_delta_rpy = torch.sum(action_delta_sq[:, :3], dim=-1)
            action_delta_thrust = action_delta_sq[:, 3]
            smooth_weight_default = float(getattr(self.cfg, "reward_action_smoothness", 0.0))
            smooth_weight_rpy_cfg = getattr(self.cfg, "reward_action_smoothness_rpy", None)
            smooth_weight_thrust_cfg = getattr(self.cfg, "reward_action_smoothness_thrust", None)
            smooth_weight_rpy = smooth_weight_default if smooth_weight_rpy_cfg is None else float(smooth_weight_rpy_cfg)
            smooth_weight_thrust = (
                smooth_weight_default if smooth_weight_thrust_cfg is None else float(smooth_weight_thrust_cfg)
            )
            rewards = rewards - smooth_weight_rpy * action_delta_rpy - smooth_weight_thrust * action_delta_thrust

        if reward_aux is not None:
            success_event = reward_aux.get("success_event")
            if success_event is not None:
                success_event = success_event.to(device=self.device, dtype=torch.bool).reshape(n_envs)
                rewards = rewards + float(self.cfg.reward_success) * success_event.to(rewards.dtype)

            timeout = reward_aux.get("timeout")
            if timeout is not None:
                timeout = timeout.to(device=self.device, dtype=torch.bool).reshape(n_envs)
                rewards = rewards - float(self.cfg.penalty_timeout) * timeout.to(rewards.dtype)

        return rewards

    def _dgppo_graph_states_from_observation(
        self,
        observations: torch.Tensor | dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(observations, dict):
            observations = observations["policy"]
        observations = torch.as_tensor(observations, device=self.device, dtype=torch.float32)
        return extract_graph_states_from_flat_obs(
            observations,
            self.graph_obs_layout,
            n_agents=self.num_agents,
        )

    @staticmethod
    def _compute_obs_dim(cfg: PosTrackingEnvCfg) -> int:
        # agent state + goal state + obstacle data
        return (
            PosTrackingEnv._compute_graph_state_dim(cfg)
            + PosTrackingEnv._compute_goal_state_dim(cfg)
            + PosTrackingEnv._compute_obstacle_obs_count(cfg) * PosTrackingEnv._compute_obstacle_obs_dim(cfg)
        )

    @staticmethod
    def _compute_graph_state_dim(cfg: PosTrackingEnvCfg) -> int:
        return 15

    @staticmethod
    def _compute_goal_state_dim(cfg: PosTrackingEnvCfg) -> int:
        return 6

    @staticmethod
    def _compute_obstacle_obs_count(cfg: PosTrackingEnvCfg) -> int:
        if cfg.obstacle_observation_mode == "none":
            return 0
        if cfg.obstacle_observation_mode == "ray_caster":
            if cfg.ray_caster_observation_mode == "ray_ordered_hits":
                return max(1, int(cfg.ray_caster_num_rays))
            return max(1, int(cfg.ray_caster_top_k_hits))
        if cfg.obstacle_observation_mode == "pillars" and cfg.enable_pillars:
            if cfg.enable_pursuit_evasion_curriculum:
                return max(1, int(cfg.pursuit_max_static_obstacles))
            return len(cfg.pillar_positions_xy)
        return 0

    @staticmethod
    def _prepare_initial_obstacle_slots(cfg: PosTrackingEnvCfg) -> None:
        if not cfg.enable_pursuit_evasion_curriculum:
            cfg.pursuit_spawn_static_obstacles = len(cfg.pillar_positions_xy) if cfg.enable_pillars else 0
            cfg.pursuit_spawn_dynamic_obstacles = 0
            return

        max_static = max(0, int(cfg.pursuit_max_static_obstacles))
        max_dynamic = max(0, int(cfg.pursuit_max_dynamic_obstacles))
        cfg.pursuit_lazy_obstacle_spawning = False
        cfg.pursuit_spawn_static_obstacles = max_static
        cfg.pursuit_spawn_dynamic_obstacles = max_dynamic

    @staticmethod
    def _configured_static_obstacle_slots(cfg: PosTrackingEnvCfg) -> int:
        if not cfg.enable_pillars:
            return 0
        if cfg.enable_pursuit_evasion_curriculum:
            return max(1, int(cfg.pursuit_max_static_obstacles))
        return len(cfg.pillar_positions_xy)

    @staticmethod
    def _configured_dynamic_obstacle_slots(cfg: PosTrackingEnvCfg) -> int:
        if not cfg.enable_pursuit_evasion_curriculum:
            return 0
        return max(0, int(cfg.pursuit_max_dynamic_obstacles))

    @staticmethod
    def _configured_spawned_static_obstacles(cfg: PosTrackingEnvCfg) -> int:
        if not cfg.enable_pillars:
            return 0
        if not cfg.enable_pursuit_evasion_curriculum:
            return len(cfg.pillar_positions_xy)
        return min(max(0, int(cfg.pursuit_spawn_static_obstacles)), max(0, int(cfg.pursuit_max_static_obstacles)))

    @staticmethod
    def _configured_spawned_dynamic_obstacles(cfg: PosTrackingEnvCfg) -> int:
        if not cfg.enable_pursuit_evasion_curriculum:
            return 0
        return min(max(0, int(cfg.pursuit_spawn_dynamic_obstacles)), max(0, int(cfg.pursuit_max_dynamic_obstacles)))

    @staticmethod
    def _compute_obstacle_obs_dim(cfg: PosTrackingEnvCfg) -> int:
        if cfg.obstacle_observation_mode == "none":
            return 0
        if cfg.obstacle_observation_mode == "ray_caster":
            return 1 if cfg.ray_caster_observation_data == "distances" else 2
        return 2

    @staticmethod
    def _make_ray_caster_cfg(cfg: PosTrackingEnvCfg) -> MultiMeshRayCasterCfg:
        horizontal_span = float(cfg.ray_caster_horizontal_fov_range[1] - cfg.ray_caster_horizontal_fov_range[0])
        num_rays = max(1, int(cfg.ray_caster_num_rays))
        full_circle = abs(abs(horizontal_span) - 360.0) < 1e-6
        horizontal_res = abs(horizontal_span) / (num_rays if full_circle else max(1, num_rays - 1))
        # The policy observation ray hits are computed analytically from the
        # obstacle tensors. The sensor is still useful for yaw-aligned ray
        # directions, but it does not need every obstacle mesh as a target.
        mesh_targets: list[MultiMeshRayCasterCfg.RaycastTargetCfg] = [
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr=cfg.terrain.prim_path,
                is_shared=True,
                track_mesh_transforms=False,
            )
        ]

        return MultiMeshRayCasterCfg(
            prim_path="/World/envs/env_.*/Robot/body",
            update_period=0.0,
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=cfg.ray_caster_offset),
            mesh_prim_paths=mesh_targets,
            ray_alignment="yaw",
            pattern_cfg=patterns.LidarPatternCfg(
                channels=1,
                vertical_fov_range=(0.0, 0.0),
                horizontal_fov_range=cfg.ray_caster_horizontal_fov_range,
                horizontal_res=horizontal_res,
            ),
            max_distance=float(cfg.ray_caster_max_distance),
            debug_vis=bool(cfg.ray_caster_debug_vis),
        )

    def _get_ray_obstacle_points_xy(
        self,
        env_origins: torch.Tensor,
        agent_pos: torch.Tensor,
        *,
        mode: str | None = None,
    ) -> torch.Tensor:
        xy, _dist = self._get_ray_obstacle_hits(env_origins, agent_pos, mode=mode)
        return xy

    def _get_ray_obstacle_distances(
        self,
        env_origins: torch.Tensor,
        agent_pos: torch.Tensor,
        *,
        mode: str | None = None,
    ) -> torch.Tensor:
        _xy, dist = self._get_ray_obstacle_hits(env_origins, agent_pos, mode=mode)
        return dist / float(self.cfg.ray_caster_max_distance)

    def _get_ray_obstacle_hits(
        self,
        env_origins: torch.Tensor,
        agent_pos: torch.Tensor,
        *,
        mode: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._ray_caster is None:
            raise RuntimeError(
                "Ray-caster obstacle observations requested, but the ray-caster sensor is not available."
            )

        mode = mode or self.cfg.ray_caster_observation_mode
        if mode not in {"ray_ordered_hits", "top_k_hits"}:
            raise ValueError(f"Unsupported ray-caster observation mode '{mode}'.")

        want = max(1, int(self.cfg.ray_caster_num_rays))
        if mode == "top_k_hits":
            want = max(1, int(self.cfg.ray_caster_top_k_hits))

        sensor_w = self._ray_caster.data.pos_w
        max_dist = float(self.cfg.ray_caster_max_distance)
        miss_dist = max_dist + 1e3

        inside_obstacle = self._agent_center_inside_ray_obstacle_mask(agent_pos)

        ray_dirs_w = getattr(self._ray_caster, "_ray_directions_w", None)
        if ray_dirs_w is None or ray_dirs_w.ndim != 3:
            raise RuntimeError(
                "Ray-caster obstacle observations requested, but ray directions are not available or have unexpected "
                "shape."
            )

        rays = ray_dirs_w.shape[1]
        if rays < want:
            raise RuntimeError(
                f"Ray-caster produced {rays} rays, but '{mode}' observation requested {want} points."
            )

        ray_dirs_w = ray_dirs_w.to(device=sensor_w.device, dtype=sensor_w.dtype)
        ray_dirs_w = ray_dirs_w / torch.norm(ray_dirs_w, dim=-1, keepdim=True).clamp_min(1e-6)
        miss_w = sensor_w[:, None, :] + ray_dirs_w * miss_dist
        miss_xy = miss_w[..., :2] - env_origins[:, None, :2]

        dist = self._analytic_ray_distances(sensor_w, ray_dirs_w, env_origins, max_dist)
        hits_w = sensor_w[:, None, :] + ray_dirs_w * dist.clamp(max=max_dist).unsqueeze(-1)
        xy = hits_w[..., :2] - env_origins[:, None, :2]
        ok = torch.isfinite(dist) & (dist <= max_dist)
        obs_dist = torch.where(ok, dist, torch.full_like(dist, max_dist))
        xy = torch.where(ok.unsqueeze(-1), xy, miss_xy)
        if inside_obstacle.any():
            current_xy = agent_pos[:, None, :2].expand(-1, rays, -1)
            xy = torch.where(inside_obstacle[:, None, None], current_xy, xy)
            ok = ok | inside_obstacle[:, None]
            dist = torch.where(inside_obstacle[:, None], torch.zeros_like(dist), dist)
            obs_dist = torch.where(inside_obstacle[:, None], torch.zeros_like(obs_dist), obs_dist)

        if mode == "top_k_hits":
            dist = torch.where(ok, dist, torch.full_like(dist, float("inf")))
            k = min(want, rays)
            _, order = torch.topk(dist, k=k, dim=1, largest=False)
            xy = torch.gather(xy, 1, order.unsqueeze(-1).expand(-1, -1, 2))
            obs_dist = torch.gather(obs_dist, 1, order)
        else:
            k = min(want, rays)
            xy = xy[:, :k]
            obs_dist = obs_dist[:, :k]

        return xy, obs_dist

    def _analytic_ray_distances(
        self,
        sensor_w: torch.Tensor,
        ray_dirs_w: torch.Tensor,
        env_origins: torch.Tensor,
        max_dist: float,
    ) -> torch.Tensor:
        inf = float(max_dist) + 1e6
        origin_xy = sensor_w[:, :2] - env_origins[:, :2]
        origin_z = sensor_w[:, 2] - env_origins[:, 2]
        dirs_xy = ray_dirs_w[..., :2]
        dirs_xy = dirs_xy / torch.norm(dirs_xy, dim=-1, keepdim=True).clamp_min(1e-6)
        best = torch.full(dirs_xy.shape[:2], inf, device=self.device, dtype=dirs_xy.dtype)

        if self.cfg.enable_walls:
            best = torch.minimum(best, self._analytic_wall_ray_distances(origin_xy, origin_z, dirs_xy, inf))

        if self.cfg.enable_pillars and self._max_static_obstacles > 0:
            best = torch.minimum(
                best,
                self._analytic_cylinder_ray_distances(
                    origin_xy,
                    origin_z,
                    dirs_xy,
                    self._static_obstacle_positions_xy,
                    self._static_obstacle_active,
                    float(self.cfg.pillar_radius),
                    float(self.cfg.arena_min[2]),
                    float(self.cfg.arena_min[2] + self.cfg.pillar_height),
                    inf,
                ),
            )

        if self._pursuit_enabled and self._max_dynamic_obstacles > 0:
            best = torch.minimum(
                best,
                self._analytic_cylinder_ray_distances(
                    origin_xy,
                    origin_z,
                    dirs_xy,
                    self._dynamic_obstacle_positions[:, :, :2],
                    self._dynamic_obstacle_active,
                    float(self.cfg.pursuit_dynamic_obstacle_radius),
                    float(self.cfg.arena_min[2]),
                    float(self.cfg.arena_min[2] + self.cfg.pursuit_dynamic_obstacle_height),
                    inf,
                ),
            )

        return best

    def _analytic_cylinder_ray_distances(
        self,
        origin_xy: torch.Tensor,
        origin_z: torch.Tensor,
        dirs_xy: torch.Tensor,
        centers_xy: torch.Tensor,
        active: torch.Tensor,
        radius: float,
        z_min: float,
        z_max: float,
        inf: float,
    ) -> torch.Tensor:
        if centers_xy.shape[1] == 0:
            return torch.full(dirs_xy.shape[:2], inf, device=self.device, dtype=dirs_xy.dtype)

        origin = origin_xy[:, None, None, :]
        dirs = dirs_xy[:, :, None, :]
        centers = centers_xy[:, None, :, :]
        oc = origin - centers
        b = torch.sum(dirs * oc, dim=-1)
        c = torch.sum(oc * oc, dim=-1) - float(radius) ** 2
        disc = b * b - c
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
        t0 = -b - sqrt_disc
        t1 = -b + sqrt_disc
        t = torch.where(t0 >= 0.0, t0, t1)

        z_ok = (origin_z >= z_min) & (origin_z <= z_max)
        valid = (disc >= 0.0) & (t >= 0.0) & active[:, None, :] & z_ok[:, None, None]
        t = torch.where(valid, t, torch.full_like(t, inf))
        return torch.amin(t, dim=-1)

    def _analytic_wall_ray_distances(
        self,
        origin_xy: torch.Tensor,
        origin_z: torch.Tensor,
        dirs_xy: torch.Tensor,
        inf: float,
    ) -> torch.Tensor:
        z_ok = (origin_z >= float(self.cfg.arena_min[2])) & (origin_z <= float(self.cfg.arena_max[2]))
        margin = float(self.cfg.wall_extra_margin)
        thickness = float(self.cfg.wall_thickness)
        x_min = float(self.cfg.arena_min[0])
        x_max = float(self.cfg.arena_max[0])
        y_min = float(self.cfg.arena_min[1])
        y_max = float(self.cfg.arena_max[1])
        boxes = (
            (x_max + margin, x_max + margin + thickness, y_min - margin, y_max + margin),
            (x_min - margin - thickness, x_min - margin, y_min - margin, y_max + margin),
            (x_min - margin, x_max + margin, y_max + margin, y_max + margin + thickness),
            (x_min - margin, x_max + margin, y_min - margin - thickness, y_min - margin),
        )

        best = torch.full(dirs_xy.shape[:2], inf, device=self.device, dtype=dirs_xy.dtype)
        for x0, x1, y0, y1 in boxes:
            lo = torch.tensor((x0, y0), device=self.device, dtype=dirs_xy.dtype)
            hi = torch.tensor((x1, y1), device=self.device, dtype=dirs_xy.dtype)
            dist = self._analytic_box_ray_distance(origin_xy, dirs_xy, lo, hi, inf)
            dist = torch.where(z_ok[:, None], dist, torch.full_like(dist, inf))
            best = torch.minimum(best, dist)
        return best

    def _analytic_box_ray_distance(
        self,
        origin_xy: torch.Tensor,
        dirs_xy: torch.Tensor,
        lo: torch.Tensor,
        hi: torch.Tensor,
        inf: float,
    ) -> torch.Tensor:
        origin = origin_xy[:, None, :]
        eps = 1e-8
        parallel = torch.abs(dirs_xy) < eps
        inside_axis = (origin >= lo.view(1, 1, 2)) & (origin <= hi.view(1, 1, 2))
        dirs = torch.where(parallel, torch.ones_like(dirs_xy), dirs_xy)
        t1 = (lo.view(1, 1, 2) - origin) / dirs
        t2 = (hi.view(1, 1, 2) - origin) / dirs
        near = torch.minimum(t1, t2)
        far = torch.maximum(t1, t2)

        near = torch.where(parallel & inside_axis, torch.full_like(near, -inf), near)
        far = torch.where(parallel & inside_axis, torch.full_like(far, inf), far)
        near = torch.where(parallel & ~inside_axis, torch.full_like(near, inf), near)
        far = torch.where(parallel & ~inside_axis, torch.full_like(far, -inf), far)

        t_near = torch.amax(near, dim=-1)
        t_far = torch.amin(far, dim=-1)
        t = torch.where(t_near >= 0.0, t_near, torch.zeros_like(t_near))
        hit = (t_far >= t) & (t_far >= 0.0)
        return torch.where(hit, t, torch.full_like(t, inf))

    def _agent_center_inside_ray_obstacle_mask(self, agent_pos: torch.Tensor) -> torch.Tensor:
        # Two types of obstacles: pillars and walls of the arena.

        inside = torch.zeros(agent_pos.shape[0], dtype=torch.bool, device=agent_pos.device)

        z = agent_pos[:, 2]
        if self.cfg.enable_pillars and self._max_static_obstacles > 0:
            dxy = torch.linalg.vector_norm(agent_pos[:, None, :2] - self._static_obstacle_positions_xy, dim=-1)
            inside_pillar_xy = torch.any((dxy <= self._pillar_radius) & self._static_obstacle_active, dim=1)
            inside_pillar_z = (z >= float(self.cfg.arena_min[2])) & (z <= self._pillar_top_z)
            inside = inside | (inside_pillar_xy & inside_pillar_z)

        if self._pursuit_enabled and self._max_dynamic_obstacles > 0:
            dxy = torch.linalg.vector_norm(agent_pos[:, None, :2] - self._dynamic_obstacle_positions[:, :, :2], dim=-1)
            inside_dynamic_xy = torch.any(
                (dxy <= float(self.cfg.pursuit_dynamic_obstacle_radius)) & self._dynamic_obstacle_active,
                dim=1,
            )
            top_z = float(self.cfg.arena_min[2] + self.cfg.pursuit_dynamic_obstacle_height)
            inside_dynamic_z = (z >= float(self.cfg.arena_min[2])) & (z <= top_z)
            inside = inside | (inside_dynamic_xy & inside_dynamic_z)

        if self.cfg.enable_walls:
            x = agent_pos[:, 0]
            y = agent_pos[:, 1]

            arena_min = self.cfg.arena_min
            arena_max = self.cfg.arena_max
            margin = float(self.cfg.wall_extra_margin)
            thickness = float(self.cfg.wall_thickness)

            def between(v: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
                return (v >= lo) & (v <= hi)

            z_in_wall = between(z, float(arena_min[2]), float(arena_max[2]))

            x_wall_lo = float(arena_min[0]) - margin - thickness
            x_wall_hi = float(arena_max[0]) + margin + thickness
            y_wall_lo = float(arena_min[1]) - margin - thickness
            y_wall_hi = float(arena_max[1]) + margin + thickness

            # The four walls are just four thin slabs around the arena.
            in_left_wall = between(x, x_wall_lo, float(arena_min[0]) - margin)
            in_right_wall = between(x, float(arena_max[0]) + margin, x_wall_hi)
            in_bottom_wall = between(y, y_wall_lo, float(arena_min[1]) - margin)
            in_top_wall = between(y, float(arena_max[1]) + margin, y_wall_hi)

            inside_x_wall = (in_left_wall | in_right_wall) & between(y, y_wall_lo, y_wall_hi)
            inside_y_wall = (in_bottom_wall | in_top_wall) & between(x, x_wall_lo, x_wall_hi)

            inside |= z_in_wall & (inside_x_wall | inside_y_wall)

        return inside

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

    def _yaw_sin_cos(self, quat: torch.Tensor) -> torch.Tensor:
        yaw = math_utils.euler_xyz_from_quat(quat)[2].unsqueeze(-1)
        return torch.cat([torch.sin(yaw), torch.cos(yaw)], dim=-1)

    def _arena_limit_masks(self, pos_local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Inner task limits: altitude is floor/ceiling, xy is the arena wall face.
        altitude = (pos_local[:, 2] < self._arena_min_safe[2]) | (pos_local[:, 2] > self._arena_max_safe[2])
        xy = torch.any(
            (pos_local[:, :2] < self._arena_min_safe[:2]) | (pos_local[:, :2] > self._arena_max_safe[:2]),
            dim=-1,
        )
        return altitude, xy

    def _outer_boundary_mask(self, pos_local: torch.Tensor) -> torch.Tensor:
        # Outer envelope: wall thickness in xy, altitude_outer_margin in z.
        below = pos_local < self._outer_limit_min
        above = pos_local > self._outer_limit_max
        return torch.any(below | above, dim=-1)

    def _pillar_collision_mask(self, pos_local: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enable_pillars:
            return torch.zeros(pos_local.shape[0], dtype=torch.bool, device=self.device)

        xy = pos_local[:, None, :2]
        dxy = torch.linalg.vector_norm(xy - self._static_obstacle_positions_xy, dim=-1)
        inside_radius = (dxy <= self._pillar_collision_radius) & self._static_obstacle_active
        inside_height = (pos_local[:, 2] >= self.cfg.arena_min[2]) & (pos_local[:, 2] <= self._pillar_top_z)
        hit_static = torch.any(inside_radius, dim=1) & inside_height

        if not self._pursuit_enabled or self._max_dynamic_obstacles == 0:
            return hit_static

        dyn_dxy = torch.linalg.vector_norm(xy - self._dynamic_obstacle_positions[:, :, :2], dim=-1)
        dyn_inside = (dyn_dxy <= self._dynamic_collision_radius) & self._dynamic_obstacle_active
        dyn_top_z = float(self.cfg.arena_min[2] + self.cfg.pursuit_dynamic_obstacle_height)
        dyn_height = (pos_local[:, 2] >= self.cfg.arena_min[2]) & (pos_local[:, 2] <= dyn_top_z)
        return hit_static | (torch.any(dyn_inside, dim=1) & dyn_height)

    def _clip_agent_root_state_in_sim(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        origins = self._terrain.env_origins[env_ids]
        state = self._robot.data.root_state_w[env_ids].clone()
        pos = state[:, :3] - origins

        if not self.cfg.enable_clip_states:
            return pos

        clipped_pos = torch.clamp(pos, min=self._outer_limit_min, max=self._outer_limit_max)
        finite = torch.isfinite(pos).all(dim=-1)
        changed = finite & torch.any(pos != clipped_pos, dim=-1)
        if not changed.any():
            return pos

        changed_ids = env_ids[changed]

        pose = state[changed, :7].clone()
        pose[:, :3] = clipped_pos[changed] + self._terrain.env_origins[changed_ids]
        self._robot.write_root_pose_to_sim(pose, changed_ids)

        vel = state[changed, 7:].clone()
        vel[:, :3] = self._clip_boundary_velocity(pos[changed], vel[:, :3])
        self._robot.write_root_velocity_to_sim(vel, changed_ids)

        if self._ray_caster is not None:
            self._ray_caster._is_outdated[changed_ids] = True

        pos[changed] = clipped_pos[changed]
        return pos

    def _clip_boundary_velocity(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        lower_hit = pos < self._outer_limit_min
        upper_hit = pos > self._outer_limit_max

        vel = torch.where(lower_hit & (vel < 0.0), torch.zeros_like(vel), vel)
        vel = torch.where(upper_hit & (vel > 0.0), torch.zeros_like(vel), vel)
        return vel

    @staticmethod
    def _masked_penalty(reference: torch.Tensor, mask: torch.Tensor, weight: float) -> torch.Tensor:
        penalty = torch.zeros_like(reference)
        penalty[mask] = float(weight)
        return penalty

    def _update_success_flags(self, pos_local: torch.Tensor) -> torch.Tensor:
        pos_error = torch.norm(self._reference_pos - pos_local, dim=-1)
        within = pos_error < self.cfg.pos_tolerance
        if self.cfg.flag_yaw_tracking:
            yaw_err, _ = self._yaw_features(self._robot.data.root_quat_w)
            within = within & (torch.abs(yaw_err.squeeze(-1)) < self.cfg.yaw_tolerance)

        self._success_counter = torch.where(within, self._success_counter + 1, torch.zeros_like(self._success_counter))
        return self._success_counter >= self._success_hold_steps

    def _maybe_update_reference(self) -> None:
        if self._pursuit_enabled:
            self._update_pursuit_episode_motion()
            return
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
            safety_radius = self._reference_pillar_clearance
            # Tiny offset avoids floating-point ties at exactly safety_radius.
            safe_radius = safety_radius + 0.05
            for _ in range(12):
                dxy = torch.cdist(pos[:, :2], self._pillar_positions_xy)
                colliding = torch.any(dxy <= safety_radius, dim=1)
                if not colliding.any():
                    break
                count = int(colliding.sum().item())
                resample = torch.rand(count, 3, device=self.device)
                pos[colliding] = low + (high - low) * resample

            # Deterministic fallback: iteratively project away from the nearest violating pillar.
            default_dirs = torch.tensor([1.0, 0.0], device=self.device, dtype=torch.float32).view(1, 2)
            for _ in range(24):
                dxy = torch.cdist(pos[:, :2], self._pillar_positions_xy)
                colliding = torch.any(dxy <= safety_radius, dim=1)
                if not colliding.any():
                    break
                nearest_idx = torch.argmin(dxy[colliding], dim=1)
                nearest_pillars = self._pillar_positions_xy[nearest_idx]
                vectors = pos[colliding, :2] - nearest_pillars
                norms = torch.norm(vectors, dim=1, keepdim=True)
                unit = torch.where(
                    norms > 1e-6,
                    vectors / norms.clamp_min(1e-6),
                    default_dirs.expand(vectors.shape[0], -1),
                )
                projected = nearest_pillars + unit * safe_radius
                pos[colliding, :2] = torch.clamp(projected, min=low[:2], max=high[:2])

            final_dxy = torch.cdist(pos[:, :2], self._pillar_positions_xy)
            final_colliding = torch.any(final_dxy <= safety_radius, dim=1)
            if final_colliding.any():
                # Last resort: sample XY from a dense random candidate set and pick known-safe points.
                candidate_count = 4096
                candidate_xy = low[:2] + (high[:2] - low[:2]) * torch.rand(candidate_count, 2, device=self.device)
                candidate_dxy = torch.cdist(candidate_xy, self._pillar_positions_xy)
                safe_candidates = candidate_xy[torch.all(candidate_dxy > safety_radius, dim=1)]
                if safe_candidates.shape[0] == 0:
                    raise RuntimeError(
                        "Reference sampling failed: no feasible XY location satisfies pillar clearance in current "
                        "bounds."
                    )
                count = int(final_colliding.sum().item())
                pick_idx = torch.randint(0, safe_candidates.shape[0], (count,), device=self.device)
                pos[final_colliding, :2] = safe_candidates[pick_idx]

                final_dxy = torch.cdist(pos[:, :2], self._pillar_positions_xy)
                final_colliding = torch.any(final_dxy <= safety_radius, dim=1)
                if final_colliding.any():
                    raise RuntimeError("Reference sampling failed: some goals still intersect pillar clearance.")
        yaw = torch.empty(env_ids.shape[0], 1, device=self.device).uniform_(self._ref_yaw_min, self._ref_yaw_max)
        self._reference_pos[env_ids] = pos
        self._reference_yaw[env_ids] = yaw

    # ---------------------------------------------------------------------
    # Pursuit-evasion episode generation
    # ---------------------------------------------------------------------

    def _setup_obstacle_views(self) -> None:
        self._static_obstacle_view = None
        self._dynamic_obstacle_view = None
        self._static_obstacle_view_indices = []
        self._dynamic_obstacle_view_indices = []
        self._static_obstacle_view_index.fill_(-1)
        self._dynamic_obstacle_view_index.fill_(-1)

        if self._spawned_static_obstacles > 0 and self.cfg.enable_pillars:
            self._static_obstacle_view = XformPrimView(
                "/World/envs/env_.*/Pillars/Pillar.*",
                device=self.device,
                validate_xform_ops=False,
            )
            self._static_obstacle_view_index[:, : self._spawned_static_obstacles] = self._obstacle_view_indices(
                self._static_obstacle_view,
                "Pillar",
                self._spawned_static_obstacles,
            )
            self._static_obstacle_view_indices = self._static_obstacle_view_index.detach().cpu().tolist()

        if self._spawned_dynamic_obstacles > 0:
            self._dynamic_obstacle_view = XformPrimView(
                "/World/envs/env_.*/DynamicObstacles/Dynamic.*",
                device=self.device,
                validate_xform_ops=False,
            )
            self._dynamic_obstacle_view_index[:, : self._spawned_dynamic_obstacles] = self._obstacle_view_indices(
                self._dynamic_obstacle_view,
                "Dynamic",
                self._spawned_dynamic_obstacles,
            )
            self._dynamic_obstacle_view_indices = self._dynamic_obstacle_view_index.detach().cpu().tolist()

    def _obstacle_view_indices(self, view: XformPrimView, prefix: str, slots: int) -> torch.Tensor:
        indices = torch.full((self.num_envs, slots), -1, dtype=torch.long, device=self.device)
        for idx, path in enumerate(view.prim_paths):
            env_id = -1
            slot_id = -1
            for part in path.split("/"):
                if part.startswith("env_"):
                    env_id = int(part[4:])
                elif part.startswith(prefix) and part[len(prefix) :].isdigit():
                    slot_id = int(part[len(prefix) :])
            if 0 <= env_id < self.num_envs and 0 <= slot_id < slots:
                indices[env_id, slot_id] = idx
        if torch.any(indices < 0):
            raise RuntimeError(f"Obstacle view for {prefix} did not find all {self.num_envs} x {slots} prims.")
        return indices

    def _resample_pursuit_scenarios(self, env_ids: torch.Tensor) -> None:
        """Assign fresh pooled layouts and episode parameters to resetting environments."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        phases = self._sample_curriculum_phases(env_ids.numel())
        needed_phases = set(torch.unique(phases).tolist())
        for phase in tuple(self._scenario_pools):
            if phase not in needed_phases:
                del self._scenario_pools[phase]
        for phase in needed_phases:
            if phase not in self._scenario_pools:
                self._scenario_pools[phase] = {
                    name: tensor.to(self.device)
                    for name, tensor in self._scenario_pools_cpu[phase].items()
                }

        for phase in needed_phases:
            pool = self._scenario_pools[phase]
            phase_mask = phases == phase
            phase_env_ids = env_ids[phase_mask]
            count = phase_env_ids.numel()
            if count == 0:
                continue

            pool_rows = torch.randint(0, pool["path_type"].shape[0], (count,), device=self.device)
            speed = self._sample_evader_episode_speeds(count)
            z = self._sample_evader_zs(count)
            step_dist = self._evader_step_distances_batch(speed)
            waypoints = self._grid_polylines_waypoints(pool["evader_xy"][pool_rows], step_dist, z)

            candidate_count = pool["pursuer_candidate_count"][pool_rows]
            candidate_pick = torch.floor(torch.rand(count, device=self.device) * candidate_count).to(torch.long)
            candidate_xy = pool["pursuer_candidates_xy"][pool_rows, candidate_pick]

            self._scenario_phase[phase_env_ids] = phase
            self._scenario_fallback[phase_env_ids] = False
            self._evader_path_type[phase_env_ids] = pool["path_type"][pool_rows]
            self._evader_waypoints[phase_env_ids] = waypoints
            self._pursuer_start_pos[phase_env_ids, :2] = candidate_xy
            self._pursuer_start_pos[phase_env_ids, 2] = z
            self._reference_pos[phase_env_ids] = waypoints[:, 0]
            self._reference_yaw[phase_env_ids, 0] = self._first_waypoint_yaws(waypoints)

            self._static_obstacle_positions_xy[phase_env_ids] = pool["static_xy"][pool_rows]
            self._static_obstacle_active[phase_env_ids] = pool["static_active"][pool_rows]
            self._dynamic_obstacle_waypoints[phase_env_ids] = pool["dynamic_waypoints"][pool_rows]
            self._dynamic_obstacle_active[phase_env_ids] = pool["dynamic_active"][pool_rows]
            self._dynamic_obstacle_positions[phase_env_ids] = pool["dynamic_waypoints"][pool_rows, 0]

        self._move_static_obstacles(env_ids)
        self._move_dynamic_obstacles(env_ids)
        self._refresh_debug_pillars()

    def _grid_polylines_waypoints(
        self,
        xy: torch.Tensor,
        step_dist: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = xy.shape[0]
        waypoints = torch.zeros(batch_size, self._path_waypoint_count, 3, device=self.device)
        seg_len = torch.linalg.vector_norm(xy[:, 1:] - xy[:, :-1], dim=-1).clamp_min(1e-6)
        cumulative = torch.cat(
            (torch.zeros(batch_size, 1, device=self.device), torch.cumsum(seg_len, dim=1)),
            dim=1,
        )
        target = torch.cat(
            (torch.zeros(batch_size, 1, device=self.device), torch.cumsum(step_dist, dim=1)),
            dim=1,
        )
        target = torch.minimum(target, cumulative[:, -1:])
        seg = torch.searchsorted(cumulative[:, 1:].contiguous(), target.contiguous())
        seg = seg.clamp(max=seg_len.shape[1] - 1)
        start = torch.gather(xy, 1, seg.unsqueeze(-1).expand(-1, -1, 2))
        end = torch.gather(xy, 1, (seg + 1).unsqueeze(-1).expand(-1, -1, 2))
        start_dist = torch.gather(cumulative, 1, seg)
        length = torch.gather(seg_len, 1, seg)
        tau = ((target - start_dist) / length).unsqueeze(-1)
        waypoints[:, :, :2] = start * (1.0 - tau) + end * tau
        waypoints[:, :, 2] = z.view(-1, 1)
        return waypoints

    def _first_waypoint_yaws(self, waypoints: torch.Tensor) -> torch.Tensor:
        dt_steps = (self._path_waypoint_steps[1:] - self._path_waypoint_steps[:-1]).to(waypoints.dtype)
        velocity = (waypoints[:, 1:, :2] - waypoints[:, :-1, :2]) / (
            dt_steps.view(1, -1, 1) * self._step_dt
        ).clamp_min(self._step_dt)
        moving = torch.linalg.vector_norm(velocity, dim=-1) > 0.05
        first = torch.argmax(moving.to(torch.int64), dim=1)
        selected = torch.gather(velocity, 1, first.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
        yaw = torch.atan2(selected[:, 1], selected[:, 0])
        return torch.where(moving.any(dim=1), yaw, torch.zeros_like(yaw))

    def _curriculum_progress(self) -> float:
        total = int(self.cfg.pursuit_curriculum_total_steps)
        if total <= 0:
            total = int(getattr(self.cfg, "total_timesteps", 1))
        return min(1.0, max(0.0, float(self.common_step_counter) / max(1, total)))

    def _sample_curriculum_phases(self, count: int) -> torch.Tensor:
        count = max(0, int(count))
        if count == 0:
            return torch.empty(0, device=self.device, dtype=torch.long)

        progress = self._curriculum_progress()
        fractions = self.cfg.pursuit_curriculum_phase_fractions
        if not fractions:
            return torch.ones(count, device=self.device, dtype=torch.long)

        end = 0.0
        for phase, fraction in enumerate(fractions, start=1):
            fraction = max(0.0, float(fraction))
            end += fraction
            if fraction == 0.0:
                continue
            if progress <= end:
                blend_ratio = max(0.0, min(1.0, float(self.cfg.pursuit_curriculum_blend_fraction)))
                blend_width = fraction * blend_ratio
                next_phase = next(
                    (
                        idx
                        for idx in range(phase + 1, len(fractions) + 1)
                        if float(fractions[idx - 1]) > 0.0
                    ),
                    None,
                )
                if blend_width > 0.0 and next_phase is not None:
                    prob_next = max(0.0, min(1.0, (progress - (end - blend_width)) / blend_width))
                    draws = torch.rand(count, device=self.device)
                    return torch.where(
                        draws < prob_next,
                        torch.full_like(draws, next_phase, dtype=torch.long),
                        torch.full_like(draws, phase, dtype=torch.long),
                    )
                return torch.full((count,), phase, device=self.device, dtype=torch.long)
        enabled = [phase for phase, fraction in enumerate(fractions, start=1) if float(fraction) > 0.0]
        return torch.full((count,), enabled[-1] if enabled else 1, device=self.device, dtype=torch.long)

    def _sample_evader_episode_speeds(self, count: int) -> torch.Tensor:
        speed_range = getattr(self.cfg, "pursuit_evader_speed_range", None)
        if speed_range is None:
            speed = max(0.0, float(getattr(self.cfg, "pursuit_evader_speed", 0.0)))
            return torch.full((count,), speed, device=self.device)
        lo = max(0.0, float(speed_range[0]))
        hi = max(0.0, float(speed_range[1]))
        if hi < lo:
            lo, hi = hi, lo
        if hi <= lo:
            return torch.full((count,), lo, device=self.device)
        return torch.empty(count, device=self.device).uniform_(lo, hi)

    def _evader_step_distances_batch(self, speed: torch.Tensor) -> torch.Tensor:
        dt_steps = (self._path_waypoint_steps[1:] - self._path_waypoint_steps[:-1]).to(torch.float32)
        return speed.view(-1, 1) * dt_steps.view(1, -1) * self._step_dt

    def _sample_evader_zs(self, count: int) -> torch.Tensor:
        clearance = float(self.cfg.pursuit_evader_wall_clearance)
        lo = self._arena_min_safe[2] + clearance
        hi = self._arena_max_safe[2] - clearance
        if float(hi) <= float(lo):
            lo = self._arena_min_safe[2]
            hi = self._arena_max_safe[2]
        return lo + (hi - lo) * torch.rand(count, device=self.device)

    def _update_pursuit_episode_motion(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        step_ids = torch.clamp(self.episode_length_buf[env_ids], 0, self._path_steps - 1).to(torch.long)
        pos, vel = self._eval_evader_waypoints(env_ids, step_ids)
        speed_xy = torch.linalg.vector_norm(vel[:, :2], dim=-1)
        yaw = torch.atan2(vel[:, 1], vel[:, 0]).view(-1, 1)
        self._reference_pos[env_ids] = pos
        self._current_evader_vel[env_ids] = vel
        self._reference_yaw[env_ids] = torch.where(speed_xy.view(-1, 1) > 0.05, yaw, self._reference_yaw[env_ids])
        if self._max_dynamic_obstacles > 0:
            active_env_ids = env_ids[torch.any(self._dynamic_obstacle_active[env_ids], dim=1)]
            if active_env_ids.numel() > 0:
                active_steps = torch.clamp(self.episode_length_buf[active_env_ids], 0, self._path_steps - 1).to(torch.long)
                self._dynamic_obstacle_positions[active_env_ids], _ = self._eval_dynamic_obstacle_waypoints(
                    active_env_ids, active_steps
                )
                self._move_dynamic_obstacles(active_env_ids)

    def _spawn_yaw_quat(self, env_ids: torch.Tensor) -> torch.Tensor:
        delta = self._reference_pos[env_ids] - self._pursuer_start_pos[env_ids]
        yaw = torch.atan2(delta[:, 1], delta[:, 0])
        zeros = torch.zeros_like(yaw)
        return math_utils.quat_from_euler_xyz(zeros, zeros, yaw)

    def _move_static_obstacles(self, env_ids: torch.Tensor) -> None:
        if not self._render_obstacles:
            return
        slots = self._spawned_static_obstacles
        if self._static_obstacle_view is None or slots == 0:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        active = self._static_obstacle_active[env_ids, :slots]
        xy = self._static_obstacle_positions_xy[env_ids, :slots]
        inactive = self._inactive_obstacle_xy(slots).view(1, -1, 2)
        xy = torch.where(active.unsqueeze(-1), xy, inactive.expand_as(xy))
        pos = torch.zeros(env_ids.shape[0], slots, 3, device=self.device)
        pos[:, :, :2] = xy
        pos[:, :, 2] = self._static_center_z()
        pos = pos + self._terrain.env_origins[env_ids].view(-1, 1, 3)
        env_id_list = env_ids.detach().cpu().tolist()
        indices = [idx for env_id in env_id_list for idx in self._static_obstacle_view_indices[env_id][:slots]]
        self._static_obstacle_view.set_world_poses(pos.reshape(-1, 3), indices=indices)
        self._mark_ray_caster_outdated(env_ids)

    def _move_dynamic_obstacles(self, env_ids: torch.Tensor) -> None:
        if not self._render_obstacles:
            return
        slots = self._spawned_dynamic_obstacles
        if self._dynamic_obstacle_view is None or slots == 0:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        active = self._dynamic_obstacle_active[env_ids, :slots]
        pos = self._dynamic_obstacle_positions[env_ids, :slots]
        inactive = self._inactive_dynamic_pos(slots).view(1, -1, 3)
        pos = torch.where(active.unsqueeze(-1), pos, inactive.expand_as(pos))
        pos = pos + self._terrain.env_origins[env_ids].view(-1, 1, 3)
        env_id_list = env_ids.detach().cpu().tolist()
        indices = [idx for env_id in env_id_list for idx in self._dynamic_obstacle_view_indices[env_id][:slots]]
        self._dynamic_obstacle_view.set_world_poses(pos.reshape(-1, 3), indices=indices)
        self._mark_ray_caster_outdated(env_ids)

    def _refresh_debug_pillars(self) -> None:
        if not self._pursuit_enabled or self._spawned_static_obstacles == 0:
            return
        active = self._static_obstacle_active[0, : self._spawned_static_obstacles]
        self._pillar_positions_xy = self._static_obstacle_positions_xy[
            0, : self._spawned_static_obstacles
        ][active].detach().clone()

    def _refresh_ray_caster_meshes(self) -> None:
        if self._ray_caster is None:
            return

        ray_cfg = self._make_ray_caster_cfg(self.cfg)
        self._ray_caster.cfg.mesh_prim_paths = ray_cfg.mesh_prim_paths
        targets = []
        for target in ray_cfg.mesh_prim_paths:
            if isinstance(target, str):
                target = MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr=target,
                    track_mesh_transforms=False,
                )
            target.prim_expr = target.prim_expr.format(ENV_REGEX_NS="/World/envs/env_.*")
            targets.append(target)

        self._ray_caster._raycast_targets_cfg = targets
        self._ray_caster._num_meshes_per_env = {}
        self._ray_caster._initialize_warp_meshes()
        self._mark_ray_caster_outdated(self._robot._ALL_INDICES)

    def _mark_ray_caster_outdated(self, env_ids: torch.Tensor) -> None:
        if self._ray_caster is None:
            return
        try:
            self._ray_caster._is_outdated[env_ids] = True
        except Exception:
            pass

    def _eval_evader_waypoints(self, env_ids: torch.Tensor, step_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._eval_waypoints(self._evader_waypoints, env_ids, step_ids)

    def _eval_dynamic_obstacle_waypoints(
        self, env_ids: torch.Tensor, step_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._eval_waypoints(self._dynamic_obstacle_waypoints, env_ids, step_ids)

    def _eval_waypoints(
        self, waypoints: torch.Tensor, env_ids: torch.Tensor, step_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seg = torch.searchsorted(self._path_waypoint_steps[1:], step_ids).clamp(max=self._path_waypoint_count - 2)
        start_steps = self._path_waypoint_steps[seg]
        end_steps = self._path_waypoint_steps[seg + 1]
        duration = (end_steps - start_steps).to(waypoints.dtype).clamp_min(1.0)
        tau = ((step_ids - start_steps).to(waypoints.dtype) / duration).clamp(0.0, 1.0)
        p0 = waypoints[env_ids, seg]
        p1 = waypoints[env_ids, seg + 1]
        view_shape = (tau.shape[0],) + (1,) * (p0.ndim - 1)
        tau = tau.view(view_shape)
        pos = p0 * (1.0 - tau) + p1 * tau
        vel = (p1 - p0) / (duration.view(view_shape) * self._step_dt).clamp_min(self._step_dt)
        return pos, vel

    def _dense_waypoint_trajectory(
        self, waypoints: torch.Tensor, env_ids: torch.Tensor, step_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seg = torch.searchsorted(self._path_waypoint_steps[1:], step_ids).clamp(max=self._path_waypoint_count - 2)
        start_steps = self._path_waypoint_steps[seg]
        end_steps = self._path_waypoint_steps[seg + 1]
        duration = (end_steps - start_steps).to(waypoints.dtype).clamp_min(1.0)
        tau = ((step_ids - start_steps).to(waypoints.dtype) / duration).clamp(0.0, 1.0)
        p0 = waypoints[env_ids][:, seg]
        p1 = waypoints[env_ids][:, seg + 1]
        view_shape = (1, tau.shape[0]) + (1,) * (p0.ndim - 2)
        tau = tau.view(view_shape)
        duration = duration.view(view_shape)
        pos = p0 * (1.0 - tau) + p1 * tau
        vel = (p1 - p0) / (duration * self._step_dt).clamp_min(self._step_dt)
        return pos.clone(), vel.clone()

    def _safe_xy_bounds(self, margin: float) -> tuple[torch.Tensor, torch.Tensor]:
        margin = float(margin)
        return self._arena_min_safe[:2] + margin, self._arena_max_safe[:2] - margin

    def _polyline_sample(self, points: torch.Tensor, steps: int, *, closed: bool = False) -> torch.Tensor:
        if closed:
            points = torch.cat((points, points[:1]), dim=0)
        seg_len = torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1).clamp_min(1e-6)
        total = torch.sum(seg_len)
        target = torch.linspace(0.0, float(total.item()), steps, device=self.device)
        cumulative = torch.cat((torch.zeros(1, device=self.device), torch.cumsum(seg_len, dim=0)))
        seg = torch.searchsorted(cumulative[1:], target).clamp(max=seg_len.shape[0] - 1)
        tau = ((target - cumulative[seg]) / seg_len[seg]).view(-1, 1)
        return points[seg] * (1.0 - tau) + points[seg + 1] * tau

    def _inactive_obstacle_xy(self, slots: int) -> torch.Tensor:
        idx = torch.arange(max(0, slots), device=self.device, dtype=torch.float32)
        x = self._arena_min_safe[0] - 6.0 - 0.35 * idx
        y = torch.full_like(x, float(self._arena_min_safe[1]) - 6.0)
        return torch.stack((x, y), dim=-1)

    def _inactive_dynamic_pos(self, slots: int) -> torch.Tensor:
        xy = self._inactive_obstacle_xy(slots)
        z = torch.full((xy.shape[0], 1), self._dynamic_center_z(), device=self.device)
        return torch.cat((xy, z), dim=-1)

    def _static_center_z(self) -> float:
        return float(self.cfg.arena_min[2] + 0.5 * self.cfg.pillar_height)

    def _dynamic_center_z(self) -> float:
        return float(self.cfg.arena_min[2] + 0.5 * self.cfg.pursuit_dynamic_obstacle_height)

    def _ray_miss_xy(self, agent_pos: torch.Tensor) -> torch.Tensor:
        miss = agent_pos[:, :2].clone()
        miss[:, 0] += float(self.cfg.ray_caster_max_distance) + 1000.0
        return miss

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

        cfg_eye = getattr(self.cfg, "camera_view_eye", None)
        cfg_target = getattr(self.cfg, "camera_view_target", None)
        if cfg_eye is not None and cfg_target is not None:
            eye = tuple(float(v) for v in cfg_eye)
            target = tuple(float(v) for v in cfg_target)
            set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
            return

        env_min, _ = torch.min(env_origins, dim=0)
        env_max, _ = torch.max(env_origins, dim=0)
        arena_min = torch.tensor(self.cfg.arena_min, device=self.device, dtype=torch.float32)
        arena_max = torch.tensor(self.cfg.arena_max, device=self.device, dtype=torch.float32)
        world_min = env_min + arena_min
        world_max = env_max + arena_max

        center = 0.5 * (world_min + world_max)
        extent = world_max - world_min
        extent_scale = float(getattr(self.cfg, "camera_view_extent_scale", 1.35))
        top_height = max(float(extent[0]), float(extent[1])) * extent_scale + float(extent[2])

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

    def _spawn_arena_pillars(self, start_slot: int = 0, end_slot: int | None = None) -> None:
        slot_count = self._configured_spawned_static_obstacles(self.cfg)
        start_slot = max(0, int(start_slot))
        end_slot = slot_count if end_slot is None else min(slot_count, int(end_slot))
        if end_slot <= start_slot:
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
            for pillar_id in range(start_slot, end_slot):
                if self.cfg.enable_pursuit_evasion_curriculum:
                    x_local = self.cfg.arena_min[0] - 6.0 - 0.35 * pillar_id
                    y_local = self.cfg.arena_min[1] - 6.0
                else:
                    x_local, y_local = self.cfg.pillar_positions_xy[pillar_id]
                path = f"{base}/Pillar{pillar_id}"
                if prim_utils.is_prim_path_valid(path):
                    continue
                pillar_cfg.func(path, pillar_cfg, translation=(x_local, y_local, center_z_local))

    def _spawn_dynamic_obstacles(self, start_slot: int = 0, end_slot: int | None = None) -> None:
        slot_count = self._configured_spawned_dynamic_obstacles(self.cfg)
        start_slot = max(0, int(start_slot))
        end_slot = slot_count if end_slot is None else min(slot_count, int(end_slot))
        if end_slot <= start_slot:
            return

        center_z_local = self.cfg.arena_min[2] + 0.5 * self.cfg.pursuit_dynamic_obstacle_height
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.12, 0.08))
        obstacle_cfg = sim_utils.CylinderCfg(
            radius=self.cfg.pursuit_dynamic_obstacle_radius,
            height=self.cfg.pursuit_dynamic_obstacle_height,
            visual_material=material,
            copy_from_source=False,
        )

        for env_id in range(self.scene.cfg.num_envs):
            base = f"/World/envs/env_{env_id}/DynamicObstacles"
            for obstacle_id in range(start_slot, end_slot):
                x_local = self.cfg.arena_min[0] - 7.5 - 0.35 * obstacle_id
                y_local = self.cfg.arena_min[1] - 7.5
                path = f"{base}/Dynamic{obstacle_id}"
                if prim_utils.is_prim_path_valid(path):
                    continue
                obstacle_cfg.func(path, obstacle_cfg, translation=(x_local, y_local, center_z_local))

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

    def get_dgppo_reward_auxiliary_data(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().clone() for key, value in self._dgppo_reward_aux.items()}

    def get_last_episode_status(self) -> torch.Tensor:
        return self._last_episode_status

    def get_last_done_reasons(self) -> torch.Tensor:
        return self.get_last_episode_status()

    def get_episode_status_counts(self) -> torch.Tensor:
        return self._episode_status_counts.clone()

    def get_last_step_snapshot(self) -> dict[str, torch.Tensor]:
        return self._last_step_snapshot

    def get_curriculum_metrics(self) -> dict[str, float]:
        if not self._pursuit_enabled:
            return {}
        metrics = {
            "progress": self._curriculum_progress(),
            "fallback_fraction": float(self._scenario_fallback.float().mean().item()),
        }
        for phase in range(1, len(self.cfg.pursuit_curriculum_phase_fractions) + 1):
            metrics[f"phase_{phase}_fraction"] = float((self._scenario_phase == phase).float().mean().item())
        return metrics

    def get_reference_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._reference_pos.clone(), self._reference_yaw.clone()

    def get_evader_trajectory(self) -> tuple[torch.Tensor, torch.Tensor]:
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        steps = torch.arange(self._path_steps, device=self.device, dtype=torch.long)
        pos, vel = self._dense_waypoint_trajectory(self._evader_waypoints, env_ids, steps)
        return pos, vel

    def get_obstacle_trajectories(self) -> dict[str, torch.Tensor]:
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        steps = torch.arange(self._path_steps, device=self.device, dtype=torch.long)
        dynamic_pos, dynamic_vel = self._dense_waypoint_trajectory(self._dynamic_obstacle_waypoints, env_ids, steps)
        return {
            "static_xy": self._static_obstacle_positions_xy.clone(),
            "static_active": self._static_obstacle_active.clone(),
            "dynamic_pos": dynamic_pos,
            "dynamic_vel": dynamic_vel,
            "dynamic_active": self._dynamic_obstacle_active.clone(),
            "phase": self._scenario_phase.clone(),
            "fallback": self._scenario_fallback.clone(),
            "fallback_count": self._scenario_fallback_count.clone(),
        }
