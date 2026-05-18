#!/usr/bin/env python3
"""Benchmark Crazyflie position-tracking policies or baseline controllers."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Benchmark Crazyflie position tracking policies.")
parser.add_argument("--task", type=str, default="PosTracking-v0", help="Gym task to run.")
parser.add_argument("--num-envs", type=int, default=None, help="Override number of environments.")
parser.add_argument("--num-steps", type=int, default=500, help="Simulation steps to run.")
parser.add_argument(
    "--num-episodes",
    type=int,
    default=None,
    help="Stop once this many episodes have completed (across all environments).",
)
parser.add_argument(
    "--policy-mode",
    choices=["rl", "baseline", "random"],
    default="rl",
    help="Policy mode: RL policy, baseline PID position controller, or random actions.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt).")
parser.add_argument("--artifact", type=str, default=None, help="WandB artifact for policy checkpoint.")
parser.add_argument("--artifact-file", type=str, default=None, help="Specific file inside wandb artifact.")
parser.add_argument(
    "--actor-cfg",
    type=str,
    default="source/isaac_pursuit_evasion/deployment/cfg/actor_pos_tracking_ray_cfg.yml",
    help="Actor config name/path for policy loading.",
)
parser.add_argument(
    "--control-mode",
    choices=["RL_velocity", "RL_rates"],
    default=None,
    help="Override env control mode.",
)
parser.add_argument("--yaw-tracking", action="store_true", help="Enable yaw tracking.")
parser.add_argument("--no-yaw-tracking", action="store_true", help="Disable yaw tracking.")
parser.add_argument("--ref-update-interval", type=float, default=None, help="Reference update interval (seconds).")
parser.add_argument("--video", action="store_true", help="Record a short video of the rollout.")
parser.add_argument("--video-length", type=int, default=500, help="Length of the recorded video (steps).")
parser.add_argument("--spawn-cameras", action="store_true", help="Force-enable cameras even in headless mode.")
parser.add_argument("--disable-cameras", action="store_true", help="Disable cameras during evaluation.")
parser.add_argument("--save-camera-images", action="store_true", help="Save per-step FPV frames for all environments.")
parser.add_argument("--camera-overlay-text", action="store_true", help="Overlay info on saved FPV frames.")
parser.add_argument("--log-dir", type=Path, default=Path("logs/pos_tracking/benchmark"), help="Output directory.")
parser.add_argument("--log-episodes", action="store_true", help="Save per-step traces to HDF5.")
parser.add_argument("--log-actions", action="store_true", help="Save per-episode action sequences to .npz.")
parser.add_argument("--log-observations", action="store_true", help="Store observations when logging episodes.")
parser.add_argument("--exp-id", type=str, help="Optional experiment identifier appended to logs.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")

AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=False)

args_cli, hydra_args = parser.parse_known_args()
if args_cli.video or args_cli.save_camera_images or args_cli.spawn_cameras:
    args_cli.enable_cameras = True
if args_cli.disable_cameras:
    args_cli.enable_cameras = False

# hydra_task_config requires args to be passed via sys.argv
import sys
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_tasks.utils.hydra import hydra_task_config

from source.isaac_pursuit_evasion.deployment.actor_policy_loader import (
    load_actor_from_checkpoint,
    load_actor_from_wandb,
    load_actor_policy_config,
)

# Ensure tasks are registered with Gym.
import source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pos_tracking  # noqa: F401


def _apply_env_overrides_from_agent_cfg(env_cfg: Any, agent_cfg: Any) -> None:
    if not isinstance(agent_cfg, dict):
        return
    env_overrides = agent_cfg.get("env", agent_cfg.get("environment", None))
    if not isinstance(env_overrides, dict):
        return
    for key, value in env_overrides.items():
        if key.startswith("_"):
            continue
        if not hasattr(env_cfg, key):
            print(f"[WARN] Ignoring agent env override '{key}': env config has no such attribute.")
            continue
        setattr(env_cfg, key, value)


def _resolve_path(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


class PolicyRunner:
    def __init__(self, actor, device: str):
        self.actor = actor
        self.device = torch.device(device)
        self.obs_scaler = getattr(self.actor, "obs_scaler", None)
        self.expected_obs_dim = self._infer_obs_dim(actor)

    @classmethod
    def from_checkpoint(cls, checkpoint: str, actor_cfg: str | None, device: str) -> "PolicyRunner":
        cfg = load_actor_policy_config(actor_cfg)
        actor = load_actor_from_checkpoint(checkpoint, cfg, device=device)
        return cls(actor, device=device)

    @classmethod
    def from_wandb(
        cls,
        artifact: str,
        actor_cfg: str | None,
        device: str,
        artifact_file: str | None = None,
    ) -> "PolicyRunner":
        cfg = load_actor_policy_config(actor_cfg)
        actor = load_actor_from_wandb(
            artifact,
            artifact_file=artifact_file,
            cfg=cfg,
            device=device,
        )
        return cls(actor, device=device)

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(self.device, dtype=torch.float32)
        if self.expected_obs_dim is not None and obs.shape[-1] != self.expected_obs_dim:
            raise ValueError(
                f"Policy expects obs_dim={self.expected_obs_dim}, but the environment produced "
                f"obs_dim={obs.shape[-1]}. Check --actor-cfg and the benchmark env overrides."
            )
        if self.obs_scaler and self.obs_scaler.mean is not None and self.obs_scaler.std is not None:
            mean = self.obs_scaler.mean.to(self.device)
            std = self.obs_scaler.std.to(self.device)
            obs = (obs - mean) / (std + 1e-6)
        with torch.no_grad():
            action = self.actor.act(obs, deterministic=True)
        return action

    @staticmethod
    def _infer_obs_dim(actor) -> int | None:
        for module in getattr(actor, "net_container", []):
            if isinstance(module, torch.nn.Linear):
                return int(module.in_features)
        return None


@dataclass
class EpisodeBuffer:
    env_id: int
    include_obs: bool
    timesteps: list[int] = field(default_factory=list)
    states: list[torch.Tensor] = field(default_factory=list)
    references: list[torch.Tensor] = field(default_factory=list)
    errors: list[torch.Tensor] = field(default_factory=list)
    yaw_errors: list[torch.Tensor] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    reward_components: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    observations: list[torch.Tensor] = field(default_factory=list)
    flags: list[torch.Tensor] = field(default_factory=list)

    def reset(self) -> None:
        self.timesteps.clear()
        self.states.clear()
        self.references.clear()
        self.errors.clear()
        self.yaw_errors.clear()
        self.actions.clear()
        self.rewards.clear()
        self.reward_components.clear()
        self.observations.clear()
        self.flags.clear()

    def append(
        self,
        timestep: int,
        state: torch.Tensor,
        reference: torch.Tensor,
        pos_error: torch.Tensor,
        yaw_error: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        reward_components: dict[str, torch.Tensor],
        observation: torch.Tensor | None,
        flags: torch.Tensor,
    ) -> None:
        self.timesteps.append(timestep)
        self.states.append(state.clone())
        self.references.append(reference.clone())
        self.errors.append(pos_error.clone())
        self.yaw_errors.append(yaw_error.clone())
        self.actions.append(action.clone())
        self.rewards.append(float(reward))
        for name, value in reward_components.items():
            self.reward_components.setdefault(name, []).append(value.clone())
        if self.include_obs and observation is not None:
            self.observations.append(observation.clone())
        self.flags.append(flags.clone())

    def to_episode(self, done_reason: int) -> EpisodeData | None:
        if not self.timesteps:
            return None
        episode = EpisodeData()
        data: dict[str, Any] = {
            "timesteps": torch.tensor(self.timesteps, dtype=torch.int32),
            "state": torch.stack(self.states, dim=0),
            "reference": torch.stack(self.references, dim=0),
            "pos_error": torch.stack(self.errors, dim=0),
            "yaw_error": torch.stack(self.yaw_errors, dim=0),
            "actions": torch.stack(self.actions, dim=0),
            "reward": torch.tensor(self.rewards, dtype=torch.float32),
            "flags": torch.stack(self.flags, dim=0),
            "done_reason": torch.tensor([done_reason], dtype=torch.int32),
        }
        if self.include_obs and self.observations:
            data["observations"] = torch.stack(self.observations, dim=0)
        if self.reward_components:
            data["reward_components"] = {name: torch.stack(vals, dim=0) for name, vals in self.reward_components.items()}
        episode.data = data
        episode.env_id = self.env_id
        return episode


class EpisodeLogger:
    def __init__(self, env, dataset_path: Path, include_obs: bool, metadata: dict[str, Any], task_name: str):
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.include_obs = include_obs
        self.num_envs = self.env.num_envs
        self.buffers = [EpisodeBuffer(i, include_obs) for i in range(self.num_envs)]
        self.step_counters = torch.zeros(self.num_envs, dtype=torch.int64)
        self.dataset_path = dataset_path
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_handler = HDF5DatasetFileHandler()
        self.file_handler.create(str(dataset_path), env_name=task_name)
        payload = {
            "done_reason_map": {str(k): v for k, v in self.env.DONE_REASON_MAP.items()},
            **metadata,
        }
        payload["step_dt"] = float(getattr(self.env, "step_dt", 0.0))
        self.file_handler.add_env_args(payload)
        self.total_episodes = 0

    def log_step(self, obs: Any, actions: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor) -> None:
        done = (terminated | truncated).detach().clone().cpu()
        env = self.env
        env_origins = env._terrain.env_origins
        pos_local = env._robot.data.root_pos_w - env_origins
        vel_world = env._robot.data.root_lin_vel_w
        quat = env._robot.data.root_quat_w
        ang_vel = env._robot.data.root_ang_vel_b
        ref_pos, ref_yaw = env.get_reference_pose()
        yaw = euler_xyz_from_quat(quat)[2].unsqueeze(-1)
        yaw_error = _wrap_angle(ref_yaw - yaw)
        pos_error = torch.norm(ref_pos - pos_local, dim=-1, keepdim=True)
        state = torch.cat([pos_local, vel_world, quat, ang_vel], dim=-1)
        reference = torch.cat([ref_pos, ref_yaw], dim=-1)
        altitude_limit, xy_limit = env._arena_limit_masks(pos_local)
        flags = torch.stack([altitude_limit, xy_limit], dim=-1).to(torch.float32)
        rewards = env.get_last_rewards().detach().clone().cpu()
        reward_components = env.get_last_reward_components()
        observations = None
        if self.include_obs and isinstance(obs, dict) and "policy" in obs:
            observations = torch.as_tensor(obs["policy"]).detach().clone().cpu()

        for env_id in range(self.num_envs):
            buffer = self.buffers[env_id]
            buffer.append(
                timestep=int(self.step_counters[env_id].item()),
                state=state[env_id].detach().clone().cpu(),
                reference=reference[env_id].detach().clone().cpu(),
                pos_error=pos_error[env_id].detach().clone().cpu(),
                yaw_error=yaw_error[env_id].detach().clone().cpu(),
                action=actions[env_id].detach().clone().cpu(),
                reward=rewards[env_id].item(),
                reward_components={name: tensor[env_id].detach().clone().cpu() for name, tensor in reward_components.items()},
                observation=None if observations is None else observations[env_id],
                flags=flags[env_id].detach().clone().cpu(),
            )
            self.step_counters[env_id] += 1
            if done[env_id].item():
                reason = int(env.get_last_episode_status()[env_id].item())
                episode = buffer.to_episode(reason)
                if episode is not None:
                    self.file_handler.write_episode(episode)
                    self.total_episodes += 1
                buffer.reset()
                self.step_counters[env_id] = 0

    def close(self) -> None:
        self.file_handler.flush()
        self.file_handler.close()


@dataclass
class ActionBuffer:
    actions: list[torch.Tensor] = field(default_factory=list)

    def reset(self) -> None:
        self.actions.clear()

    def append(self, action: torch.Tensor) -> None:
        self.actions.append(action.clone())

    def to_numpy(self) -> np.ndarray:
        if not self.actions:
            return np.empty((0, 0), dtype=np.float32)
        return torch.stack(self.actions, dim=0).cpu().numpy()


class ActionLogger:
    def __init__(self, env, output_dir: Path):
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.num_envs = self.env.num_envs
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffers = [ActionBuffer() for _ in range(self.num_envs)]
        self.episode_counts = [0 for _ in range(self.num_envs)]
        self.total_episodes = 0

    def log_step(self, actions: torch.Tensor, done_mask: torch.Tensor) -> None:
        done = done_mask.detach().clone().cpu()
        for env_id in range(self.num_envs):
            self.buffers[env_id].append(actions[env_id].detach().clone().cpu())
            if done[env_id].item():
                self._flush(env_id)
                self.buffers[env_id].reset()

    def _flush(self, env_id: int) -> None:
        self.episode_counts[env_id] += 1
        self.total_episodes += 1
        filename = self.output_dir / f"env_{env_id:03d}_ep_{self.episode_counts[env_id]:04d}.npz"
        np.savez_compressed(filename, actions=self.buffers[env_id].to_numpy())


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    log_dir = _resolve_path(args_cli.log_dir)
    if not args_cli.exp_id:
        safe_task = args_cli.task.replace("/", "-")
        args_cli.exp_id = f"{safe_task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = log_dir / args_cli.exp_id
    log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = log_dir / "videos"
    episodes_path = log_dir / "episodes.hdf5"
    metrics_path = log_dir / "metrics.json"

    env_cfg.scene.num_envs = args_cli.num_envs or env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    _apply_env_overrides_from_agent_cfg(env_cfg, agent_cfg)
    env_cfg.domain_randomization.enable = False
    env_cfg.use_position_controller = args_cli.policy_mode == "baseline"
    if args_cli.control_mode:
        env_cfg.control_mode = args_cli.control_mode
    if args_cli.yaw_tracking:
        env_cfg.flag_yaw_tracking = True
    if args_cli.no_yaw_tracking:
        env_cfg.flag_yaw_tracking = False
    if args_cli.ref_update_interval is not None:
        env_cfg.ref_update_interval_s = float(args_cli.ref_update_interval)

    if args_cli.seed is not None:
        env_cfg.seed = int(args_cli.seed)

    env_cfg.debug_vis = (not args_cli.headless) or args_cli.video
    env_cfg.debug_visualizer = env_cfg.debug_vis
    if args_cli.spawn_cameras or args_cli.video or args_cli.save_camera_images:
        env_cfg.enable_cameras = True
    if args_cli.disable_cameras:
        env_cfg.enable_cameras = False
    env_cfg.save_camera_images = args_cli.save_camera_images
    if args_cli.save_camera_images:
        env_cfg.camera_image_dir = str(video_dir / "camera_frames")
        env_cfg.camera_overlay_text = args_cli.camera_overlay_text

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if args_cli.video:
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Recording rollout video to: {video_dir}")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            name_prefix="pos-tracking",
            disable_logger=True,
        )

    device = env.unwrapped.device if hasattr(env, "unwrapped") else torch.device("cpu")
    policy = None
    if args_cli.policy_mode == "rl":
        if args_cli.artifact:
            policy = PolicyRunner.from_wandb(
                args_cli.artifact,
                args_cli.actor_cfg,
                device=str(device),
                artifact_file=args_cli.artifact_file,
            )
        elif args_cli.checkpoint:
            policy = PolicyRunner.from_checkpoint(args_cli.checkpoint, args_cli.actor_cfg, device=str(device))
        else:
            print("[WARN] No checkpoint provided; falling back to random actions.")
            args_cli.policy_mode = "random"

    obs, _ = env.reset()
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    action_dim = base_env.cfg.action_space

    episode_logger = (
        EpisodeLogger(env, episodes_path, args_cli.log_observations, {"task": args_cli.task}, args_cli.task)
        if args_cli.log_episodes
        else None
    )
    action_logger = ActionLogger(env, log_dir / "actions") if args_cli.log_actions else None

    total_steps = 0
    sum_pos_error = 0.0
    sum_pos_error_sq = 0.0
    sum_yaw_error = 0.0
    sum_reward = 0.0
    sum_components: dict[str, float] = {}
    total_samples = 0

    episode_steps = torch.zeros(base_env.num_envs, dtype=torch.int64)
    episode_lengths: list[int] = []
    success_times: list[float] = []
    success_count = 0
    crash_count = 0
    oob_count = 0
    timeout_count = 0
    invalid_count = 0
    total_episodes = 0

    for _ in range(args_cli.num_steps):
        if args_cli.num_episodes is not None and total_episodes >= args_cli.num_episodes:
            break

        if args_cli.policy_mode == "baseline":
            actions = torch.zeros(base_env.num_envs, action_dim, device=device)
        elif args_cli.policy_mode == "random":
            actions = torch.empty(base_env.num_envs, action_dim, device=device).uniform_(-1.0, 1.0)
        else:
            obs_tensor = torch.as_tensor(obs["policy"], device=device)
            actions = policy(obs_tensor)

        obs, reward, terminated, truncated, _ = env.step(actions)
        done = terminated | truncated

        env_origins = base_env._terrain.env_origins
        pos_local = base_env._robot.data.root_pos_w - env_origins
        ref_pos, ref_yaw = base_env.get_reference_pose()
        pos_error = torch.norm(ref_pos - pos_local, dim=-1)
        sum_pos_error += pos_error.sum().item()
        sum_pos_error_sq += (pos_error ** 2).sum().item()
        total_samples += pos_error.numel()

        if base_env.cfg.flag_yaw_tracking:
            yaw = euler_xyz_from_quat(base_env._robot.data.root_quat_w)[2]
            yaw_error = torch.abs(_wrap_angle(ref_yaw.squeeze(-1) - yaw))
            sum_yaw_error += yaw_error.sum().item()

        sum_reward += reward.sum().item()
        components = base_env.get_last_reward_components()
        for name, tensor in components.items():
            sum_components[name] = sum_components.get(name, 0.0) + tensor.sum().item()

        if episode_logger is not None:
            episode_logger.log_step(obs, actions, terminated, truncated)
        if action_logger is not None:
            action_logger.log_step(actions, done)

        episode_steps += 1
        if done.any():
            done_ids = torch.nonzero(done).squeeze(-1)
            reasons = base_env.get_last_episode_status()[done_ids]
            for env_id, reason in zip(done_ids.tolist(), reasons.tolist()):
                total_episodes += 1
                length = int(episode_steps[env_id].item())
                episode_lengths.append(length)
                if reason == 1:
                    success_count += 1
                    success_times.append(length * base_env.step_dt)
                elif reason == 2:
                    crash_count += 1
                elif reason == 3:
                    oob_count += 1
                elif reason == 4:
                    timeout_count += 1
                elif reason == 5:
                    invalid_count += 1
                episode_steps[env_id] = 0

        total_steps += 1

    if episode_logger is not None:
        episode_logger.close()

    mean_pos_error = sum_pos_error / max(1, total_samples)
    rms_pos_error = (sum_pos_error_sq / max(1, total_samples)) ** 0.5
    mean_yaw_error = sum_yaw_error / max(1, total_samples) if base_env.cfg.flag_yaw_tracking else None
    mean_reward = sum_reward / max(1, total_samples)

    metrics = {
        "task": args_cli.task,
        "control_mode": base_env.cfg.control_mode,
        "policy_mode": args_cli.policy_mode,
        "num_steps": total_steps,
        "num_episodes": total_episodes,
        "mean_pos_error": mean_pos_error,
        "rms_pos_error": rms_pos_error,
        "mean_yaw_error": mean_yaw_error,
        "success_rate": success_count / max(1, total_episodes),
        "mean_time_to_success": (sum(success_times) / max(1, len(success_times))) if success_times else None,
        "crash_rate": crash_count / max(1, total_episodes),
        "out_of_bounds_rate": oob_count / max(1, total_episodes),
        "timeout_rate": timeout_count / max(1, total_episodes),
        "invalid_rate": invalid_count / max(1, total_episodes),
        "mean_episode_length": (sum(episode_lengths) / max(1, len(episode_lengths))) if episode_lengths else None,
        "mean_total_reward": mean_reward,
        "reward_components": {name: total / max(1, total_samples) for name, total in sum_components.items()},
        "video_dir": str(video_dir) if args_cli.video else None,
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        import json
        json.dump(metrics, f, indent=2)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
