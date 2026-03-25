#!/usr/bin/env python3
"""Benchmark and autotune the trajectory-tracking task driven by the Lee controller."""
from __future__ import annotations
import argparse
import itertools
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from isaaclab.app import AppLauncher
from isaaclab.utils.datasets import HDF5DatasetFileHandler, EpisodeData


TASK_NAME = "Trajectory-Tracking-v0"
DEFAULT_SEARCH_SPACE = {
    "position_gain_scale": {"min": 0.2, "max": 5.0, "log": True},
    "velocity_gain_scale": {"min": 0.2, "max": 5.0, "log": True},
    "attitude_gain_scale": {"min": 0.2, "max": 5.0, "log": True},
    "rate_gain_scale": {"min": 0.2, "max": 5.0, "log": True},
}


def collect_env_metadata(env) -> dict:
    """Return serializable metadata describing the current environment setup."""
    metadata: dict[str, object] = {}
    metadata["num_envs"] = int(getattr(env, "num_envs", 0))
    step_dt = getattr(env, "step_dt", None)
    if step_dt is None:
        sim_cfg = getattr(env, "sim", None)
        sim_dt = getattr(getattr(sim_cfg, "cfg", sim_cfg), "dt", None)
        decimation = getattr(getattr(env, "cfg", None), "decimation", 1)
        if sim_dt is not None:
            step_dt = float(sim_dt) * float(decimation)
        else:
            step_dt = 0.0
    metadata["step_dt"] = float(step_dt)

    trajectory_specs = []
    task_cfg = getattr(env, "cfg", None)
    task_cfg = getattr(task_cfg, "task", task_cfg)
    specs = getattr(task_cfg, "trajectory_specs", None)
    if specs is not None:
        for spec in specs:
            trajectory_specs.append(
                {
                    "name": getattr(spec, "name", ""),
                    "count": int(getattr(spec, "count", 0)),
                }
            )
    metadata["trajectory_specs"] = trajectory_specs
    metadata["enable_yaw_tracking"] = bool(
        getattr(task_cfg, "enable_yaw_tracking", getattr(env, "_enable_yaw_tracking", False))
    )
    return metadata


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark / tune Lee controller for trajectory tracking.")
    parser.add_argument("--task", type=str, default=TASK_NAME, help="Gym task name to load.")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of simulation steps per evaluation.")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments.")
    parser.add_argument("--video", action="store_true", help="Record a video for the baseline run.")
    parser.add_argument("--video-length", type=int, default=100, help="Video length in steps.")
    parser.add_argument(
        "--video-folder",
        type=Path,
        default=Path("logs/trajectory_following/videos"),
        help="Folder where recorded videos are stored.",
    )
    parser.add_argument("--optimize", action="store_true", help="Enable controller gain optimization.")
    parser.add_argument("--search-method", choices=("random", "grid"), default="random", help="Search strategy.")
    parser.add_argument("--search-space", type=Path, help="Optional JSON describing the gain search space.")
    parser.add_argument("--num-trials", type=int, default=25, help="Number of evaluations for random/grid search.")
    parser.add_argument("--opt-steps", type=int, default=400, help="Number of steps per trial during optimization.")
    parser.add_argument("--collision-weight", type=float, default=50.0, help="Penalty weight for collisions.")
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("logs/trajectory_following/plots"),
        help="Directory to store summary plots.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("logs/trajectory_following/metrics"),
        help="Directory where per-run metrics JSON files are written.",
    )
    parser.add_argument("--log-episodes", action="store_true", help="Dump full state/reward traces to HDF5.")
    parser.add_argument("--log-observations", action="store_true", help="Store observations when logging episodes.")
    parser.add_argument("--yaw-command", action="store_true", help="Enable yaw references from the trajectory.")
    parser.add_argument("--hover-count", type=int, help="Override number of hover trajectories.")
    parser.add_argument("--circular-count", type=int, help="Override number of circular trajectories.")
    parser.add_argument("--lemniscate-count", type=int, help="Override number of lemniscate trajectories.")
    parser.add_argument(
        "--exp-id",
        type=str,
        help="Optional experiment identifier appended to plot/metric/video directories.",
    )
    parser.add_argument("--skip-app-close", action="store_true", help="Skip closing simulation app (avoids hanging).")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_search_space(path: Optional[Path]) -> Dict[str, dict]:
    if path is None:
        return DEFAULT_SEARCH_SPACE
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def iter_grid(space: Dict[str, dict], max_trials: int) -> Iterable[dict]:
    options = []
    for name, spec in space.items():
        if "values" in spec:
            vals = spec["values"]
        else:
            num = spec.get("num", 4)
            vals = np.linspace(spec["min"], spec["max"], num).tolist()
        options.append([(name, float(v)) for v in vals])
    for idx, combo in enumerate(itertools.product(*options)):
        if max_trials and idx >= max_trials:
            break
        yield dict(combo)


def sample_random(space: Dict[str, dict]) -> dict:
    sample = {}
    for name, spec in space.items():
        if "values" in spec:
            sample[name] = float(random.choice(spec["values"]))
            continue
        low, high = float(spec["min"]), float(spec["max"])
        if spec.get("log", False):
            sample[name] = float(math.exp(random.uniform(math.log(low), math.log(high))))
        else:
            sample[name] = float(random.uniform(low, high))
    return sample


def build_overrides(scales: dict, base_cfg: dict) -> dict:
    overrides = {}
    mapping = {
        "position_gain": ("k_pos", "position_gain_scale"),
        "velocity_gain": ("k_vel", "velocity_gain_scale"),
        "attitude_gain": ("k_att", "attitude_gain_scale"),
        "angular_rate_gain": ("k_rate", "rate_gain_scale"),
    }
    for key, (attr, scale_key) in mapping.items():
        scale = scales.get(scale_key, 1.0)
        values = (torch.tensor(base_cfg[key], dtype=torch.float32) * scale).tolist()
        overrides[attr] = values
    return overrides


def build_trajectory_specs_from_args(args):
    from source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.trajectories.tracking_env import (
        TrajectorySpecConfig,
    )
    if args.hover_count is None and args.circular_count is None and args.lemniscate_count is None:
        return None
    specs = []
    if args.hover_count:
        specs.append(TrajectorySpecConfig(name="hovertrajectory", count=int(args.hover_count)))
    if args.circular_count:
        specs.append(TrajectorySpecConfig(name="circulartrajectory", count=int(args.circular_count)))
    if args.lemniscate_count:
        specs.append(TrajectorySpecConfig(name="lemniscatetrajectory", count=int(args.lemniscate_count)))
    return specs


def create_env(args) -> gym.Env:
    """Create the Gym environment once and reuse it across all benchmark runs."""
    import isaaclab_tasks  # noqa: F401
    import source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.trajectories  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
    )
    specs = build_trajectory_specs_from_args(args)
    if specs:
        env_cfg.trajectory_specs = tuple(specs)
    if args.yaw_command:
        env_cfg.enable_yaw_tracking = True

    render_mode = "rgb_array" if args.video else None
    env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)
    if args.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(args.video_folder),
            episode_trigger=lambda episode_id: episode_id == 0,
            video_length=args.video_length,
            name_prefix="trajectory-following",
            disable_logger=True,
        )
    env.reset()
    return env


class EpisodeLogger:
    """Episode-wise logger that records per-env trajectories into HDF5."""

    def __init__(self, output_path: Path, env_name: str, include_obs: bool) -> None:
        self.include_obs = include_obs
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_handler = HDF5DatasetFileHandler()
        self.file_handler.create(str(output_path), env_name=env_name)
        self.buffers: list[EpisodeData] = []
        self.metadata_written = False
        self._cached_pos: Optional[torch.Tensor] = None
        self._cached_vel: Optional[torch.Tensor] = None
        self._cached_ang: Optional[torch.Tensor] = None
        self._env_origins: Optional[torch.Tensor] = None
        self.step_counters: Optional[torch.Tensor] = None

    def begin(self, env) -> None:
        self.base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.num_envs = self.base_env.num_envs
        self.buffers = [EpisodeData() for _ in range(self.num_envs)]
        self.step_counters = torch.zeros(self.num_envs, dtype=torch.int64)
        self._env_origins = self.base_env._terrain.env_origins.detach().clone().cpu()
        if not self.metadata_written:
            metadata = collect_env_metadata(self.base_env)
            metadata["num_envs"] = self.num_envs
            self.file_handler.add_env_args(metadata)
            self.metadata_written = True
        self.capture_state()

    def capture_state(self) -> None:
        if not hasattr(self, "base_env"):
            return
        with torch.no_grad():
            pos = self.base_env._robot.data.root_pos_w.detach().clone().cpu()
            vel = self.base_env._robot.data.root_lin_vel_w.detach().clone().cpu()
            ang = self.base_env._robot.data.root_ang_vel_b.detach().clone().cpu()
            if self._env_origins is not None:
                pos = pos - self._env_origins
            self._cached_pos = pos
            self._cached_vel = vel
            self._cached_ang = ang

    def _ensure_env_length(self, tensor: torch.Tensor, fill: bool = False) -> torch.Tensor:
        tensor = tensor.reshape(-1)
        if tensor.numel() == 1 and self.num_envs > 1:
            tensor = tensor.repeat(self.num_envs)
        if tensor.numel() != self.num_envs:
            raise ValueError(f"Tensor length {tensor.numel()} does not match num_envs {self.num_envs}")
        return tensor

    def log_step(
        self,
        metrics: dict,
        *,
        obs: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
        terminated=None,
        truncated=None,
    ):
        if self._cached_pos is None or self._cached_vel is None or self._cached_ang is None:
            self.capture_state()
        pos = self._cached_pos
        vel = self._cached_vel
        ang = self._cached_ang

        if reward is None:
            reward_tensor = -metrics["position_error"].detach().cpu()
        else:
            reward_tensor = torch.as_tensor(reward).detach().cpu().to(torch.float32)
        if reward_tensor.dim() == 0:
            reward_tensor = reward_tensor.repeat(self.num_envs)
        reward_tensor = reward_tensor.reshape(-1)
        if reward_tensor.numel() != self.num_envs:
            if reward_tensor.numel() == 1:
                reward_tensor = reward_tensor.repeat(self.num_envs)
            else:
                raise ValueError(f"Reward tensor length {reward_tensor.numel()} does not match num_envs {self.num_envs}")
        reward_tensor = reward_tensor.unsqueeze(-1)

        obs_cpu = None
        if self.include_obs and obs is not None:
            if torch.is_tensor(obs):
                obs_cpu = obs.detach().cpu()
            else:
                obs_cpu = torch.as_tensor(obs)

        if terminated is not None:
            term_tensor = torch.as_tensor(terminated, dtype=torch.bool)
        else:
            term_tensor = torch.zeros(self.num_envs, dtype=torch.bool)
        term_tensor = self._ensure_env_length(term_tensor)

        if truncated is not None:
            trunc_tensor = torch.as_tensor(truncated, dtype=torch.bool)
        else:
            trunc_tensor = torch.zeros(self.num_envs, dtype=torch.bool)
        trunc_tensor = self._ensure_env_length(trunc_tensor)

        for env_id in range(self.num_envs):
            episode = self.buffers[env_id]
            if episode.is_empty():
                episode.add("initial_state/pos", pos[env_id].unsqueeze(0))
                episode.add("initial_state/velocity", vel[env_id].unsqueeze(0))
                episode.add("initial_state/angular_velocity", ang[env_id].unsqueeze(0))
            episode.add(
                "timesteps",
                torch.tensor([[self.step_counters[env_id].item()]], dtype=torch.int64),
            )
            episode.add("states/position", pos[env_id].unsqueeze(0))
            episode.add("states/velocity", vel[env_id].unsqueeze(0))
            episode.add("states/angular_velocity", ang[env_id].unsqueeze(0))
            episode.add("reward", reward_tensor[env_id].unsqueeze(0))
            if obs_cpu is not None:
                episode.add("observations/policy", obs_cpu[env_id].unsqueeze(0))

            done_reason = 0
            if term_tensor[env_id]:
                done_reason = 1
            elif trunc_tensor[env_id]:
                done_reason = 2

            if done_reason != 0:
                self._finalize_episode(env_id, done_reason)
            else:
                self.step_counters[env_id] += 1

    def _stack_lists(self, data):
        if isinstance(data, dict):
            return {k: self._stack_lists(v) for k, v in data.items()}
        if isinstance(data, list):
            if not data:
                return torch.empty(0)
            if isinstance(data[0], torch.Tensor):
                return torch.cat(data, dim=0) if data[0].ndim > 0 else torch.stack(data, dim=0)
            return torch.tensor(data)
        return data

    def _finalize_episode(self, env_id: int, reason: int):
        episode = self.buffers[env_id]
        if episode.is_empty():
            return
        episode.add(
            "done/reason",
            torch.tensor([[reason]], dtype=torch.int32),
        )
        episode.env_id = env_id
        episode.data = self._stack_lists(episode.data)
        self.file_handler.write_episode(episode)
        self.buffers[env_id] = EpisodeData()
        self.step_counters[env_id] = 0

    def finalize(self):
        for env_id in range(self.num_envs):
            if not self.buffers[env_id].is_empty():
                self._finalize_episode(env_id, reason=0)
        self.file_handler.flush()
        self.file_handler.close()



def run_benchmark(
    env,
    args,
    controller_overrides: Optional[dict],
    *,
    record_history: bool,
    num_steps: Optional[int] = None,
    label: str = "run",
    log_episodes: bool = False,
) -> dict:
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    if controller_overrides:
        base_env.apply_controller_overrides(controller_overrides)

    steps = num_steps or args.num_steps
    history = {"mean_error": [], "max_error": [], "collisions": []} if record_history else None
    total_error = 0.0
    total_collisions = 0

    base_env._collision_counts.zero_()
    base_env._latest_collision_mask = torch.zeros_like(base_env._latest_collision_mask)
    env.reset()

    episode_logger = None
    if log_episodes:
        episodes_dir = args.metrics_dir / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = episodes_dir / f"{label}_episodes.hdf5"
        episode_logger = EpisodeLogger(dataset_path, args.task, args.log_observations)
        episode_logger.begin(env)

    num_envs = base_env.num_envs
    action_dim = base_env.single_action_space.shape[0]
    actions = torch.zeros(num_envs, action_dim, device=base_env.device)

    for _ in tqdm.tqdm(range(steps), desc=f"{label:>9}"):
        obs, reward, terminated, truncated, _ = env.step(actions)
        metrics = base_env.get_tracking_metrics()
        errors = metrics["position_error"].detach().cpu().numpy()
        collisions = int(metrics["collision_mask"].sum().item())
        total_error += errors.mean()
        total_collisions += collisions
        if history is not None:
            history["mean_error"].append(float(errors.mean()))
            history["max_error"].append(float(errors.max()))
            history["collisions"].append(collisions)
        if episode_logger:
            policy_obs = obs.get("policy") if isinstance(obs, dict) else obs
            episode_logger.log_step(
                metrics,
                obs=policy_obs,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
            episode_logger.capture_state()
        render_mode = getattr(env, "render_mode", None)
        if render_mode is None and hasattr(env, "unwrapped"):
            render_mode = getattr(env.unwrapped, "render_mode", None)
        if render_mode == "human":
            env.render()

    if episode_logger:
        episode_logger.finalize()

    mean_error = total_error / steps
    collision_rate = total_collisions / (steps * num_envs)
    return {
        "mean_error": mean_error,
        "collision_rate": collision_rate,
        "history": history,
        "controller_overrides": controller_overrides,
    }


def run_optimization(env, args, base_cfg, search_space):
    best = None
    best_score = float("inf")
    history = []
    
    if args.search_method == "grid":
        candidates = iter_grid(search_space, args.num_trials)
    else:
        candidates = (sample_random(search_space) for _ in range(args.num_trials))
    
    for idx, scales in enumerate(candidates):        
        overrides = build_overrides(scales, base_cfg)
        result = run_benchmark(
            env,
            args,
            overrides,
            record_history=False,
            num_steps=args.opt_steps,
            label=f"trial_{idx:02d}",
        )
        score = result["mean_error"] + args.collision_weight * result["collision_rate"]
        history.append({"trial": idx, "scales": scales, "metrics": result, "score": score})
        
        if score < best_score:
            best = overrides
            best_score = score
        
        if args.search_method == "random" and idx + 1 >= args.num_trials:
            break
    
    return best, history


def plot_comparison(baseline: dict, optimized: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = len(baseline["mean_error"])
    x = np.arange(steps)
    
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    ax[0].plot(x, baseline["mean_error"], label="Baseline")
    ax[0].plot(x, optimized["mean_error"], label="Optimized")
    ax[0].set_ylabel("Mean position error [m]")
    ax[0].grid(True)
    ax[0].legend()
    
    ax[1].plot(x, baseline["collisions"], label="Baseline")
    ax[1].plot(x, optimized["collisions"], label="Optimized")
    ax[1].set_ylabel("# collisions / step")
    ax[1].set_xlabel("Simulation step")
    ax[1].grid(True)
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "error_collision_comparison.png")
    plt.close(fig)


def save_metrics(result: dict, label: str, args, env_metadata: Optional[dict] = None):
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": label,
        "mean_error": result["mean_error"],
        "collision_rate": result["collision_rate"],
        "controller_overrides": result["controller_overrides"],
    }
    if env_metadata:
        payload["env_metadata"] = env_metadata
    path = args.metrics_dir / f"{label}_metrics.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_cli()

    if args.exp_id:
        args.video_folder = args.video_folder / args.exp_id
        args.plot_dir = args.plot_dir / args.exp_id
        args.metrics_dir = args.metrics_dir / args.exp_id

    args.video_folder = args.video_folder.expanduser().resolve()
    args.plot_dir = args.plot_dir.expanduser().resolve()
    args.metrics_dir = args.metrics_dir.expanduser().resolve()

    if args.video:
        args.enable_cameras = True

    reset_dir(args.metrics_dir)
    reset_dir(args.plot_dir)
    if args.video:
        reset_dir(args.video_folder)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    env = create_env(args)
    from source.isaac_pursuit_evasion.controllers.config import load_controller_config

    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    env_metadata = collect_env_metadata(base_env)
    metadata_path = args.metrics_dir / "env_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(env_metadata, f, indent=2)

    base_cfg = load_controller_config("lee_controller", "vaporx5")
    search_space = load_search_space(args.search_space)

    print("Running baseline evaluation...")
    baseline_result = run_benchmark(
        env,
        args,
        controller_overrides=None,
        record_history=True,
        label="baseline",
        log_episodes=args.log_episodes,
    )
    save_metrics(baseline_result, "baseline", args, env_metadata=env_metadata)
    print(
        f"Baseline - Mean Error: {baseline_result['mean_error']:.4f}, "
        f"Collision Rate: {baseline_result['collision_rate']:.4f}"
    )

    if args.optimize:
        print("\nRunning optimization...")
        best_overrides, opt_history = run_optimization(env, args, base_cfg, search_space)

        if best_overrides is not None:
            print("\nEvaluating optimized controller...")
            optimized_result = run_benchmark(
                env,
                args,
                controller_overrides=best_overrides,
                record_history=True,
                label="optimized",
                log_episodes=args.log_episodes,
            )
            save_metrics(optimized_result, "optimized", args, env_metadata=env_metadata)
            print(
                f"Optimized - Mean Error: {optimized_result['mean_error']:.4f}, "
                f"Collision Rate: {optimized_result['collision_rate']:.4f}"
            )
            plot_comparison(baseline_result["history"], optimized_result["history"], args.plot_dir)
        else:
            print("Optimization search space did not produce any candidates.")

        history_payload = {"env_metadata": env_metadata, "trials": opt_history}
        history_path = args.metrics_dir / "optimization_history.json"
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history_payload, f, indent=2)

        print(f"\nOptimization complete! Results saved to {args.metrics_dir}")
    else:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(baseline_result["history"]["mean_error"])
        plt.title("Baseline mean position error")
        plt.xlabel("step")
        plt.ylabel("mean error [m]")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(args.plot_dir / "baseline_error.png")
        plt.close()
        print(f"\nBaseline evaluation complete! Results saved to {args.metrics_dir}")

    # env.close()
    # simulation_app.close


if __name__ == "__main__":
    main()
