#!/usr/bin/env python3
"""Record a small set of position-tracking rollouts from a trained local checkpoint."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Record rollout videos for a trained PosTracking policy checkpoint.")
parser.add_argument("--task", type=str, default="PosTracking-RL-rates-v0", help="Gym task to run.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to local model checkpoint (.pt).")
parser.add_argument(
    "--actor-cfg",
    type=str,
    default=None,
    help=(
        "Actor policy config file. If omitted, architecture is inferred from checkpoint "
        "(obs/action dims + hidden sizes)."
    ),
)
parser.add_argument("--num-rollouts", type=int, default=3, help="Number of episodes to record.")
parser.add_argument("--num-envs", type=int, default=1, help="Number of environments (use 1 for clean videos).")
parser.add_argument("--max-steps", type=int, default=15000, help="Safety limit for total simulation steps.")
parser.add_argument("--seed", type=int, default=None, help="Optional environment seed.")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("logs/pos_tracking/rollouts"),
    help="Root directory where rollout videos are stored.",
)
parser.add_argument("--exp-id", type=str, default=None, help="Optional experiment ID appended to output directory.")

AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)

args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True

# hydra_task_config requires args to be passed via sys.argv
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
import gymnasium as gym
import torch
from isaaclab_tasks.utils.hydra import hydra_task_config

from source.isaac_pursuit_evasion.deployment.actor_policy_loader import (
    ActorPolicyConfig,
    load_actor_from_checkpoint,
    load_actor_policy_config,
)

# Ensure tasks are registered with Gym.
import source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pos_tracking  # noqa: F401


def _resolve_path(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _get_local_xy(base_env) -> torch.Tensor:
    """Return drone XY positions in local env coordinates (num_envs, 2)."""
    root_pos_w = base_env._robot.data.root_pos_w
    env_origins = base_env._terrain.env_origins
    pos_local = root_pos_w - env_origins
    return pos_local[:, :2].detach().cpu()


def _get_reference_local_xy(base_env) -> torch.Tensor:
    """Return reference goal XY in local env coordinates (num_envs, 2)."""
    return base_env._reference_pos[:, :2].detach().cpu()


def _plot_xy_rollouts(
    output_dir: Path,
    env_cfg,
    trajectories: list[list[tuple[float, float]]],
    goals_xy: list[tuple[float, float]],
) -> Path | None:
    """Save a 2D XY trajectory plot with arena bounds and obstacles."""
    if not trajectories:
        return None
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"[WARN] Could not import matplotlib to plot trajectories: {exc}")
        return None

    fig, ax = plt.subplots(figsize=(8, 7))
    arena_min = env_cfg.arena_min
    arena_max = env_cfg.arena_max
    width = arena_max[0] - arena_min[0]
    height = arena_max[1] - arena_min[1]
    ax.add_patch(
        Rectangle(
            (arena_min[0], arena_min[1]),
            width,
            height,
            fill=False,
            lw=2.0,
            ls="--",
            ec="black",
            label="Arena bounds",
        )
    )

    if getattr(env_cfg, "enable_pillars", False):
        for idx, (px, py) in enumerate(getattr(env_cfg, "pillar_positions_xy", ())):
            pillar = Circle(
                (float(px), float(py)),
                radius=float(env_cfg.pillar_radius),
                color="dimgray",
                alpha=0.35,
                ec="black",
                lw=1.0,
                label="Pillar obstacle" if idx == 0 else None,
            )
            ax.add_patch(pillar)

    cmap = plt.cm.get_cmap("tab10", max(1, len(trajectories)))
    for rollout_idx, points in enumerate(trajectories):
        if len(points) < 2:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color = cmap(rollout_idx)
        ax.plot(xs, ys, color=color, lw=1.8, label=f"Rollout {rollout_idx + 1}")
        ax.scatter(xs[0], ys[0], color=color, marker="o", s=24)
        ax.scatter(xs[-1], ys[-1], color=color, marker="x", s=35)
        if rollout_idx < len(goals_xy):
            goal = goals_xy[rollout_idx]
            ax.scatter(
                goal[0],
                goal[1],
                color=color,
                marker="*",
                s=260,
                edgecolors="black",
                linewidths=0.8,
                zorder=30,
            )

    ax.set_title("Drone XY trajectories")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncols=2)
    fig.tight_layout()

    plot_path = output_dir / "trajectory_xy.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def _extract_policy_state_dict(payload: Any) -> Mapping[str, Any] | None:
    if isinstance(payload, torch.nn.Module):
        return payload.state_dict()
    if not isinstance(payload, Mapping):
        return None

    if "policy" in payload and isinstance(payload["policy"], Mapping):
        return payload["policy"]
    for container_key in ("models", "model", "model_state_dict", "state_dict"):
        container = payload.get(container_key)
        if isinstance(container, Mapping):
            if "policy" in container and isinstance(container["policy"], Mapping):
                return container["policy"]
            for prefix in ("policy", "models.policy", "model.policy"):
                token = f"{prefix}."
                filtered = {key[len(token) :]: value for key, value in container.items() if key.startswith(token)}
                if filtered:
                    return filtered

    for prefix in ("policy", "models.policy", "model.policy"):
        token = f"{prefix}."
        filtered = {key[len(token) :]: value for key, value in payload.items() if key.startswith(token)}
        if filtered:
            return filtered

    if any(key.startswith("net_container.") for key in payload.keys()):
        return payload
    return None


def _infer_actor_cfg_from_checkpoint(checkpoint: Path) -> ActorPolicyConfig:
    payload = torch.load(str(checkpoint), map_location="cpu")
    state_dict = _extract_policy_state_dict(payload)
    if state_dict is None:
        raise ValueError(f"Unable to locate policy weights in checkpoint: {checkpoint}")

    first_layer = state_dict.get("net_container.0.weight")
    if first_layer is None or not isinstance(first_layer, torch.Tensor):
        raise ValueError("Could not infer obs/action dimensions (missing net_container.0.weight).")

    obs_dim = int(first_layer.shape[1])
    hidden_layers: list[int] = []
    layer_indices: list[int] = []
    pattern = re.compile(r"^net_container\.(\d+)\.weight$")
    for key in state_dict.keys():
        match = pattern.match(key)
        if match:
            layer_indices.append(int(match.group(1)))
    for idx in sorted(layer_indices):
        w_key = f"net_container.{idx}.weight"
        weight = state_dict.get(w_key)
        if isinstance(weight, torch.Tensor):
            hidden_layers.append(int(weight.shape[0]))
    if not hidden_layers:
        raise ValueError("Could not infer hidden layers from checkpoint weights.")
    action_dim = hidden_layers.pop()
    return ActorPolicyConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_layers=hidden_layers,
        activation="elu",
        log_std_init=0.0,
    )


class PolicyRunner:
    def __init__(self, actor, device: str):
        self.actor = actor
        self.device = torch.device(device)
        self.obs_scaler = getattr(self.actor, "obs_scaler", None)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        actor_cfg: str | None,
        device: str,
    ) -> "PolicyRunner":
        checkpoint_path = _resolve_path(Path(checkpoint))
        if actor_cfg:
            cfg = load_actor_policy_config(actor_cfg)
        else:
            cfg = _infer_actor_cfg_from_checkpoint(checkpoint_path)
            print(
                "[INFO] Inferred actor config from checkpoint: "
                f"obs_dim={cfg.obs_dim}, action_dim={cfg.action_dim}, hidden_layers={list(cfg.hidden_layers)}"
            )
        actor = load_actor_from_checkpoint(str(checkpoint_path), cfg, device=device)
        return cls(actor, device=device)

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(self.device, dtype=torch.float32)
        if self.obs_scaler and self.obs_scaler.mean is not None and self.obs_scaler.std is not None:
            mean = self.obs_scaler.mean.to(self.device)
            std = self.obs_scaler.std.to(self.device)
            obs = (obs - mean) / (std + 1e-6)
        with torch.no_grad():
            action = self.actor.act(obs, deterministic=True)
        return action


def _default_checkpoint() -> Path:
    root = Path(__file__).resolve().parents[2]
    candidates = sorted((root / "logs" / "skrl" / "training").glob("**/checkpoints/*.pt"))
    if not candidates:
        raise FileNotFoundError("No local checkpoints found under logs/skrl/training/**/checkpoints/*.pt")
    return candidates[-1]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, _agent_cfg: dict):
    run_id = args_cli.exp_id or f"{args_cli.task.replace('/', '-')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir = _resolve_path(args_cli.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = args_cli.checkpoint or str(_default_checkpoint())
    checkpoint = str(_resolve_path(Path(checkpoint)))
    print(f"[INFO] Checkpoint: {checkpoint}")
    print(f"[INFO] Video directory: {video_dir}")

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    if args_cli.seed is not None:
        env_cfg.seed = int(args_cli.seed)
    fixed_goal_candidates = ((-1.8, 0.0), (1.5, 0.0), (-1.0, 0.0))
    fixed_goal_budget = min(3, int(args_cli.num_rollouts))
    env_cfg.debug_vis = True
    env_cfg.debug_visualizer = True
    env_cfg.enable_cameras = True

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_dir),
        episode_trigger=lambda episode_id: episode_id < int(args_cli.num_rollouts),
        name_prefix="pos-tracking-rollout",
        disable_logger=True,
    )

    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    device = str(base_env.device)
    policy = PolicyRunner.from_checkpoint(checkpoint, args_cli.actor_cfg, device=device)
    fixed_goal_z = float((env_cfg.ref_pos_min[2] + env_cfg.ref_pos_max[2]) * 0.5)
    next_fixed_goal_idx = 0

    def _assign_fixed_goals(env_ids: list[int]) -> None:
        nonlocal next_fixed_goal_idx
        if fixed_goal_budget <= 0:
            return
        for env_id in env_ids:
            if next_fixed_goal_idx >= fixed_goal_budget:
                break
            goal_xy = fixed_goal_candidates[next_fixed_goal_idx]
            base_env._reference_pos[env_id, 0] = float(goal_xy[0])
            base_env._reference_pos[env_id, 1] = float(goal_xy[1])
            base_env._reference_pos[env_id, 2] = fixed_goal_z
            if hasattr(base_env, "_reference_timer"):
                base_env._reference_timer[env_id] = 0.0
            next_fixed_goal_idx += 1

    obs, _ = env.reset()
    _assign_fixed_goals(list(range(int(args_cli.num_envs))))
    current_paths: list[list[tuple[float, float]]] = [[] for _ in range(int(args_cli.num_envs))]
    current_goals: list[tuple[float, float] | None] = [None for _ in range(int(args_cli.num_envs))]
    finished_paths: list[list[tuple[float, float]]] = []
    finished_goals: list[tuple[float, float]] = []
    for env_id, xy in enumerate(_get_local_xy(base_env)):
        current_paths[env_id].append((float(xy[0].item()), float(xy[1].item())))
    ref_xy = _get_reference_local_xy(base_env)
    for env_id, goal in enumerate(ref_xy):
        current_goals[env_id] = (float(goal[0].item()), float(goal[1].item()))

    episode_count = 0
    total_steps = 0
    while episode_count < int(args_cli.num_rollouts) and total_steps < int(args_cli.max_steps):
        obs_tensor = torch.as_tensor(obs["policy"], device=base_env.device)
        actions = policy(obs_tensor)
        obs, _, terminated, truncated, _ = env.step(actions)
        xy_now = _get_local_xy(base_env)

        done = terminated | truncated
        if bool(done.any().item()):
            done_ids = torch.nonzero(done, as_tuple=False).flatten().tolist()
            done_idx_set = set(done_ids)
            for env_id, xy in enumerate(xy_now):
                if env_id not in done_idx_set:
                    current_paths[env_id].append((float(xy[0].item()), float(xy[1].item())))
            for env_id in done_ids:
                if len(finished_paths) < int(args_cli.num_rollouts):
                    finished_paths.append(current_paths[env_id].copy())
                    if current_goals[env_id] is not None:
                        finished_goals.append(current_goals[env_id])
                current_paths[env_id].clear()
                current_paths[env_id].append((float(xy_now[env_id, 0].item()), float(xy_now[env_id, 1].item())))
            _assign_fixed_goals(done_ids)

            ref_xy = _get_reference_local_xy(base_env)
            for env_id, goal in enumerate(ref_xy):
                current_goals[env_id] = (float(goal[0].item()), float(goal[1].item()))
            done_eps = int(done.to(torch.int32).sum().item())
            episode_count += done_eps
            print(f"[INFO] Completed rollouts: {episode_count}/{args_cli.num_rollouts}")
        else:
            for env_id, xy in enumerate(xy_now):
                current_paths[env_id].append((float(xy[0].item()), float(xy[1].item())))
        total_steps += 1

    env.close()
    plot_path = _plot_xy_rollouts(
        output_dir,
        env_cfg,
        finished_paths[: int(args_cli.num_rollouts)],
        finished_goals[: int(args_cli.num_rollouts)],
    )
    if plot_path is not None:
        print(f"[INFO] XY trajectory plot: {plot_path}")
    print(f"[INFO] Finished. Recorded up to {min(episode_count, int(args_cli.num_rollouts))} rollout videos in: {video_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
