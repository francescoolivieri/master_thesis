#!/usr/bin/env python3
"""Sample and plot position-tracking curriculum scenarios for generator debugging."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Plot sampled position-tracking curriculum scenarios.")
parser.add_argument("--task", type=str, default="PosTracking-RL-velocity-v0", help="Gym task name.")
parser.add_argument("--num-envs", type=int, default=1, help="Number of envs to instantiate. Use 1 for this tool.")
parser.add_argument("--samples-per-phase", type=int, default=10, help="How many scenarios to plot for each phase.")
parser.add_argument(
    "--phases",
    type=str,
    default=None,
    help="Comma-separated curriculum phases to sample. Defaults to all configured phases.",
)
parser.add_argument(
    "--attempts",
    type=int,
    default=None,
    help="Override pursuit_scenario_attempts for each sampled scenario.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible samples.")
parser.add_argument(
    "--log-dir",
    type=Path,
    default=Path("logs/pos_tracking/generator_debug"),
    help="Directory where plots and metadata will be written.",
)

AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import torch
from isaaclab_tasks.utils.hydra import hydra_task_config

import source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pos_tracking  # noqa: F401


def _parse_phases(phases_arg: str | None, phase_count: int) -> list[int]:
    if phases_arg is None:
        return list(range(1, phase_count + 1))
    phases: list[int] = []
    for token in phases_arg.split(","):
        token = token.strip()
        if not token:
            continue
        phase = int(token)
        if phase < 1 or phase > phase_count:
            raise ValueError(f"Phase {phase} is out of range 1..{phase_count}.")
        phases.append(phase)
    if not phases:
        raise ValueError("No valid phases were provided.")
    return phases


def _to_float_list(values: torch.Tensor) -> list[float]:
    return [float(x) for x in values.detach().cpu().tolist()]


def _dense_path(waypoints: torch.Tensor, samples: int = 250) -> torch.Tensor:
    if waypoints.shape[0] <= 1:
        return waypoints
    seg = waypoints[1:] - waypoints[:-1]
    seg_len = torch.linalg.vector_norm(seg[:, :2], dim=-1).clamp_min(1e-6)
    cumulative = torch.cat((torch.zeros(1, device=waypoints.device), torch.cumsum(seg_len, dim=0)))
    target = torch.linspace(0.0, float(cumulative[-1].item()), samples, device=waypoints.device)
    seg_ids = torch.searchsorted(cumulative[1:], target).clamp(max=seg_len.shape[0] - 1)
    tau = ((target - cumulative[seg_ids]) / seg_len[seg_ids]).view(-1, 1)
    return waypoints[seg_ids] * (1.0 - tau) + waypoints[seg_ids + 1] * tau


def _plot_contact_sheet(image_paths: list[Path], output_path: Path, title: str) -> None:
    cols = 2
    rows = int(math.ceil(len(image_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(9.5, 4.8 * rows))
    axes_list = axes.ravel().tolist() if hasattr(axes, "ravel") else [axes]
    for ax, path in zip(axes_list, image_paths):
        image = plt.imread(path)
        ax.imshow(image)
        ax.set_title(path.stem, fontsize=9)
        ax.axis("off")
    for ax in axes_list[len(image_paths):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_scenario(base_env: Any, scenario: dict[str, Any], phase: int, sample_id: int, output_path: Path) -> dict[str, Any]:
    static_xy = scenario["static_xy"]
    static_active = scenario["static_active"]
    dynamic_waypoints = scenario["dynamic_waypoints"]
    dynamic_active = scenario["dynamic_active"]
    evader_waypoints = scenario["evader_waypoints"]
    pursuer_start = scenario["pursuer_start"]

    static_xy_cpu = static_xy.detach().cpu()
    static_active_cpu = static_active.detach().cpu()
    dynamic_waypoints_cpu = dynamic_waypoints.detach().cpu()
    dynamic_active_cpu = dynamic_active.detach().cpu()
    evader_waypoints_cpu = evader_waypoints.detach().cpu()
    pursuer_start_cpu = pursuer_start.detach().cpu()

    dense_evader = _dense_path(evader_waypoints_cpu)
    time_axis = torch.arange(evader_waypoints_cpu.shape[0], dtype=torch.float32) * float(base_env._step_dt) * float(
        base_env._path_waypoint_stride
    )

    fig = plt.figure(figsize=(10.5, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_z = fig.add_subplot(gs[0, 1])

    arena_min = _to_float_list(base_env._arena_min_safe[:2])
    arena_max = _to_float_list(base_env._arena_max_safe[:2])
    ax_xy.add_patch(
        Rectangle(
            (arena_min[0], arena_min[1]),
            arena_max[0] - arena_min[0],
            arena_max[1] - arena_min[1],
            fill=False,
            lw=1.6,
            ec="black",
        )
    )

    wall_clearance = float(base_env.cfg.pursuit_evader_wall_clearance)
    safe_lo = _to_float_list(base_env._arena_min_safe[:2] + wall_clearance)
    safe_hi = _to_float_list(base_env._arena_max_safe[:2] - wall_clearance)
    ax_xy.add_patch(
        Rectangle(
            (safe_lo[0], safe_lo[1]),
            safe_hi[0] - safe_lo[0],
            safe_hi[1] - safe_lo[1],
            fill=False,
            lw=1.0,
            ls="--",
            ec="#888888",
        )
    )

    ax_xy.scatter(
        [float(pursuer_start_cpu[0])],
        [float(pursuer_start_cpu[1])],
        s=70,
        c="#0b5fff",
        marker="o",
        label="pursuer start",
        zorder=5,
    )
    ax_xy.scatter(
        [float(evader_waypoints_cpu[0, 0])],
        [float(evader_waypoints_cpu[0, 1])],
        s=90,
        c="#00843d",
        marker="*",
        label="evader start",
        zorder=6,
    )
    ax_xy.plot(
        dense_evader[:, 0].numpy(),
        dense_evader[:, 1].numpy(),
        color="#159f5b",
        lw=2.2,
        label="evader path",
        zorder=4,
    )
    ax_xy.scatter(
        evader_waypoints_cpu[:, 0].numpy(),
        evader_waypoints_cpu[:, 1].numpy(),
        s=14,
        c="#159f5b",
        alpha=0.75,
        zorder=4,
    )

    for idx in torch.nonzero(static_active_cpu, as_tuple=False).flatten().tolist():
        center = static_xy_cpu[idx]
        ax_xy.add_patch(
            Circle(
                (float(center[0]), float(center[1])),
                radius=float(base_env.cfg.pillar_radius),
                facecolor="#e59800",
                edgecolor="#9a6400",
                alpha=0.55,
                lw=1.2,
            )
        )
        ax_xy.add_patch(
            Circle(
                (float(center[0]), float(center[1])),
                radius=float(base_env._static_grid_radius()),
                fill=False,
                edgecolor="#e59800",
                alpha=0.25,
                lw=0.9,
                ls=":",
            )
        )

    for idx in torch.nonzero(dynamic_active_cpu, as_tuple=False).flatten().tolist():
        rail = dynamic_waypoints_cpu[:, idx]
        ax_xy.plot(
            rail[:, 0].numpy(),
            rail[:, 1].numpy(),
            color="#b21f66",
            lw=1.8,
            alpha=0.9,
            label="dynamic rail" if idx == 0 else None,
        )
        ax_xy.scatter(
            [float(rail[0, 0]), float(rail[-1, 0])],
            [float(rail[0, 1]), float(rail[-1, 1])],
            c="#b21f66",
            s=26,
            alpha=0.9,
        )
        ax_xy.add_patch(
            Circle(
                (float(rail[0, 0]), float(rail[0, 1])),
                radius=float(base_env.cfg.pursuit_dynamic_obstacle_radius),
                facecolor="#f3b2d0",
                edgecolor="#b21f66",
                alpha=0.35,
                lw=1.0,
            )
        )

    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.set_title("Top-down generator sample")
    ax_xy.grid(alpha=0.18)
    ax_xy.legend(loc="upper right", fontsize=8)

    ax_z.plot(time_axis.numpy(), evader_waypoints_cpu[:, 2].numpy(), color="#159f5b", lw=2.2)
    ax_z.scatter(time_axis.numpy(), evader_waypoints_cpu[:, 2].numpy(), color="#159f5b", s=16)
    ax_z.axhline(float(pursuer_start_cpu[2]), color="#0b5fff", lw=1.2, ls="--", label="pursuer z")
    ax_z.set_xlabel("time [s]")
    ax_z.set_ylabel("z [m]")
    ax_z.set_title("Evader altitude profile")
    ax_z.grid(alpha=0.18)
    ax_z.legend(loc="upper right", fontsize=8)

    static_count = int(static_active_cpu.sum().item())
    dynamic_count = int(dynamic_active_cpu.sum().item())
    fallback = bool(scenario["fallback"])
    path_type = int(scenario["path_type"])
    fig.suptitle(
        f"Phase {phase} sample {sample_id:02d} | static={static_count} dynamic={dynamic_count} "
        f"| path_type={path_type} | fallback={fallback}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "phase": int(phase),
        "sample_id": int(sample_id),
        "plot_path": str(output_path),
        "fallback": fallback,
        "path_type": path_type,
        "static_count": static_count,
        "dynamic_count": dynamic_count,
        "pursuer_start": _to_float_list(pursuer_start_cpu),
        "evader_start": _to_float_list(evader_waypoints_cpu[0]),
        "evader_goal": _to_float_list(evader_waypoints_cpu[-1]),
    }


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, _agent_cfg) -> None:
    env_cfg.scene.num_envs = max(1, int(args_cli.num_envs))
    env_cfg.enable_pursuit_evasion_curriculum = True
    env_cfg.domain_randomization.enable = False
    env_cfg.enable_cameras = False
    env_cfg.enable_ray_caster = False
    env_cfg.debug_vis = False
    env_cfg.debug_visualizer = False
    env_cfg.obstacle_observation_mode = "pillars"
    if args_cli.seed is not None:
        env_cfg.seed = int(args_cli.seed)
        torch.manual_seed(int(args_cli.seed))
    if args_cli.attempts is not None:
        env_cfg.pursuit_scenario_attempts = max(1, int(args_cli.attempts))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args_cli.log_dir.expanduser().resolve() / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        env.reset()
        phase_count = len(tuple(base_env.cfg.pursuit_curriculum_phase_fractions))
        phases = _parse_phases(args_cli.phases, phase_count)

        metadata: dict[str, Any] = {
            "task": args_cli.task,
            "seed": args_cli.seed,
            "samples_per_phase": int(args_cli.samples_per_phase),
            "phases": phases,
            "output_dir": str(output_dir),
            "phase_fractions": [float(x) for x in base_env.cfg.pursuit_curriculum_phase_fractions],
            "samples": [],
        }

        for phase in phases:
            base_env._ensure_obstacle_slots_for_phase(phase)
            phase_dir = output_dir / f"phase_{phase}"
            phase_dir.mkdir(parents=True, exist_ok=True)
            image_paths: list[Path] = []
            for sample_idx in range(1, int(args_cli.samples_per_phase) + 1):
                scenario = base_env._sample_pursuit_scenario_with_fallback(
                    phase,
                    int(base_env.cfg.pursuit_scenario_attempts),
                )
                image_path = phase_dir / f"sample_{sample_idx:02d}.png"
                sample_meta = _plot_scenario(base_env, scenario, phase, sample_idx, image_path)
                metadata["samples"].append(sample_meta)
                image_paths.append(image_path)
            _plot_contact_sheet(image_paths, phase_dir / "contact_sheet.png", title=f"Phase {phase} samples")

        metadata_path = output_dir / "generator_samples.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"[INFO] Wrote generator plots to: {output_dir}")
        print(f"[INFO] Metadata: {metadata_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
