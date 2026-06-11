#!/usr/bin/env python3
"""Plot random rows from the cached pursuit-evasion scenario pools."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Plot random cached pursuit scenarios.")
parser.add_argument("--task", type=str, default="PosTracking-RL-velocity-v0", help="Gym task name.")
parser.add_argument("--pool-size", type=int, default=5000, help="Pool size used to resolve the cache.")
parser.add_argument("--samples-per-phase", type=int, default=4, help="Random rows plotted from each phase.")
parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
parser.add_argument(
    "--log-dir",
    type=Path,
    default=Path("logs/pos_tracking/scenario_pool_plots"),
    help="Output directory.",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, device="cpu")
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaac_pursuit_evasion.tasks.direct.pos_tracking.pos_tracking_env_cfg import (
    pos_tracking_rates_cfg,
    pos_tracking_velocity_cfg,
)
from isaac_pursuit_evasion.tasks.direct.pos_tracking.scenario_pool import (
    enabled_pool_phases,
    load_scenario_pools,
    path_layout,
    scenario_pool_cache_path,
)


def _plot_row(
    cfg,
    phase: int,
    row_id: int,
    pool: dict[str, torch.Tensor],
    output_path: Path,
) -> dict[str, object]:
    path = pool["evader_xy"][row_id]
    static_xy = pool["static_xy"][row_id]
    static_active = pool["static_active"][row_id]
    dynamic_waypoints = pool["dynamic_waypoints"][row_id]
    dynamic_active = pool["dynamic_active"][row_id]
    candidate_count = int(pool["pursuer_candidate_count"][row_id].item())
    candidates = pool["pursuer_candidates_xy"][row_id, :candidate_count]
    candidate_id = int(torch.randint(0, candidate_count, (1,)).item())
    pursuer_xy = candidates[candidate_id]

    arena_min = torch.tensor(cfg.arena_min) + float(cfg.arena_margin)
    arena_max = torch.tensor(cfg.arena_max) - float(cfg.arena_margin)
    static_radius = float(cfg.pillar_radius)
    static_clearance = float(cfg.pillar_radius + cfg.pursuit_evader_radius + cfg.pursuit_obstacle_clearance)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.add_patch(
        Rectangle(
            (float(arena_min[0]), float(arena_min[1])),
            float(arena_max[0] - arena_min[0]),
            float(arena_max[1] - arena_min[1]),
            fill=False,
            color="black",
            lw=1.6,
        )
    )

    for slot in torch.nonzero(static_active, as_tuple=False).flatten().tolist():
        center = static_xy[slot]
        ax.add_patch(
            Circle(
                (float(center[0]), float(center[1])),
                static_radius,
                facecolor="#e59800",
                edgecolor="#8a5c00",
                alpha=0.6,
            )
        )
        ax.add_patch(
            Circle(
                (float(center[0]), float(center[1])),
                static_clearance,
                fill=False,
                edgecolor="#e59800",
                alpha=0.28,
                ls=":",
            )
        )

    for slot in torch.nonzero(dynamic_active, as_tuple=False).flatten().tolist():
        rail = dynamic_waypoints[:, slot, :2]
        ax.plot(rail[:, 0], rail[:, 1], color="#b21f66", lw=1.8, label="dynamic rail" if slot == 0 else None)
        ax.scatter(rail[0, 0], rail[0, 1], color="#b21f66", s=34)

    if candidate_count:
        ax.scatter(candidates[:, 0], candidates[:, 1], color="#4f83cc", s=12, alpha=0.25, label="valid starts")
    ax.scatter(pursuer_xy[0], pursuer_xy[1], color="#0b5fff", s=75, label="sampled pursuer", zorder=6)
    ax.plot(path[:, 0], path[:, 1], color="#159f5b", lw=2.0, label="canonical evader path")
    ax.scatter(path[0, 0], path[0, 1], color="#00843d", marker="*", s=110, label="evader start", zorder=7)

    static_count = int(static_active.sum().item())
    dynamic_count = int(dynamic_active.sum().item())
    ax.set_title(
        f"Phase {phase}, pool row {row_id} | static={static_count}, "
        f"dynamic={dynamic_count}, starts={candidate_count}"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "phase": phase,
        "row": row_id,
        "plot": str(output_path),
        "static_count": static_count,
        "dynamic_count": dynamic_count,
        "pursuer_candidate_count": candidate_count,
    }


def _contact_sheet(paths: list[Path], output_path: Path, title: str) -> None:
    cols = min(2, len(paths))
    rows = int(math.ceil(len(paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 6.2 * rows))
    axes_list = axes.ravel().tolist() if hasattr(axes, "ravel") else [axes]
    for ax, path in zip(axes_list, paths):
        ax.imshow(plt.imread(path))
        ax.axis("off")
    for ax in axes_list[len(paths):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main() -> None:
    cfg_factory = pos_tracking_rates_cfg if "rates" in args.task.lower() else pos_tracking_velocity_cfg
    cfg = cfg_factory(num_envs=1)
    cfg.enable_pursuit_evasion_curriculum = True
    cfg.enable_walls = True
    cfg.enable_pillars = True
    cfg.pursuit_scenario_pool_size = max(1, int(args.pool_size))

    step_dt, _, waypoint_steps = path_layout(cfg)
    pools = load_scenario_pools(cfg, waypoint_steps, step_dt)
    torch.manual_seed(int(args.seed))

    output_dir = args.log_dir.expanduser().resolve() / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "cache": str(scenario_pool_cache_path(cfg, waypoint_steps, step_dt)),
        "seed": int(args.seed),
        "samples_per_phase": int(args.samples_per_phase),
        "samples": [],
    }

    for phase in enabled_pool_phases(cfg):
        pool = pools[phase]
        count = min(max(1, int(args.samples_per_phase)), int(pool["path_type"].shape[0]))
        row_ids = torch.randperm(pool["path_type"].shape[0])[:count].tolist()
        paths: list[Path] = []
        for sample_id, row_id in enumerate(row_ids, start=1):
            path = output_dir / f"phase_{phase}_sample_{sample_id:02d}.png"
            metadata["samples"].append(_plot_row(cfg, phase, row_id, pool, path))
            paths.append(path)
        _contact_sheet(paths, output_dir / f"phase_{phase}_contact_sheet.png", f"Phase {phase} pool samples")

    metadata_path = output_dir / "scenario_pool_samples.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote scenario pool plots to: {output_dir}")
    print(f"[INFO] Metadata: {metadata_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
