#!/usr/bin/env python3
"""Analyze pursuit-evasion or trajectory-tracking datasets and generate summary plots."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab.utils.math import matrix_from_quat
import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]


def flatten_samples(samples: list[np.ndarray]) -> np.ndarray:
    if not samples:
        return np.empty(0, dtype=float)
    return np.concatenate(samples)


def body_z_alignment(quat: np.ndarray) -> np.ndarray:
    """Return dot(+Z_body, +Z_world) using rotation from the quaternion (expects wxyz)."""
    if quat.size == 0:
        return np.empty(0, dtype=float)
    q_torch = torch.as_tensor(quat, dtype=torch.float32, device="cpu")
    rot = matrix_from_quat(q_torch)  # (..., 3, 3)
    # The rotated body +Z axis is the third column of the rotation matrix
    body_up_world = rot[..., 2, 2]
    return body_up_world.cpu().numpy()

def parse_cli() -> tuple[argparse.Namespace, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Analyze logged pursuit-evasion datasets.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Exact path to a dataset file. If omitted, the latest file under --dataset-root is used.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("logs/pursuit_evasion/benchmark"),
        help="Root directory used to auto-discover datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/pursuit_evasion_analysis"),
        help="Directory where plots/statistics will be stored.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=50,
        help="Maximum number of trajectories to overlay in trajectory plots.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config (keys mirroring CLI arguments) to avoid editing args repeatedly.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=("auto", "pursuit", "tracking"),
        default="auto",
        help="Dataset template to expect ('auto' infers from file metadata).",
    )
    defaults = parser.parse_args([])
    args = parser.parse_args()
    return args, defaults


def apply_config(args: argparse.Namespace, defaults: argparse.Namespace) -> argparse.Namespace:
    if args.config and args.config.exists():
        with open(args.config, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, value in data.items():
            if hasattr(defaults, key) and getattr(args, key, None) == getattr(defaults, key, None):
                if key.endswith("_dir") or key.startswith("dataset"):
                    setattr(args, key, Path(value))
                else:
                    setattr(args, key, value)
    return args


def resolve_repo_path(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def discover_latest_dataset(root: Path) -> Optional[Path]:
    candidates = sorted(root.glob("**/*.hdf5"))
    return candidates[-1] if candidates else None


def load_episodes(handler: HDF5DatasetFileHandler, device: str = "cpu") -> Iterable:
    for name in handler.get_episode_names():
        episode = handler.load_episode(name, device=device)
        if episode is not None:
            yield episode


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_hist2d(data: np.ndarray, title: str, xlabel: str, ylabel: str, output_path: Path):
    if data.size == 0:
        return
    plt.figure(figsize=(6, 5))
    plt.hist2d(data[:, 0], data[:, 1], bins=40, cmap="viridis")
    plt.colorbar(label="count")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_hist1d(data: np.ndarray, title: str, xlabel: str, output_path: Path):
    if data.size == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=40, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def load_action_sequences(actions_dir: Optional[Path]) -> list[np.ndarray]:
    if actions_dir is None or not actions_dir.exists():
        return []
    sequences: list[np.ndarray] = []
    for file_path in sorted(actions_dir.glob("*.npz")):
        try:
            data = np.load(file_path)
        except Exception:
            continue
        if "actions" not in data:
            continue
        actions = np.asarray(data["actions"])
        if actions.size == 0:
            continue
        sequences.append(actions)
    return sequences


def compute_action_variation_metrics(sequences: list[np.ndarray]) -> tuple[dict[str, Any], np.ndarray]:
    if not sequences:
        return {}, np.empty(0, dtype=float)
    norm_samples: list[np.ndarray] = []
    abs_samples: list[np.ndarray] = []
    per_agent_norm: dict[str, list[np.ndarray]] = {"pursuer": [], "evader": []}
    per_agent_abs: dict[str, list[np.ndarray]] = {"pursuer": [], "evader": []}

    for seq in sequences:
        if seq.shape[0] < 2:
            continue
        diff = np.diff(seq, axis=0)  # (T-1, agent, dim)
        diff_norm = np.linalg.norm(diff, axis=-1)
        norm_samples.append(diff_norm.reshape(-1))
        diff_abs = np.abs(diff)
        abs_samples.append(diff_abs.reshape(-1, diff_abs.shape[-1]))
        if diff.shape[1] >= 1:
            per_agent_norm["pursuer"].append(diff_norm[:, 0])
            per_agent_abs["pursuer"].append(diff_abs[:, 0, :].reshape(-1, diff_abs.shape[-1]))
        if diff.shape[1] >= 2:
            per_agent_norm["evader"].append(diff_norm[:, 1])
            per_agent_abs["evader"].append(diff_abs[:, 1, :].reshape(-1, diff_abs.shape[-1]))

    if not norm_samples:
        return {}, np.empty(0, dtype=float)

    norm_all = np.concatenate(norm_samples)
    abs_all = np.concatenate(abs_samples) if abs_samples else np.empty((0, seq.shape[-1]))
    metrics: dict[str, Any] = {
        "mean_norm": float(np.mean(norm_all)),
        "max_norm": float(np.max(norm_all)),
    }
    if abs_all.size:
        metrics["per_dim_mean"] = np.mean(abs_all, axis=0).tolist()
        metrics["per_dim_max"] = np.max(abs_all, axis=0).tolist()

    per_agent_metrics: dict[str, Any] = {}
    for agent_name, samples in per_agent_norm.items():
        if not samples:
            continue
        merged = np.concatenate(samples)
        agent_entry = {
            "mean_norm": float(np.mean(merged)),
            "max_norm": float(np.max(merged)),
        }
        abs_samples_agent = per_agent_abs.get(agent_name, [])
        if abs_samples_agent:
            merged_abs = np.concatenate(abs_samples_agent)
            if merged_abs.size:
                agent_entry["per_dim_mean"] = np.mean(merged_abs, axis=0).tolist()
                agent_entry["per_dim_max"] = np.max(merged_abs, axis=0).tolist()
        per_agent_metrics[agent_name] = agent_entry
    if per_agent_metrics:
        metrics["per_agent"] = per_agent_metrics

    return metrics, norm_all


def plot_done_reasons(reason_ids: list[int], reason_map: dict[int, str], output_path: Path):
    unique, counts = np.unique(reason_ids, return_counts=True)
    labels = [reason_map.get(int(idx), str(idx)) for idx in unique]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts)
    plt.title("Termination reasons")
    plt.ylabel("episodes")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def summarize_stats(samples: list[np.ndarray], title_prefix: str, output_path: Path):
    if not samples:
        return
    merged = np.concatenate(samples)
    stats = {
        "mean": np.mean(merged),
        "std": np.std(merged),
        "min": np.min(merged),
        "max": np.max(merged),
    }
    plt.figure(figsize=(5, 3))
    plt.bar(stats.keys(), stats.values(), color="steelblue")
    plt.title(f"{title_prefix} stats")
    plt.ylabel("magnitude")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_trajectories_3d(trajectories: list[np.ndarray], title: str, output_path: Path):
    if not trajectories:
        return
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for traj in trajectories:
        if traj.size == 0:
            continue
        spread = np.ptp(traj, axis=0)
        max_extent = float(np.max(spread))
        if not np.isfinite(max_extent):
            continue
        if max_extent < 0.5:
            sample = traj[:: max(1, traj.shape[0] // 50 + 1)]
            ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], alpha=0.6, s=8)
        else:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_relative_distance_traces(
    traces: list[np.ndarray],
    step_dt: float,
    capture_distance: Optional[float],
    output_path: Path,
):
    if not traces:
        return
    plt.figure(figsize=(8, 4))
    for trace in traces:
        if step_dt > 0.0:
            t = np.arange(trace.shape[0]) * step_dt
            plt.plot(t, trace, alpha=0.4)
        else:
            plt.plot(trace, alpha=0.4)
    plt.title("Relative distance over time")
    plt.xlabel("time [s]" if step_dt > 0.0 else "step")
    plt.ylabel("distance [m]")
    if capture_distance is not None:
        if step_dt > 0.0:
            plt.axhline(
                capture_distance,
                linestyle="--",
                linewidth=1.2,
                color="black",
                label="capture threshold",
            )
            plt.legend()
        else:
            plt.axhline(
                capture_distance,
                linestyle="--",
                linewidth=1.2,
                color="black",
            )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def analyze_pursuit_dataset(
    handler: HDF5DatasetFileHandler,
    args: argparse.Namespace,
    env_args: dict,
    step_dt: float,
    dataset_dir: Optional[Path] = None,
):
    done_reason_map = {int(k): v for k, v in env_args.get("done_reason_map", {}).items()} if env_args else {}
    capture_distance = float(env_args.get("capture_distance", 0.5)) if env_args else 0.5
    tracking_enabled = bool(env_args.get("flag_tracking", False)) or env_args.get("benchmark_task") == "tracking"
    tracking_lower = capture_distance
    tracking_upper = float(env_args.get("tracking_boundary_distance", tracking_lower)) if env_args else tracking_lower
    controller_env_map = env_args.get("controller_env_map", {}) if env_args else {}
    pursuer_map = controller_env_map.get("pursuer", {}) if isinstance(controller_env_map, dict) else {}
    evader_map = controller_env_map.get("evader", {}) if isinstance(controller_env_map, dict) else {}
    # Determine adversary side: prefer the side with multiple controllers, else evader by default.
    if len(evader_map) > 1:
        adversary_side = "evader"
    elif len(pursuer_map) > 1:
        adversary_side = "pursuer"
    else:
        adversary_side = "evader"
    adversary_env_to_name = {}
    for name, ids in (evader_map if adversary_side == "evader" else pursuer_map).items():
        for env_id in ids:
            adversary_env_to_name[int(env_id)] = name

    initial_positions_pursuer = []
    initial_positions_evader = []
    initial_velocities_pursuer = []
    initial_velocities_evader = []
    pursuer_reward_totals = []
    evader_reward_totals = []
    done_reasons = []
    pursuer_trajs = []
    evader_trajs = []
    pursuer_vel_samples = []
    evader_vel_samples = []
    pursuer_rate_samples = []
    evader_rate_samples = []
    pursuer_acc_samples = []
    evader_acc_samples = []
    pursuer_upright_samples = []
    evader_upright_samples = []
    pursuer_inverted_episodes = 0
    evader_inverted_episodes = 0
    relative_distance_samples = []
    relative_speed_samples = []
    relative_traces = []
    min_relative_distances = []
    capture_times = []
    escape_times = []
    capture_final_dists = []
    episode_durations = []
    pursuer_reward_components = defaultdict(list)  # per-episode sums
    evader_reward_components = defaultdict(list)
    evader_max_displacement = []
    pursuer_oob_positions = []
    pursuer_oob_durations = []
    pursuer_oob_min_alt = []
    pursuer_oob_episode_ids = []
    adversary_stats = defaultdict(lambda: {"episodes": 0, "wins": 0, "captures": 0, "escapes": 0, "timeouts": 0})
    tracking_error_sum = 0.0
    tracking_error_max = 0.0
    tracking_inside = 0
    tracking_count = 0

    for ep_idx, episode in enumerate(load_episodes(handler)):
        data = episode.data
        pursuer_state = data["pursuer"]["state"].cpu().numpy()
        evader_state = data["evader"]["state"].cpu().numpy()
        pursuer_reward = data["pursuer"]["reward"].cpu().numpy()
        evader_reward = data["evader"]["reward"].cpu().numpy()
        done_reason = int(data["done_reason"].cpu().numpy()[0])
        env_id = getattr(episode, "env_id", None)
        if env_id is None:
            env_id = ep_idx
        adversary_name = None
        if env_id is not None and env_id in adversary_env_to_name:
            adversary_name = adversary_env_to_name[env_id]
            stats = adversary_stats[adversary_name]
            stats["episodes"] += 1
            if done_reason == 1:
                stats["captures"] += 1
            elif done_reason in (2, 5):
                stats["escapes"] += 1
                stats["timeouts"] += int(done_reason == 5)
            # Wins depend on who is adversary: if adversary is evader, pursuer wins on capture; if adversary is pursuer, evader wins on escape/timeout.
            if adversary_side == "evader" and done_reason == 1:
                stats["wins"] += 1
            if adversary_side == "pursuer" and done_reason in (2, 5):
                stats["wins"] += 1

        initial_positions_pursuer.append(pursuer_state[0, :3])
        initial_positions_evader.append(evader_state[0, :3])
        initial_velocities_pursuer.append(pursuer_state[0, 7:10])
        initial_velocities_evader.append(evader_state[0, 7:10])
        pursuer_reward_totals.append(np.sum(pursuer_reward))
        evader_reward_totals.append(np.sum(evader_reward))
        done_reasons.append(done_reason)

        if len(pursuer_trajs) < args.max_trajectories:
            pursuer_trajs.append(pursuer_state[:, :3])
        if len(evader_trajs) < args.max_trajectories:
            evader_trajs.append(evader_state[:, :3])

        pursuer_vel_samples.append(np.linalg.norm(pursuer_state[:, 7:10], axis=1))
        evader_vel_samples.append(np.linalg.norm(evader_state[:, 7:10], axis=1))
        pursuer_rate_samples.append(np.linalg.norm(pursuer_state[:, 10:13], axis=1))
        evader_rate_samples.append(np.linalg.norm(evader_state[:, 10:13], axis=1))
        if step_dt > 0.0:
            purs_acc = np.diff(pursuer_state[:, 7:10], axis=0) / step_dt
            evd_acc = np.diff(evader_state[:, 7:10], axis=0) / step_dt
            pursuer_acc_samples.append(np.linalg.norm(purs_acc, axis=1))
            evader_acc_samples.append(np.linalg.norm(evd_acc, axis=1))

        rel = pursuer_state[:, :3] - evader_state[:, :3]
        rel_dist = np.linalg.norm(rel, axis=1)
        relative_distance_samples.append(rel_dist)
        if len(relative_traces) < args.max_trajectories:
            relative_traces.append(rel_dist)
        min_relative_distances.append(np.min(rel_dist))
        if tracking_enabled:
            below = np.clip(tracking_lower - rel_dist, 0.0, None)
            above = np.clip(rel_dist - tracking_upper, 0.0, None)
            error = below + above
            tracking_error_sum += float(np.sum(error))
            tracking_error_max = max(tracking_error_max, float(np.max(error)))
            tracking_inside += int(np.sum((rel_dist >= tracking_lower) & (rel_dist <= tracking_upper)))
            tracking_count += int(rel_dist.size)
        rel_speed = np.linalg.norm(pursuer_state[:, 7:10] - evader_state[:, 7:10], axis=1)
        relative_speed_samples.append(rel_speed)
        duration = rel_dist.shape[0] * step_dt if step_dt > 0.0 else rel_dist.shape[0]
        episode_durations.append(duration)

        pursuer_components = data["pursuer"].get("reward_components") if "pursuer" in data else None
        if pursuer_components:
            for name, tensor in pursuer_components.items():
                arr = tensor.cpu().numpy()
                pursuer_reward_components[name].append(float(arr.sum()))
        evader_components = data["evader"].get("reward_components") if "evader" in data else None
        if evader_components:
            for name, tensor in evader_components.items():
                arr = tensor.cpu().numpy()
                evader_reward_components[name].append(float(arr.sum()))

        disp = np.linalg.norm(evader_state[:, :3] - evader_state[0:1, :3], axis=1)
        evader_max_displacement.append(float(np.max(disp)))

        if done_reason == 3:
            pursuer_oob_positions.append(pursuer_state[-1, :3])
            pursuer_oob_durations.append(duration)
            pursuer_oob_min_alt.append(np.min(pursuer_state[:, 2]))
            pursuer_oob_episode_ids.append(ep_idx)

        if step_dt > 0.0:
            duration = rel_dist.shape[0] * step_dt
        else:
            duration = rel_dist.shape[0]
        if done_reason == 1:
            capture_times.append(duration)
            capture_final_dists.append(rel_dist[-1])
        elif done_reason in (2, 5):
            escape_times.append(duration)

        purs_upright = body_z_alignment(pursuer_state[:, 3:7])
        evd_upright = body_z_alignment(evader_state[:, 3:7])
        pursuer_upright_samples.append(purs_upright)
        evader_upright_samples.append(evd_upright)
        if np.any(purs_upright < 0.0):
            pursuer_inverted_episodes += 1
        if np.any(evd_upright < 0.0):
            evader_inverted_episodes += 1

    if not initial_positions_pursuer:
        print("No episodes found in dataset.")
        return

    pursuer_init = np.array(initial_positions_pursuer)
    evader_init = np.array(initial_positions_evader)

    plot_hist2d(pursuer_init[:, :2], "Pursuer initial XY positions", "x [m]", "y [m]", args.output_dir / "pursuer_xy.png")
    plot_hist2d(evader_init[:, :2], "Evader initial XY positions", "x [m]", "y [m]", args.output_dir / "evader_xy.png")
    plot_hist2d(pursuer_init[:, [0, 2]], "Pursuer initial XZ positions", "x [m]", "z [m]", args.output_dir / "pursuer_xz.png")
    plot_hist2d(pursuer_init[:, [1, 2]], "Pursuer initial YZ positions", "y [m]", "z [m]", args.output_dir / "pursuer_yz.png")
    plot_hist2d(evader_init[:, [0, 2]], "Evader initial XZ positions", "x [m]", "z [m]", args.output_dir / "evader_xz.png")
    plot_hist2d(evader_init[:, [1, 2]], "Evader initial YZ positions", "y [m]", "z [m]", args.output_dir / "evader_yz.png")

    pursuer_speed0 = np.linalg.norm(np.array(initial_velocities_pursuer), axis=1)
    evader_speed0 = np.linalg.norm(np.array(initial_velocities_evader), axis=1)
    plot_hist1d(pursuer_speed0, "Pursuer initial speed", "speed [m/s]", args.output_dir / "pursuer_initial_speed.png")
    plot_hist1d(evader_speed0, "Evader initial speed", "speed [m/s]", args.output_dir / "evader_initial_speed.png")

    plot_hist1d(np.array(pursuer_reward_totals), "Pursuer total rewards", "reward", args.output_dir / "pursuer_rewards.png")
    plot_hist1d(np.array(evader_reward_totals), "Evader total rewards", "reward", args.output_dir / "evader_rewards.png")
    plot_done_reasons(done_reasons, done_reason_map, args.output_dir / "done_reasons.png")

    summarize_stats(pursuer_vel_samples, "Pursuer velocity", args.output_dir / "pursuer_velocity_stats.png")
    summarize_stats(evader_vel_samples, "Evader velocity", args.output_dir / "evader_velocity_stats.png")
    summarize_stats(pursuer_rate_samples, "Pursuer body-rate", args.output_dir / "pursuer_bodyrate_stats.png")
    summarize_stats(evader_rate_samples, "Evader body-rate", args.output_dir / "evader_bodyrate_stats.png")
    summarize_stats(pursuer_acc_samples, "Pursuer acceleration", args.output_dir / "pursuer_acc_stats.png")
    summarize_stats(evader_acc_samples, "Evader acceleration", args.output_dir / "evader_acc_stats.png")

    plot_trajectories_3d(pursuer_trajs, "Sample pursuer trajectories (3D)", args.output_dir / "pursuer_trajs.png")
    plot_trajectories_3d(evader_trajs, "Sample evader trajectories (3D)", args.output_dir / "evader_trajs.png")

    purs_upright_all = flatten_samples(pursuer_upright_samples)
    evader_upright_all = flatten_samples(evader_upright_samples)
    if purs_upright_all.size:
        plot_hist1d(
            purs_upright_all,
            "Pursuer body-up alignment",
            "dot(+Z_body, +Z_world)",
            args.output_dir / "pursuer_upright_hist.png",
        )
    if evader_upright_all.size:
        plot_hist1d(
            evader_upright_all,
            "Evader body-up alignment",
            "dot(+Z_body, +Z_world)",
            args.output_dir / "evader_upright_hist.png",
        )

    if relative_distance_samples:
        plot_hist1d(
            np.concatenate(relative_distance_samples),
            "Relative distance distribution",
            "distance [m]",
            args.output_dir / "relative_distance_hist.png",
        )
    if relative_speed_samples:
        plot_hist1d(
            np.concatenate(relative_speed_samples),
            "Relative speed distribution",
            "speed [m/s]",
            args.output_dir / "relative_speed_hist.png",
        )
    if min_relative_distances:
        plot_hist1d(
            np.array(min_relative_distances),
            "Minimum distance per episode",
            "distance [m]",
            args.output_dir / "relative_distance_min.png",
        )
    plot_relative_distance_traces(
        relative_traces,
        step_dt,
        capture_distance,
        args.output_dir / "relative_distance_traces.png",
    )
    if capture_times:
        plot_hist1d(
            np.array(capture_times),
            "Capture durations",
            "time [s]" if step_dt > 0.0 else "steps",
            args.output_dir / "capture_times.png",
        )
    if escape_times:
        plot_hist1d(
            np.array(escape_times),
            "Escape durations",
            "time [s]" if step_dt > 0.0 else "steps",
            args.output_dir / "escape_times.png",
        )
    if capture_final_dists:
        plot_hist1d(
            np.array(capture_final_dists),
            "Distance at capture",
            "distance [m]",
            args.output_dir / "capture_distance.png",
        )
    if episode_durations:
        label = "time [s]" if step_dt > 0.0 else "steps"
        plot_hist1d(
            np.array(episode_durations),
            "Episode durations",
            label,
            args.output_dir / "episode_durations.png",
        )
    if evader_max_displacement:
        plot_hist1d(
            np.array(evader_max_displacement),
            "Evader max displacement per episode",
            "max displacement [m]",
            args.output_dir / "evader_max_displacement.png",
        )

    if pursuer_oob_positions:
        pos = np.array(pursuer_oob_positions)
        plot_hist2d(
            pos[:, :2],
            "Pursuer OOB terminal XY",
            "x [m]",
            "y [m]",
            args.output_dir / "pursuer_oob_xy.png",
        )
        plot_hist1d(
            np.array(pursuer_oob_min_alt),
            "Pursuer OOB minimum altitude",
            "z [m]",
            args.output_dir / "pursuer_oob_min_alt.png",
        )
        label = "time [s]" if step_dt > 0.0 else "steps"
        plot_hist1d(
            np.array(pursuer_oob_durations),
            "Pursuer OOB episode durations",
            label,
            args.output_dir / "pursuer_oob_durations.png",
        )
        with (args.output_dir / "pursuer_oob_episodes.txt").open("w", encoding="utf-8") as f:
            f.write("Episode indices with pursuer out-of-bounds:\n")
            for idx, dur, alt in zip(pursuer_oob_episode_ids, pursuer_oob_durations, pursuer_oob_min_alt):
                f.write(f"episode {idx}: duration={dur:.3f}, min_alt={alt:.3f}\n")

    def plot_reward_component_distributions(samples: dict, agent: str):
        for name, arrays in samples.items():
            values = np.asarray(arrays, dtype=float).reshape(-1)
            if values.size == 0:
                continue
            plot_hist1d(
                values,
                f"{agent.capitalize()} reward component: {name}",
                "episode sum",
                args.output_dir / f"{agent}_reward_{name}.png",
            )

    plot_reward_component_distributions(pursuer_reward_components, "pursuer")
    plot_reward_component_distributions(evader_reward_components, "evader")

    total_eps = len(done_reasons)
    capture_rate = done_reasons.count(1) / total_eps if total_eps else 0.0
    escape_count = done_reasons.count(2) + done_reasons.count(5)
    escape_rate = escape_count / total_eps if total_eps else 0.0
    timeout_rate = done_reasons.count(5) / total_eps if total_eps else 0.0
    oob_rate = (done_reasons.count(3) + done_reasons.count(4)) / total_eps if total_eps else 0.0
    adversary_win_rates = {}
    for name, stats in adversary_stats.items():
        eps = max(1, stats["episodes"])
        adversary_win_rates[name] = {
            "episodes": stats["episodes"],
            "wins": stats["wins"],
            "win_rate": stats["wins"] / eps,
            "captures": stats["captures"],
            "escapes": stats["escapes"],
            "timeouts": stats["timeouts"],
        }
    summary = {
        "episodes": total_eps,
        "capture_rate": capture_rate,
        "escape_rate": escape_rate,
        "timeout_rate": timeout_rate,
        "out_of_bounds_rate": oob_rate,
        "mean_capture_time": float(np.mean(capture_times)) if capture_times else None,
        "mean_escape_time": float(np.mean(escape_times)) if escape_times else None,
        "mean_relative_distance": float(np.mean(np.concatenate(relative_distance_samples))) if relative_distance_samples else None,
        "mean_relative_speed": float(np.mean(np.concatenate(relative_speed_samples))) if relative_speed_samples else None,
        "mean_episode_duration": float(np.mean(episode_durations)) if episode_durations else None,
        "pursuer_oob_count": int(len(pursuer_oob_positions)),
    }
    if adversary_win_rates:
        summary["adversary_win_rates"] = adversary_win_rates
    if pursuer_reward_components:
        comp_summary = {}
        for name, arrays in pursuer_reward_components.items():
            flat = np.asarray(arrays, dtype=float).reshape(-1)
            if flat.size:
                comp_summary[name] = float(np.mean(flat))
        if comp_summary:
            summary["pursuer_reward_components"] = comp_summary
    if evader_reward_components:
        comp_summary = {}
        for name, arrays in evader_reward_components.items():
            flat = np.asarray(arrays, dtype=float).reshape(-1)
            if flat.size:
                comp_summary[name] = float(np.mean(flat))
        if comp_summary:
            summary["evader_reward_components"] = comp_summary
    if purs_upright_all.size:
        summary["pursuer_upright_mean"] = float(np.mean(purs_upright_all))
        summary["pursuer_upright_std"] = float(np.std(purs_upright_all))
        summary["pursuer_inversion_rate"] = float(np.mean(purs_upright_all < 0.0))
        summary["pursuer_inverted_episode_frac"] = float(pursuer_inverted_episodes / total_eps)
    if evader_upright_all.size:
        summary["evader_upright_mean"] = float(np.mean(evader_upright_all))
        summary["evader_upright_std"] = float(np.std(evader_upright_all))
        summary["evader_inversion_rate"] = float(np.mean(evader_upright_all < 0.0))
        summary["evader_inverted_episode_frac"] = float(evader_inverted_episodes / total_eps)
    if evader_max_displacement:
        summary["evader_max_displacement_mean"] = float(np.mean(evader_max_displacement))
        summary["evader_max_displacement_max"] = float(np.max(evader_max_displacement))
    if pursuer_oob_positions:
        summary["pursuer_oob_min_alt_mean"] = float(np.mean(pursuer_oob_min_alt))
        summary["pursuer_oob_duration_mean"] = float(np.mean(pursuer_oob_durations))
    if tracking_enabled and tracking_count > 0:
        summary["tracking_metrics"] = {
            "mean_error": tracking_error_sum / tracking_count,
            "max_error": tracking_error_max,
            "inside_rate": tracking_inside / tracking_count,
            "capture_distance": tracking_lower,
            "tracking_boundary_distance": tracking_upper,
        }

    actions_dir = dataset_dir / "actions" if dataset_dir is not None else None
    action_sequences = load_action_sequences(actions_dir)
    action_metrics, action_norms = compute_action_variation_metrics(action_sequences)
    if action_metrics:
        summary["action_variation"] = action_metrics
    if action_norms.size:
        plot_hist1d(
            action_norms,
            "Action delta norm",
            "||a_t - a_{t-1}||",
            args.output_dir / "action_delta_norm_hist.png",
        )
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Analysis complete. Plots saved to {args.output_dir}")


def analyze_tracking_dataset(handler: HDF5DatasetFileHandler, args: argparse.Namespace, env_args: dict, step_dt: float):
    initial_positions = []
    initial_velocities = []
    reward_totals = []
    trajectory_samples = []
    speed_samples = []
    rate_samples = []
    acc_samples = []
    done_reasons = []

    for episode in load_episodes(handler):
        data = episode.data
        states = data.get("states", {})
        if not states:
            continue
        pos = states.get("position")
        vel = states.get("velocity")
        if pos is None or vel is None:
            continue
        pos_np = pos.cpu().numpy()
        vel_np = vel.cpu().numpy()
        if pos_np.size == 0 or vel_np.size == 0:
            continue
        initial_positions.append(pos_np[0])
        initial_velocities.append(vel_np[0])
        if len(trajectory_samples) < args.max_trajectories:
            trajectory_samples.append(pos_np)
        speed_samples.append(np.linalg.norm(vel_np, axis=1))

        ang = states.get("angular_velocity")
        if ang is not None:
            ang_np = ang.cpu().numpy()
            if ang_np.size > 0:
                rate_samples.append(np.linalg.norm(ang_np, axis=1))

        if step_dt > 0.0 and vel_np.shape[0] > 1:
            acc = np.diff(vel_np, axis=0) / step_dt
            acc_samples.append(np.linalg.norm(acc, axis=1))

        reward = data.get("reward")
        if reward is not None:
            reward_totals.append(float(reward.cpu().numpy().sum()))

        done = data.get("done", {})
        if isinstance(done, dict) and "reason" in done:
            reason_vals = done["reason"].cpu().numpy().astype(int).flatten().tolist()
            done_reasons.extend(reason_vals)

    if not initial_positions:
        print("No episodes found in dataset.")
        return

    init_pos = np.array(initial_positions)
    plot_hist2d(init_pos[:, :2], "Initial XY positions", "x [m]", "y [m]", args.output_dir / "initial_xy.png")
    plot_hist2d(init_pos[:, [0, 2]], "Initial XZ positions", "x [m]", "z [m]", args.output_dir / "initial_xz.png")
    plot_hist2d(init_pos[:, [1, 2]], "Initial YZ positions", "y [m]", "z [m]", args.output_dir / "initial_yz.png")

    init_speed = np.linalg.norm(np.array(initial_velocities), axis=1)
    plot_hist1d(init_speed, "Initial speed magnitude", "speed [m/s]", args.output_dir / "initial_speed.png")

    if reward_totals:
        plot_hist1d(np.array(reward_totals), "Episode reward totals", "reward", args.output_dir / "reward_totals.png")

    if speed_samples:
        merged_speed = np.concatenate(speed_samples)
        plot_hist1d(merged_speed, "Speed distribution", "speed [m/s]", args.output_dir / "speed_hist.png")
        summarize_stats(speed_samples, "Velocity magnitude", args.output_dir / "velocity_stats.png")
    if rate_samples:
        merged_rate = np.concatenate(rate_samples)
        plot_hist1d(merged_rate, "Body-rate distribution", "rad/s", args.output_dir / "bodyrate_hist.png")
        summarize_stats(rate_samples, "Body-rate", args.output_dir / "bodyrate_stats.png")
    if acc_samples:
        merged_acc = np.concatenate(acc_samples)
        plot_hist1d(merged_acc, "Acceleration distribution", "m/s^2", args.output_dir / "acceleration_hist.png")
        summarize_stats(acc_samples, "Acceleration", args.output_dir / "acceleration_stats.png")

    reason_map = {0: "unfinished", 1: "terminated", 2: "truncated"}
    if done_reasons:
        plot_done_reasons(done_reasons, reason_map, args.output_dir / "done_reasons.png")

    plot_trajectories_3d(trajectory_samples, "Sample trajectories (3D)", args.output_dir / "trajectories_3d.png")

    print(f"Tracking dataset analysis complete. Plots saved to {args.output_dir}")


def main():
    args, defaults = parse_cli()
    args = apply_config(args, defaults)
    args.dataset = resolve_repo_path(args.dataset)
    args.dataset_root = resolve_repo_path(args.dataset_root)
    args.output_dir = resolve_repo_path(args.output_dir) or (_REPO_ROOT / "logs/pursuit_evasion_analysis").resolve()
    defaults.output_dir = resolve_repo_path(defaults.output_dir)
    defaults.dataset_root = resolve_repo_path(defaults.dataset_root)
    if args.dataset is None:
        latest = discover_latest_dataset(args.dataset_root)
        if latest is None:
            raise FileNotFoundError(f"No dataset found under {args.dataset_root}")
        args.dataset = latest
        print(f"[INFO] Using latest dataset: {args.dataset}")
    elif args.dataset.is_dir():
        candidate = args.dataset / "episodes.hdf5"
        if candidate.exists():
            args.dataset = candidate
        else:
            raise FileNotFoundError(f"No episodes.hdf5 found in dataset directory {args.dataset}")

    dataset_dir = args.dataset.parent if args.dataset.is_file() else args.dataset

    # Auto-select output directory if not specified
    if args.output_dir == defaults.output_dir and args.dataset is not None:
        base_dir = args.dataset.parent if args.dataset.is_file() else args.dataset
        benchmark_root = base_dir.parent
        if benchmark_root.name == "benchmark":
            analysis_root = benchmark_root.parent / "analysis"
        else:
            analysis_root = benchmark_root / "analysis"
        args.output_dir = analysis_root / base_dir.name

    handler = HDF5DatasetFileHandler()
    handler.open(str(args.dataset), mode="r")
    env_args_raw = handler._hdf5_data_group.attrs.get("env_args", "{}")
    env_args = json.loads(env_args_raw) if isinstance(env_args_raw, str) else {}
    step_dt = float(env_args.get("step_dt", 0.0)) if env_args else 0.0

    env_name = env_args.get("env_name", "") if env_args else ""
    inferred_type = "tracking" if "Trajectory-Tracking" in env_name else "pursuit"
    dataset_type = args.dataset_type if args.dataset_type != "auto" else inferred_type
    if args.output_dir == defaults.output_dir and dataset_type == "tracking":
        args.output_dir = Path("logs/trajectory_following_analysis")
    ensure_dir(args.output_dir)
    print(f"[INFO] Dataset type resolved to '{dataset_type}'.")

    metadata_path = args.output_dir / "env_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(env_args, f, indent=2)

    if dataset_type == "tracking":
        analyze_tracking_dataset(handler, args, env_args, step_dt)
    else:
        analyze_pursuit_dataset(handler, args, env_args, step_dt, dataset_dir=dataset_dir)
    handler.close()


if __name__ == "__main__":
    main()
