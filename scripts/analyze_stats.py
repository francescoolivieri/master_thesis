#!/usr/bin/env python3
"""Summarize position-tracking benchmark logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from isaaclab.utils.datasets import HDF5DatasetFileHandler


def _resolve(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def _discover_latest(root: Path, pattern: str) -> Optional[Path]:
    candidates = sorted(root.glob(f"**/{pattern}"))
    return candidates[-1] if candidates else None


def _compute_metrics_from_dataset(path: Path) -> dict[str, Any]:
    handler = HDF5DatasetFileHandler()
    handler.open(str(path))
    total_steps = 0
    sum_pos_error = 0.0
    sum_pos_error_sq = 0.0
    sum_yaw_error = 0.0
    sum_reward = 0.0
    reward_components: dict[str, float] = {}

    success_count = 0
    crash_count = 0
    oob_count = 0
    timeout_count = 0
    invalid_count = 0
    episode_lengths: list[int] = []
    success_times: list[float] = []

    env_args_raw = handler._hdf5_data_group.attrs.get("env_args", "{}")
    env_args = json.loads(env_args_raw) if isinstance(env_args_raw, str) else {}
    step_dt = env_args.get("step_dt", None)

    for name in handler.get_episode_names():
        episode = handler.load_episode(name, device="cpu")
        if episode is None:
            continue
        data = episode.data
        pos_error = np.asarray(data.get("pos_error", []))
        if pos_error.size == 0:
            state = np.asarray(data.get("state", []))
            ref = np.asarray(data.get("reference", []))
            if state.size and ref.size:
                pos = state[:, :3]
                ref_pos = ref[:, :3]
                pos_error = np.linalg.norm(ref_pos - pos, axis=-1, keepdims=True)
        pos_error = pos_error.reshape(-1)
        if pos_error.size:
            sum_pos_error += float(pos_error.sum())
            sum_pos_error_sq += float((pos_error ** 2).sum())
            total_steps += pos_error.size

        yaw_error = np.asarray(data.get("yaw_error", []))
        yaw_error = yaw_error.reshape(-1)
        if yaw_error.size:
            sum_yaw_error += float(np.abs(yaw_error).sum())

        reward = np.asarray(data.get("reward", []))
        if reward.size:
            sum_reward += float(reward.sum())

        components = data.get("reward_components", {})
        for key, values in components.items():
            arr = np.asarray(values).reshape(-1)
            if arr.size:
                reward_components[key] = reward_components.get(key, 0.0) + float(arr.sum())

        reason = int(np.asarray(data.get("done_reason", [0]))[0])
        length = pos_error.size if pos_error.size else int(np.asarray(data.get("timesteps", [])).shape[0])
        episode_lengths.append(length)
        if reason == 1:
            success_count += 1
            if step_dt is not None:
                success_times.append(length * float(step_dt))
        elif reason == 2:
            crash_count += 1
        elif reason == 3:
            oob_count += 1
        elif reason == 4:
            timeout_count += 1
        elif reason == 5:
            invalid_count += 1

    handler.close()

    mean_pos_error = sum_pos_error / max(1, total_steps)
    rms_pos_error = (sum_pos_error_sq / max(1, total_steps)) ** 0.5
    mean_yaw_error = sum_yaw_error / max(1, total_steps) if sum_yaw_error else None
    mean_reward = sum_reward / max(1, total_steps)

    return {
        "num_episodes": len(episode_lengths),
        "num_steps": total_steps,
        "mean_pos_error": mean_pos_error,
        "rms_pos_error": rms_pos_error,
        "mean_yaw_error": mean_yaw_error,
        "success_rate": success_count / max(1, len(episode_lengths)),
        "mean_time_to_success": (sum(success_times) / max(1, len(success_times))) if success_times else None,
        "crash_rate": crash_count / max(1, len(episode_lengths)),
        "out_of_bounds_rate": oob_count / max(1, len(episode_lengths)),
        "timeout_rate": timeout_count / max(1, len(episode_lengths)),
        "invalid_rate": invalid_count / max(1, len(episode_lengths)),
        "mean_episode_length": (sum(episode_lengths) / max(1, len(episode_lengths))) if episode_lengths else None,
        "mean_total_reward": mean_reward,
        "reward_components": {k: v / max(1, total_steps) for k, v in reward_components.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize pos-tracking benchmark logs.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/pos_tracking/benchmark"), help="Log root.")
    parser.add_argument("--metrics", type=Path, default=None, help="Path to metrics.json.")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to episodes.hdf5.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    log_dir = _resolve(args.log_dir)
    metrics_path = _resolve(args.metrics)
    dataset_path = _resolve(args.dataset)

    if metrics_path is None and log_dir is not None:
        metrics_path = _discover_latest(log_dir, "metrics.json")
    if dataset_path is None and log_dir is not None:
        dataset_path = _discover_latest(log_dir, "episodes.hdf5")

    summary: dict[str, Any] = {}
    if metrics_path and metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    elif dataset_path and dataset_path.exists():
        summary = _compute_metrics_from_dataset(dataset_path)
    else:
        raise FileNotFoundError("No metrics.json or episodes.hdf5 found to analyze.")

    print("Pos-Tracking Benchmark Summary")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  - {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")

    if args.output:
        output_path = _resolve(args.output)
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
