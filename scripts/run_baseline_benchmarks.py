#!/usr/bin/env python3
"""Run baseline pursuit–evasion benchmarks and analyze the results automatically."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_SCRIPT = REPO_ROOT / "scripts" / "benchmark" / "bench_pursuit_evasion.py"
ANALYZE_SCRIPT = REPO_ROOT / "scripts" / "analyze_episode_stats.py"
BENCHMARK_ROOT = REPO_ROOT / "logs" / "pursuit_evasion" / "benchmark"

DEFAULT_TASKS = (
    "Bench-frpn-vs-apf",
    "Bench-frpn-vs-hover",
    "Bench-frpn-vs-trajectories",
    "Bench-slowfrpn-vs-apf",
)


def _run_cmd(cmd: Sequence[str]) -> int:
    print(f"[INFO] Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


def _exp_id(task: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{task}-{timestamp}"


def run_benchmark(
    task: str, num_envs: int, num_steps: int, num_episodes: int | None, extra_args: Iterable[str]
) -> Path | None:
    exp_id = _exp_id(task)
    cmd: List[str] = [
        sys.executable,
        str(BENCH_SCRIPT),
        "--task",
        task,
        "--num-envs",
        str(num_envs),
        "--num-steps",
        str(num_steps),
        "--log-episodes",
        "--log-observations",
        "--log-actions",
        "--exp-id",
        exp_id,
    ]
    if num_episodes is not None:
        cmd.extend(["--num-episodes", str(num_episodes)])
    cmd.extend(extra_args)
    ret = _run_cmd(cmd)
    if ret != 0:
        print(f"[WARN] Benchmark for task '{task}' failed with exit code {ret}")
        return None

    dataset = BENCHMARK_ROOT / exp_id / "episodes.hdf5"
    if not dataset.exists():
        print(f"[WARN] Benchmark completed but dataset not found at {dataset}")
        return None
    return dataset


def analyze_dataset(dataset: Path, extra_args: Iterable[str]) -> int:
    cmd: List[str] = [
        sys.executable,
        str(ANALYZE_SCRIPT),
        "--dataset",
        str(dataset),
    ]
    cmd.extend(extra_args)
    return _run_cmd(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline pursuit–evasion benchmarks and analyze them.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="Task ids to benchmark (Gym ids registered in pursuit_evasion/__init__.py).",
    )
    parser.add_argument("--num-envs", type=int, default=1024, help="Number of environments per run.")
    parser.add_argument("--num-steps", type=int, default=500, help="Number of steps per run.")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Stop each benchmark after collecting this many episodes (per task).",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Pass --headless to the benchmark script (disables viewer)."
    )
    parser.add_argument(
        "--task-mode",
        type=str,
        default="auto",
        choices=["auto", "pursuit_evasion", "tracking"],
        help="Override tracking flag for benchmarks (default: auto uses task config).",
    )
    parser.add_argument("--video", action="store_true", help="Record video in the benchmark script.")
    parser.add_argument(
        "--skip-analysis", action="store_true", help="Skip analyze_episode_stats after running benchmarks."
    )
    parser.add_argument("--disable-cameras", action="store_true", help="Disable FPV cameras during benchmarks.")
    parser.add_argument("--save-camera-images", action="store_true", help="Save per-step FPV images during benchmarks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extra_bench_args: List[str] = []
    if args.headless:
        extra_bench_args.append("--headless")
    if args.video:
        extra_bench_args.append("--video")
    if args.disable_cameras:
        extra_bench_args.append("--disable-cameras")
    if args.save_camera_images:
        extra_bench_args.append("--save-camera-images")
    if args.task_mode != "auto":
        extra_bench_args.extend(["--task-mode", args.task_mode])

    extra_analyze_args: List[str] = []

    datasets: list[Path] = []
    for task in args.tasks:
        dataset = run_benchmark(task, args.num_envs, args.num_steps, args.num_episodes, extra_bench_args)
        if dataset:
            datasets.append(dataset)

    if args.skip_analysis or not datasets:
        return

    for dataset in datasets:
        ret = analyze_dataset(dataset, extra_analyze_args)
        if ret != 0:
            print(f"[WARN] Analysis failed for dataset {dataset} with exit code {ret}")


if __name__ == "__main__":
    main()
