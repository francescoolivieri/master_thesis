# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import re
import sys
import time
import types
from pathlib import Path
from typing import Any, Mapping, Optional
from omegaconf import DictConfig, OmegaConf

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--vidoe", dest="video", action="store_true", help="Alias for --video.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--video-interval-frames",
    type=int,
    default=None,
    help="Interval between video recordings in frames (num_envs * steps).",
)
# Backward compatible alias (deprecated)
parser.add_argument(
    "--video_interval_frames",
    dest="video_interval_frames",
    type=int,
    default=None,
    help=argparse.SUPPRESS,
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--total_frames",
    type=int,
    default=None,
    help="Total frames to collect during training (converted to timesteps via num_envs). Overrides --max_iterations.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO", "DGPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--wandb-name", type=str, help="Override WandB run name (agent.agent.experiment.wandb_kwargs.name).")
parser.add_argument("--wandb-id", type=str, help="Override WandB run id (agent.agent.experiment.wandb_kwargs.id).")
parser.add_argument(
    "--wandb-project", type=str, help="Override WandB project (agent.agent.experiment.wandb_kwargs.project)."
)
parser.add_argument(
    "--wandb-entity", type=str, help="Override WandB entity (agent.agent.experiment.wandb_kwargs.entity)."
)
parser.add_argument("--wandb-dir", type=str, help="Override WandB local dir (agent.agent.experiment.wandb_kwargs.dir).")
parser.add_argument("--disable-fpv-cameras", action="store_true", help="Disable FPV cameras in the environment.")
parser.add_argument("--disable-evader-cameras", action="store_true", help="Disable only evader FPV cameras.")
parser.add_argument(
    "--domain-randomization",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Enable/disable domain randomization (overrides env config).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments

args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.disable_fpv_cameras:
    # Keep rendering cameras enabled; only disable FPV sensors in env config later.
    args_cli.enable_cameras = True


def _normalize_overrides(overrides: list[str]) -> list[str]:
    """Support shorter alias 'agent.experiment.*' by expanding to 'agent.agent.experiment.*'."""
    normalized: list[str] = []
    for item in overrides:
        if item.startswith("agent.experiment."):
            normalized.append("agent.agent." + item[len("agent.") :])
        else:
            normalized.append(item)
    return normalized


# Inject WandB overrides into Hydra args so users don't need full dotted paths.
wandb_overrides: list[str] = []
if args_cli.wandb_name:
    wandb_overrides.append(f"agent.agent.experiment.wandb_kwargs.name={args_cli.wandb_name}")
if args_cli.wandb_id:
    wandb_overrides.append(f"agent.agent.experiment.wandb_kwargs.id={args_cli.wandb_id}")
if args_cli.wandb_project:
    wandb_overrides.append(f"agent.agent.experiment.wandb_kwargs.project={args_cli.wandb_project}")
if args_cli.wandb_entity:
    wandb_overrides.append(f"agent.agent.experiment.wandb_kwargs.entity={args_cli.wandb_entity}")
if args_cli.wandb_dir:
    wandb_overrides.append(f"agent.agent.experiment.wandb_kwargs.dir={args_cli.wandb_dir}")

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + _normalize_overrides(hydra_args + wandb_overrides)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Workaround for a known bug in omni.replicator.core annotator_utils.py where
# /rtx/dataWindowNDC/* settings return None when not initialised, causing
# `None - None` arithmetic in _resize_data_for_overscan.  Setting
# fitOutputToDataWindow=True makes the guard condition evaluate to False and
# skips the overscan-resize block entirely.
import carb as _carb

_carb.settings.get_settings().set("/rtx/dataWindow/fitOutputToDataWindow", True)

import gymnasium as gym
import os
import random
from datetime import datetime

import numpy as np
from isaaclab.utils import math as math_utils
import torch

import omni
import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct  # noqa: F401
from source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pos_tracking.pos_tracking_env import (
    DONE_REASON_LABELS,
)


class PerformanceTracker:
    """Compute environment and frame throughput for logging and monitoring."""

    def __init__(
        self,
        num_envs: int,
        num_envs_fn: Optional[callable] = None,
        window_sec: float = 2.0,
        console_interval_sec: float = 30.0,
    ) -> None:
        self._num_envs = max(1, num_envs)
        self._num_envs_fn = num_envs_fn
        self._window_sec = max(1e-3, window_sec)
        self._console_interval_sec = max(1e-3, console_interval_sec)
        now = time.perf_counter()
        self._last_time = now
        self._last_console_time = now
        self._last_step = 0
        self.env_steps_per_sec = 0.0
        self.frames_per_sec = 0.0

    def update(self, current_step: int) -> Optional[tuple[dict[str, float], bool]]:
        """Update the running throughput estimates.

        Returns a tuple with the latest metrics and a flag indicating if they should be printed.
        """
        now = time.perf_counter()
        elapsed = now - self._last_time
        if elapsed < self._window_sec:
            return None
        steps = current_step - self._last_step
        if steps <= 0:
            self._last_time = now
            return None

        num_envs = self._num_envs_fn() if self._num_envs_fn else self._num_envs
        num_envs = max(1, int(num_envs))
        self._num_envs = num_envs

        self.env_steps_per_sec = steps / max(elapsed, 1e-9)
        self.frames_per_sec = self.env_steps_per_sec * num_envs
        self._last_time = now
        self._last_step = current_step

        should_print = now - self._last_console_time >= self._console_interval_sec
        if should_print:
            self._last_console_time = now
        return (
            {
                "perf/env_steps_per_sec": self.env_steps_per_sec,
                "perf/frames_per_sec": self.frames_per_sec,
            },
            should_print,
        )


class WandbBridge:
    """Thin wrapper around wandb to keep imports optional and failures non-fatal."""

    def __init__(self) -> None:
        self._module = None
        self._failed = False

    def _ensure_module(self):
        if self._module or self._failed:
            return self._module
        try:
            import wandb  # type: ignore

            self._module = wandb
        except Exception:
            self._failed = True
            self._module = None
        return self._module

    def is_active(self) -> bool:
        module = self._ensure_module()
        return bool(module and module.run is not None)

    def _log_kwargs(self, module, step: Optional[int]) -> dict[str, int]:
        if step is None:
            return {}
        try:
            run = module.run
            settings = getattr(run, "settings", None)
            if bool(getattr(settings, "sync_tensorboard", False)):
                # W&B rejects explicit steps when TensorBoard sync is enabled.
                return {}
        except Exception:
            pass
        return {"step": int(step)}

    def log(self, data: dict[str, float], step: Optional[int] = None) -> None:
        module = self._ensure_module()
        if module and module.run is not None:
            try:
                module.log(data, **self._log_kwargs(module, step))
            except Exception:
                pass

    def log_histogram(self, key: str, values, step: Optional[int] = None, title: Optional[str] = None) -> None:
        module = self._ensure_module()
        if module and module.run is not None:
            try:
                import numpy as _np
                import torch

                array = _np.asarray(values, dtype=_np.float32).flatten()
                if array.size == 0:
                    return
                tensor = torch.as_tensor(array)
                min_val = float(torch.min(tensor))
                max_val = float(torch.max(tensor))
                data: list[list[float]] = []
                if max_val - min_val < 1e-6:
                    data = [[min_val]]
                else:
                    counts, edges = torch.histogram(tensor, bins=10)
                    for idx in range(len(counts)):
                        count = int(counts[idx].item())
                        if count <= 0:
                            continue
                        center = float((edges[idx].item() + edges[idx + 1].item()) * 0.5)
                        repeats = max(1, min(count, 200))
                        data.extend([[center]] * repeats)
                        if len(data) > 5000:
                            break
                if not data:
                    data = [[float(torch.mean(tensor).item())]]
                table = module.Table(columns=["scores"], data=data)
                plot = module.plot.histogram(table, "scores", title=title or key)
                module.log({key: plot}, **self._log_kwargs(module, step))
            except Exception:
                pass

    def log_bar(self, key: str, labels: list[str], values: list[float], step: Optional[int] = None, title: Optional[str] = None) -> None:
        module = self._ensure_module()
        if module and module.run is not None:
            try:
                if not labels or not values:
                    return
                table = module.Table(columns=["label", "value"], data=[[label, float(val)] for label, val in zip(labels, values)])
                plot = module.plot.bar(table, "label", "value", title=title or key)
                module.log({key: plot}, **self._log_kwargs(module, step))
            except Exception:
                pass

    def log_video(self, path: Path, key: str, fps: int, step: Optional[int] = None) -> None:
        module = self._ensure_module()
        if module and module.run is not None and path.exists():
            try:
                module.log({key: module.Video(str(path), format="mp4", caption=path.stem)}, step=None)
            except Exception:
                pass

    def get_run_name(self) -> Optional[str]:
        """Return the active wandb run name (or id as fallback)."""
        module = self._ensure_module()
        if module and module.run is not None:
            run = module.run
            name = getattr(run, "name", None)
            if name:
                return str(name)
            run_id = getattr(run, "id", None)
            if run_id:
                return str(run_id)
        return None

    def finish(self, exit_code: Optional[int] = None) -> None:
        module = self._ensure_module()
        if module and module.run is not None:
            try:
                kwargs = {}
                if exit_code is not None:
                    kwargs["exit_code"] = int(exit_code)
                module.finish(**kwargs)
            except Exception:
                pass

    def log_artifact(
        self,
        artifact_name: str,
        files: list[Path],
        *,
        artifact_type: str = "model",
        description: Optional[str] = None,
        aliases: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Log a wandb artifact built from a list of files."""
        module = self._ensure_module()
        if not (module and module.run is not None):
            return False
        paths: list[Path] = []
        for path in files:
            try:
                if path.exists():
                    paths.append(path)
            except Exception:
                continue
        if not paths:
            return False
        try:
            artifact = module.Artifact(artifact_name, type=artifact_type, description=description, metadata=metadata)
            for path in paths:
                artifact.add_file(str(path))
            module.log_artifact(artifact, aliases=aliases)
            return True
        except Exception:
            return False


class WandbVideoLogger:
    """Upload new video files from a directory to wandb at a controlled cadence."""

    def __init__(
        self,
        bridge: WandbBridge,
        video_directory: Path,
        fps: int,
        log_key: str = "videos/train",
        check_interval_sec: float = 30.0,
    ) -> None:
        self._bridge = bridge
        self._video_directory = video_directory
        self._fps = fps
        self._log_key = log_key
        self._check_interval_sec = max(1.0, check_interval_sec)
        self._last_check = 0.0
        self._uploaded: set[str] = set()

    def poll(self, step: Optional[int] = None) -> None:
        if not self._bridge.is_active():
            return
        now = time.perf_counter()
        if now - self._last_check < self._check_interval_sec:
            return
        self._last_check = now
        if not self._video_directory.exists():
            return
        for video_path in sorted(self._video_directory.glob("*.mp4")):
            if video_path.name in self._uploaded:
                continue
            self._bridge.log_video(video_path, self._log_key, self._fps, step=step)
            self._uploaded.add(video_path.name)


class WandbCheckpointUploader:
    """Upload newly written checkpoints to wandb as artifacts."""

    def __init__(
        self,
        bridge: WandbBridge,
        checkpoints_dir: Path,
        *,
        artifact_type: str = "model",
        metadata: Optional[dict] = None,
    ) -> None:
        self._bridge = bridge
        self._dir = checkpoints_dir
        self._artifact_type = artifact_type
        self._metadata = metadata or {}
        self._known_mtimes: dict[str, float] = {}
        self._record_existing()

    def _record_existing(self) -> None:
        if not self._dir.exists():
            return
        for path in self._dir.glob("*.pt"):
            try:
                self._known_mtimes[path.name] = path.stat().st_mtime
            except FileNotFoundError:
                continue

    def _sanitize_artifact_name(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        safe = "".join(ch if (ch.isalnum() or ch in "-._:") else "-" for ch in str(name))
        safe = safe.strip("-._:")
        return safe or None

    def _extract_step(self, stem: str) -> Optional[int]:
        match = re.search(r"([0-9]+)$", stem)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
        return None

    def _build_aliases(self, path: Path) -> list[str]:
        aliases: list[str] = ["latest"]
        stem = path.stem
        if stem.startswith("best"):
            aliases.append("best")
        step_val = self._extract_step(stem)
        if step_val is not None:
            aliases.append(f"step-{step_val}")
        safe_stem = stem.replace(" ", "_").replace("/", "_")
        if safe_stem and safe_stem not in aliases:
            aliases.append(safe_stem)
        return aliases

    def upload_new(self) -> None:
        if not self._bridge.is_active():
            return
        run_name = self._bridge.get_run_name()
        sanitized_name = self._sanitize_artifact_name(run_name)
        names_to_try: list[str] = []
        if run_name:
            names_to_try.append(str(run_name))
        if sanitized_name and sanitized_name not in names_to_try:
            names_to_try.append(sanitized_name)
        if not names_to_try:
            return
        if not self._dir.exists():
            return
        for path in sorted(self._dir.glob("*.pt")):
            try:
                mtime = path.stat().st_mtime
            except FileNotFoundError:
                continue
            prev_mtime = self._known_mtimes.get(path.name)
            if prev_mtime is not None and mtime <= prev_mtime:
                continue

            aliases = self._build_aliases(path)
            description = f"Checkpoint {path.name}"
            step_val = self._extract_step(path.stem)
            metadata = dict(self._metadata) if self._metadata else {}
            if run_name:
                metadata.setdefault("run_name", run_name)
            if step_val is not None:
                metadata["step"] = step_val
            if not metadata:
                metadata = None
            uploaded = False
            for name in names_to_try:
                uploaded = self._bridge.log_artifact(
                    name,
                    [path],
                    artifact_type=self._artifact_type,
                    description=description,
                    aliases=aliases,
                    metadata=metadata,
                )
                if uploaded:
                    break
            if uploaded:
                self._known_mtimes[path.name] = mtime


def _build_evader_strategy_map(base_env) -> dict[int, str]:
    assignment = getattr(base_env, "_evader_controller_assignment", {})
    mapping: dict[int, str] = {}
    for name, cfg in assignment.items():
        env_ids = cfg.get("env_ids")
        if env_ids is None:
            continue
        ids = env_ids.detach().clone()
        if ids.is_cuda:
            ids = ids.cpu()
        for env_id in ids.tolist():
            mapping[int(env_id)] = name
    if not mapping and hasattr(base_env, "num_envs"):
        mapping = {env_id: "evader_default" for env_id in range(base_env.num_envs)}
    return mapping


def _frames_to_timesteps(total_frames: Optional[int], num_envs: int) -> Optional[int]:
    """Convert a total frame budget (across all envs) into vectorized timesteps."""
    if total_frames is None:
        return None
    return max(1, int(math.ceil(total_frames / max(1, num_envs))))


def _log_histogram_plot(
    bridge: WandbBridge,
    key: str,
    values: list[np.ndarray | torch.Tensor],
    step: int,
    title: Optional[str] = None,
) -> None:
    if not values:
        return
    arrays = []
    for val in values:
        if isinstance(val, torch.Tensor):
            arrays.append(val.detach().cpu().numpy().reshape(-1))
        else:
            arrays.append(np.asarray(val).reshape(-1))
    merged = np.concatenate(arrays)
    if merged.size == 0:
        return
    bridge.log_histogram(key, merged, step=step, title=title)


def _log_bar_plot(
    bridge: WandbBridge,
    key: str,
    labels: list[str],
    values: list[float],
    step: int,
    title: Optional[str] = None,
) -> None:
    if not labels or not values:
        return
    bridge.log_bar(key, labels, values, step=step, title=title)


def _safe_mean(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.mean().item())
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return None
        return float(arr.mean())
    except Exception:
        return None


def _reward_components_to_metrics(components: Any) -> dict[str, float]:
    if not isinstance(components, dict):
        return {}
    metrics: dict[str, float] = {}
    if any(isinstance(val, dict) for val in components.values()):
        for agent, agent_components in components.items():
            if not isinstance(agent_components, dict):
                continue
            for name, tensor in agent_components.items():
                mean_val = _safe_mean(tensor)
                if mean_val is not None:
                    metrics[f"Rewards/{agent}/{name}"] = mean_val
    else:
        for name, tensor in components.items():
            mean_val = _safe_mean(tensor)
            if mean_val is not None:
                metrics[f"Rewards/{name}"] = mean_val
    return metrics


def _termination_rates_to_metrics(base_env: Any) -> dict[str, float]:
    if not hasattr(base_env, "get_last_done_reasons"):
        return {}
    try:
        reasons = base_env.get_last_done_reasons()
    except Exception:
        return {}
    if not isinstance(reasons, torch.Tensor):
        reasons = torch.as_tensor(reasons)
    if reasons.numel() == 0:
        return {}
    reasons = reasons.view(-1).to(device="cpu")
    num_envs = max(1, reasons.numel())
    num_done = int((reasons != 0).sum().item())
    metrics: dict[str, float] = {
        "Env/termination_rate/any": float(num_done / num_envs),
    }
    if num_done <= 0:
        return metrics
    mapping = getattr(base_env, "DONE_REASON_MAP", None) or getattr(base_env, "DONE_REASON_LABELS", None) or {}
    if isinstance(mapping, dict):
        for idx, label in mapping.items():
            idx_int = int(idx)
            if idx_int == 0:
                continue
            rate = float((reasons == idx_int).float().sum().item() / num_done)
            if rate > 0.0:
                metrics[f"Env/termination_rate/{label}"] = rate
    return metrics


def _tracking_error_metrics(base_env: Any) -> dict[str, float]:
    if not hasattr(base_env, "get_reference_pose") or not hasattr(base_env, "_robot"):
        return {}
    try:
        ref_pos, ref_yaw = base_env.get_reference_pose()
    except Exception:
        return {}
    try:
        pos_w = base_env._robot.data.root_pos_w
    except Exception:
        return {}
    env_origins = getattr(getattr(base_env, "_terrain", None), "env_origins", None)
    if env_origins is not None:
        pos_local = pos_w - env_origins
    else:
        pos_local = pos_w
    pos_err = torch.norm(ref_pos - pos_local, dim=-1)
    metrics = {
        "Env/pos_error_mean": float(pos_err.mean().item()),
        "Env/pos_error_max": float(pos_err.max().item()),
    }
    if ref_yaw is not None:
        try:
            yaw = math_utils.euler_xyz_from_quat(base_env._robot.data.root_quat_w)[2]
            yaw_err = (ref_yaw.view(-1) - yaw + math.pi) % (2.0 * math.pi) - math.pi
            yaw_abs = torch.abs(yaw_err)
            metrics["Env/yaw_error_mean"] = float(yaw_abs.mean().item())
            metrics["Env/yaw_error_max"] = float(yaw_abs.max().item())
        except Exception:
            pass
    return metrics


def _estimate_values(agent, states: torch.Tensor) -> torch.Tensor:
    if not hasattr(agent, "value") or agent.value is None:
        return torch.zeros(states.shape[0], device=states.device)
    state_preprocessor = getattr(
        agent, "_observation_preprocessor", getattr(agent, "_state_preprocessor", lambda x: x)
    )
    value_preprocessor = getattr(agent, "_value_preprocessor", lambda x, inverse=False: x)
    with torch.no_grad():
        inputs = {"states": state_preprocessor(states)}
        values, _, _ = agent.value.act(inputs, role="value")
        values = value_preprocessor(values, inverse=True)
    return values.squeeze(-1)


def _summarize_tensor(tensor: Any, *, max_items: int = 6) -> str:
    if tensor is None:
        return "not found"
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy().reshape(-1)
    else:
        arr = np.asarray(tensor).reshape(-1)
    if arr.size == 0:
        return "empty"
    stats = f"len={arr.size}, min={arr.min():.4g}, max={arr.max():.4g}, mean={arr.mean():.4g}"
    sample = arr[: min(max_items, arr.size)]
    sample_str = np.array2string(sample, precision=4, separator=", ")
    if arr.size > max_items:
        sample_str = sample_str[:-1] + ", ...]"
    return f"{stats}, sample={sample_str}"


def _extract_preprocessor_stats(preprocessor: Any) -> tuple[str, str]:
    if preprocessor is None:
        return "not found", "not found"
    mean = None
    std = None
    for key in ("mean", "running_mean", "_mean"):
        if hasattr(preprocessor, key):
            mean = getattr(preprocessor, key)
            break
    for key in ("std", "running_std", "_std"):
        if hasattr(preprocessor, key):
            std = getattr(preprocessor, key)
            break
    if std is None:
        for key in ("var", "running_var", "running_variance", "variance", "_var"):
            if hasattr(preprocessor, key):
                var = getattr(preprocessor, key)
                if var is not None:
                    std = torch.sqrt(torch.clamp(torch.as_tensor(var), min=0.0))
                break
    return _summarize_tensor(mean), _summarize_tensor(std)


def _log_agent_preprocessors(agent: Any, agent_cfg: dict) -> None:
    cfg_agent = agent_cfg.get("agent", {}) if isinstance(agent_cfg, dict) else {}
    cfg_state = cfg_agent.get("observation_preprocessor", cfg_agent.get("state_preprocessor"))
    cfg_value = cfg_agent.get("value_preprocessor")
    print(f"[INFO] Agent preprocessor config: observation/state={cfg_state}, value={cfg_value}")

    state_pre = getattr(agent, "_observation_preprocessor", getattr(agent, "_state_preprocessor", None))
    value_pre = getattr(agent, "_value_preprocessor", None)

    def _describe(pre: Any) -> str:
        if pre is None:
            return "not found"
        name = pre.__class__.__name__
        size = getattr(pre, "size", None)
        device = getattr(pre, "device", None)
        mean_str, std_str = _extract_preprocessor_stats(pre)
        return f"{name}(size={size}, device={device}) mean={mean_str} std={std_str}"

    print(f"[INFO] Agent preprocessor runtime (observation/state): {_describe(state_pre)}")
    print(f"[INFO] Agent preprocessor runtime (value): {_describe(value_pre)}")


def _find_checkpoint_path(checkpoint_dir: Path, target_step: int) -> tuple[Optional[Path], Optional[int]]:
    if target_step <= 0:
        return None, None
    if not checkpoint_dir.exists():
        return None, None
    best_path = None
    best_step = None
    for file in checkpoint_dir.glob("agent_*.pt"):
        try:
            step_str = file.stem.split("_")[-1]
            step_val = int(step_str)
        except ValueError:
            continue
        if step_val <= target_step and (best_step is None or step_val > best_step):
            best_step = step_val
            best_path = file
    return best_path, best_step


def _resolve_local_checkpoint(path_str: str) -> Optional[str]:
    """Resolve a checkpoint path that may be relative to the workspace or assets."""
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    if path.exists():
        return str(path)
    try:
        return retrieve_file_path(path_str)
    except Exception:
        return None


def _download_wandb_artifact(artifact_cfg: dict[str, Any]) -> Optional[str]:
    """Download a WandB artifact and return the first .pt file (or requested file) path."""
    artifact_path = artifact_cfg.get("artifact") or artifact_cfg.get("path")
    if not artifact_path:
        return None
    target_file = artifact_cfg.get("file")
    local_dir = artifact_cfg.get("local_dir")
    try:
        import wandb  # type: ignore
    except Exception as exc:
        print(f"[WARN] Unable to import wandb to download artifact '{artifact_path}': {exc}")
        return None

    try:
        api = wandb.Api()
        download_kwargs = {"root": local_dir} if local_dir else {}
        artifact = api.artifact(artifact_path)
        download_dir = Path(artifact.download(**download_kwargs))
        if target_file:
            candidate = download_dir / target_file
            if candidate.exists():
                return str(candidate)
        pt_files = sorted(download_dir.rglob("*.pt"))
        if pt_files:
            return str(pt_files[-1])
        print(f"[WARN] No .pt checkpoints found in WandB artifact '{artifact_path}' (downloaded to {download_dir}).")
    except Exception as exc:
        print(f"[WARN] Failed to download WandB artifact '{artifact_path}': {exc}")
    return None


def _resolve_checkpoint_spec(spec: Any) -> Optional[str]:
    """Resolve a checkpoint specification to a local path (supports WandB artifacts)."""
    if spec is None:
        return None

    # Simple string: try local path first, otherwise treat as WandB artifact id.
    if isinstance(spec, str):
        local = _resolve_local_checkpoint(spec)
        if local:
            return local
        return _download_wandb_artifact({"artifact": spec})

    # Mapping-like (DictConfig or dict) with checkpoint or wandb_artifact keys.
    if isinstance(spec, Mapping):
        checkpoint = spec.get("checkpoint") or spec.get("path")
        if checkpoint:
            local = _resolve_local_checkpoint(checkpoint)
            if local:
                return local

        artifact_cfg = spec.get("wandb_artifact") or spec.get("artifact")
        if artifact_cfg:
            if isinstance(artifact_cfg, str):
                artifact_cfg = {"artifact": artifact_cfg}
            return _download_wandb_artifact(artifact_cfg)

    return None


def run_policy_evaluation(
    env,
    base_env,
    agent,
    *,
    wandb_bridge: WandbBridge,
    video_logger: Optional[WandbVideoLogger],
    eval_name: str,
    step: int,
    stochastic_eval: bool,
    eval_steps: Optional[int] = None,
) -> None:
    if base_env is None or not hasattr(base_env, "_pursuer"):
        return
    if not hasattr(base_env, "step_dt"):
        return

    strategy_map = _build_evader_strategy_map(base_env)
    strategy_stats: dict[str, dict[str, list[float] | int]] = {
        "overall": {"episodes": 0, "captures": 0, "durations": []},
    }
    for name in strategy_map.values():
        strategy_stats.setdefault(name, {"episodes": 0, "captures": 0, "durations": []})

    lin_hist: list[np.ndarray] = []
    ang_hist: list[np.ndarray] = []
    acc_hist: list[np.ndarray] = []
    action_hist: list[np.ndarray] = []
    value_hist: list[np.ndarray] = []
    reason_samples: list[int] = []

    base_horizon = int(getattr(base_env, "max_episode_length", 0))
    horizon_multiplier = 1
    eval_horizon = (eval_steps or base_horizon) * horizon_multiplier
    if eval_horizon <= 0:
        return

    agent.set_running_mode("eval")
    states, _ = env.reset()
    value_estimates = _estimate_values(agent, states)
    device = getattr(base_env, "device", states.device)
    num_envs = getattr(base_env, "num_envs", states.shape[0])
    episode_returns = torch.zeros(num_envs, device=device, dtype=torch.float32)
    episode_steps = torch.zeros(num_envs, device=device, dtype=torch.float32)
    prev_lin_vel = base_env._pursuer.data.root_lin_vel_w.detach().clone()
    step_dt = float(base_env.step_dt)

    for timestep in range(eval_horizon):
        with torch.no_grad():
            outputs = agent.act(states, timestep=timestep, timesteps=eval_horizon)
            actions = outputs[0]
            if not stochastic_eval and isinstance(outputs[-1], dict):
                actions = outputs[-1].get("mean_actions", actions)
            next_states, rewards, terminated, truncated, infos = env.step(actions)

        action_hist.append(actions.detach().clone().cpu().numpy())
        value_hist.append(value_estimates.detach().clone().cpu().numpy())

        lin_vel = base_env._pursuer.data.root_lin_vel_w.detach().clone()
        ang_vel = base_env._pursuer.data.root_ang_vel_b.detach().clone()
        lin_speed = torch.norm(lin_vel, dim=-1).cpu().numpy()
        ang_speed = torch.norm(ang_vel, dim=-1).cpu().numpy()
        accel = torch.norm((lin_vel - prev_lin_vel) / max(step_dt, 1e-6), dim=-1).cpu().numpy()
        lin_hist.append(lin_speed)
        ang_hist.append(ang_speed)
        acc_hist.append(accel)
        prev_lin_vel = lin_vel

        episode_returns += rewards.view(-1).to(episode_returns.device)
        episode_steps += 1.0

        done_mask = (terminated | truncated)
        done_ids = torch.nonzero(done_mask, as_tuple=False).flatten()
        if done_ids.numel() > 0:
            reasons = base_env.get_last_done_reasons().detach().clone().cpu()
            for env_id in done_ids.tolist():
                reason = int(reasons[env_id].item())
                reason_samples.append(reason)
                duration_val = float((episode_steps[env_id] * step_dt).item())
                strategy = strategy_map.get(env_id, "overall")
                stats = strategy_stats.setdefault(strategy, {"episodes": 0, "captures": 0, "durations": []})
                stats["episodes"] = int(stats["episodes"]) + 1
                stats["captures"] = int(stats["captures"]) + (1 if reason == 1 else 0)
                if isinstance(stats["durations"], list):
                    stats["durations"].append(duration_val)
                strategy_stats["overall"]["episodes"] = int(strategy_stats["overall"]["episodes"]) + 1
                strategy_stats["overall"]["captures"] = int(strategy_stats["overall"]["captures"]) + (1 if reason == 1 else 0)
                if isinstance(strategy_stats["overall"]["durations"], list):
                    strategy_stats["overall"]["durations"].append(duration_val)
                episode_returns[env_id] = 0.0
                episode_steps[env_id] = 0.0

        states = next_states
        value_estimates = _estimate_values(agent, states)

    env.reset()
    agent.set_running_mode("train")

    summary: dict[str, float] = {}
    for name, stats in strategy_stats.items():
        episodes = int(stats["episodes"])
        if episodes <= 0:
            continue
        durations_list = stats["durations"] if isinstance(stats["durations"], list) else []
        capture_rate = (float(stats["captures"]) / episodes) if episodes > 0 else 0.0
        mean_duration = float(np.mean(durations_list)) if durations_list else 0.0
        if name == "overall":
            summary["Eval/capture_rate"] = capture_rate
            summary["Eval/mean_episode_duration"] = mean_duration
        else:
            summary[f"Eval/capture_rate/{name}"] = capture_rate
            summary[f"Eval/mean_episode_duration/{name}"] = mean_duration

    if summary:
        wandb_bridge.log(summary, step=step)

    # Histograms disabled per user request
    # _log_histogram_plot(...)
    if reason_samples:
        from collections import Counter

        counter = Counter(reason_samples)
        labels: list[str] = []
        counts: list[float] = []
        for idx, label in DONE_REASON_LABELS.items():
            if idx == 0:
                continue
            value = counter.get(idx, 0)
            if value <= 0:
                continue
            labels.append(label)
            counts.append(float(value))
        _log_bar_plot(wandb_bridge, "Eval/termination_reasons", labels, counts, step=step, title="Termination reasons")

    if video_logger is not None:
        video_logger.poll(step=step)
# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    num_envs_cfg = int(env_cfg.scene.num_envs)

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations / frame budget for training
    train_timesteps = _frames_to_timesteps(args_cli.total_frames, num_envs_cfg)
    if train_timesteps is None and args_cli.max_iterations:
        rollouts = agent_cfg.get("agent", {}).get("rollouts", 1)
        train_timesteps = int(args_cli.max_iterations * rollouts)
    if train_timesteps is not None:
        agent_cfg["trainer"]["timesteps"] = int(train_timesteps)
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # Default WandB/run naming from environment config (if provided) unless CLI/hydra already set it.
    env_run_name = getattr(env_cfg, "wandb_run_name", None)
    experiment_cfg = agent_cfg.get("agent", {}).get("experiment", {})
    wandb_kwargs = experiment_cfg.get("wandb_kwargs", {})
    if env_run_name:
        if not wandb_kwargs.get("name"):
            wandb_kwargs["name"] = env_run_name
            experiment_cfg["wandb_kwargs"] = wandb_kwargs
        if not experiment_cfg.get("experiment_name"):
            experiment_cfg["experiment_name"] = env_run_name
        agent_cfg["agent"]["experiment"] = experiment_cfg

    # Disable FPV sensors while keeping rendering cameras if requested.
    if args_cli.disable_fpv_cameras and hasattr(env_cfg, "enable_cameras"):
        env_cfg.enable_cameras = False
    if args_cli.disable_evader_cameras and hasattr(env_cfg, "enable_evader_cameras"):
        env_cfg.enable_evader_cameras = False
    if args_cli.domain_randomization is not None and hasattr(env_cfg, "domain_randomization"):
        env_cfg.domain_randomization.enable = bool(args_cli.domain_randomization)
            
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # Provide the agent config to the environment so QuadrotorManager can rebuild RL opponents from checkpoints.
    if getattr(env_cfg, "skrl_agent_cfg", None) is None:
        if isinstance(agent_cfg, DictConfig):
            env_cfg.skrl_agent_cfg = OmegaConf.to_container(agent_cfg, resolve=True)
        else:
            env_cfg.skrl_agent_cfg = getattr(agent_cfg, "to_dict", lambda: agent_cfg)()

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # get checkpoint path (to resume training), preferring CLI and falling back to env config warmstart
    resume_source = None
    resume_path = _resolve_checkpoint_spec(args_cli.checkpoint) if args_cli.checkpoint else None
    if resume_path:
        resume_source = "--checkpoint"
    elif args_cli.checkpoint:
        print(f"[WARN] Failed to resolve checkpoint provided via --checkpoint: {args_cli.checkpoint}")

    warmstart_spec = getattr(env_cfg, "training_warmstart", None)
    if resume_path is None and warmstart_spec:
        resume_path = _resolve_checkpoint_spec(warmstart_spec)
        resume_source = "training_warmstart"
        if resume_path is None:
            print(f"[WARN] Training warmstart could not be resolved: {warmstart_spec}")

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    frames_per_step = max(1, num_envs_cfg)
    video_interval_steps = _frames_to_timesteps(args_cli.video_interval_frames, frames_per_step)
    if video_interval_steps is None:
        video_interval_steps = 8000  # fallback to previous default in steps
    video_interval_steps = max(1, int(video_interval_steps))

    # create isaac environment
    derived_timesteps = train_timesteps if train_timesteps is not None else agent_cfg.get("trainer", {}).get("timesteps", 0)
    if not derived_timesteps:
        derived_timesteps = getattr(env_cfg, "total_timesteps", 0)
    env_cfg.total_timesteps = int(derived_timesteps) if derived_timesteps else 0
    if args_cli.total_frames is not None:
        target_frames = int(args_cli.total_frames)
        print(
            f"[INFO] Training for ~{target_frames} frames across {num_envs_cfg} envs "
            f"({env_cfg.total_timesteps} timesteps)."
        )
    elif env_cfg.total_timesteps:
        approx_frames = env_cfg.total_timesteps * frames_per_step
        print(
            f"[INFO] Training for {env_cfg.total_timesteps} timesteps "
            f"(~{approx_frames} frames across {num_envs_cfg} envs)."
        )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    step_dt = getattr(base_env, "step_dt", None)
    video_directory: Optional[Path] = None

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_directory = Path(os.path.join(log_dir, "videos", "train"))
        video_directory.mkdir(parents=True, exist_ok=True)
        video_kwargs = {
            "video_folder": str(video_directory),
            "step_trigger": lambda step: step % video_interval_steps == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`
    
    # monitoring helpers
    def _perf_num_envs() -> int:
        return (
            getattr(env, "num_envs", None)
            or getattr(base_env, "num_envs", None)
            or getattr(env_cfg.scene, "num_envs", None)
            or getattr(env_cfg, "num_envs", None)
            or 1
        )

    num_envs = int(_perf_num_envs())
    performance_tracker = PerformanceTracker(num_envs=num_envs, num_envs_fn=_perf_num_envs)
    wandb_bridge = WandbBridge()
    checkpoint_metadata = {
        "task": args_cli.task,
        "algorithm": algorithm,
        "ml_framework": args_cli.ml_framework,
    }
    checkpoint_metadata = {key: value for key, value in checkpoint_metadata.items() if value is not None}
    checkpoint_uploader: Optional[WandbCheckpointUploader] = None
    video_logger: Optional[WandbVideoLogger] = None
    if args_cli.video and video_directory is not None:
        fps_guess = None
        if step_dt:
            try:
                fps_guess = int(max(1.0, round(1.0 / step_dt)))
            except Exception:
                fps_guess = None
        if fps_guess is None:
            fps_guess = 30
        video_logger = WandbVideoLogger(
            bridge=wandb_bridge,
            video_directory=video_directory,
            fps=fps_guess,
            check_interval_sec=max(5.0, video_interval_steps * (step_dt or 0.0) if video_interval_steps else 30.0),
        )

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    print("Obs space:", env.single_observation_space, "Action space:", env.single_action_space)
    
    if algorithm != "dgppo":
        runner = Runner(env, agent_cfg)
    else:
        
        # skrl's default runner is not compatible with DGPPO, so we use our ad-hoc.
        # DGPPORunner exposes the same interface as the default runner.
        from source.isaac_pursuit_evasion.dgppo.dgppo_runner import DGPPORunner
        runner = DGPPORunner(env, agent_cfg)
        
    _log_agent_preprocessors(runner.agent, agent_cfg)
    checkpoints_dir = Path(getattr(runner.agent, "experiment_dir", log_dir)) / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_uploader = WandbCheckpointUploader(
        bridge=wandb_bridge,
        checkpoints_dir=checkpoints_dir,
        metadata=checkpoint_metadata,
    )

    # augment agent post-interaction hook with performance and media logging
    if hasattr(runner, "agent") and hasattr(runner.agent, "post_interaction"):
        original_post_interaction = runner.agent.post_interaction

        def _post_interaction_with_monitoring(self, timestep: int, timesteps: int) -> None:  # type: ignore[override]
            original_post_interaction(timestep=timestep, timesteps=timesteps)
            current_step = timestep + 1
            update = performance_tracker.update(current_step)
            if update is not None:
                current_num_envs = _perf_num_envs()
                metrics, should_print = update
                if not hasattr(self, "_warned_num_envs_mismatch"):
                    expected_envs = getattr(env_cfg.scene, "num_envs", current_num_envs)
                    if expected_envs and expected_envs != current_num_envs:
                        print(
                            f"[Perf] Detected num_envs={current_num_envs} (cfg requested {expected_envs}); "
                            "frames/s uses detected value."
                        )
                    self._warned_num_envs_mismatch = True
                system_metrics = {
                    "System/env_steps_per_sec": metrics["perf/env_steps_per_sec"],
                    "System/frames_per_sec": metrics["perf/frames_per_sec"],
                    "System/num_envs": current_num_envs,
                    "System/total_frames": current_step * current_num_envs,
                }
                self.track_data("System/env_steps_per_sec", system_metrics["System/env_steps_per_sec"])
                self.track_data("System/frames_per_sec", system_metrics["System/frames_per_sec"])
                self.track_data("System/num_envs", system_metrics["System/num_envs"])
                self.track_data("System/total_frames", system_metrics["System/total_frames"])
                wandb_bridge.log(system_metrics, step=current_step)
                env_metrics: dict[str, float] = {}
                if hasattr(base_env, "get_last_reward_components"):
                    try:
                        components = base_env.get_last_reward_components()
                        env_metrics.update(_reward_components_to_metrics(components))
                    except Exception:
                        pass
                try:
                    env_metrics.update(_tracking_error_metrics(base_env))
                except Exception:
                    pass
                try:
                    env_metrics.update(_termination_rates_to_metrics(base_env))
                except Exception:
                    pass
                if env_metrics:
                    wandb_bridge.log(env_metrics, step=current_step)
                if should_print:
                    print(
                        "[Perf] step={:,} env_steps/s={:.1f} frames/s={:.1f}".format(
                            current_step,
                            metrics["perf/env_steps_per_sec"],
                            metrics["perf/frames_per_sec"],
                        )
                    )
            if video_logger is not None:
                video_logger.poll(step=current_step)

        runner.agent.post_interaction = types.MethodType(_post_interaction_with_monitoring, runner.agent)

    if checkpoint_uploader is not None and hasattr(runner, "agent") and hasattr(runner.agent, "write_checkpoint"):
        original_write_checkpoint = runner.agent.write_checkpoint

        def _write_checkpoint_with_upload(self, timestep: int, timesteps: int) -> None:  # type: ignore[override]
            original_write_checkpoint(timestep, timesteps)
            try:
                checkpoint_uploader.upload_new()
            except Exception:
                pass

        runner.agent.write_checkpoint = types.MethodType(_write_checkpoint_with_upload, runner.agent)

    # load checkpoint (if specified)
    if resume_path:
        source_label = resume_source or "checkpoint"
        print(f"[INFO] Loading model checkpoint from ({source_label}): {resume_path}")
        runner.agent.load(resume_path)

    # try:
    #     run_policy_evaluation(
    #         env,
    #         base_env,
    #         runner.agent,
    #         wandb_bridge=wandb_bridge,
    #         video_logger=video_logger,
    #         eval_name="step0",
    #         step=0,
    #         stochastic_eval=runner.trainer.stochastic_evaluation,
    #         eval_steps=getattr(base_env, "max_episode_length", None),
    #     )
    # except Exception as exc:
    #     print(f"[WARN] Initial evaluation failed: {exc}")

    run_failed = False
    try:
        # run training
        runner.run()

        total_timesteps = int(agent_cfg.get("trainer", {}).get("timesteps", getattr(runner.trainer, "timesteps", 0)))
        if total_timesteps > 0:
            runner.agent.write_checkpoint(total_timesteps, total_timesteps)
            if checkpoint_uploader is not None:
                checkpoint_uploader.upload_new()
    except Exception:
        run_failed = True
        raise
    finally:
        # Always close run/simulator so failed jobs still show correctly in W&B.
        wandb_bridge.finish(exit_code=1 if run_failed else 0)
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
