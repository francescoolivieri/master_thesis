#!/usr/bin/env python3
"""Evaluate pursuit–evasion setups with configurable pursuer/evader strategies."""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------#
# CLI (pre-hydra)
# -----------------------------------------------------------------------------#
parser = argparse.ArgumentParser(description="Evaluate pursuit–evasion controllers / policies.")
parser.add_argument("--task", type=str, default="PursuitEvasion-Benchmark", help="Gym task to run.")
parser.add_argument(
    "--task-mode",
    type=str,
    default="auto",
    choices=["auto", "pursuit_evasion", "tracking"],
    help="Override env flag_tracking for benchmark runs (default: auto uses task config).",
)
parser.add_argument("--num-envs", type=int, default=None, help="Override number of environments.")
parser.add_argument("--num-steps", type=int, default=500, help="Simulation steps to run.")
parser.add_argument(
    "--num-episodes",
    type=int,
    default=None,
    help="Stop once this many episodes have completed (across all environments).",
)
parser.add_argument("--video", action="store_true", help="Record a short video of the rollout.")
parser.add_argument("--video-length", type=int, default=500, help="Length of the recorded video (steps).")
parser.add_argument("--spawn-cameras", action="store_true", help="Force-enable cameras even in headless mode.")
parser.add_argument("--disable-cameras", action="store_true", help="Disable FPV cameras during evaluation.")
parser.add_argument("--save-camera-images", action="store_true", help="Save per-step FPV frames for all environments.")
parser.add_argument("--camera-overlay-text", action="store_true", help="Overlay target status/angle on saved FPV frames.")
parser.add_argument("--log-dir", type=Path, default=Path("logs/pursuit_evasion/benchmark"), help="Output directory.")
parser.add_argument("--log-episodes", action="store_true", help="Save per-step traces to HDF5.")
parser.add_argument("--log-actions", action="store_true", help="Save per-episode action sequences to .npz.")
parser.add_argument("--log-observations", action="store_true", help="Store observations when logging episodes.")
parser.add_argument("--wandb-name", type=str, help="Override WandB run name (agent.agent.experiment.wandb_kwargs.name).")
parser.add_argument("--wandb-id", type=str, help="Override WandB run id (agent.agent.experiment.wandb_kwargs.id).")
parser.add_argument("--wandb-project", type=str, help="Override WandB project (agent.agent.experiment.wandb_kwargs.project).")
parser.add_argument("--wandb-entity", type=str, help="Override WandB entity (agent.agent.experiment.wandb_kwargs.entity).")
parser.add_argument("--wandb-dir", type=str, help="Override WandB local dir (agent.agent.experiment.wandb_kwargs.dir).")
parser.add_argument("--exp-id", type=str, help="Optional experiment identifier appended to logs.")
parser.add_argument("--tracy", action="store_true", help="Enable Tracy profiling (requires tracy client).")
parser.add_argument("--tracy-port", type=int, default=8086, help="Tracy server port (default: 8086).")
parser.add_argument("--tracy-server", type=str, default="127.0.0.1", help="Tracy server address (default: 127.0.0.1).")
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["PPO", "AMP", "IPPO", "MAPPO"],
    help="Which skrl algorithm entry to use (maps to gym-registered agent cfg entry point).",
)
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help="Explicit agent config entry point key (overrides --algorithm). Leave empty to use the gym-registered skrl cfg.",
)
parser.add_argument("--disable_fabric", action="store_true", help="Disable fabric and use USD I/O operations.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--pursuer-artifact", type=str, help="WandB artifact for pursuer policy (e.g. entity/project/run:alias).")
parser.add_argument("--pursuer-artifact-file", type=str, help="Specific file inside pursuer artifact (optional).")
parser.add_argument("--evader-artifacts", type=str, help="Comma-separated WandB artifacts for evader policies.")
parser.add_argument("--evader-artifact-file", type=str, help="Specific file inside each evader artifact (optional).")
parser.add_argument("--pursuer-actor-cfg", type=str, help="Actor config name/path for pursuer RL controllers.")
parser.add_argument("--pursuer-critic-cfg", type=str, help="Critic config name/path for pursuer RL controllers.")
parser.add_argument("--evader-actor-cfg", type=str, help="Actor config name/path for evader RL controllers.")
parser.add_argument("--evader-critic-cfg", type=str, help="Critic config name/path for evader RL controllers.")
parser.add_argument("--policy-name", type=str, help="Optional label for the evaluated policy (used in exp-id/metadata).")


AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=False)

args_cli, hydra_args = parser.parse_known_args()
if args_cli.video or args_cli.save_camera_images or args_cli.spawn_cameras:
    args_cli.enable_cameras = True
if args_cli.disable_cameras:
    args_cli.enable_cameras = False

# Enable Tracy profiling if requested.
if args_cli.tracy:
    os.environ["OMNI_TRACY"] = "1"
    os.environ["OMNI_TRACY_PORT"] = str(args_cli.tracy_port)
    os.environ["OMNI_TRACY_SERVER"] = args_cli.tracy_server
    # Launch tracy client automatically unless user overrides.
    os.environ.setdefault("OMNI_TRACY_LAUNCH", "1")
    # Ensure Kit loads the tracy extension at startup with Python/carb channels enabled.
    tracy_kit_flags = [
        "--enable", "omni.kit.profiler.tracy",
        "--/profiler/enabled=true",
        "--/app/profilerBackend=tracy",
        "--/app/profileFromStart=true",
        "--/profiler/channels/python/enabled=true",
        "--/profiler/channels/carb.profiler/enabled=true",
    ]
    existing_kit = getattr(args_cli, "kit_args", "")
    args_cli.kit_args = (existing_kit + " " + " ".join(tracy_kit_flags)).strip()
    print(
        f"[INFO] Tracy profiling enabled on {os.environ['OMNI_TRACY_SERVER']}:{os.environ['OMNI_TRACY_PORT']} "
        "(set OMNI_TRACY_* to override)."
    )
else:
    tracy_kit_args = []

# Resolve agent cfg entry point (matches other play scripts)
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent


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

normal_overrides_only = _normalize_overrides(hydra_args + wandb_overrides)
sys.argv = [sys.argv[0]] + normal_overrides_only
launcher_kwargs = {}
if args_cli.tracy:
    launcher_kwargs["profiler_backend"] = ["tracy"]
app_launcher = AppLauncher(args_cli, **launcher_kwargs)
simulation_app = app_launcher.app

if args_cli.tracy:
    try:
        import omni.kit.app

        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        ext_mgr.set_extension_enabled_immediate("omni.kit.profiler.tracy", True)
        simulation_app.set_setting("/profiler/enabled", True)
        simulation_app.set_setting("/app/profilerBackend", "tracy")
        simulation_app.set_setting("/privacy/externalBuild", 0)
        simulation_app.set_setting("/app/profileFromStart", True)
        simulation_app.set_setting("/profiler/gpu", True)
        simulation_app.set_setting("/profiler/gpu/tracyInject/enabled", True)
        simulation_app.set_setting("/profiler/channels/carb.profiler/enabled", True)
        simulation_app.set_setting("/profiler/channels/python/enabled", True)
        simulation_app.set_setting("/profiler/channels/carb.tasking/enabled", False)
        simulation_app.set_setting("/profiler/channels/carb.events/enabled", False)
        print("[INFO] Tracy extension enabled via settings; waiting for client connection.")
    except Exception as err:
        print(f"[WARN] Failed to enable Tracy extension: {err}")

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils.hydra import hydra_task_config
from tqdm import trange

# Derive a default exp-id from task name to avoid overwriting runs.
if not args_cli.exp_id:
    base_name = args_cli.policy_name or args_cli.task
    safe_task = base_name.replace("/", "-")
    args_cli.exp_id = f"{safe_task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Ensure tasks are registered with Gym.
import source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pursuit_evasion  # noqa: F401
from source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pursuit_evasion.pursuit_evasion_env import (
    DONE_REASON_LABELS,
)

TRACKING_BENCHMARK_DEFINITION = (
    "Tracking benchmark: pursuer tracks the evader without capture, aiming to stay within "
    "[capture_distance, tracking_boundary_distance] meters. Metrics include mean/max tracking error "
    "and fraction of time inside the band."
)


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
def _resolve_path(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _controller_specs_to_dict(specs: Iterable[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for spec in specs or ():
        result.append(
            {
                "name": getattr(spec, "name", ""),
                "count": int(getattr(spec, "count", 0)),
                "config": getattr(spec, "config", None),
            }
        )
    return result


def collect_env_metadata(env, env_cfg, *, task_name: str, benchmark_task: str | None = None) -> dict[str, Any]:
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    step_dt = getattr(base_env, "step_dt", getattr(base_env.sim.cfg, "dt", 0.01) * getattr(base_env.cfg, "decimation", 1))
    flag_tracking = bool(getattr(env_cfg, "flag_tracking", False))
    benchmark_task = benchmark_task or ("tracking" if flag_tracking else "pursuit_evasion")

    def _controller_map(role: str) -> dict[str, list[int]]:
        assignment = getattr(base_env, f"_{role}_controller_assignment", {})
        mapping: dict[str, list[int]] = {}
        for name, cfg in assignment.items():
            env_ids = cfg.get("env_ids")
            if env_ids is None:
                continue
            ids = env_ids.detach().clone()
            if ids.is_cuda:
                ids = ids.cpu()
            mapping[name] = [int(x) for x in ids.tolist()]
        return mapping

    return {
        "task": task_name,
        "benchmark_task": benchmark_task,
        "num_envs": int(getattr(base_env, "num_envs", 0)),
        "step_dt": float(step_dt),
        "training_agent": getattr(env_cfg, "training_agent", ""),
        "flag_tracking": flag_tracking,
        "capture_distance": float(getattr(env_cfg, "capture_distance", 0.0)),
        "tracking_boundary_distance": float(getattr(env_cfg, "tracking_boundary_distance", 0.0)),
        "tracking_definition": TRACKING_BENCHMARK_DEFINITION if flag_tracking else None,
        "pursuer_controllers": _controller_specs_to_dict(getattr(env_cfg, "pursuer_controllers", ())),
        "evader_controllers": _controller_specs_to_dict(getattr(env_cfg, "evader_controllers", ())),
        "controller_env_map": {
            "pursuer": _controller_map("pursuer"),
            "evader": _controller_map("evader"),
        },
    }


def _is_rl_spec(spec: Any) -> bool:
    name = str(getattr(spec, "name", "")).lower()
    kind = getattr(spec, "kind", None)
    return kind in {"rl_velocity", "rl_bodyrates", "rl_policy"} or name.startswith("rl_")


def _apply_artifact_override(
    specs: Iterable[Any],
    artifacts: list[str],
    artifact_file: str | None,
    defaults: dict[str, Any] | None = None,
) -> None:
    names = [a for a in artifacts if a]
    if not names:
        return
    rl_specs = [spec for spec in specs if _is_rl_spec(spec)]
    if not rl_specs:
        return

    for idx, spec in enumerate(rl_specs):
        name = names[idx % len(names)]
        cfg = copy.deepcopy(getattr(spec, "config", None)) or {}
        cfg.pop("artifact_name", None)
        cfg.pop("wandb_artifact_name", None)
        artifact_path = name
        if defaults:
            entity = defaults.get("entity")
            project = defaults.get("project")
            alias = defaults.get("alias")
            if entity and project and "/" not in artifact_path:
                artifact_path = f"{entity}/{project}/{artifact_path}"
            if alias and ":" not in artifact_path:
                artifact_path = f"{artifact_path}:{alias}"
        artifact_cfg: dict[str, Any] = {"artifact": artifact_path}
        if artifact_file:
            artifact_cfg["file"] = artifact_file
        if defaults and defaults.get("dir"):
            artifact_cfg["local_dir"] = defaults["dir"]
        cfg["wandb_artifact"] = artifact_cfg
        spec.config = cfg
        if getattr(spec, "kind", None) is None:
            spec.kind = "rl_velocity"


def _apply_policy_cfg_override(
    specs: Iterable[Any],
    actor_cfg: str | None,
    critic_cfg: str | None,
) -> None:
    if not actor_cfg and not critic_cfg:
        return
    rl_specs = [spec for spec in specs if _is_rl_spec(spec)]
    if not rl_specs:
        return
    for spec in rl_specs:
        cfg = copy.deepcopy(getattr(spec, "config", None)) or {}
        if actor_cfg:
            cfg["actor_cfg"] = actor_cfg
        if critic_cfg:
            cfg["critic_cfg"] = critic_cfg
        spec.config = cfg
        if getattr(spec, "kind", None) is None:
            spec.kind = "rl_velocity"


# -----------------------------------------------------------------------------#
# Episode logging
# -----------------------------------------------------------------------------#
class EpisodeLogger:
    """Logs per-environment trajectories for offline analysis."""

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
            "done_reason_map": {str(k): v for k, v in DONE_REASON_LABELS.items()},
            "capture_distance": float(self.env.cfg.capture_distance),
            "tracking_boundary_distance": float(getattr(self.env.cfg, "tracking_boundary_distance", 0.0)),
            "flag_tracking": bool(getattr(self.env.cfg, "flag_tracking", False)),
            **metadata,
        }
        self.file_handler.add_env_args(payload)
        self._origins_cpu = self.env._terrain.env_origins.detach().clone().cpu()
        self.total_episodes = 0

    def _split_obs(self, obs: Any) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Best-effort extraction of pursuer/evader observations from env outputs."""
        if not self.include_obs:
            return None, None
        if isinstance(obs, dict):
            if "pursuer" in obs and "evader" in obs:
                return (
                    torch.as_tensor(obs["pursuer"]).detach().clone().cpu(),
                    torch.as_tensor(obs["evader"]).detach().clone().cpu(),
                )
            if "policy" in obs:
                tensor = torch.as_tensor(obs["policy"]).detach().clone().cpu()
                return tensor, tensor
        if isinstance(obs, torch.Tensor):
            tensor = obs.detach().clone().cpu()
            return tensor, tensor
        return None, None

    def log_step(self, obs: Any, terminated: torch.Tensor, truncated: torch.Tensor) -> None:
        done = (terminated | truncated).detach().clone().cpu()
        with torch.no_grad():
            pursuer_state = self.env._pursuer.data.root_state_w.detach().clone().cpu()
            evader_state = self.env._evader.data.root_state_w.detach().clone().cpu()
            pursuer_state[:, :3] -= self._origins_cpu
            evader_state[:, :3] -= self._origins_cpu

        pursuer_rewards, evader_rewards = self.env.get_last_rewards()
        pursuer_rewards = pursuer_rewards.detach().clone().cpu()
        evader_rewards = evader_rewards.detach().clone().cpu()
        reasons = self.env.get_last_done_reasons().detach().clone().cpu()
        reward_components = self.env.get_last_reward_components()
        pursuer_comp = {
            name: tensor.detach().clone().cpu() for name, tensor in reward_components.get("pursuer", {}).items()
        }
        evader_comp = {
            name: tensor.detach().clone().cpu() for name, tensor in reward_components.get("evader", {}).items()
        }
        obs_pursuer, obs_evader = self._split_obs(obs)

        for env_id in range(self.num_envs):
            buffer = self.buffers[env_id]
            buffer.append(
                timestep=int(self.step_counters[env_id].item()),
                pursuer_state=pursuer_state[env_id],
                evader_state=evader_state[env_id],
                pursuer_reward=pursuer_rewards[env_id].item(),
                evader_reward=evader_rewards[env_id].item(),
                obs_pursuer=None if obs_pursuer is None else obs_pursuer[env_id],
                obs_evader=None if obs_evader is None else obs_evader[env_id],
                pursuer_components={k: v[env_id] for k, v in pursuer_comp.items()},
                evader_components={k: v[env_id] for k, v in evader_comp.items()},
            )
            self.step_counters[env_id] += 1
            if done[env_id].item():
                episode = buffer.to_episode(int(reasons[env_id].item()))
                if episode is not None:
                    episode.env_id = int(env_id)
                    self.file_handler.write_episode(episode)
                    self.total_episodes += 1
                buffer.reset()
                self.step_counters[env_id] = 0

    def close(self) -> None:
        self.file_handler.flush()
        self.file_handler.close()


@dataclass
class EpisodeBuffer:
    env_id: int
    include_obs: bool
    timesteps: list[int] = field(default_factory=list)
    pursuer_states: list[torch.Tensor] = field(default_factory=list)
    evader_states: list[torch.Tensor] = field(default_factory=list)
    pursuer_rewards: list[float] = field(default_factory=list)
    evader_rewards: list[float] = field(default_factory=list)
    obs_pursuer: list[torch.Tensor] = field(default_factory=list)
    obs_evader: list[torch.Tensor] = field(default_factory=list)
    pursuer_reward_components: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    evader_reward_components: dict[str, list[torch.Tensor]] = field(default_factory=dict)

    def reset(self) -> None:
        self.timesteps.clear()
        self.pursuer_states.clear()
        self.evader_states.clear()
        self.pursuer_rewards.clear()
        self.evader_rewards.clear()
        self.obs_pursuer.clear()
        self.obs_evader.clear()
        self.pursuer_reward_components.clear()
        self.evader_reward_components.clear()

    def append(
        self,
        timestep: int,
        pursuer_state: torch.Tensor,
        evader_state: torch.Tensor,
        pursuer_reward: float,
        evader_reward: float,
        obs_pursuer: Optional[torch.Tensor] = None,
        obs_evader: Optional[torch.Tensor] = None,
        pursuer_components: Optional[dict[str, torch.Tensor]] = None,
        evader_components: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        self.timesteps.append(timestep)
        self.pursuer_states.append(pursuer_state.clone())
        self.evader_states.append(evader_state.clone())
        self.pursuer_rewards.append(float(pursuer_reward))
        self.evader_rewards.append(float(evader_reward))
        if self.include_obs:
            self.obs_pursuer.append(obs_pursuer.clone() if obs_pursuer is not None else torch.empty(0))
            self.obs_evader.append(obs_evader.clone() if obs_evader is not None else torch.empty(0))
        if pursuer_components:
            for name, value in pursuer_components.items():
                self.pursuer_reward_components.setdefault(name, []).append(value.clone())
        if evader_components:
            for name, value in evader_components.items():
                self.evader_reward_components.setdefault(name, []).append(value.clone())

    def to_episode(self, done_reason: int) -> EpisodeData | None:
        if not self.timesteps:
            return None
        episode = EpisodeData()
        data: dict[str, torch.Tensor | dict] = {
            "timesteps": torch.tensor(self.timesteps, dtype=torch.int32),
            "pursuer": {
                "state": torch.stack(self.pursuer_states, dim=0),
                "reward": torch.tensor(self.pursuer_rewards, dtype=torch.float32),
            },
            "evader": {
                "state": torch.stack(self.evader_states, dim=0),
                "reward": torch.tensor(self.evader_rewards, dtype=torch.float32),
            },
            "done_reason": torch.tensor([done_reason], dtype=torch.int32),
        }
        if self.include_obs and self.obs_pursuer:
            data["observations"] = {
                "pursuer": torch.stack(self.obs_pursuer, dim=0),
                "evader": torch.stack(self.obs_evader, dim=0),
            }
        if self.pursuer_reward_components:
            data["pursuer"]["reward_components"] = {
                name: torch.stack(values, dim=0) for name, values in self.pursuer_reward_components.items()
            }   
        if self.evader_reward_components:
            data["evader"]["reward_components"] = {
                name: torch.stack(values, dim=0) for name, values in self.evader_reward_components.items()
            }
        episode.data = data
        episode.env_id = self.env_id
        return episode


@dataclass
class ActionBuffer:
    actions: list[torch.Tensor] = field(default_factory=list)

    def reset(self) -> None:
        self.actions.clear()

    def append(self, action: torch.Tensor) -> None:
        self.actions.append(action.clone())

    def to_numpy(self, action_dim: int) -> np.ndarray:
        if not self.actions:
            return np.empty((0, 2, action_dim), dtype=np.float32)
        return torch.stack(self.actions, dim=0).cpu().numpy()


class ActionLogger:
    """Save per-episode action sequences to .npz."""

    def __init__(self, env, output_dir: Path, action_dim: int, agent_order: Sequence[str] = ("pursuer", "evader")):
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.num_envs = self.env.num_envs
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffers = [ActionBuffer() for _ in range(self.num_envs)]
        self.episode_counts = [0 for _ in range(self.num_envs)]
        self.total_episodes = 0
        self.action_dim = int(action_dim)
        self.agent_order = tuple(agent_order)
        self._write_metadata()

    def _write_metadata(self) -> None:
        meta_path = self.output_dir / "actions_metadata.json"
        if meta_path.exists():
            return
        payload = {
            "agent_order": list(self.agent_order),
            "action_dim": self.action_dim,
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _flush(self, env_id: int, done_reason: int | None = None) -> None:
        buffer = self.buffers[env_id]
        actions = buffer.to_numpy(self.action_dim)
        episode_idx = self.episode_counts[env_id]
        filename = self.output_dir / f"env_{env_id:03d}_episode_{episode_idx:05d}.npz"
        np.savez_compressed(
            filename,
            actions=actions,
            env_id=int(env_id),
            episode_idx=int(episode_idx),
            done_reason=-1 if done_reason is None else int(done_reason),
        )
        buffer.reset()
        self.episode_counts[env_id] += 1
        self.total_episodes += 1

    def log_step(
        self,
        pursuer_actions: torch.Tensor,
        evader_actions: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        done_reasons: torch.Tensor | None = None,
    ) -> None:
        done = (terminated | truncated).detach().clone().cpu()
        reasons_cpu = done_reasons.detach().clone().cpu() if done_reasons is not None else None
        for env_id in range(self.num_envs):
            action = torch.stack([pursuer_actions[env_id], evader_actions[env_id]], dim=0).detach().cpu()
            self.buffers[env_id].append(action)
            if done[env_id].item():
                reason_val = None
                if reasons_cpu is not None:
                    reason_val = int(reasons_cpu[env_id].item())
                self._flush(env_id, reason_val)

    def close(self) -> None:
        return


# -----------------------------------------------------------------------------#
# Rollout
# -----------------------------------------------------------------------------#
def run_rollout(
    env,
    *,
    num_steps: int,
    num_episodes: int | None = None,
    episode_logger: EpisodeLogger | None = None,
    action_logger: ActionLogger | None = None,
) -> dict[str, Any]:
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    device = torch.device(getattr(base_env, "device", "cpu"))
    action_dim = base_env.action_space.shape[-1] if hasattr(base_env.action_space, "shape") else 4
    actions = torch.zeros(base_env.num_envs, action_dim, device=device)
    episode_steps = torch.zeros(base_env.num_envs, device=device)

    total_pursuer_reward = torch.zeros(base_env.num_envs, device=device)
    total_evader_reward = torch.zeros_like(total_pursuer_reward)
    done_counts = {k: 0 for k in DONE_REASON_LABELS}
    relative_distance: list[float] = []
    capture_times: list[float] = []
    escape_times: list[float] = []
    episode_log_samples: dict[str, list[float]] = defaultdict(list)
    episodes_completed = 0
    steps_executed = 0
    step_dt = getattr(base_env, "step_dt", getattr(base_env.sim.cfg, "dt", 0.01) * base_env.cfg.decimation)
    tracking_enabled = bool(getattr(base_env.cfg, "flag_tracking", False))
    tracking_error_sum = 0.0
    tracking_error_max = 0.0
    tracking_inside = 0
    tracking_count = 0
    tracking_lower = float(getattr(base_env.cfg, "capture_distance", 0.0))
    tracking_upper = float(getattr(base_env.cfg, "tracking_boundary_distance", 0.0))

    obs, _ = env.reset()

    for _ in trange(num_steps, desc="evaluate"):
        steps_executed += 1
        obs, _, terminated, truncated, _ = env.step(actions)
        pursuer_rewards, evader_rewards = base_env.get_last_rewards()
        total_pursuer_reward += pursuer_rewards
        total_evader_reward += evader_rewards
        dones = terminated | truncated
        episode_steps += 1

        pursuer_state = base_env._pursuer.data.root_state_w
        evader_state = base_env._evader.data.root_state_w
        rel_vec = evader_state[:, :3] - pursuer_state[:, :3]
        rel_dist = torch.norm(rel_vec, dim=-1)
        relative_distance.append(float(rel_dist.mean().item()))
        if tracking_enabled:
            below = torch.clamp(tracking_lower - rel_dist, min=0.0)
            above = torch.clamp(rel_dist - tracking_upper, min=0.0)
            error = below + above
            tracking_error_sum += float(error.sum().item())
            tracking_error_max = max(tracking_error_max, float(error.max().item()))
            tracking_inside += int(((rel_dist >= tracking_lower) & (rel_dist <= tracking_upper)).sum().item())
            tracking_count += int(rel_dist.numel())

        reasons = base_env.get_last_done_reasons()
        if episode_logger:
            episode_logger.log_step(obs, terminated, truncated)
        if action_logger:
            action_logger.log_step(
                base_env._pursuer_actions,
                base_env._evader_actions,
                terminated,
                truncated,
                reasons,
            )

        done_mask = dones
        if done_mask.any():
            reasons_done = reasons[done_mask].to(torch.int64)
            unique, counts = torch.unique(reasons_done, return_counts=True)
            for reason, count in zip(unique.tolist(), counts.tolist()):
                done_counts[reason] = done_counts.get(reason, 0) + int(count)

            elapsed = (episode_steps[done_mask] * step_dt).detach().cpu()
            reasons_cpu = reasons_done.detach().cpu()
            if (reasons_cpu == 1).any():
                capture_times.extend(elapsed[reasons_cpu == 1].tolist())
            if (reasons_cpu == 2).any() or (reasons_cpu == 5).any():
                escape_mask = (reasons_cpu == 2) | (reasons_cpu == 5)
                escape_times.extend(elapsed[escape_mask].tolist())

            log_data = getattr(base_env, "extras", {}).get("log")
            if isinstance(log_data, dict):
                for key, value in log_data.items():
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 1:
                            val = float(value.item())
                        else:
                            val = float(value.mean().item())
                    else:
                        try:
                            val = float(value)
                        except (TypeError, ValueError):
                            continue
                    episode_log_samples[key].append(val)

            episode_steps[done_mask] = 0
            episodes_completed += int(done_mask.sum().item())

        if getattr(env, "render_mode", None) == "human":
            env.render()
        if num_episodes is not None and episodes_completed >= num_episodes:
            break

    if episode_logger:
        episode_logger.close()
    if action_logger:
        action_logger.close()

    episodes_completed = max(episodes_completed, sum(done_counts.values()))
    if episode_logger:
        episodes_completed = max(episodes_completed, episode_logger.total_episodes)
    total_episodes = max(episodes_completed, 1)
    capture_rate = done_counts.get(1, 0) / total_episodes
    escape_rate = (done_counts.get(2, 0) + done_counts.get(5, 0)) / total_episodes
    timeout_rate = done_counts.get(5, 0) / total_episodes
    out_of_bounds_rate = (done_counts.get(3, 0) + done_counts.get(4, 0)) / total_episodes
    episode_stats = {key: float(np.mean(values)) for key, values in episode_log_samples.items() if values}
    tracking_metrics = None
    if tracking_enabled and tracking_count > 0:
        tracking_metrics = {
            "mean_error": tracking_error_sum / tracking_count,
            "max_error": tracking_error_max,
            "inside_rate": tracking_inside / tracking_count,
            "capture_distance": tracking_lower,
            "tracking_boundary_distance": tracking_upper,
        }

    metrics = {
        "num_steps": num_steps,
        "steps_executed": steps_executed,
        "target_episodes": num_episodes,
        "total_episodes": total_episodes,
        "mean_pursuer_reward": float(total_pursuer_reward.mean().item()),
        "mean_evader_reward": float(total_evader_reward.mean().item()),
        "capture_rate": capture_rate,
        "escape_rate": escape_rate,
        "timeout_rate": timeout_rate,
        "out_of_bounds_rate": out_of_bounds_rate,
        "mean_relative_distance": float(np.mean(relative_distance)) if relative_distance else 0.0,
        "capture_time_avg": float(np.mean(capture_times)) if capture_times else None,
        "escape_time_avg": float(np.mean(escape_times)) if escape_times else None,
        "done_counts": {DONE_REASON_LABELS.get(k, str(k)): v for k, v in done_counts.items()},
        "history": {"relative_distance": relative_distance},
    }
    if episode_stats:
        metrics["episode_stats"] = episode_stats
    if tracking_metrics is not None:
        metrics["tracking_metrics"] = tracking_metrics
    return metrics


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#
@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, agent_cfg: dict):
    # Resolve output locations
    log_dir = _resolve_path(args_cli.log_dir)
    if args_cli.exp_id:
        log_dir = log_dir / args_cli.exp_id
    video_dir = log_dir / "videos"
    episodes_path = log_dir / "episodes.hdf5"
    metrics_path = log_dir / "metrics.json"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Basic overrides for evaluation mode
    env_cfg.scene.num_envs = args_cli.num_envs or env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.training_agent = ""
    env_cfg.use_managers_for_trained_agent = True
    env_cfg.domain_randomization.enable = False
    if args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        env_cfg.seed = int(args_cli.seed)
    elif getattr(env_cfg, "seed", None) is None:
        agent_seed = agent_cfg.get("seed") if isinstance(agent_cfg, dict) else None
        if agent_seed is not None:
            env_cfg.seed = int(agent_seed)
    if args_cli.task_mode != "auto":
        env_cfg.flag_tracking = args_cli.task_mode == "tracking"
    env_cfg.debug_vis = (not args_cli.headless) or args_cli.video
    if args_cli.video:
        env_cfg.flag_draw_velocity_markers = True
    if args_cli.spawn_cameras or args_cli.video or args_cli.save_camera_images:
        env_cfg.enable_cameras = True
    if args_cli.disable_cameras:
        env_cfg.enable_cameras = False
    env_cfg.save_camera_images = args_cli.save_camera_images
    if args_cli.save_camera_images:
        env_cfg.camera_image_dir = str(video_dir / "camera_frames")
        env_cfg.camera_overlay_text = args_cli.camera_overlay_text
    env_cfg.total_timesteps = agent_cfg["trainer"].get("timesteps", getattr(env_cfg, "total_timesteps", 0))

    # Override controllers to use WandB artifacts if provided
    # from source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pursuit_evasion.pursuit_evasion_env import ControllerSpec

    wandb_cfg = agent_cfg.get("agent", {}).get("experiment", {}).get("wandb_kwargs", {}) if agent_cfg else {}
    default_entity = wandb_cfg.get("entity")
    default_project = wandb_cfg.get("project")
    artifact_defaults: dict[str, Any] = {}
    if default_entity and default_project:
        artifact_defaults.update({"entity": default_entity, "project": default_project, "alias": "latest"})
    if wandb_cfg.get("dir"):
        artifact_defaults["dir"] = wandb_cfg["dir"]
    env_cfg.wandb_artifact_defaults = artifact_defaults or None
    pursuer_artifacts = [args_cli.pursuer_artifact] if args_cli.pursuer_artifact else []
    evader_artifacts = (
        [item.strip() for item in args_cli.evader_artifacts.split(",") if item.strip()]
        if args_cli.evader_artifacts
        else []
    )
    if getattr(env_cfg, "pursuer_controllers", None):
        _apply_artifact_override(
            env_cfg.pursuer_controllers,
            pursuer_artifacts,
            args_cli.pursuer_artifact_file,
            artifact_defaults,
        )
    if getattr(env_cfg, "evader_controllers", None):
        _apply_artifact_override(
            env_cfg.evader_controllers,
            evader_artifacts,
            args_cli.evader_artifact_file,
            artifact_defaults,
        )
    if getattr(env_cfg, "pursuer_controllers", None):
        _apply_policy_cfg_override(
            env_cfg.pursuer_controllers,
            args_cli.pursuer_actor_cfg,
            args_cli.pursuer_critic_cfg,
        )
    if getattr(env_cfg, "evader_controllers", None):
        _apply_policy_cfg_override(
            env_cfg.evader_controllers,
            args_cli.evader_actor_cfg,
            args_cli.evader_critic_cfg,
        )

    # Create environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if args_cli.video:
        video_dir.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            name_prefix="pursuit-evasion",
            disable_logger=True,
        )

    benchmark_task = (
        "tracking" if (args_cli.task_mode == "tracking" or (args_cli.task_mode == "auto" and env_cfg.flag_tracking))
        else "pursuit_evasion"
    )
    env_metadata = collect_env_metadata(env, env_cfg, task_name=args_cli.task, benchmark_task=benchmark_task)
    if args_cli.policy_name:
        env_metadata["policy_name"] = args_cli.policy_name

    episode_logger = (
        EpisodeLogger(env, episodes_path, args_cli.log_observations, env_metadata, args_cli.task)
        if args_cli.log_episodes
        else None
    )
    action_logger = None
    if args_cli.log_actions or args_cli.log_episodes:
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        action_dim = int(getattr(base_env._pursuer_actions, "shape", [0, 4])[-1])
        action_logger = ActionLogger(env, log_dir / "actions", action_dim=action_dim)

    metrics = run_rollout(
        env,
        num_steps=args_cli.num_steps,
        num_episodes=args_cli.num_episodes,
        episode_logger=episode_logger,
        action_logger=action_logger,
    )
    metrics.update(
        {
            "env_metadata": env_metadata,
            "timestamp": datetime.now().isoformat(),
        }
    )
    if action_logger is not None:
        metrics["actions_dir"] = str((log_dir / "actions").resolve())
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
