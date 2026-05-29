#!/usr/bin/env python3
"""Benchmark Crazyflie position-tracking policies or baseline controllers."""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Benchmark Crazyflie position tracking policies.")
parser.add_argument("--task", type=str, default="PosTracking-v0", help="Gym task to run.")
parser.add_argument(
    "--algorithm",
    choices=["ppo", "dgppo"],
    default="ppo",
    help="Checkpoint family to evaluate.",
)
parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel benchmark environments.")
parser.add_argument("--num-steps", type=int, default=2500, help="Simulation steps to run.")
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
    "--trained-agent-cfg",
    type=Path,
    default=None,
    help="Agent YAML used to train the checkpoint. Defaults to params/agent.yaml next to the checkpoint/run.",
)
parser.add_argument(
    "--trained-env-cfg",
    type=Path,
    default=None,
    help="Env YAML used to train the checkpoint. Defaults to params/env.yaml next to the checkpoint/run.",
)
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
parser.add_argument(
    "--benchmark-profile",
    choices=["pillar_random", "fixed", "random"],
    default="pillar_random",
    help="Goal schedule to evaluate.",
)
parser.add_argument(
    "--fixed-goals",
    type=str,
    default=None,
    help="Semicolon-separated fixed goals as x,y or x,y,z. Defaults to points behind the configured pillars.",
)
parser.add_argument("--fixed-goal-repeats", type=int, default=1, help="How many episodes to run for each fixed goal.")
parser.add_argument("--num-random-episodes", type=int, default=12, help="Random-target episodes after fixed goals.")
parser.add_argument(
    "--allow-safety-continuation",
    action="store_true",
    help="Do not terminate episodes on safety violations. The benchmark default terminates immediately.",
)
parser.add_argument("--video", action="store_true", help="Record a video of the first benchmark rollouts.")
parser.add_argument(
    "--video-length",
    type=int,
    default=None,
    help="Length of the recorded video (steps). Defaults to --num-steps so the first fixed rollouts are covered.",
)
parser.add_argument(
    "--video-trigger",
    choices=["step", "episode"],
    default="step",
    help="Start video after the first step, or at episode reset. Step avoids Isaac render reset stalls.",
)
parser.add_argument(
    "--video-episodes",
    type=int,
    default=None,
    help="Maximum completed episodes to record when --video-trigger=episode. Defaults to the fixed-goal count.",
)
parser.add_argument("--spawn-cameras", action="store_true", help="Force-enable cameras even in headless mode.")
parser.add_argument("--disable-cameras", action="store_true", help="Disable cameras during evaluation.")
parser.add_argument("--save-camera-images", action="store_true", help="Save per-step FPV frames for all environments.")
parser.add_argument("--camera-overlay-text", action="store_true", help="Overlay info on saved FPV frames.")
parser.add_argument(
    "--visualize-rays",
    action="store_true",
    help="Enable ray-caster debug visualization in the viewport/video. Best with --num-envs 1.",
)
parser.add_argument(
    "--camera-eye",
    type=str,
    default=None,
    help="Override viewport camera eye as x,y,z in world coordinates.",
)
parser.add_argument(
    "--camera-target",
    type=str,
    default=None,
    help="Override viewport camera target as x,y,z in world coordinates.",
)
parser.add_argument("--log-dir", type=Path, default=Path("logs/pos_tracking/benchmark"), help="Output directory.")
parser.add_argument("--log-episodes", action="store_true", help="Save per-step traces to HDF5.")
parser.add_argument("--log-actions", action="store_true", help="Save per-episode action sequences to .npz.")
parser.add_argument("--log-observations", action="store_true", help="Store observations when logging episodes.")
parser.add_argument("--save-episode-csv", action="store_true", help="Save a per-episode CSV summary.")
parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib rollout plots.")
parser.add_argument(
    "--no-terminate-on-success",
    action="store_true",
    help="Keep benchmark episodes running after the success tolerance is reached.",
)
parser.add_argument("--exp-id", type=str, help="Optional experiment identifier appended to logs.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")

AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=False)

args_cli, hydra_args = parser.parse_known_args()
if args_cli.video or args_cli.save_camera_images or args_cli.spawn_cameras:
    args_cli.enable_cameras = True
    os.environ["ENABLE_CAMERAS"] = "1"
    if not os.environ.get("DISPLAY"):
        args_cli.headless = True
if args_cli.disable_cameras:
    args_cli.enable_cameras = False
    os.environ["ENABLE_CAMERAS"] = "0"

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
    ActorPolicyConfig,
    load_actor_from_checkpoint,
    load_actor_from_wandb,
    load_actor_policy_config,
)
from source.isaac_pursuit_evasion.dgppo.dgppo_config import DGPPOAgentCfg
from source.isaac_pursuit_evasion.dgppo.dgppo_models import DGPPOPolicy
from source.isaac_pursuit_evasion.dgppo.utils import (
    NUM_TYPE_INDICATORS,
    build_graph_data,
    extract_graph_states_from_flat_obs,
    zero_policy_rnn_states_for_done,
)

# Ensure tasks are registered with Gym.
import source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pos_tracking  # noqa: F401

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _apply_env_overrides_from_agent_cfg(env_cfg: Any, agent_cfg: Any) -> None:
    agent_cfg = _as_plain_container(agent_cfg)
    if not isinstance(agent_cfg, Mapping):
        return
    env_overrides = agent_cfg.get("env", agent_cfg.get("environment", None))
    env_overrides = _as_plain_container(env_overrides)
    if not isinstance(env_overrides, Mapping):
        return
    for key, value in env_overrides.items():
        key_str = str(key)
        if key_str.startswith("_"):
            continue
        if not hasattr(env_cfg, key_str):
            print(f"[WARN] Ignoring agent env override '{key}': env config has no such attribute.")
            continue
        setattr(env_cfg, key_str, _as_plain_container(value))


_TRAINING_ENV_COMPAT_KEYS = (
    "arena_min",
    "arena_max",
    "arena_margin",
    "altitude_outer_margin",
    "enable_walls",
    "wall_thickness",
    "wall_extra_margin",
    "enable_pillars",
    "pillar_positions_xy",
    "pillar_radius",
    "pillar_height",
    "drone_collision_radius",
    "drone_name",
    "control_mode",
    "vel_scale",
    "yaw_rate_scale",
    "thrust_to_weight",
    "flag_yaw_tracking",
    "flag_penalize_linvel",
    "flag_action_smoothness_penalty",
    "include_yaw_in_observations",
    "include_yaw_with_ray_caster",
    "enable_obstacle_observations",
    "obstacle_observation_mode",
    "enable_ray_caster",
    "ray_caster_observation_mode",
    "ray_caster_observation_data",
    "ray_caster_top_k_hits",
    "ray_caster_num_rays",
    "ray_caster_max_distance",
    "ray_caster_horizontal_fov_range",
    "ray_caster_offset",
    "reference_obstacle_clearance",
    "reward_pos",
    "reward_pos_scale",
    "reward_yaw",
    "reward_body_rates",
    "reward_lin_vel",
    "reward_action_smoothness",
    "penalty_altitude_limit",
    "penalty_xy_boundary",
    "penalty_pillar_collision",
    "pos_tolerance",
    "yaw_tolerance",
    "success_hold_time_s",
    "terminate_on_success",
    "enable_clip_states",
)


def _apply_env_overrides_from_training_env_cfg(env_cfg: Any, env_cfg_data: Mapping[str, Any] | None) -> None:
    env_cfg_data = _as_plain_container(env_cfg_data)
    if not isinstance(env_cfg_data, Mapping):
        return

    applied: list[str] = []
    for key in _TRAINING_ENV_COMPAT_KEYS:
        if key not in env_cfg_data or not hasattr(env_cfg, key):
            continue
        setattr(env_cfg, key, _as_plain_container(env_cfg_data[key]))
        applied.append(key)
    if applied:
        print(f"[INFO] Applied training env params: {', '.join(sorted(applied))}")


def _path_candidates(path: Path) -> list[Path]:
    if path.is_absolute():
        return [path]
    roots = (Path.cwd(), _PROJECT_ROOT, _PROJECT_ROOT.parent)
    candidates: list[Path] = []
    for root in roots:
        candidate = root / path
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _resolve_existing_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    for item in _path_candidates(candidate):
        if item.exists():
            return item.resolve()
    return candidate.resolve()


def _resolve_path(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _parse_xyz(value: str, *, arg_name: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{arg_name} expects exactly three comma-separated values: x,y,z")
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise ValueError(f"{arg_name} expects numeric values, got {value!r}") from exc


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


def _action_dim_from_env(base_env: Any) -> int:
    cfg_action_space = getattr(getattr(base_env, "cfg", None), "action_space", None)
    if isinstance(cfg_action_space, int):
        return int(cfg_action_space)
    action_space = getattr(base_env, "action_space", None)
    shape = getattr(action_space, "shape", None)
    if shape:
        return int(shape[0])
    if cfg_action_space is not None:
        return int(cfg_action_space)
    raise AttributeError("Could not infer action dimension from environment.")


def _obs_dim_from_env(base_env: Any) -> int:
    cfg_obs_space = getattr(getattr(base_env, "cfg", None), "observation_space", None)
    if isinstance(cfg_obs_space, int):
        return int(cfg_obs_space)
    observation_space = getattr(base_env, "observation_space", None)
    if observation_space is not None:
        policy_space = None
        if hasattr(observation_space, "spaces"):
            policy_space = observation_space.spaces.get("policy")
        shape = getattr(policy_space, "shape", None)
        if shape:
            return int(shape[0])
        shape = getattr(observation_space, "shape", None)
        if shape:
            return int(shape[0])
    if cfg_obs_space is not None:
        return int(cfg_obs_space)
    raise AttributeError("Could not infer observation dimension from environment.")


def _as_int_tuple(values: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if values is None:
        return default
    if isinstance(values, int):
        return (int(values),)
    return tuple(int(v) for v in values)


def _strip_prefix(state_dict: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    token = f"{prefix}."
    return {key[len(token) :]: value for key, value in state_dict.items() if key.startswith(token)}


def _looks_like_dgppo_policy_state(state_dict: Mapping[str, Any]) -> bool:
    prefixes = ("gnn.", "mlp.", "rnn.", "scale_hid.", "mean_head.", "std_head.")
    return any(str(key).startswith(prefixes) for key in state_dict.keys())


def _extract_dgppo_policy_state_dict(payload: Any) -> Mapping[str, Any] | None:
    if isinstance(payload, torch.nn.Module):
        return payload.state_dict()
    if not isinstance(payload, Mapping):
        return None

    direct = payload.get("policy")
    if isinstance(direct, torch.nn.Module):
        return direct.state_dict()
    if isinstance(direct, Mapping):
        return direct

    for container_key in ("models", "model", "modules", "checkpoint_modules", "state_dict", "model_state_dict"):
        container = payload.get(container_key)
        if not isinstance(container, Mapping):
            continue
        direct = container.get("policy")
        if isinstance(direct, torch.nn.Module):
            return direct.state_dict()
        if isinstance(direct, Mapping):
            return direct
        for prefix in ("policy", "models.policy", "model.policy", "modules.policy"):
            filtered = _strip_prefix(container, prefix)
            if filtered:
                return filtered

    for prefix in ("policy", "models.policy", "model.policy", "modules.policy"):
        filtered = _strip_prefix(payload, prefix)
        if filtered:
            return filtered

    if _looks_like_dgppo_policy_state(payload):
        return payload
    return None


def _download_wandb_artifact(artifact: str, artifact_file: str | None = None) -> str:
    try:
        import wandb  # type: ignore
    except Exception as exc:
        raise ImportError("wandb is required to download artifacts.") from exc

    api = wandb.Api()
    artifact_obj = api.artifact(artifact)
    download_dir = Path(artifact_obj.download())
    if artifact_file:
        candidate = download_dir / artifact_file
        if candidate.exists():
            return str(candidate)
    pt_files = sorted(download_dir.rglob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt checkpoint found in artifact {artifact}")
    return str(pt_files[-1])


def _as_plain_container(value: Any) -> Any:
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf  # type: ignore
    except Exception:
        return value
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise ImportError("PyYAML is required to parse agent YAML configs.") from exc

    class IsaacYamlLoader(yaml.SafeLoader):
        pass

    def _construct_python_tuple(loader, node):
        return tuple(loader.construct_sequence(node))

    def _construct_python_tag(loader, tag_suffix, node):
        if isinstance(node, yaml.ScalarNode):
            return loader.construct_scalar(node)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)
        return None

    IsaacYamlLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)
    IsaacYamlLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_python_tag)

    path = _resolve_existing_path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=IsaacYamlLoader) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def _latest_agent_checkpoint(checkpoints_dir: Path) -> Path | None:
    if not checkpoints_dir.exists() or not checkpoints_dir.is_dir():
        return None
    numbered: list[tuple[int, Path]] = []
    fallback: list[Path] = []
    for file in checkpoints_dir.glob("agent_*.pt"):
        try:
            numbered.append((int(file.stem.split("_")[-1]), file))
        except ValueError:
            fallback.append(file)
    if numbered:
        return max(numbered, key=lambda item: item[0])[1].resolve()
    if fallback:
        return sorted(fallback)[-1].resolve()
    pt_files = sorted(checkpoints_dir.glob("*.pt"))
    return pt_files[-1].resolve() if pt_files else None


def _resolve_checkpoint_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    candidate = _resolve_existing_path(path)
    if not candidate.exists():
        return candidate
    if candidate.is_file():
        return candidate.resolve()

    # Accept either ".../<run>/checkpoints" or the skrl run directory itself.
    if candidate.name == "checkpoints":
        checkpoint = _latest_agent_checkpoint(candidate)
    else:
        checkpoint = _latest_agent_checkpoint(candidate / "checkpoints")
        if checkpoint is None:
            checkpoint = _latest_agent_checkpoint(candidate)
    if checkpoint is None:
        raise FileNotFoundError(f"No .pt checkpoint found under directory: {candidate}")
    print(f"[INFO] Using latest checkpoint from directory: {checkpoint}")
    return checkpoint


def _checkpoint_run_dir(checkpoint: str | Path | None) -> Path | None:
    checkpoint_path = _resolve_checkpoint_path(checkpoint)
    if checkpoint_path is None or not checkpoint_path.exists():
        return None
    if checkpoint_path.is_dir():
        return checkpoint_path.parent if checkpoint_path.name == "checkpoints" else checkpoint_path
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    if checkpoint_path.parent.name == "params":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _checkpoint_params_yaml(checkpoint: str | Path | None, filename: str) -> Path | None:
    run_dir = _checkpoint_run_dir(checkpoint)
    if run_dir is None:
        return None
    candidate = run_dir / "params" / filename
    return candidate if candidate.exists() else None


def _agent_cfg_for_checkpoint(default_cfg: Mapping[str, Any], checkpoint: str | Path | None) -> dict[str, Any]:
    if args_cli.trained_agent_cfg is not None:
        cfg_path = _resolve_existing_path(args_cli.trained_agent_cfg)
        print(f"[INFO] Loading trained agent config: {cfg_path}")
        return _load_yaml_mapping(cfg_path)

    cfg_path = _checkpoint_params_yaml(checkpoint, "agent.yaml")
    if cfg_path is not None:
        print(f"[INFO] Loading trained agent config next to checkpoint: {cfg_path}")
        return _load_yaml_mapping(cfg_path)

    return dict(default_cfg)


def _env_cfg_for_checkpoint(checkpoint: str | Path | None) -> dict[str, Any]:
    if args_cli.trained_env_cfg is not None:
        cfg_path = _resolve_existing_path(args_cli.trained_env_cfg)
        print(f"[INFO] Loading trained env config: {cfg_path}")
        return _load_yaml_mapping(cfg_path)

    cfg_path = _checkpoint_params_yaml(checkpoint, "env.yaml")
    if cfg_path is not None:
        print(f"[INFO] Loading trained env config next to checkpoint: {cfg_path}")
        return _load_yaml_mapping(cfg_path)

    return {}


def _infer_algorithm_from_agent_cfg(agent_cfg_data: Mapping[str, Any]) -> str | None:
    cfg = _as_plain_container(agent_cfg_data)
    if not isinstance(cfg, Mapping):
        return None

    agent_section = cfg.get("agent")
    agent_section = _as_plain_container(agent_section)
    agent_section = agent_section if isinstance(agent_section, Mapping) else {}

    class_name = str(agent_section.get("class", "")).lower()
    if class_name == "ppo" or isinstance(cfg.get("models"), Mapping):
        return "ppo"

    dgppo_keys = {
        "alpha",
        "cbf_eps",
        "cbf_weight",
        "obs_radius",
        "lr_policy",
        "lr_vl",
        "lr_vh",
        "vl_loss_scale",
        "vh_loss_scale",
        "gnn",
    }
    if any(key in agent_section for key in dgppo_keys):
        return "dgppo"
    return None


def _validate_checkpoint_algorithm(agent_cfg_data: Mapping[str, Any], checkpoint: str | Path | None) -> None:
    if args_cli.policy_mode != "rl":
        return
    inferred = _infer_algorithm_from_agent_cfg(agent_cfg_data)
    if inferred is None or inferred == args_cli.algorithm:
        return

    checkpoint_hint = f" for checkpoint {checkpoint}" if checkpoint else ""
    raise ValueError(
        f"The loaded training config{checkpoint_hint} looks like {inferred.upper()}, "
        f"but --algorithm {args_cli.algorithm} was selected. "
        f"Re-run with --algorithm {inferred} so the benchmark builds the matching policy loader."
    )


def _ppo_actor_cfg_from_agent_cfg(agent_cfg_data: Mapping[str, Any], base_env: Any) -> ActorPolicyConfig | None:
    if not isinstance(agent_cfg_data, Mapping):
        return None
    models_cfg = agent_cfg_data.get("models")
    if not isinstance(models_cfg, Mapping):
        return None
    policy_cfg = models_cfg.get("policy")
    if not isinstance(policy_cfg, Mapping):
        return None
    network_cfg = policy_cfg.get("network")
    if not isinstance(network_cfg, Sequence) or len(network_cfg) == 0:
        return None
    first_net = network_cfg[0]
    if not isinstance(first_net, Mapping):
        return None
    hidden_layers = first_net.get("layers")
    if not isinstance(hidden_layers, Sequence):
        return None
    activation = str(first_net.get("activations", "elu")).lower()
    log_std_init = float(policy_cfg.get("initial_log_std", policy_cfg.get("log_std_init", 0.0)))
    return ActorPolicyConfig(
        obs_dim=_obs_dim_from_env(base_env),
        action_dim=_action_dim_from_env(base_env),
        hidden_layers=[int(layer) for layer in hidden_layers],
        activation=activation,
        log_std_init=log_std_init,
    )


def _extract_ppo_policy_state_dict(payload: Any) -> Mapping[str, Any] | None:
    if isinstance(payload, torch.nn.Module):
        return payload.state_dict()
    if not isinstance(payload, Mapping):
        return None
    direct = payload.get("policy")
    if isinstance(direct, Mapping):
        return direct
    for container_key in ("models", "model", "model_state_dict", "state_dict"):
        container = payload.get(container_key)
        if not isinstance(container, Mapping):
            continue
        direct = container.get("policy")
        if isinstance(direct, Mapping):
            return direct
        for prefix in ("policy", "models.policy", "model.policy"):
            filtered = _strip_prefix(container, prefix)
            if filtered:
                return filtered
    for prefix in ("policy", "models.policy", "model.policy"):
        filtered = _strip_prefix(payload, prefix)
        if filtered:
            return filtered
    if any(str(key).startswith("net_container.") for key in payload.keys()):
        return payload
    return None


def _ppo_actor_cfg_from_checkpoint(checkpoint: str | Path, fallback_cfg: ActorPolicyConfig) -> ActorPolicyConfig:
    payload = torch.load(str(_resolve_path(Path(checkpoint))), map_location="cpu")
    state_dict = _extract_ppo_policy_state_dict(payload)
    if state_dict is None:
        return fallback_cfg

    linear_layers: list[tuple[int, torch.Tensor]] = []
    for key, value in state_dict.items():
        key_str = str(key)
        if not key_str.startswith("net_container.") or not key_str.endswith(".weight"):
            continue
        parts = key_str.split(".")
        if len(parts) < 3:
            continue
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            linear_layers.append((int(parts[1]), value))
    if not linear_layers:
        return fallback_cfg
    linear_layers.sort(key=lambda item: item[0])
    obs_dim = int(linear_layers[0][1].shape[1])
    action_dim = int(linear_layers[-1][1].shape[0])
    hidden_layers = [int(weight.shape[0]) for _idx, weight in linear_layers[:-1]]
    return ActorPolicyConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_layers=hidden_layers or list(fallback_cfg.hidden_layers),
        activation=fallback_cfg.activation,
        log_std_init=fallback_cfg.log_std_init,
    )


class PolicyRunner:
    def __init__(self, actor, device: str, base_env: Any | None = None):
        self.actor = actor
        self.device = torch.device(device)
        self.base_env = base_env
        self.obs_scaler = getattr(self.actor, "obs_scaler", None)
        self.expected_obs_dim = self._infer_obs_dim(actor)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        actor_cfg: str | None,
        *,
        base_env: Any | None = None,
        agent_cfg_data: Mapping[str, Any] | None = None,
        device: str,
    ) -> "PolicyRunner":
        cfg = None
        if base_env is not None and agent_cfg_data is not None:
            cfg = _ppo_actor_cfg_from_agent_cfg(agent_cfg_data, base_env)
            if cfg is not None:
                print("[INFO] Built PPO actor config from checkpoint training config.", flush=True)
        if cfg is None:
            cfg = load_actor_policy_config(actor_cfg)
        cfg = _ppo_actor_cfg_from_checkpoint(checkpoint, cfg)
        print(
            "[INFO] PPO actor config: "
            f"obs_dim={cfg.obs_dim}, action_dim={cfg.action_dim}, hidden_layers={list(cfg.hidden_layers)}",
            flush=True,
        )
        actor = load_actor_from_checkpoint(checkpoint, cfg, device=device)
        return cls(actor, device=device, base_env=base_env)

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
            obs = self._adapt_observation(obs)
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

    def _adapt_observation(self, obs: torch.Tensor) -> torch.Tensor:
        if self.expected_obs_dim is None or obs.shape[-1] == self.expected_obs_dim:
            return obs
        if self.base_env is None:
            raise ValueError(
                f"Policy expects obs_dim={self.expected_obs_dim}, but the environment produced obs_dim={obs.shape[-1]}."
            )
        if self.expected_obs_dim == 14:
            return self._legacy_14d_observation()
        if self.expected_obs_dim >= 11 and (self.expected_obs_dim - 11) % 2 == 0:
            return self._legacy_graph_observation(obs, state_dim=8, n_obstacles=(self.expected_obs_dim - 11) // 2)
        raise ValueError(
            f"Policy expects obs_dim={self.expected_obs_dim}, but the environment produced obs_dim={obs.shape[-1]}. "
            "No known benchmark observation adapter matches this checkpoint."
        )

    def _legacy_14d_observation(self) -> torch.Tensor:
        env = self.base_env
        env_origins = env._terrain.env_origins
        pos_local = env._robot.data.root_pos_w - env_origins
        ref_pos, _ref_yaw = env.get_reference_pose()
        pos_error = ref_pos - pos_local
        dist = torch.linalg.vector_norm(pos_error, dim=-1, keepdim=True)
        return torch.cat(
            [
                pos_error,
                dist,
                env._robot.data.root_lin_vel_w,
                env._robot.data.root_ang_vel_b,
                env._robot.data.root_quat_w,
            ],
            dim=-1,
        ).to(self.device, dtype=torch.float32)

    def _legacy_graph_observation(self, obs: torch.Tensor, *, state_dim: int, n_obstacles: int) -> torch.Tensor:
        env = self.base_env
        agent_state, goal_state, obs_state = extract_graph_states_from_flat_obs(
            obs,
            env.graph_obs_layout,
            n_agents=int(getattr(env, "num_agents", 1)),
        )
        n_envs, n_agents, _ = agent_state.shape
        if n_agents != 1:
            raise ValueError("Legacy PPO graph observation adapter currently supports one agent.")
        adapted_agent = agent_state.new_zeros(n_envs, state_dim)
        adapted_agent[:, : min(6, state_dim)] = agent_state[:, 0, : min(6, state_dim)]
        if state_dim == 8:
            quat = env._robot.data.root_quat_w.to(device=agent_state.device, dtype=agent_state.dtype)
            yaw = euler_xyz_from_quat(quat)[2]
            adapted_agent[:, 6] = torch.sin(yaw)
            adapted_agent[:, 7] = torch.cos(yaw)
        goal = goal_state[:, 0, :3]
        obstacle_xy = obs_state[:, :n_obstacles, :2].reshape(n_envs, -1)
        missing_obstacle_values = n_obstacles * 2 - obstacle_xy.shape[-1]
        if missing_obstacle_values > 0:
            obstacle_xy = torch.cat([obstacle_xy, obstacle_xy.new_zeros(n_envs, missing_obstacle_values)], dim=-1)
        return torch.cat([adapted_agent, goal, obstacle_xy], dim=-1).to(self.device, dtype=torch.float32)


class DGPPOPolicyRunner:
    """Actor-only DGPPO evaluator using the same graph/flat backbone as training."""

    def __init__(
        self,
        policy: DGPPOPolicy,
        base_env: Any,
        agent_cfg: DGPPOAgentCfg,
        *,
        model_state_dim: int,
        device: str,
    ):
        self.policy = policy
        self.base_env = base_env
        self.agent_cfg = agent_cfg
        self.model_state_dim = int(model_state_dim)
        self.device = torch.device(device)
        self.n_agents = int(getattr(base_env, "num_agents", 1))
        self.rnn_state = None
        if self.policy.use_rnn:
            self.rnn_state = self.policy.initialize_carry(
                num_sequences=int(base_env.num_envs) * self.n_agents,
                device=self.device,
            )
        print(f"[INFO] Moving DGPPO policy to device: {self.device}", flush=True)
        self.policy.to(self.device)
        self.policy.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        agent_cfg_data: Mapping[str, Any],
        base_env: Any,
        device: str,
    ) -> "DGPPOPolicyRunner":
        cfg_data = agent_cfg_data.get("agent", agent_cfg_data) if isinstance(agent_cfg_data, Mapping) else {}
        agent_cfg = DGPPOAgentCfg.from_dict(cfg_data)

        print(f"[INFO] Loading DGPPO checkpoint: {checkpoint}", flush=True)
        payload = torch.load(str(_resolve_path(Path(checkpoint))), map_location="cpu")
        print("[INFO] Extracting DGPPO policy weights.", flush=True)
        state_dict = _extract_dgppo_policy_state_dict(payload)
        if state_dict is None:
            raise ValueError(f"Unable to locate DGPPO policy weights in checkpoint: {checkpoint}")
        model_state_dim = cls._infer_model_state_dim(state_dict)
        print(f"[INFO] Building DGPPO policy module (graph_state_dim={model_state_dim}).", flush=True)
        policy = cls._make_policy(agent_cfg, base_env, device, model_state_dim=model_state_dim)
        print("[INFO] Applying DGPPO policy weights.", flush=True)
        missing, unexpected = policy.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[WARN] DGPPO policy checkpoint load mismatch (missing={missing}, unexpected={unexpected}).")
        runner = cls(policy, base_env=base_env, agent_cfg=agent_cfg, model_state_dim=model_state_dim, device=device)
        print("[INFO] DGPPO policy ready.", flush=True)
        return runner

    @staticmethod
    def _infer_model_state_dim(state_dict: Mapping[str, Any]) -> int:
        query_weight = state_dict.get("gnn.gnn_layers.0.query.weight")
        if not isinstance(query_weight, torch.Tensor):
            raise ValueError("Could not infer DG-PPO graph state dim from checkpoint policy weights.")
        return int(query_weight.shape[1]) - NUM_TYPE_INDICATORS

    @staticmethod
    def _make_policy(
        agent_cfg: DGPPOAgentCfg,
        base_env: Any,
        device: str,
        *,
        model_state_dim: int,
    ) -> DGPPOPolicy:
        layout = base_env.graph_obs_layout
        n_agents = int(getattr(base_env, "num_agents", 1))
        action_dim = _action_dim_from_env(base_env)
        node_dim = int(model_state_dim) + NUM_TYPE_INDICATORS
        edge_dim = int(model_state_dim)
        gnn_cfg = agent_cfg.gnn
        rnn_cfg = agent_cfg.rnn
        model_cfg = agent_cfg.model
        return DGPPOPolicy(
            node_dim=node_dim,
            edge_dim=edge_dim,
            n_agents=n_agents,
            action_dim=action_dim,
            gnn_layers=int(gnn_cfg.get("policy_layers", 1)),
            gnn_out_dim=int(gnn_cfg.get("policy_out_dim", gnn_cfg.get("out_dim", 64))),
            gnn_msg_dim=int(gnn_cfg.get("msg_dim", 32)),
            gnn_heads=int(gnn_cfg.get("n_heads", 3)),
            mlp_hid=_as_int_tuple(model_cfg.get("policy_mlp_hid"), (128, 64)),
            scale_hid=int(model_cfg.get("scale_hid", 64)),
            scale_final=float(model_cfg.get("scale_final", 0.01)),
            std_dev_init=float(model_cfg.get("std_dev_init", 0.5)),
            std_dev_min=float(model_cfg.get("std_dev_min", 1e-5)),
            use_rnn=bool(agent_cfg.use_rnn),
            rnn_cell=str(rnn_cfg.get("cell", "gru")),
            rnn_hidden=int(rnn_cfg.get("hidden", 64)),
            rnn_layers=int(rnn_cfg.get("layers", 1)),
            device=device,
        )

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(self.device, dtype=torch.float32)
        graph_input = self._build_policy_input(obs)
        with torch.no_grad():
            action, _log_prob, _mean, new_rnn = self.policy.act(
                graph_input,
                self.rnn_state,
                deterministic=True,
            )
        if new_rnn is not None:
            self.rnn_state = new_rnn
        return action.reshape(int(self.base_env.num_envs), -1)

    def reset_done(self, done: torch.Tensor) -> None:
        if self.rnn_state is None:
            return
        self.rnn_state = zero_policy_rnn_states_for_done(
            self.rnn_state,
            done.to(self.device),
            n_agents=self.n_agents,
        )

    def _build_policy_input(self, obs: torch.Tensor):
        agent_state, goal_state, obs_state = extract_graph_states_from_flat_obs(
            obs,
            self.base_env.graph_obs_layout,
            n_agents=self.n_agents,
        )
        agent_state, goal_state, obs_state = self._adapt_graph_states_for_model(agent_state, goal_state, obs_state)
        return build_graph_data(
            agent_state=agent_state,
            goal_state=goal_state,
            obs_state=obs_state,
            obs_radius=float(self.agent_cfg.obs_radius),
        )

    def _adapt_graph_states_for_model(
        self,
        agent_state: torch.Tensor,
        goal_state: torch.Tensor,
        obs_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_state_dim = int(agent_state.shape[-1])
        if current_state_dim == self.model_state_dim:
            return agent_state, goal_state, obs_state

        n_envs, n_agents, _ = agent_state.shape
        state_dim = self.model_state_dim
        adapted_agent = agent_state.new_zeros(n_envs, n_agents, state_dim)
        copy_dim = min(6, current_state_dim, state_dim)
        if copy_dim > 0:
            adapted_agent[..., :copy_dim] = agent_state[..., :copy_dim]
        if state_dim == 8 and current_state_dim >= 6:
            quat = self.base_env._robot.data.root_quat_w.to(device=agent_state.device, dtype=agent_state.dtype)
            yaw = euler_xyz_from_quat(quat)[2]
            adapted_agent[:, :, 6] = torch.sin(yaw).view(n_envs, 1).expand(-1, n_agents)
            adapted_agent[:, :, 7] = torch.cos(yaw).view(n_envs, 1).expand(-1, n_agents)
        elif state_dim > copy_dim:
            extra_dim = min(current_state_dim, state_dim) - copy_dim
            if extra_dim > 0:
                adapted_agent[..., copy_dim : copy_dim + extra_dim] = agent_state[
                    ..., copy_dim : copy_dim + extra_dim
                ]

        adapted_goal = goal_state.new_zeros(n_envs, goal_state.shape[1], state_dim)
        adapted_goal[..., : min(3, state_dim)] = goal_state[..., : min(3, state_dim)]
        adapted_obs = obs_state.new_zeros(n_envs, obs_state.shape[1], state_dim)
        adapted_obs[..., : min(2, state_dim)] = obs_state[..., : min(2, state_dim)]
        return adapted_agent, adapted_goal, adapted_obs


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
        snapshot = _get_step_snapshot(env, use_last_step_snapshot=True)
        pos_local = snapshot["pos_local"]
        vel_world = snapshot["vel_world"]
        quat = snapshot["quat"]
        ang_vel = snapshot["ang_vel"]
        ref_pos = snapshot["ref_pos"]
        ref_yaw = snapshot["ref_yaw"]
        yaw = euler_xyz_from_quat(quat)[2].unsqueeze(-1)
        yaw_error = _wrap_angle(ref_yaw - yaw)
        pos_error = torch.norm(ref_pos - pos_local, dim=-1, keepdim=True)
        state = torch.cat([pos_local, vel_world, quat, ang_vel], dim=-1)
        reference = torch.cat([ref_pos, ref_yaw], dim=-1)
        altitude_limit = snapshot["altitude_limit"]
        xy_limit = snapshot["xy_limit"]
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


@dataclass
class GoalScenario:
    kind: str
    label: str
    goal: tuple[float, float, float]


@dataclass
class ActiveRollout:
    env_id: int
    scenario: GoalScenario
    positions: list[tuple[float, float]] = field(default_factory=list)
    goals: list[tuple[float, float]] = field(default_factory=list)
    pos_errors: list[float] = field(default_factory=list)
    yaw_errors: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    action_norms: list[float] = field(default_factory=list)
    altitude_violations: int = 0
    boundary_violations: int = 0
    pillar_collisions: int = 0
    min_pillar_clearance: float | None = None
    min_ray_clearance: float | None = None

    def append(
        self,
        *,
        position_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        pos_error: float,
        yaw_error: float | None,
        reward: float,
        action_norm: float,
        altitude_violation: bool,
        boundary_violation: bool,
        pillar_collision: bool,
        pillar_clearance: float | None,
        ray_clearance: float | None,
    ) -> None:
        self.positions.append(position_xy)
        self.goals.append(goal_xy)
        self.pos_errors.append(float(pos_error))
        if yaw_error is not None:
            self.yaw_errors.append(float(yaw_error))
        self.rewards.append(float(reward))
        self.action_norms.append(float(action_norm))
        self.altitude_violations += int(altitude_violation)
        self.boundary_violations += int(boundary_violation)
        self.pillar_collisions += int(pillar_collision)
        if pillar_clearance is not None:
            self.min_pillar_clearance = (
                float(pillar_clearance)
                if self.min_pillar_clearance is None
                else min(self.min_pillar_clearance, float(pillar_clearance))
            )
        if ray_clearance is not None:
            self.min_ray_clearance = (
                float(ray_clearance)
                if self.min_ray_clearance is None
                else min(self.min_ray_clearance, float(ray_clearance))
            )


@dataclass
class EpisodeResult:
    episode_id: int
    env_id: int
    scenario_kind: str
    scenario_label: str
    target: tuple[float, float, float]
    done_reason: int
    done_label: str
    length_steps: int
    duration_s: float
    total_reward: float
    mean_pos_error: float | None
    final_pos_error: float | None
    min_pos_error: float | None
    mean_yaw_error: float | None
    max_action_norm: float | None
    altitude_violations: int
    boundary_violations: int
    pillar_collisions: int
    min_pillar_clearance: float | None
    min_ray_clearance: float | None
    path_xy: list[tuple[float, float]]
    errors: list[float]

    @property
    def safety_violation_steps(self) -> int:
        return self.altitude_violations + self.boundary_violations + self.pillar_collisions

    @property
    def success(self) -> bool:
        return self.done_reason == 1

    @property
    def safety_terminated(self) -> bool:
        return self.done_reason in {2, 3, 6}

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "env_id": self.env_id,
            "scenario_kind": self.scenario_kind,
            "scenario_label": self.scenario_label,
            "target_x": self.target[0],
            "target_y": self.target[1],
            "target_z": self.target[2],
            "done_reason": self.done_reason,
            "done_label": self.done_label,
            "length_steps": self.length_steps,
            "duration_s": self.duration_s,
            "total_reward": self.total_reward,
            "mean_pos_error": self.mean_pos_error,
            "final_pos_error": self.final_pos_error,
            "min_pos_error": self.min_pos_error,
            "mean_yaw_error": self.mean_yaw_error,
            "max_action_norm": self.max_action_norm,
            "altitude_violations": self.altitude_violations,
            "boundary_violations": self.boundary_violations,
            "pillar_collisions": self.pillar_collisions,
            "min_pillar_clearance": self.min_pillar_clearance,
            "min_ray_clearance": self.min_ray_clearance,
            "safety_terminated": self.safety_terminated,
            "success": self.success,
        }


class ScenarioManager:
    def __init__(self, base_env: Any, fixed_goals: Sequence[tuple[float, float, float]]):
        self.base_env = base_env
        self._fixed_queue = [
            GoalScenario(kind="fixed", label=f"fixed_{idx:02d}", goal=goal)
            for idx, goal in enumerate(fixed_goals)
        ]
        self._next_fixed = 0
        self.active: list[GoalScenario | None] = [None for _ in range(int(base_env.num_envs))]

    def assign(self, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            if self._next_fixed < len(self._fixed_queue):
                scenario = self._fixed_queue[self._next_fixed]
                self._next_fixed += 1
                self._write_reference(env_id, scenario.goal)
            else:
                ref = self.base_env._reference_pos[env_id].detach().cpu().tolist()
                scenario = GoalScenario(
                    kind="random",
                    label="random",
                    goal=(float(ref[0]), float(ref[1]), float(ref[2])),
                )
            self.active[int(env_id)] = scenario

    def get(self, env_id: int) -> GoalScenario:
        scenario = self.active[int(env_id)]
        if scenario is None:
            ref = self.base_env._reference_pos[env_id].detach().cpu().tolist()
            scenario = GoalScenario("random", "random", (float(ref[0]), float(ref[1]), float(ref[2])))
            self.active[int(env_id)] = scenario
        return scenario

    def _write_reference(self, env_id: int, goal: tuple[float, float, float]) -> None:
        self.base_env._reference_pos[env_id, 0] = float(goal[0])
        self.base_env._reference_pos[env_id, 1] = float(goal[1])
        self.base_env._reference_pos[env_id, 2] = float(goal[2])
        if hasattr(self.base_env, "_reference_timer"):
            self.base_env._reference_timer[env_id] = 0.0
        if hasattr(self.base_env, "_success_counter"):
            self.base_env._success_counter[env_id] = 0
        if hasattr(self.base_env, "_last_success"):
            self.base_env._last_success[env_id] = False


class RolloutRecorder:
    def __init__(self, base_env: Any, scenario_manager: ScenarioManager):
        self.base_env = base_env
        self.scenario_manager = scenario_manager
        self.active: list[ActiveRollout | None] = [None for _ in range(int(base_env.num_envs))]
        self.results: list[EpisodeResult] = []
        self.step_samples = 0
        self.altitude_violation_samples = 0
        self.boundary_violation_samples = 0
        self.pillar_collision_samples = 0

    def start(self, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            scenario = self.scenario_manager.get(int(env_id))
            self.active[int(env_id)] = ActiveRollout(env_id=int(env_id), scenario=scenario)

    def capture_step(self, actions: torch.Tensor, *, use_last_step_snapshot: bool = False) -> dict[str, torch.Tensor | None]:
        env = self.base_env
        snapshot = _get_step_snapshot(env, use_last_step_snapshot=use_last_step_snapshot)
        pos_local = snapshot["pos_local"]
        ref_pos = snapshot["ref_pos"]
        ref_yaw = snapshot["ref_yaw"]
        quat = snapshot["quat"]
        altitude_limit = snapshot["altitude_limit"]
        boundary_limit = snapshot["xy_limit"]
        pillar_collision = snapshot["pillar_collision"]
        pos_error = torch.norm(ref_pos - pos_local, dim=-1)
        yaw_abs = None
        if env.cfg.flag_yaw_tracking:
            yaw = euler_xyz_from_quat(quat)[2]
            yaw_abs = torch.abs(_wrap_angle(ref_yaw.squeeze(-1) - yaw))

        pillar_clearance = _pillar_clearance(env, pos_local)
        ray_clearance = _ray_clearance(env, pos_local)
        action_norm = torch.linalg.vector_norm(actions, dim=-1)

        return {
            "pos_local": pos_local.detach().clone(),
            "ref_pos": ref_pos.detach().clone(),
            "pos_error": pos_error.detach().clone(),
            "yaw_abs": None if yaw_abs is None else yaw_abs.detach().clone(),
            "altitude_limit": altitude_limit.detach().clone(),
            "boundary_limit": boundary_limit.detach().clone(),
            "pillar_collision": pillar_collision.detach().clone(),
            "pillar_clearance": None if pillar_clearance is None else pillar_clearance.detach().clone(),
            "ray_clearance": None if ray_clearance is None else ray_clearance.detach().clone(),
            "action_norm": action_norm.detach().clone(),
        }

    def append_step(self, sample: Mapping[str, torch.Tensor | None], rewards: torch.Tensor) -> None:
        env = self.base_env
        pos_local = sample["pos_local"]
        ref_pos = sample["ref_pos"]
        pos_error = sample["pos_error"]
        yaw_abs = sample["yaw_abs"]
        altitude_limit = sample["altitude_limit"]
        boundary_limit = sample["boundary_limit"]
        pillar_collision = sample["pillar_collision"]
        pillar_clearance = sample["pillar_clearance"]
        ray_clearance = sample["ray_clearance"]
        action_norm = sample["action_norm"]
        assert isinstance(pos_local, torch.Tensor)
        assert isinstance(ref_pos, torch.Tensor)
        assert isinstance(pos_error, torch.Tensor)
        assert isinstance(altitude_limit, torch.Tensor)
        assert isinstance(boundary_limit, torch.Tensor)
        assert isinstance(pillar_collision, torch.Tensor)
        assert isinstance(action_norm, torch.Tensor)
        self.step_samples += int(env.num_envs)
        self.altitude_violation_samples += int(altitude_limit.to(torch.int32).sum().item())
        self.boundary_violation_samples += int(boundary_limit.to(torch.int32).sum().item())
        self.pillar_collision_samples += int(pillar_collision.to(torch.int32).sum().item())

        for env_id in range(int(env.num_envs)):
            rollout = self.active[env_id]
            if rollout is None:
                continue
            rollout.append(
                position_xy=(float(pos_local[env_id, 0].item()), float(pos_local[env_id, 1].item())),
                goal_xy=(float(ref_pos[env_id, 0].item()), float(ref_pos[env_id, 1].item())),
                pos_error=float(pos_error[env_id].item()),
                yaw_error=None if yaw_abs is None else float(yaw_abs[env_id].item()),
                reward=float(rewards[env_id].item()),
                action_norm=float(action_norm[env_id].item()),
                altitude_violation=bool(altitude_limit[env_id].item()),
                boundary_violation=bool(boundary_limit[env_id].item()),
                pillar_collision=bool(pillar_collision[env_id].item()),
                pillar_clearance=None if pillar_clearance is None else float(pillar_clearance[env_id].item()),
                ray_clearance=None if ray_clearance is None else float(ray_clearance[env_id].item()),
            )

    def finish(self, env_id: int, done_reason: int) -> EpisodeResult | None:
        rollout = self.active[int(env_id)]
        if rollout is None:
            return None
        mapping = getattr(self.base_env, "DONE_REASON_MAP", {})
        done_label = str(mapping.get(int(done_reason), f"reason_{int(done_reason)}"))
        length = len(rollout.pos_errors)
        step_dt = float(getattr(self.base_env, "step_dt", getattr(self.base_env, "_step_dt", 0.0)))
        result = EpisodeResult(
            episode_id=len(self.results),
            env_id=int(env_id),
            scenario_kind=rollout.scenario.kind,
            scenario_label=rollout.scenario.label,
            target=rollout.scenario.goal,
            done_reason=int(done_reason),
            done_label=done_label,
            length_steps=length,
            duration_s=length * step_dt,
            total_reward=float(sum(rollout.rewards)),
            mean_pos_error=_safe_list_mean(rollout.pos_errors),
            final_pos_error=rollout.pos_errors[-1] if rollout.pos_errors else None,
            min_pos_error=min(rollout.pos_errors) if rollout.pos_errors else None,
            mean_yaw_error=_safe_list_mean(rollout.yaw_errors),
            max_action_norm=max(rollout.action_norms) if rollout.action_norms else None,
            altitude_violations=rollout.altitude_violations,
            boundary_violations=rollout.boundary_violations,
            pillar_collisions=rollout.pillar_collisions,
            min_pillar_clearance=rollout.min_pillar_clearance,
            min_ray_clearance=rollout.min_ray_clearance,
            path_xy=rollout.positions.copy(),
            errors=rollout.pos_errors.copy(),
        )
        self.results.append(result)
        self.active[int(env_id)] = None
        return result


def _get_step_snapshot(base_env: Any, *, use_last_step_snapshot: bool) -> dict[str, torch.Tensor]:
    if use_last_step_snapshot and hasattr(base_env, "get_last_step_snapshot"):
        snapshot = base_env.get_last_step_snapshot()
        if snapshot:
            return snapshot

    env_origins = base_env._terrain.env_origins
    pos_local = base_env._robot.data.root_pos_w - env_origins
    ref_pos, ref_yaw = base_env.get_reference_pose()
    altitude_limit, xy_limit = base_env._arena_limit_masks(pos_local)
    pillar_collision = base_env._pillar_collision_mask(pos_local)
    return {
        "pos_local": pos_local,
        "vel_world": base_env._robot.data.root_lin_vel_w,
        "quat": base_env._robot.data.root_quat_w,
        "ang_vel": base_env._robot.data.root_ang_vel_b,
        "ref_pos": ref_pos,
        "ref_yaw": ref_yaw,
        "altitude_limit": altitude_limit,
        "xy_limit": xy_limit,
        "pillar_collision": pillar_collision,
    }


def _safe_list_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_list_std(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return float((sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5)


def _distribution_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _pillar_clearance(base_env: Any, pos_local: torch.Tensor) -> torch.Tensor | None:
    pillars = getattr(base_env, "_pillar_positions_xy", None)
    if pillars is None or pillars.numel() == 0:
        return None
    dxy = torch.linalg.vector_norm(pos_local[:, None, :2] - pillars[None, :, :], dim=-1)
    return dxy.min(dim=1).values - float(getattr(base_env, "_pillar_collision_radius", 0.0))


def _ray_clearance(base_env: Any, pos_local: torch.Tensor) -> torch.Tensor | None:
    if not hasattr(base_env, "_get_ray_obstacle_points_xy") or getattr(base_env, "_ray_caster", None) is None:
        return None
    env_origins = getattr(base_env, "_terrain").env_origins
    try:
        ray_xy = base_env._get_ray_obstacle_points_xy(env_origins, pos_local)
    except Exception:
        return None
    if ray_xy.numel() == 0:
        return None
    dxy = torch.linalg.vector_norm(pos_local[:, None, :2] - ray_xy, dim=-1)
    max_dist = float(getattr(base_env.cfg, "ray_caster_max_distance", 0.0))
    dxy = torch.where(dxy > max_dist, torch.full_like(dxy, float("inf")), dxy)
    min_dxy = dxy.min(dim=1).values
    min_dxy = torch.where(torch.isfinite(min_dxy), min_dxy, torch.full_like(min_dxy, max_dist))
    return min_dxy - float(getattr(base_env.cfg, "drone_collision_radius", 0.0))


def _refresh_observations_after_reference_write(base_env: Any, obs: Any) -> Any:
    if hasattr(base_env, "_get_observations"):
        return base_env._get_observations()
    return obs


def _default_fixed_goals(env_cfg: Any) -> list[tuple[float, float, float]]:
    z = float((env_cfg.ref_pos_min[2] + env_cfg.ref_pos_max[2]) * 0.5)
    arena_min = env_cfg.arena_min
    arena_max = env_cfg.arena_max
    x_left = max(float(arena_min[0]) + 0.45, -1.8)
    x_right = min(float(arena_max[0]) - 0.45, 1.8)
    y_mid = 0.0
    y_high = min(float(arena_max[1]) - 0.45, 0.75)
    y_low = max(float(arena_min[1]) + 0.45, -0.75)
    return [
        (x_left, y_mid, z),
        (x_right, y_mid, z),
        (x_left, y_high, z),
        (x_right, y_low, z),
    ]


def _parse_fixed_goals(env_cfg: Any) -> list[tuple[float, float, float]]:
    if not args_cli.fixed_goals:
        goals = _default_fixed_goals(env_cfg)
    else:
        goals = []
        default_z = float((env_cfg.ref_pos_min[2] + env_cfg.ref_pos_max[2]) * 0.5)
        for idx, raw_goal in enumerate(args_cli.fixed_goals.split(";")):
            raw_goal = raw_goal.strip()
            if not raw_goal:
                continue
            parts = [float(part.strip()) for part in raw_goal.split(",")]
            if len(parts) == 2:
                parts.append(default_z)
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid --fixed-goals item #{idx + 1}: expected x,y or x,y,z, got '{raw_goal}'."
                )
            goals.append((parts[0], parts[1], parts[2]))

    if args_cli.benchmark_profile == "random":
        return []
    repeats = max(0, int(args_cli.fixed_goal_repeats))
    return [goal for goal in goals for _ in range(repeats)]


def _target_episode_count(fixed_goals: Sequence[tuple[float, float, float]]) -> int | None:
    if args_cli.num_episodes is not None:
        return int(args_cli.num_episodes)
    if args_cli.benchmark_profile == "fixed":
        return len(fixed_goals)
    if args_cli.benchmark_profile == "random":
        return int(args_cli.num_random_episodes)
    return len(fixed_goals) + int(args_cli.num_random_episodes)


def _write_episode_csv(path: Path, results: Sequence[EpisodeResult]) -> None:
    rows = [result.to_csv_row() for result in results]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize_results(results: Sequence[EpisodeResult], recorder: RolloutRecorder) -> dict[str, Any]:
    total = len(results)
    reason_counts: dict[str, int] = {}
    for result in results:
        reason_counts[result.done_label] = reason_counts.get(result.done_label, 0) + 1

    def values(name: str) -> list[float]:
        out = []
        for result in results:
            value = getattr(result, name)
            if value is not None:
                out.append(float(value))
        return out

    final_errors = values("final_pos_error")
    time_to_target = [float(result.duration_s) for result in results if result.success]
    safety_terminated = sum(1 for result in results if result.safety_terminated)
    success = sum(1 for result in results if result.success)
    collisions = sum(1 for result in results if result.done_reason == 6 or result.pillar_collisions > 0)

    by_scenario: dict[str, dict[str, Any]] = {}
    for kind in sorted({result.scenario_kind for result in results}):
        group = [result for result in results if result.scenario_kind == kind]
        group_final = [float(result.final_pos_error) for result in group if result.final_pos_error is not None]
        group_time_to_target = [float(result.duration_s) for result in group if result.success]
        group_collisions = sum(1 for result in group if result.done_reason == 6 or result.pillar_collisions > 0)
        by_scenario[kind] = {
            "episodes": len(group),
            "success_percent": 100.0 * sum(1 for result in group if result.success) / max(1, len(group)),
            "collision_percent": 100.0 * group_collisions / max(1, len(group)),
            "safety_termination_percent": (
                100.0 * sum(1 for result in group if result.safety_terminated) / max(1, len(group))
            ),
            "time_to_target_s": _distribution_stats(group_time_to_target),
            "final_position_error_m": _distribution_stats(group_final),
        }

    return {
        "episodes": total,
        "success_percent": 100.0 * success / max(1, total),
        "collision_percent": 100.0 * collisions / max(1, total),
        "safety_termination_percent": 100.0 * safety_terminated / max(1, total),
        "termination_counts": reason_counts,
        "time_to_target_s": _distribution_stats(time_to_target),
        "final_position_error_m": _distribution_stats(final_errors),
        "by_scenario": by_scenario,
    }


def _plot_rollouts(output_dir: Path, env_cfg: Any, results: Sequence[EpisodeResult]) -> dict[str, str]:
    if args_cli.no_plots or not results:
        return {}
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
    except Exception as exc:
        print(f"[WARN] Could not import matplotlib to plot benchmark rollouts: {exc}")
        return {}

    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(8, 7))
    arena_min = env_cfg.arena_min
    arena_max = env_cfg.arena_max
    ax.add_patch(
        Rectangle(
            (arena_min[0], arena_min[1]),
            arena_max[0] - arena_min[0],
            arena_max[1] - arena_min[1],
            fill=False,
            lw=2.0,
            ls="--",
            ec="black",
            label="Arena bounds",
        )
    )
    if getattr(env_cfg, "enable_pillars", False):
        for idx, (px, py) in enumerate(getattr(env_cfg, "pillar_positions_xy", ())):
            ax.add_patch(
                Circle(
                    (float(px), float(py)),
                    radius=float(env_cfg.pillar_radius),
                    color="dimgray",
                    alpha=0.35,
                    ec="black",
                    lw=1.0,
                    label="Pillar obstacle" if idx == 0 else None,
                )
            )

    cmap = plt.cm.get_cmap("tab20", max(1, len(results)))
    for idx, result in enumerate(results):
        if len(result.path_xy) < 2:
            continue
        xs = [p[0] for p in result.path_xy]
        ys = [p[1] for p in result.path_xy]
        color = cmap(idx)
        ax.plot(xs, ys, color=color, lw=1.5, alpha=0.9, label=result.scenario_label if idx < 12 else None)
        ax.scatter(xs[0], ys[0], color=color, marker="o", s=18)
        ax.scatter(xs[-1], ys[-1], color=color, marker="x", s=30)
        ax.scatter(result.target[0], result.target[1], color=color, marker="*", s=150, edgecolors="black", zorder=30)

    ax.set_title("Position-tracking benchmark trajectories")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7, ncols=2)
    fig.tight_layout()
    trajectory_path = output_dir / "trajectory_xy.png"
    fig.savefig(trajectory_path, dpi=180)
    plt.close(fig)
    paths["trajectory_xy"] = str(trajectory_path)

    labels = sorted({result.scenario_kind for result in results})
    if labels:
        groups = [[result for result in results if result.scenario_kind == label] for label in labels]
        time_data = [[result.duration_s for result in group if result.success] for group in groups]
        final_error_data = [
            [float(result.final_pos_error) for result in group if result.final_pos_error is not None] for group in groups
        ]
        success_data = [[1.0 if result.success else 0.0 for result in group] for group in groups]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=False)
        box_style = {
            "patch_artist": True,
            "boxprops": {"facecolor": "#d8e8f2", "edgecolor": "#243447", "linewidth": 1.0},
            "medianprops": {"color": "#b23a48", "linewidth": 1.5},
            "whiskerprops": {"color": "#243447", "linewidth": 1.0},
            "capprops": {"color": "#243447", "linewidth": 1.0},
        }

        axes[0].boxplot([data if data else [np.nan] for data in time_data], labels=labels, **box_style)
        axes[0].set_title("Time to target")
        axes[0].set_ylabel("seconds")

        axes[1].boxplot([data if data else [np.nan] for data in final_error_data], labels=labels, **box_style)
        axes[1].set_title("Final position error")
        axes[1].set_ylabel("meters")

        axes[2].boxplot(success_data, labels=labels, **box_style)
        axes[2].set_title("Success")
        axes[2].set_ylabel("0/1")
        axes[2].set_ylim(-0.05, 1.05)

        total = max(1, len(results))
        collision_pct = 100.0 * sum(
            1 for result in results if result.done_reason == 6 or result.pillar_collisions > 0
        ) / total
        success_pct = 100.0 * sum(1 for result in results if result.success) / total
        fig.suptitle(f"Success {success_pct:.1f}% | Collisions {collision_pct:.1f}%")
        for ax in axes:
            ax.grid(True, axis="y", alpha=0.25)
            ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        boxplot_path = output_dir / "benchmark_boxplots.png"
        fig.savefig(boxplot_path, dpi=180)
        plt.close(fig)
        paths["boxplots"] = str(boxplot_path)

    return paths


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
agent_cfg_entry_point = "skrl_dgppo_cfg_entry_point" if args_cli.algorithm == "dgppo" else "skrl_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, agent_cfg: dict):
    log_dir = _resolve_path(args_cli.log_dir)
    if not args_cli.exp_id:
        safe_task = args_cli.task.replace("/", "-")
        args_cli.exp_id = f"{safe_task}-{args_cli.algorithm}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = log_dir / args_cli.exp_id
    log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = log_dir / "videos"
    episodes_path = log_dir / "episodes.hdf5"
    metrics_path = log_dir / "metrics.json"
    episode_csv_path = log_dir / "episode_summary.csv"

    checkpoint = args_cli.checkpoint
    if checkpoint is None and args_cli.artifact and args_cli.algorithm == "dgppo":
        checkpoint = _download_wandb_artifact(args_cli.artifact, args_cli.artifact_file)
    if checkpoint is not None:
        checkpoint_path = _resolve_checkpoint_path(checkpoint)
        if checkpoint_path is None:
            raise ValueError("Could not resolve checkpoint path.")
        checkpoint = str(checkpoint_path)

    trained_agent_cfg = _agent_cfg_for_checkpoint(agent_cfg, checkpoint)
    trained_env_cfg = _env_cfg_for_checkpoint(checkpoint)
    _validate_checkpoint_algorithm(trained_agent_cfg, checkpoint)

    env_cfg.scene.num_envs = int(args_cli.num_envs or env_cfg.scene.num_envs)
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    _apply_env_overrides_from_agent_cfg(env_cfg, trained_agent_cfg)
    _apply_env_overrides_from_training_env_cfg(env_cfg, trained_env_cfg)
    env_cfg.domain_randomization.enable = False
    env_cfg.use_position_controller = args_cli.policy_mode == "baseline"
    if args_cli.policy_mode == "rl":
        if getattr(env_cfg, "obstacle_observation_mode", None) == "ray_caster":
            env_cfg.enable_obstacle_observations = True
            env_cfg.enable_ray_caster = True
    else:
        env_cfg.enable_obstacle_observations = True
        env_cfg.enable_ray_caster = True
        env_cfg.obstacle_observation_mode = "ray_caster"
    env_cfg.terminate_on_safety_violation = not args_cli.allow_safety_continuation
    env_cfg.terminate_on_success = not args_cli.no_terminate_on_success
    if hasattr(env_cfg, "terminate_on_out_of_boundaries"):
        env_cfg.terminate_on_out_of_boundaries = True
    env_cfg.ref_update_interval_s = 0.0
    if args_cli.control_mode:
        env_cfg.control_mode = args_cli.control_mode
    if args_cli.yaw_tracking:
        env_cfg.flag_yaw_tracking = True
    if args_cli.no_yaw_tracking:
        env_cfg.flag_yaw_tracking = False
    if args_cli.ref_update_interval is not None:
        env_cfg.ref_update_interval_s = float(args_cli.ref_update_interval)

    print(
        "[INFO] Benchmark observation layout: "
        f"obstacle_observation_mode={getattr(env_cfg, 'obstacle_observation_mode', None)}, "
        f"enable_ray_caster={getattr(env_cfg, 'enable_ray_caster', None)}, "
        f"ray_caster_observation_mode={getattr(env_cfg, 'ray_caster_observation_mode', None)}, "
        f"ray_caster_observation_data={getattr(env_cfg, 'ray_caster_observation_data', None)}, "
        f"ray_caster_top_k_hits={getattr(env_cfg, 'ray_caster_top_k_hits', None)}, "
        f"ray_caster_num_rays={getattr(env_cfg, 'ray_caster_num_rays', None)}",
        flush=True,
    )

    if args_cli.seed is not None:
        env_cfg.seed = int(args_cli.seed)

    env_cfg.debug_vis = (not args_cli.headless) or args_cli.video
    env_cfg.debug_visualizer = env_cfg.debug_vis
    if args_cli.visualize_rays:
        env_cfg.ray_caster_debug_vis = True
        env_cfg.debug_vis = True
        env_cfg.debug_visualizer = True
    if args_cli.camera_eye is not None:
        env_cfg.camera_view_eye = _parse_xyz(args_cli.camera_eye, arg_name="--camera-eye")
    if args_cli.camera_target is not None:
        env_cfg.camera_view_target = _parse_xyz(args_cli.camera_target, arg_name="--camera-target")
    if hasattr(env_cfg, "enable_fpv_camera_sensor"):
        env_cfg.enable_fpv_camera_sensor = bool(args_cli.spawn_cameras or args_cli.save_camera_images)
    if args_cli.video or args_cli.spawn_cameras or args_cli.save_camera_images:
        env_cfg.enable_cameras = True
    if args_cli.disable_cameras:
        env_cfg.enable_cameras = False
        if hasattr(env_cfg, "enable_fpv_camera_sensor"):
            env_cfg.enable_fpv_camera_sensor = False
    env_cfg.save_camera_images = args_cli.save_camera_images
    if args_cli.save_camera_images:
        env_cfg.camera_image_dir = str(video_dir / "camera_frames")
        env_cfg.camera_overlay_text = args_cli.camera_overlay_text

    fixed_goals = _parse_fixed_goals(env_cfg)
    target_episodes = _target_episode_count(fixed_goals)
    video_length = int(args_cli.video_length or args_cli.num_steps)
    video_episode_limit = args_cli.video_episodes
    if video_episode_limit is None:
        video_episode_limit = len(fixed_goals) if fixed_goals else min(4, int(target_episodes or 4))
    video_episode_limit = max(1, int(video_episode_limit))

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    if args_cli.video:
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Recording rollout video to: {video_dir}")
        video_kwargs = {
            "video_folder": str(video_dir),
            "video_length": video_length,
            "name_prefix": "pos-tracking",
            "disable_logger": True,
        }
        if args_cli.video_trigger == "episode":
            video_kwargs["episode_trigger"] = lambda episode_id: episode_id < video_episode_limit
        else:
            print(
                f"[INFO] Video trigger: first benchmark step, length={video_length} steps.",
                flush=True,
            )
            video_kwargs["step_trigger"] = lambda step: step == 0
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    device = env.unwrapped.device if hasattr(env, "unwrapped") else torch.device("cpu")
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    policy = None
    if args_cli.policy_mode == "rl":
        if args_cli.algorithm == "dgppo":
            if checkpoint is None:
                print("[WARN] No DGPPO checkpoint provided; falling back to random actions.")
                args_cli.policy_mode = "random"
            else:
                policy = DGPPOPolicyRunner.from_checkpoint(
                    checkpoint,
                    trained_agent_cfg,
                    base_env=base_env,
                    device=str(device),
                )
        elif args_cli.artifact:
            policy = PolicyRunner.from_wandb(
                args_cli.artifact,
                args_cli.actor_cfg,
                device=str(device),
                artifact_file=args_cli.artifact_file,
            )
        elif checkpoint:
            policy = PolicyRunner.from_checkpoint(
                checkpoint,
                args_cli.actor_cfg,
                base_env=base_env,
                agent_cfg_data=trained_agent_cfg,
                device=str(device),
            )
        else:
            print("[WARN] No checkpoint provided; falling back to random actions.")
            args_cli.policy_mode = "random"

    print("[INFO] Resetting benchmark environment.", flush=True)
    obs, _ = env.reset()
    print("[INFO] Environment reset complete.", flush=True)
    action_dim = _action_dim_from_env(base_env)

    scenario_manager = ScenarioManager(base_env, fixed_goals=fixed_goals)
    all_env_ids = list(range(int(base_env.num_envs)))
    scenario_manager.assign(all_env_ids)
    obs = _refresh_observations_after_reference_write(base_env, obs)
    recorder = RolloutRecorder(base_env, scenario_manager)
    recorder.start(all_env_ids)
    print(
        "[INFO] Benchmark schedule: "
        f"{len(fixed_goals)} fixed-goal episodes, "
        f"profile={args_cli.benchmark_profile}, target_episodes={target_episodes}, "
        f"terminate_on_safety_violation={base_env.cfg.terminate_on_safety_violation}, "
        f"terminate_on_success={base_env.cfg.terminate_on_success}"
    )

    episode_logger = (
        EpisodeLogger(
            env,
            episodes_path,
            args_cli.log_observations,
            {"task": args_cli.task, "algorithm": args_cli.algorithm},
            args_cli.task,
        )
        if args_cli.log_episodes
        else None
    )
    action_logger = ActionLogger(env, log_dir / "actions") if args_cli.log_actions else None

    total_steps = 0

    print("[INFO] Starting benchmark loop.", flush=True)
    for step_idx in range(args_cli.num_steps):
        if target_episodes is not None and len(recorder.results) >= target_episodes:
            break

        if args_cli.policy_mode == "baseline":
            actions = torch.zeros(base_env.num_envs, action_dim, device=device)
        elif args_cli.policy_mode == "random":
            actions = torch.empty(base_env.num_envs, action_dim, device=device).uniform_(-1.0, 1.0)
        else:
            obs_tensor = torch.as_tensor(obs["policy"], device=device)
            actions = policy(obs_tensor)

        if step_idx == 0:
            print("[INFO] Taking first benchmark environment step.", flush=True)
        obs, reward, terminated, truncated, _ = env.step(actions)
        if step_idx == 0:
            print("[INFO] First benchmark environment step complete.", flush=True)
        step_sample = recorder.capture_step(actions.detach(), use_last_step_snapshot=True)
        done = terminated | truncated
        recorder.append_step(step_sample, reward.detach())

        if episode_logger is not None:
            episode_logger.log_step(obs, actions, terminated, truncated)
        if action_logger is not None:
            action_logger.log_step(actions, done)

        if done.any():
            done_ids = torch.nonzero(done).squeeze(-1)
            reasons = base_env.get_last_episode_status()[done_ids]
            for env_id, reason in zip(done_ids.tolist(), reasons.tolist()):
                if target_episodes is None or len(recorder.results) < target_episodes:
                    result = recorder.finish(env_id, int(reason))
                    if result is not None:
                        print(
                            "[INFO] Episode "
                            f"{result.episode_id + 1}/{target_episodes or '?'} "
                            f"env={env_id} scenario={result.scenario_label} "
                            f"reason={result.done_label} final_err={result.final_pos_error}"
                        )
                if target_episodes is None or len(recorder.results) < target_episodes:
                    scenario_manager.assign([env_id])
                    obs = _refresh_observations_after_reference_write(base_env, obs)
                    recorder.start([env_id])
            if isinstance(policy, DGPPOPolicyRunner):
                policy.reset_done(done)

        total_steps += 1
        if total_steps % 100 == 0:
            print(
                f"[INFO] Benchmark progress: steps={total_steps}, episodes={len(recorder.results)}/{target_episodes or '?'}",
                flush=True,
            )

    if episode_logger is not None:
        episode_logger.close()

    if args_cli.save_episode_csv:
        _write_episode_csv(episode_csv_path, recorder.results)
    plot_paths = _plot_rollouts(log_dir, env_cfg, recorder.results)
    summary = _summarize_results(recorder.results, recorder)

    metrics = {
        "task": args_cli.task,
        "algorithm": args_cli.algorithm,
        "control_mode": base_env.cfg.control_mode,
        "policy_mode": args_cli.policy_mode,
        "checkpoint": checkpoint,
        "benchmark_profile": args_cli.benchmark_profile,
        "num_steps": total_steps,
        "target_episodes": target_episodes,
        "fixed_goals": [list(goal) for goal in fixed_goals],
        "obstacle_observation_mode": str(getattr(base_env.cfg, "obstacle_observation_mode", "")),
        "ray_caster_observation_mode": str(getattr(base_env.cfg, "ray_caster_observation_mode", "")),
        "ray_caster_observation_data": str(getattr(base_env.cfg, "ray_caster_observation_data", "")),
        "ray_caster_top_k_hits": int(getattr(base_env.cfg, "ray_caster_top_k_hits", 0)),
        "ray_caster_num_rays": int(getattr(base_env.cfg, "ray_caster_num_rays", 0)),
        "terminate_on_safety_violation": bool(base_env.cfg.terminate_on_safety_violation),
        "terminate_on_success": bool(base_env.cfg.terminate_on_success),
        "summary": summary,
        "episode_csv": str(episode_csv_path) if args_cli.save_episode_csv and recorder.results else None,
        "plots": plot_paths,
        "video_dir": str(video_dir) if args_cli.video else None,
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[INFO] Metrics: {metrics_path}")
    if args_cli.save_episode_csv and recorder.results:
        print(f"[INFO] Episode summary: {episode_csv_path}")
    for label, path in plot_paths.items():
        print(f"[INFO] Plot {label}: {path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
