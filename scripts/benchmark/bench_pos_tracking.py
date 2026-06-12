#!/usr/bin/env python3
"""Benchmark Crazyflie position-tracking policies or baseline controllers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
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
parser.add_argument("--num-steps", type=int, default=12000, help="Simulation steps to run.")
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
    "--allow-observation-adapter",
    action="store_true",
    help="Allow legacy PPO observation conversion when checkpoint and benchmark dimensions differ.",
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
    choices=["pursuit", "pillar_random", "fixed", "random"],
    default="pursuit",
    help="Goal schedule to evaluate.",
)
parser.add_argument(
    "--tests-per-difficulty",
    type=int,
    default=30,
    help="Number of pursuit benchmark scenarios to run for Easy, Medium, and Hard.",
)
parser.add_argument(
    "--evader-speed",
    type=float,
    default=None,
    help="Pursuit evader speed in m/s. Defaults to the maximum speed from the training config.",
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
from source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pos_tracking.scenario_pool import (
    ScenarioPoolBuilder,
    point_free,
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
    "episode_length_s",
    "decimation",
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
    "enable_pursuit_evasion_curriculum",
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
    "ref_pos_min",
    "ref_pos_max",
    "ref_yaw_range",
    "pursuit_curriculum_phase_fractions",
    "pursuit_curriculum_blend_fraction",
    "pursuit_phase1_fixed_evader_fraction",
    "pursuit_scenario_pool_size",
    "pursuit_grid_cell_size",
    "pursuit_grid_wall_margin",
    "pursuit_pursuer_occupied_side",
    "pursuit_evader_goal_min_distance",
    "pursuit_dynamic_rail_length_range",
    "pursuit_path_waypoint_dt",
    "pursuit_evader_speed_range",
    "pursuit_smooth_evader_path",
    "pursuit_smooth_evader_resolution",
    "pursuit_smooth_evader_validate",
    "pursuit_smooth_evader_goal_blend_distance",
    "pursuit_smooth_evader_goal_sample_min_alignment",
    "pursuit_smooth_evader_goal_min_alignment",
    "pursuit_smooth_evader_goal_attempts",
    "pursuit_evader_wall_clearance",
    "pursuit_evader_radius",
    "pursuit_max_static_obstacles",
    "pursuit_max_dynamic_obstacles",
    "pursuit_dynamic_obstacle_radius",
    "pursuit_dynamic_obstacle_height",
    "pursuit_obstacle_clearance",
    "pursuit_dynamic_max_speed",
    "pursuit_pursuer_wall_clearance",
    "pursuit_pursuer_min_evader_distance",
    "pursuit_scenario_attempts",
    "reward_pos",
    "reward_pos_scale",
    "reward_approach",
    "reward_yaw",
    "reward_body_rates",
    "reward_body_rates_roll_pitch",
    "reward_body_rates_yaw",
    "reward_lin_vel",
    "reward_action_smoothness",
    "reward_action_smoothness_rpy",
    "reward_action_smoothness_thrust",
    "reward_success",
    "penalty_timeout",
    "penalty_altitude_limit",
    "penalty_xy_boundary",
    "penalty_pillar_collision",
    "pos_tolerance",
    "yaw_tolerance",
    "success_hold_time_s",
    "terminate_on_success",
    "terminate_on_safety_violation",
    "terminate_on_out_of_boundaries",
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
    sim_cfg = _as_plain_container(env_cfg_data.get("sim"))
    if isinstance(sim_cfg, Mapping) and "dt" in sim_cfg and hasattr(env_cfg, "sim"):
        env_cfg.sim.dt = float(sim_cfg["dt"])
        applied.append("sim.dt")
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "render_interval"):
        env_cfg.sim.render_interval = int(env_cfg.decimation)
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
    for candidate in (run_dir / "params" / filename, run_dir / filename):
        if candidate.exists():
            return candidate
    return None


def _selected_training_cfg_path(checkpoint: str | Path | None, explicit: Path | None, filename: str) -> Path | None:
    if explicit is not None:
        path = _resolve_existing_path(explicit)
        return path if path.exists() else None
    return _checkpoint_params_yaml(checkpoint, filename)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_ppo_training_context(checkpoint: str | Path | None) -> tuple[Path, Path]:
    agent_path = _selected_training_cfg_path(checkpoint, args_cli.trained_agent_cfg, "agent.yaml")
    env_path = _selected_training_cfg_path(checkpoint, args_cli.trained_env_cfg, "env.yaml")
    missing = []
    if agent_path is None:
        missing.append("agent.yaml")
    if env_path is None:
        missing.append("env.yaml")
    if missing:
        raise FileNotFoundError(
            f"PPO checkpoint {checkpoint} is missing its training context: {', '.join(missing)}. "
            "Keep params/ beside checkpoints/, or pass --trained-agent-cfg and --trained-env-cfg."
        )
    return agent_path, env_path


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


def _validate_task_env_contract(env_cfg_data: Mapping[str, Any]) -> None:
    control_mode = str(env_cfg_data.get("control_mode", ""))
    task_name = str(args_cli.task).lower()
    expected = "RL_rates" if "rates" in task_name else "RL_velocity" if "velocity" in task_name else ""
    if expected and control_mode and control_mode != expected:
        raise ValueError(
            f"Training env config uses control_mode={control_mode}, but --task {args_cli.task} expects {expected}."
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


def _ppo_checkpoint_provenance(
    checkpoint: str | Path,
    agent_cfg_path: Path,
    env_cfg_path: Path,
) -> dict[str, Any]:
    checkpoint_path = _resolve_existing_path(checkpoint)
    agent_cfg_data = _load_yaml_mapping(agent_cfg_path)
    env_cfg_data = _load_yaml_mapping(env_cfg_path)
    run_dir = _checkpoint_run_dir(checkpoint_path)
    metadata_path = None
    if run_dir is not None:
        for candidate in (run_dir / "params" / "run_metadata.json", run_dir / "run_metadata.json"):
            if candidate.exists():
                metadata_path = candidate
                break
    run_metadata = None
    if metadata_path is not None and metadata_path.exists():
        run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_algorithm = str(run_metadata.get("algorithm", "")).lower()
        if metadata_algorithm and metadata_algorithm != "ppo":
            raise ValueError(
                f"Run metadata says algorithm={metadata_algorithm}, but PPO evaluation was requested."
            )
        metadata_task = run_metadata.get("task")
        if metadata_task and str(metadata_task) != str(args_cli.task):
            raise ValueError(
                f"Run metadata says task={metadata_task}, but benchmark task={args_cli.task}."
            )
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = _extract_ppo_policy_state_dict(payload)
    if state_dict is None:
        raise ValueError(f"Unable to locate PPO policy weights in checkpoint: {checkpoint_path}")

    linear_layers: list[tuple[int, torch.Tensor]] = []
    for key, value in state_dict.items():
        parts = str(key).split(".")
        if len(parts) >= 3 and parts[0] == "net_container" and parts[-1] == "weight":
            if isinstance(value, torch.Tensor) and value.ndim == 2:
                linear_layers.append((int(parts[1]), value))
    if not linear_layers:
        raise ValueError(f"Unable to infer PPO network dimensions from checkpoint: {checkpoint_path}")
    linear_layers.sort(key=lambda item: item[0])

    obs_state = payload.get("observation_preprocessor") if isinstance(payload, Mapping) else None
    scaler_mean = obs_state.get("running_mean") if isinstance(obs_state, Mapping) else None
    scaler_count = obs_state.get("current_count") if isinstance(obs_state, Mapping) else None
    checkpoint_step = None
    step_match = re.search(r"([0-9]+)$", checkpoint_path.stem)
    if step_match:
        checkpoint_step = int(step_match.group(1))
    agent_section = agent_cfg_data.get("agent", {})
    trainer_section = agent_cfg_data.get("trainer", {})
    sim_section = env_cfg_data.get("sim", {})
    training_contract = {
        "seed": env_cfg_data.get("seed", agent_cfg_data.get("seed")),
        "agent_class": agent_section.get("class") if isinstance(agent_section, Mapping) else None,
        "trainer_timesteps": trainer_section.get("timesteps") if isinstance(trainer_section, Mapping) else None,
        "episode_length_s": env_cfg_data.get("episode_length_s"),
        "sim_dt": sim_section.get("dt") if isinstance(sim_section, Mapping) else None,
        "decimation": env_cfg_data.get("decimation"),
        "control_mode": env_cfg_data.get("control_mode"),
        "obstacle_observation_mode": env_cfg_data.get("obstacle_observation_mode"),
        "ray_caster_observation_mode": env_cfg_data.get("ray_caster_observation_mode"),
        "ray_caster_observation_data": env_cfg_data.get("ray_caster_observation_data"),
        "ray_caster_num_rays": env_cfg_data.get("ray_caster_num_rays"),
        "pursuit_evader_speed_range": env_cfg_data.get("pursuit_evader_speed_range"),
        "pursuit_max_static_obstacles": env_cfg_data.get("pursuit_max_static_obstacles"),
        "pursuit_max_dynamic_obstacles": env_cfg_data.get("pursuit_max_dynamic_obstacles"),
        "observation_preprocessor": agent_section.get("observation_preprocessor")
        if isinstance(agent_section, Mapping)
        else None,
    }
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_step": checkpoint_step,
        "agent_cfg_path": str(agent_cfg_path),
        "agent_cfg_sha256": _file_sha256(agent_cfg_path),
        "env_cfg_path": str(env_cfg_path),
        "env_cfg_sha256": _file_sha256(env_cfg_path),
        "run_metadata_path": str(metadata_path) if metadata_path is not None and metadata_path.exists() else None,
        "run_metadata": run_metadata,
        "training_contract": training_contract,
        "policy_obs_dim": int(linear_layers[0][1].shape[1]),
        "policy_action_dim": int(linear_layers[-1][1].shape[0]),
        "policy_hidden_layers": [int(weight.shape[0]) for _idx, weight in linear_layers[:-1]],
        "observation_scaler_present": isinstance(scaler_mean, torch.Tensor),
        "observation_scaler_shape": list(scaler_mean.shape) if isinstance(scaler_mean, torch.Tensor) else None,
        "observation_scaler_count": float(scaler_count.item())
        if isinstance(scaler_count, torch.Tensor) and scaler_count.numel() == 1
        else None,
    }


def _validate_ppo_runtime_contract(
    base_env: Any,
    agent_cfg_data: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    env_obs_dim = _obs_dim_from_env(base_env)
    env_action_dim = _action_dim_from_env(base_env)
    policy_obs_dim = int(provenance["policy_obs_dim"])
    policy_action_dim = int(provenance["policy_action_dim"])
    if policy_action_dim != env_action_dim:
        raise ValueError(
            f"PPO checkpoint action_dim={policy_action_dim}, but benchmark environment action_dim={env_action_dim}."
        )
    if policy_obs_dim != env_obs_dim and not args_cli.allow_observation_adapter:
        raise ValueError(
            f"PPO checkpoint obs_dim={policy_obs_dim}, but restored benchmark environment obs_dim={env_obs_dim}."
        )

    agent_section = agent_cfg_data.get("agent", {}) if isinstance(agent_cfg_data, Mapping) else {}
    uses_scaler = isinstance(agent_section, Mapping) and str(
        agent_section.get("observation_preprocessor", "")
    ).lower() not in {"", "none", "null"}
    scaler_shape = provenance.get("observation_scaler_shape")
    if uses_scaler and scaler_shape is None:
        raise ValueError("PPO training config requires an observation preprocessor, but the checkpoint has no scaler.")
    if scaler_shape is not None and int(math.prod(scaler_shape)) != policy_obs_dim:
        raise ValueError(
            f"PPO observation scaler shape={scaler_shape} does not match checkpoint obs_dim={policy_obs_dim}."
        )
    print(
        "[INFO] PPO training contract verified: "
        f"obs_dim={policy_obs_dim}, action_dim={policy_action_dim}, "
        f"scaler={'yes' if scaler_shape is not None else 'no'}, "
        f"checkpoint_step={provenance.get('checkpoint_step')}, "
        f"trainer_timesteps={provenance.get('training_contract', {}).get('trainer_timesteps')}, "
        f"checkpoint_sha256={str(provenance['checkpoint_sha256'])[:12]}."
    )


class PolicyRunner:
    def __init__(
        self,
        actor,
        device: str,
        base_env: Any | None = None,
        *,
        allow_observation_adapter: bool = False,
    ):
        self.actor = actor
        self.device = torch.device(device)
        self.base_env = base_env
        self.allow_observation_adapter = bool(allow_observation_adapter)
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
        allow_observation_adapter: bool = False,
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
        actor = load_actor_from_checkpoint(checkpoint, cfg, device=device, strict=True)
        return cls(
            actor,
            device=device,
            base_env=base_env,
            allow_observation_adapter=allow_observation_adapter,
        )

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
            if not self.allow_observation_adapter:
                raise ValueError(
                    f"PPO checkpoint expects obs_dim={self.expected_obs_dim}, but the restored training "
                    f"environment produced obs_dim={obs.shape[-1]}. Use --allow-observation-adapter only "
                    "for a deliberately supported legacy checkpoint."
                )
            obs = self._adapt_observation(obs)
        if self.obs_scaler and self.obs_scaler.mean is not None and self.obs_scaler.std is not None:
            mean = self.obs_scaler.mean.to(self.device)
            std = self.obs_scaler.std.to(self.device)
            if mean.numel() != obs.shape[-1] or std.numel() != obs.shape[-1]:
                raise ValueError(
                    f"PPO observation scaler has shape {tuple(mean.shape)}, but policy input is {obs.shape[-1]}."
                )
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
        goal_copy_dim = min(6, goal_state.shape[-1], state_dim)
        adapted_goal[..., :goal_copy_dim] = goal_state[..., :goal_copy_dim]
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
    path_kind: str = ""
    static_obstacles: int = 0
    dynamic_obstacles: int = 0
    evader_start: tuple[float, float, float] | None = None
    evader_end: tuple[float, float, float] | None = None
    scenario_data: dict[str, Any] | None = field(default=None, repr=False)


DIFFICULTY_ORDER = ("Easy", "Medium", "Hard")
DIFFICULTY_PHASE = {"Easy": 2, "Medium": 3, "Hard": 4}
PATH_TYPE_CODES = {
    "straight": 0,
    "circle": 1,
    "waypoint": 2,
    "corridor": 3,
    "bottleneck": 4,
}


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
    min_safety_margin: float | None = None

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
        safety_margin: float | None,
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
        if safety_margin is not None:
            self.min_safety_margin = (
                float(safety_margin)
                if self.min_safety_margin is None
                else min(self.min_safety_margin, float(safety_margin))
            )


@dataclass
class EpisodeResult:
    episode_id: int
    env_id: int
    scenario_kind: str
    scenario_label: str
    path_kind: str
    static_obstacles: int
    dynamic_obstacles: int
    target: tuple[float, float, float]
    evader_start: tuple[float, float, float] | None
    evader_end: tuple[float, float, float] | None
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
    min_safety_margin: float | None
    path_xy: list[tuple[float, float]]
    reference_xy: list[tuple[float, float]]
    errors: list[float]
    static_obstacle_xy: list[tuple[float, float]]
    dynamic_obstacle_paths_xy: list[list[tuple[float, float]]]

    @property
    def safety_violation_steps(self) -> int:
        return self.altitude_violations + self.boundary_violations + self.pillar_collisions

    @property
    def success(self) -> bool:
        return self.done_reason == 1

    @property
    def safety_terminated(self) -> bool:
        return self.done_reason in {2, 3, 6}

    @property
    def collided(self) -> bool:
        return self.safety_terminated or self.safety_violation_steps > 0

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "env_id": self.env_id,
            "scenario_kind": self.scenario_kind,
            "scenario_label": self.scenario_label,
            "path_kind": self.path_kind,
            "static_obstacles": self.static_obstacles,
            "dynamic_obstacles": self.dynamic_obstacles,
            "target_x": self.target[0],
            "target_y": self.target[1],
            "target_z": self.target[2],
            "evader_start_x": None if self.evader_start is None else self.evader_start[0],
            "evader_start_y": None if self.evader_start is None else self.evader_start[1],
            "evader_start_z": None if self.evader_start is None else self.evader_start[2],
            "evader_end_x": None if self.evader_end is None else self.evader_end[0],
            "evader_end_y": None if self.evader_end is None else self.evader_end[1],
            "evader_end_z": None if self.evader_end is None else self.evader_end[2],
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
            "min_safety_margin": self.min_safety_margin,
            "safety_terminated": self.safety_terminated,
            "collided": self.collided,
            "success": self.success,
        }


class ScenarioManager:
    def __init__(
        self,
        base_env: Any,
        fixed_goals: Sequence[tuple[float, float, float]],
        *,
        pursuit: bool,
        tests_per_difficulty: int,
        evader_speed: float | None = None,
    ):
        self.base_env = base_env
        self.pursuit = bool(pursuit)
        self._fixed_queue = [
            GoalScenario(kind="fixed", label=f"fixed_{idx:02d}", goal=goal)
            for idx, goal in enumerate(fixed_goals)
        ]
        self._next_fixed = 0
        self._pursuit_queue: list[tuple[str, int]] = []
        if self.pursuit:
            count = max(0, int(tests_per_difficulty))
            self._pursuit_queue = [(difficulty, idx) for difficulty in DIFFICULTY_ORDER for idx in range(count)]
        self._next_pursuit = 0
        self.active: list[GoalScenario | None] = [None for _ in range(int(base_env.num_envs))]
        self._trajectory_stats: list[dict[str, float]] = []
        self._scenario_builder = ScenarioPoolBuilder(base_env.cfg, base_env.device) if self.pursuit else None
        speed_range = tuple(getattr(base_env.cfg, "pursuit_evader_speed_range", (0.0, 0.0)))
        self._evader_speed = max(0.0, float(max(speed_range) if evader_speed is None else evader_speed))

    @property
    def pursuit_target_episodes(self) -> int:
        return len(self._pursuit_queue)

    def trajectory_summary(self) -> dict[str, float | int] | None:
        if not self._trajectory_stats:
            return None
        return {
            "scenario_count": len(self._trajectory_stats),
            "requested_speed_mps": self._evader_speed,
            "min_executed_distance_m": min(item["executed_distance_m"] for item in self._trajectory_stats),
            "max_executed_speed_mps": max(item["max_speed_mps"] for item in self._trajectory_stats),
            "min_distance_ratio": min(item["distance_ratio"] for item in self._trajectory_stats),
            "min_near_target_fraction": min(item["near_target_fraction"] for item in self._trajectory_stats),
            "fallback_count": sum(int(item["fallback"]) for item in self._trajectory_stats),
        }

    def assign(self, env_ids: Sequence[int]) -> list[int]:
        assigned: list[int] = []
        for env_id in env_ids:
            env_id = int(env_id)
            if self.pursuit:
                if self._next_pursuit >= len(self._pursuit_queue):
                    self.active[env_id] = None
                    continue
                difficulty, scenario_idx = self._pursuit_queue[self._next_pursuit]
                self._next_pursuit += 1
                scenario = self._build_pursuit_scenario(difficulty, scenario_idx, env_id)
                self._apply_pursuit_scenario(env_id, scenario)
                self.active[env_id] = scenario
                assigned.append(env_id)
                continue

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
            self.active[env_id] = scenario
            assigned.append(env_id)
        return assigned

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

    def _build_pursuit_scenario(self, difficulty: str, scenario_idx: int, env_id: int) -> GoalScenario:
        for attempt in range(64):
            variant = scenario_idx + 37 * attempt
            if difficulty == "Easy":
                data = self._candidate_easy(scenario_idx, variant)
            elif difficulty == "Medium":
                data = self._candidate_medium(scenario_idx, variant)
            else:
                data = self._candidate_hard(scenario_idx, variant)
            if data is None:
                continue
            if self._valid_pursuit_scenario(data, difficulty):
                return self._scenario_from_data(difficulty, scenario_idx, data)

        data = self._fallback_env_sample(difficulty)
        return self._scenario_from_data(difficulty, scenario_idx, data)

    def _scenario_from_data(self, difficulty: str, scenario_idx: int, data: dict[str, Any]) -> GoalScenario:
        evader_pos = data["evader_pos"]
        static_active = data["static_active"]
        dynamic_active = data["dynamic_active"]
        stats = self._evader_trajectory_stats(data)
        speed_hi = max(0.0, float(max(getattr(self.base_env.cfg, "pursuit_evader_speed_range", (0.0, 0.0)))))
        speed_tolerance = max(1e-3, 1e-3 * speed_hi)
        if stats["max_speed_mps"] > speed_hi + speed_tolerance:
            raise RuntimeError(
                f"Scenario {difficulty}_{scenario_idx:02d} exceeds the training evader speed maximum: "
                f"{stats['max_speed_mps']:.3f} > {speed_hi:.3f} m/s."
            )
        if self._evader_speed > 0.0 and stats["distance_ratio"] < 0.9:
            raise RuntimeError(
                f"Scenario {difficulty}_{scenario_idx:02d} evader path is too short: "
                f"executed/requested distance ratio={stats['distance_ratio']:.3f}."
            )
        self._trajectory_stats.append(stats)
        start = evader_pos[0].detach().cpu().tolist()
        end = evader_pos[-1].detach().cpu().tolist()
        path_kind = str(data.get("path_kind", "pursuit"))
        return GoalScenario(
            kind=difficulty,
            label=f"{difficulty}_{scenario_idx:02d}",
            goal=(float(start[0]), float(start[1]), float(start[2])),
            path_kind=path_kind,
            static_obstacles=int(static_active.to(torch.int32).sum().item()),
            dynamic_obstacles=int(dynamic_active.to(torch.int32).sum().item()),
            evader_start=(float(start[0]), float(start[1]), float(start[2])),
            evader_end=(float(end[0]), float(end[1]), float(end[2])),
            scenario_data=data,
        )

    def _apply_pursuit_scenario(self, env_id: int, scenario: GoalScenario) -> None:
        data = scenario.scenario_data
        if data is None:
            raise RuntimeError(f"Pursuit scenario {scenario.label} has no tensor payload.")

        env = self.base_env
        device = env.device
        env_ids = torch.tensor([int(env_id)], device=device, dtype=torch.long)
        env.episode_length_buf[env_ids] = 0
        if hasattr(env, "reset_buf"):
            env.reset_buf[env_ids] = False
        if hasattr(env, "reset_terminated"):
            env.reset_terminated[env_ids] = False
        if hasattr(env, "reset_time_outs"):
            env.reset_time_outs[env_ids] = False

        env._scenario_phase[env_id] = int(data["phase"])
        env._scenario_fallback[env_id] = bool(data.get("fallback", False))
        env._evader_path_type[env_id] = int(data["path_type"])
        env._evader_waypoints[env_id] = self._dense_path_to_waypoints(data["evader_pos"].to(device=device))
        env._pursuer_start_pos[env_id] = data["pursuer_start"].to(device=device)
        env._reference_pos[env_id] = data["evader_pos"][0].to(device=device)
        env._reference_yaw[env_id, 0] = float(data["evader_yaw"])
        env._reference_timer[env_id] = 0.0
        env._success_counter[env_id] = 0
        env._last_success[env_id] = False

        env._static_obstacle_positions_xy[env_id] = data["static_xy"].to(device=device)
        env._static_obstacle_active[env_id] = data["static_active"].to(device=device)
        env._dynamic_obstacle_waypoints[env_id] = self._dense_dynamic_path_to_waypoints(
            data["dynamic_pos"].to(device=device)
        )
        env._dynamic_obstacle_active[env_id] = data["dynamic_active"].to(device=device)
        env._dynamic_obstacle_positions[env_id] = data["dynamic_pos"][0].to(device=device)

        if hasattr(env, "_actions"):
            env._actions[env_ids] = 0.0
            env._prev_actions[env_ids] = 0.0
            env._action_diff[env_ids] = 0.0
        if getattr(env, "_action_wrapper", None) is not None:
            env._action_wrapper.reset(env_ids)
        if getattr(env, "_baseline_controller", None) is not None:
            env._baseline_controller.reset(env_ids)

        root_state = env._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = env._pursuer_start_pos[env_ids] + env._terrain.env_origins[env_ids]
        root_state[:, 3:7] = env._spawn_yaw_quat(env_ids)
        root_state[:, 7:] = 0.0
        env._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        env._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

        env._update_pursuit_episode_motion(env_ids)
        env._move_static_obstacles(env_ids)
        env._move_dynamic_obstacles(env_ids)
        env._refresh_debug_pillars()
        env._clip_agent_root_state_in_sim(env_ids)
        if getattr(env, "_ray_caster", None) is not None:
            try:
                env._ray_caster.update(0.0, force_recompute=True)
            except Exception:
                pass

    def _candidate_easy(self, scenario_idx: int, variant: int) -> dict[str, Any] | None:
        mode = scenario_idx % 3
        if mode == 0:
            path_kind = "waypoint"
            points = [(1.45, 0.75), (0.45, 0.55), (-1.25, 0.15), (-1.55, -0.35)]
            pursuer_xy = (-1.55, -0.95)
        elif mode == 1:
            path_kind = "circle"
            xy = self._circle_xy((0.35, 0.05), radius=0.62, phase=0.35 * scenario_idx)
            pursuer_xy = (-1.45, -0.95)
            return self._pack_candidate(path_kind, xy, pursuer_xy, desired_static=2, desired_dynamic=0, variant=variant)
        else:
            path_kind = "straight"
            points = [(1.55, -0.65), (0.35, -0.25), (-1.35, 0.55)]
            pursuer_xy = (-1.55, 0.95)

        xy = self._polyline_xy(points)
        return self._pack_candidate(path_kind, xy, pursuer_xy, desired_static=2, desired_dynamic=0, variant=variant)

    def _candidate_medium(self, scenario_idx: int, variant: int) -> dict[str, Any] | None:
        desired_static = 3 + scenario_idx % 3
        if scenario_idx % 2 == 0:
            points = [(-1.55, 0.85), (-0.75, 0.75), (-0.25, 0.05), (0.55, -0.2), (1.45, -0.75)]
            pursuer_xy = (1.65, 0.95)
        else:
            points = [(1.55, -0.85), (0.75, -0.65), (0.15, 0.1), (-0.55, 0.25), (-1.45, 0.75)]
            pursuer_xy = (-1.65, -0.95)
        xy = self._polyline_xy(points)
        return self._pack_candidate(
            "corridor",
            xy,
            pursuer_xy,
            desired_static=desired_static,
            desired_dynamic=0,
            variant=variant,
        )

    def _candidate_hard(self, scenario_idx: int, variant: int) -> dict[str, Any] | None:
        mixed_counts = ((3, 1), (4, 1), (3, 2), (5, 1), (4, 2), (3, 3))
        desired_static, desired_dynamic = mixed_counts[scenario_idx % len(mixed_counts)]
        if scenario_idx % 2 == 0:
            points = [(-1.55, -0.85), (-0.75, -0.65), (-0.2, -0.1), (0.55, 0.15), (1.45, 0.85)]
            pursuer_xy = (1.65, -1.0)
        else:
            points = [(1.55, 0.85), (0.75, 0.65), (0.2, 0.05), (-0.55, -0.15), (-1.45, -0.85)]
            pursuer_xy = (-1.65, 1.0)
        xy = self._polyline_xy(points)
        return self._pack_candidate(
            "bottleneck",
            xy,
            pursuer_xy,
            desired_static=desired_static,
            desired_dynamic=desired_dynamic,
            variant=variant,
        )

    def _pack_candidate(
        self,
        path_kind: str,
        xy: torch.Tensor,
        pursuer_xy: tuple[float, float],
        *,
        desired_static: int,
        desired_dynamic: int,
        variant: int,
    ) -> dict[str, Any] | None:
        evader_xy = self._transform_xy(xy, variant, margin=float(self.base_env.cfg.pursuit_evader_wall_clearance))
        pursuer_xy_t = self._transform_xy(
            self._tensor_xy([pursuer_xy]),
            variant,
            margin=float(self.base_env.cfg.pursuit_pursuer_wall_clearance),
        )[0]
        evader_pos = self._finish_evader_xy_path(evader_xy)
        pursuer_start = self._finish_point(pursuer_xy_t)

        static_seed = self._static_seed_points(path_kind, evader_xy, pursuer_xy_t)
        static = self._make_static_slots(static_seed, evader_pos, pursuer_start, desired_static, variant)
        if static is None:
            return None
        static_xy, static_active = static

        dynamic_paths = self._dynamic_seed_paths(path_kind, variant)
        dynamic = self._make_dynamic_slots(dynamic_paths, evader_pos, static_xy, static_active, desired_dynamic)
        if dynamic is None:
            return None
        dynamic_pos, dynamic_vel, dynamic_active = dynamic

        return self._pack_scenario_data(
            path_kind=path_kind,
            evader_pos=evader_pos,
            pursuer_start=pursuer_start,
            static_xy=static_xy,
            static_active=static_active,
            dynamic_pos=dynamic_pos,
            dynamic_vel=dynamic_vel,
            dynamic_active=dynamic_active,
            fallback=False,
        )

    def _evader_trajectory_stats(self, data: Mapping[str, Any]) -> dict[str, float]:
        evader_pos = data["evader_pos"]
        evader_vel = data["evader_vel"]
        speed = torch.linalg.vector_norm(evader_vel, dim=-1)
        executed_distance = torch.linalg.vector_norm(evader_pos[1:] - evader_pos[:-1], dim=-1).sum()
        duration_s = max(0.0, float(evader_pos.shape[0] - 1) * float(self.base_env.step_dt))
        requested_distance = self._evader_speed * duration_s
        near_target = speed >= 0.95 * self._evader_speed if self._evader_speed > 0.0 else speed <= 0.05
        return {
            "executed_distance_m": float(executed_distance.item()),
            "max_speed_mps": float(speed.max().item()),
            "distance_ratio": float(executed_distance.item()) / max(requested_distance, 1e-6),
            "near_target_fraction": float(near_target.to(torch.float32).mean().item()),
            "fallback": float(bool(data.get("fallback", False))),
        }

    def _pack_scenario_data(
        self,
        *,
        path_kind: str,
        evader_pos: torch.Tensor,
        pursuer_start: torch.Tensor,
        static_xy: torch.Tensor,
        static_active: torch.Tensor,
        dynamic_pos: torch.Tensor,
        dynamic_vel: torch.Tensor,
        dynamic_active: torch.Tensor,
        fallback: bool,
    ) -> dict[str, Any]:
        evader_pos = self._waypoints_to_dense(self._dense_path_to_waypoints(evader_pos))
        evader_vel = self._path_velocity(evader_pos)
        speed = torch.linalg.vector_norm(evader_vel[:, :2], dim=-1)
        moving = torch.nonzero(speed > 0.05).squeeze(-1)
        first_idx = int(moving[0].item()) if moving.numel() > 0 else 0
        first_vel = evader_vel[first_idx]
        yaw = torch.atan2(first_vel[1], first_vel[0])
        difficulty = self._difficulty_for_counts(static_active, dynamic_active)
        return {
            "phase": DIFFICULTY_PHASE[difficulty],
            "path_type": PATH_TYPE_CODES.get(path_kind, 0),
            "path_kind": path_kind,
            "evader_pos": evader_pos,
            "evader_vel": evader_vel,
            "evader_yaw": float(yaw.item()),
            "pursuer_start": pursuer_start,
            "static_xy": static_xy,
            "static_active": static_active,
            "dynamic_pos": dynamic_pos,
            "dynamic_vel": dynamic_vel,
            "dynamic_active": dynamic_active,
            "fallback": bool(fallback),
        }

    def _difficulty_for_counts(self, static_active: torch.Tensor, dynamic_active: torch.Tensor) -> str:
        if int(dynamic_active.to(torch.int32).sum().item()) > 0:
            return "Hard"
        if int(static_active.to(torch.int32).sum().item()) >= 3:
            return "Medium"
        return "Easy"

    def _fallback_env_sample(self, difficulty: str) -> dict[str, Any]:
        phase = DIFFICULTY_PHASE[difficulty]
        attempts = max(32, int(getattr(self.base_env.cfg, "pursuit_scenario_attempts", 300)) // 3)
        assert self._scenario_builder is not None
        for _ in range(attempts):
            sampled = self._scenario_builder.sample_scenario(phase, evader_speed=self._evader_speed)
            if sampled is not None:
                sampled["fallback"] = True
                sampled["path_kind"] = f"env_phase_{phase}"
                return self._dense_env_sample(sampled)
        sampled = self._scenario_builder.fallback_scenario(phase)
        sampled["fallback"] = True
        sampled["path_kind"] = f"env_fallback_phase_{phase}"
        return self._dense_env_sample(sampled)

    def _dense_env_sample(self, data: dict[str, Any]) -> dict[str, Any]:
        if "evader_pos" in data:
            return data

        env = self.base_env
        out = dict(data)
        evader_pos = self._waypoints_to_dense(data["evader_waypoints"])
        evader_vel = self._path_velocity(evader_pos)
        dynamic_wp = data["dynamic_waypoints"]
        if dynamic_wp.shape[1] > 0:
            dynamic_pos = torch.stack(
                [self._waypoints_to_dense(dynamic_wp[:, slot]) for slot in range(dynamic_wp.shape[1])],
                dim=1,
            )
            dynamic_vel = torch.stack(
                [self._path_velocity(dynamic_pos[:, slot]) for slot in range(dynamic_pos.shape[1])],
                dim=1,
            )
        else:
            dynamic_pos = dynamic_wp.new_zeros(env._path_steps, 0, 3)
            dynamic_vel = torch.zeros_like(dynamic_pos)
        out["evader_pos"] = evader_pos
        out["evader_vel"] = evader_vel
        out["dynamic_pos"] = dynamic_pos
        out["dynamic_vel"] = dynamic_vel
        return out

    def _valid_pursuit_scenario(self, data: dict[str, Any], difficulty: str) -> bool:
        env = self.base_env
        evader_pos = data["evader_pos"]
        evader_vel = data["evader_vel"]
        pursuer_start = data["pursuer_start"]
        static_xy = data["static_xy"]
        static_active = data["static_active"]
        dynamic_pos = data["dynamic_pos"]
        dynamic_active = data["dynamic_active"]

        lo = env._arena_min_safe + float(env.cfg.pursuit_evader_wall_clearance)
        hi = env._arena_max_safe - float(env.cfg.pursuit_evader_wall_clearance)
        if not bool(torch.all((evader_pos >= lo) & (evader_pos <= hi)).item()):
            return False
        speed = torch.linalg.vector_norm(evader_vel, dim=-1)
        evader_speed_hi = max(0.0, float(max(getattr(env.cfg, "pursuit_evader_speed_range", (0.0, 0.0)))))
        speed_tolerance = max(1e-3, 1e-3 * evader_speed_hi)
        if bool(torch.any(speed > evader_speed_hi + speed_tolerance).item()):
            return False
        stats = self._evader_trajectory_stats(data)
        if self._evader_speed > 0.0 and stats["distance_ratio"] < 0.9:
            return False

        start_dist = torch.linalg.vector_norm(pursuer_start[:2] - evader_pos[0, :2])
        if float(start_dist.item()) < float(env.cfg.pursuit_pursuer_min_evader_distance):
            return False
        if not point_free(env.cfg, pursuer_start, static_xy, static_active, dynamic_pos, dynamic_active):
            return False

        static_count = int(static_active.to(torch.int32).sum().item())
        dynamic_count = int(dynamic_active.to(torch.int32).sum().item())
        if difficulty == "Easy" and (static_count < 1 or static_count > 2 or dynamic_count != 0):
            return False
        if difficulty == "Medium" and (static_count < 2 or static_count > 5 or dynamic_count != 0):
            return False
        total_count = static_count + dynamic_count
        if difficulty == "Hard" and (total_count < 4 or total_count > 6 or dynamic_count < 1 or dynamic_count > 3):
            return False
        return True

    def _make_static_slots(
        self,
        candidates: torch.Tensor,
        evader_pos: torch.Tensor,
        pursuer_start: torch.Tensor,
        desired_count: int,
        variant: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        env = self.base_env
        slots = int(getattr(env, "_max_static_obstacles", 0))
        if desired_count > slots:
            return None
        xy = env._inactive_obstacle_xy(slots).clone()
        active = torch.zeros(slots, dtype=torch.bool, device=env.device)
        if desired_count <= 0:
            return xy, active

        ordered = [candidate for candidate in candidates]
        ordered.extend(self._static_grid_candidates(variant))
        placed = 0
        for candidate in ordered:
            if placed >= desired_count:
                break
            if not self._static_candidate_ok(candidate, xy[:placed], evader_pos, pursuer_start):
                continue
            xy[placed] = candidate
            active[placed] = True
            placed += 1
        if placed < desired_count:
            return None
        return xy, active

    def _static_candidate_ok(
        self,
        candidate: torch.Tensor,
        placed: torch.Tensor,
        evader_pos: torch.Tensor,
        pursuer_start: torch.Tensor,
    ) -> bool:
        env = self.base_env
        lo, hi = env._safe_xy_bounds(float(env.cfg.pillar_radius) + float(env.cfg.pursuit_obstacle_clearance))
        if not bool(torch.all((candidate >= lo) & (candidate <= hi)).item()):
            return False
        safe_evader = (
            float(env.cfg.pillar_radius)
            + float(env.cfg.pursuit_evader_radius)
            + float(getattr(env.cfg, "pursuit_evader_tube_margin", 0.0))
        )
        d_evader = torch.linalg.vector_norm(evader_pos[:, :2] - candidate, dim=-1).min()
        if float(d_evader.item()) <= safe_evader:
            return False
        safe_start = (
            float(env.cfg.pillar_radius)
            + float(env.cfg.drone_collision_radius)
            + float(env.cfg.pursuit_obstacle_clearance)
        )
        d_start = torch.linalg.vector_norm(pursuer_start[:2] - candidate)
        if float(d_start.item()) <= safe_start:
            return False
        if placed.numel() > 0:
            d_static = torch.linalg.vector_norm(placed - candidate, dim=-1)
            safe_static = 2.0 * float(env.cfg.pillar_radius) + float(env.cfg.pursuit_obstacle_clearance)
            if bool(torch.any(d_static <= safe_static).item()):
                return False
        return True

    def _make_dynamic_slots(
        self,
        candidates: Sequence[torch.Tensor],
        evader_pos: torch.Tensor,
        static_xy: torch.Tensor,
        static_active: torch.Tensor,
        desired_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        env = self.base_env
        slots = int(getattr(env, "_max_dynamic_obstacles", 0))
        if desired_count > slots:
            return None
        pos = env._inactive_dynamic_pos(slots).view(1, slots, 3).repeat(env._path_steps, 1, 1)
        vel = torch.zeros_like(pos)
        active = torch.zeros(slots, dtype=torch.bool, device=env.device)
        if desired_count <= 0:
            return pos, vel, active

        placed = 0
        for path in candidates:
            if placed >= desired_count:
                break
            path_vel = self._path_velocity(path)
            if not self._dynamic_path_valid(
                path,
                evader_pos,
                static_xy,
                static_active,
                pos[:, :placed],
                active[:placed],
            ):
                continue
            pos[:, placed] = path
            vel[:, placed] = path_vel
            active[placed] = True
            placed += 1
        if placed < desired_count:
            return None
        return pos, vel, active

    def _static_seed_points(self, path_kind: str, evader_xy: torch.Tensor, pursuer_xy: torch.Tensor) -> torch.Tensor:
        midpoint = 0.52 * evader_xy[0] + 0.48 * pursuer_xy
        line = evader_xy[0] - pursuer_xy
        norm = torch.linalg.vector_norm(line).clamp_min(1e-6)
        perp = torch.stack((-line[1], line[0])) / norm
        if path_kind in {"straight", "waypoint", "circle"}:
            points = torch.stack(
                (
                    midpoint + 0.10 * perp,
                    evader_xy[min(evader_xy.shape[0] // 3, evader_xy.shape[0] - 1)] - 0.55 * perp,
                    evader_xy[min(evader_xy.shape[0] // 2, evader_xy.shape[0] - 1)] + 0.65 * perp,
                )
            )
            return points

        base = self._tensor_xy(
            [
                (-1.15, -0.15),
                (-0.75, 0.55),
                (-0.25, -0.75),
                (0.35, 0.65),
                (0.85, -0.45),
                (1.25, 0.25),
                (-1.45, -0.85),
                (1.45, 0.85),
                (0.0, 1.15),
                (0.0, -1.15),
            ]
        )
        return base

    def _static_grid_candidates(self, variant: int) -> list[torch.Tensor]:
        env = self.base_env
        lo, hi = env._safe_xy_bounds(float(env.cfg.pillar_radius) + float(env.cfg.pursuit_obstacle_clearance))
        xs = torch.linspace(float(lo[0]), float(hi[0]), 6, device=env.device)
        ys = torch.linspace(float(lo[1]), float(hi[1]), 5, device=env.device)
        grid = [torch.stack((x, y)) for x in xs for y in ys]
        if not grid:
            return []
        shift = variant % len(grid)
        return grid[shift:] + grid[:shift]

    def _dynamic_seed_paths(self, path_kind: str, variant: int) -> list[torch.Tensor]:
        if path_kind != "bottleneck":
            return []
        phase = 0.45 * (variant % 7)
        return [
            self._dynamic_sine_path((0.05, 0.05), (0.0, 1.0), amp=1.00, phase=phase + math.pi / 2.0),
            self._dynamic_sine_path((-0.55, -0.25), (1.0, 0.0), amp=0.95, phase=phase),
            self._dynamic_sine_path((0.65, 0.35), (1.0, -0.35), amp=0.90, phase=phase + math.pi),
            self._dynamic_sine_path((0.0, -0.75), (1.0, 0.0), amp=1.00, phase=phase + 0.5 * math.pi),
        ]

    def _dynamic_sine_path(
        self,
        center: tuple[float, float],
        direction: tuple[float, float],
        *,
        amp: float,
        phase: float,
    ) -> torch.Tensor:
        env = self.base_env
        t = torch.linspace(0.0, 1.0, env._path_steps, device=env.device)
        direction_t = self._tensor_xy([direction])[0]
        direction_t = direction_t / torch.linalg.vector_norm(direction_t).clamp_min(1e-6)
        center_t = self._tensor_xy([center])[0]
        xy = center_t + torch.sin(2.0 * math.pi * t + phase).view(-1, 1) * float(amp) * direction_t
        radius = float(env.cfg.pursuit_dynamic_obstacle_radius) + float(env.cfg.pursuit_obstacle_clearance)
        lo, hi = env._safe_xy_bounds(radius)
        xy = torch.clamp(xy, min=lo, max=hi)
        z = torch.full((env._path_steps, 1), env._dynamic_center_z(), device=env.device)
        return torch.cat((xy, z), dim=-1)

    def _polyline_xy(self, points: Sequence[tuple[float, float]]) -> torch.Tensor:
        route = self._tensor_xy(points)
        route = torch.cat((route, torch.flip(route[1:-1], dims=(0,)), route[:1]), dim=0)
        return self._sample_repeating_route(route)

    def _circle_xy(self, center: tuple[float, float], *, radius: float, phase: float) -> torch.Tensor:
        theta = torch.linspace(0.0, 2.0 * math.pi, 129, device=self.base_env.device) + float(phase)
        center_t = self._tensor_xy([center])[0]
        route = center_t + float(radius) * torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
        return self._sample_repeating_route(route)

    def _sample_repeating_route(self, route: torch.Tensor) -> torch.Tensor:
        env = self.base_env
        seg_len = torch.linalg.vector_norm(route[1:] - route[:-1], dim=-1).clamp_min(1e-6)
        cumulative = torch.cat((torch.zeros(1, device=env.device), torch.cumsum(seg_len, dim=0)))
        elapsed = torch.arange(env._path_steps, device=env.device, dtype=torch.float32) * float(env.step_dt)
        distance = torch.remainder(elapsed * self._evader_speed, cumulative[-1])
        segment = torch.searchsorted(cumulative[1:], distance).clamp(max=seg_len.shape[0] - 1)
        alpha = ((distance - cumulative[segment]) / seg_len[segment]).view(-1, 1)
        return route[segment] * (1.0 - alpha) + route[segment + 1] * alpha

    def _transform_xy(self, xy: torch.Tensor, variant: int, *, margin: float) -> torch.Tensor:
        out = xy.clone()
        if variant % 2:
            out[..., 0] = -out[..., 0]
        if (variant // 2) % 2:
            out[..., 1] = -out[..., 1]
        shift = torch.tensor(
            [0.12 * math.sin(1.37 * variant), 0.10 * math.cos(1.91 * variant)],
            device=self.base_env.device,
            dtype=torch.float32,
        )
        out = out + shift
        lo, hi = self.base_env._safe_xy_bounds(float(margin))
        return torch.clamp(out, min=lo, max=hi)

    def _finish_evader_xy_path(self, xy: torch.Tensor) -> torch.Tensor:
        env = self.base_env
        lo = env._arena_min_safe + float(env.cfg.pursuit_evader_wall_clearance)
        hi = env._arena_max_safe - float(env.cfg.pursuit_evader_wall_clearance)
        z = 0.5 * (float(lo[2]) + float(hi[2]))
        pos = torch.zeros(env._path_steps, 3, device=env.device)
        pos[:, :2] = xy
        pos[:, 2] = z
        return pos

    def _finish_point(self, xy: torch.Tensor) -> torch.Tensor:
        env = self.base_env
        lo = env._arena_min_safe + float(env.cfg.pursuit_pursuer_wall_clearance)
        hi = env._arena_max_safe - float(env.cfg.pursuit_pursuer_wall_clearance)
        z = 0.5 * (float(lo[2]) + float(hi[2]))
        return torch.tensor([float(xy[0]), float(xy[1]), z], device=env.device, dtype=torch.float32)

    def _dense_path_to_waypoints(self, path: torch.Tensor) -> torch.Tensor:
        steps = self.base_env._path_waypoint_steps.to(device=path.device)
        indices = torch.clamp(steps, max=path.shape[0] - 1)
        return path[indices]

    def _dense_dynamic_path_to_waypoints(self, path: torch.Tensor) -> torch.Tensor:
        steps = self.base_env._path_waypoint_steps.to(device=path.device)
        indices = torch.clamp(steps, max=path.shape[0] - 1)
        return path[indices]

    def _waypoints_to_dense(self, waypoints: torch.Tensor) -> torch.Tensor:
        env = self.base_env
        step_ids = torch.arange(env._path_steps, device=waypoints.device)
        waypoint_steps = env._path_waypoint_steps.to(device=waypoints.device)
        segment = torch.searchsorted(waypoint_steps[1:], step_ids).clamp(max=waypoints.shape[0] - 2)
        start_steps = waypoint_steps[segment]
        duration = (waypoint_steps[segment + 1] - start_steps).to(waypoints.dtype).clamp_min(1.0)
        alpha = ((step_ids - start_steps).to(waypoints.dtype) / duration).view(
            (-1,) + (1,) * (waypoints.ndim - 1)
        )
        return waypoints[segment] * (1.0 - alpha) + waypoints[segment + 1] * alpha

    def _path_velocity(self, path: torch.Tensor) -> torch.Tensor:
        velocity = torch.zeros_like(path)
        if path.shape[0] > 1:
            velocity[1:] = (path[1:] - path[:-1]) / float(self.base_env.step_dt)
            velocity[0] = velocity[1]
        return velocity

    def _dynamic_path_valid(
        self,
        path: torch.Tensor,
        evader_pos: torch.Tensor,
        static_xy: torch.Tensor,
        static_active: torch.Tensor,
        placed_paths: torch.Tensor,
        placed_active: torch.Tensor,
    ) -> bool:
        env = self.base_env
        radius = float(env.cfg.pursuit_dynamic_obstacle_radius)
        lo, hi = env._safe_xy_bounds(radius + float(env.cfg.pursuit_obstacle_clearance))
        if not bool(torch.all((path[:, :2] >= lo) & (path[:, :2] <= hi)).item()):
            return False

        safe_evader = radius + float(env.cfg.pursuit_evader_radius) + float(env.cfg.pursuit_obstacle_clearance)
        if bool(torch.any(torch.linalg.vector_norm(path[:, :2] - evader_pos[:, :2], dim=-1) <= safe_evader).item()):
            return False

        if bool(static_active.any().item()):
            distance = torch.linalg.vector_norm(
                path[:, None, :2] - static_xy[None, :, :],
                dim=-1,
            )
            safe_static = radius + float(env.cfg.pillar_radius) + float(env.cfg.pursuit_obstacle_clearance)
            if bool(torch.any((distance <= safe_static) & static_active.view(1, -1)).item()):
                return False

        if placed_paths.numel() > 0 and bool(placed_active.any().item()):
            distance = torch.linalg.vector_norm(path[:, None, :2] - placed_paths[:, :, :2], dim=-1)
            safe_dynamic = 2.0 * radius + float(env.cfg.pursuit_obstacle_clearance)
            if bool(torch.any((distance <= safe_dynamic) & placed_active.view(1, -1)).item()):
                return False
        return True

    def _tensor_xy(self, values: Sequence[tuple[float, float]]) -> torch.Tensor:
        return torch.tensor(values, device=self.base_env.device, dtype=torch.float32)


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
        safety_margin = _safety_margin(env, pos_local)
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
            "safety_margin": None if safety_margin is None else safety_margin.detach().clone(),
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
        safety_margin = sample["safety_margin"]
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
                safety_margin=None if safety_margin is None else float(safety_margin[env_id].item()),
            )

    def finish(self, env_id: int, done_reason: int) -> EpisodeResult | None:
        rollout = self.active[int(env_id)]
        if rollout is None:
            return None
        static_obstacle_xy, dynamic_obstacle_paths_xy = _scenario_obstacle_geometry(rollout.scenario)
        mapping = getattr(self.base_env, "DONE_REASON_MAP", {})
        done_label = str(mapping.get(int(done_reason), f"reason_{int(done_reason)}"))
        length = len(rollout.pos_errors)
        step_dt = float(getattr(self.base_env, "step_dt", getattr(self.base_env, "_step_dt", 0.0)))
        result = EpisodeResult(
            episode_id=len(self.results),
            env_id=int(env_id),
            scenario_kind=rollout.scenario.kind,
            scenario_label=rollout.scenario.label,
            path_kind=rollout.scenario.path_kind,
            static_obstacles=rollout.scenario.static_obstacles,
            dynamic_obstacles=rollout.scenario.dynamic_obstacles,
            target=rollout.scenario.goal,
            evader_start=rollout.scenario.evader_start,
            evader_end=rollout.scenario.evader_end,
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
            min_safety_margin=rollout.min_safety_margin,
            path_xy=rollout.positions.copy(),
            reference_xy=rollout.goals.copy(),
            errors=rollout.pos_errors.copy(),
            static_obstacle_xy=static_obstacle_xy,
            dynamic_obstacle_paths_xy=dynamic_obstacle_paths_xy,
        )
        self.results.append(result)
        self.active[int(env_id)] = None
        return result


def _scenario_obstacle_geometry(
    scenario: GoalScenario,
) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
    data = scenario.scenario_data
    if data is None:
        return [], []

    static_xy = data.get("static_xy")
    static_active = data.get("static_active")
    static_obstacles: list[tuple[float, float]] = []
    if isinstance(static_xy, torch.Tensor) and isinstance(static_active, torch.Tensor):
        for point in static_xy[static_active].detach().cpu().tolist():
            static_obstacles.append((float(point[0]), float(point[1])))

    dynamic_pos = data.get("dynamic_pos")
    dynamic_active = data.get("dynamic_active")
    dynamic_paths: list[list[tuple[float, float]]] = []
    if isinstance(dynamic_pos, torch.Tensor) and isinstance(dynamic_active, torch.Tensor):
        active_slots = torch.nonzero(dynamic_active).squeeze(-1).detach().cpu().tolist()
        positions = dynamic_pos.detach().cpu()
        for slot in active_slots:
            dynamic_paths.append(
                [(float(point[0]), float(point[1])) for point in positions[:, int(slot), :2].tolist()]
            )
    return static_obstacles, dynamic_paths


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


def _ordered_result_groups(results: Sequence[EpisodeResult]) -> list[tuple[str, list[EpisodeResult]]]:
    labels = list(DIFFICULTY_ORDER)
    labels.extend(sorted({result.scenario_kind for result in results if result.scenario_kind not in labels}))
    return [(label, [result for result in results if result.scenario_kind == label]) for label in labels]


def _finite_values(values: Sequence[float | None]) -> list[float]:
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            out.append(value)
    return out


def _performance_summary_rows(results: Sequence[EpisodeResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for difficulty, group in _ordered_result_groups(results):
        if not group:
            continue
        captures = sum(1 for result in group if result.success)
        collisions = sum(1 for result in group if result.collided)
        capture_times = [float(result.duration_s) for result in group if result.success]
        margins = _finite_values([result.min_safety_margin for result in group])
        rows.append(
            {
                "difficulty": difficulty,
                "episodes": len(group),
                "capture_rate_percent": 100.0 * captures / max(1, len(group)),
                "collision_rate_percent": 100.0 * collisions / max(1, len(group)),
                "time_to_capture_s": _safe_list_mean(capture_times),
                "min_margin_m": _safe_list_mean(margins),
            }
        )
    return rows


def _pillar_clearance(base_env: Any, pos_local: torch.Tensor) -> torch.Tensor | None:
    pillars = getattr(base_env, "_static_obstacle_positions_xy", None)
    active = getattr(base_env, "_static_obstacle_active", None)
    if pillars is None or active is None or pillars.numel() == 0:
        return None
    dxy = torch.linalg.vector_norm(pos_local[:, None, :2] - pillars, dim=-1)
    clearance = dxy - float(getattr(base_env, "_pillar_collision_radius", 0.0))
    clearance = torch.where(active, clearance, torch.full_like(clearance, float("inf")))
    if not bool(torch.isfinite(clearance).any().item()):
        return None
    return clearance.min(dim=1).values


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


def _safety_margin(base_env: Any, pos_local: torch.Tensor) -> torch.Tensor | None:
    if not hasattr(base_env, "_arena_min_safe") or not hasattr(base_env, "_arena_max_safe"):
        return None

    lower = pos_local - base_env._arena_min_safe.view(1, 3)
    upper = base_env._arena_max_safe.view(1, 3) - pos_local
    margin = torch.cat((lower, upper), dim=-1).min(dim=1).values

    static_xy = getattr(base_env, "_static_obstacle_positions_xy", None)
    static_active = getattr(base_env, "_static_obstacle_active", None)
    if static_xy is not None and static_active is not None and static_xy.numel() > 0:
        dxy = torch.linalg.vector_norm(pos_local[:, None, :2] - static_xy, dim=-1)
        static_margin = dxy - float(getattr(base_env, "_pillar_collision_radius", 0.0))
        static_margin = torch.where(static_active, static_margin, torch.full_like(static_margin, float("inf")))
        static_min = static_margin.min(dim=1).values
        margin = torch.minimum(margin, static_min)

    dynamic_pos = getattr(base_env, "_dynamic_obstacle_positions", None)
    dynamic_active = getattr(base_env, "_dynamic_obstacle_active", None)
    if dynamic_pos is not None and dynamic_active is not None and dynamic_pos.numel() > 0:
        dxy = torch.linalg.vector_norm(pos_local[:, None, :2] - dynamic_pos[:, :, :2], dim=-1)
        dynamic_margin = dxy - float(getattr(base_env, "_dynamic_collision_radius", 0.0))
        dynamic_margin = torch.where(dynamic_active, dynamic_margin, torch.full_like(dynamic_margin, float("inf")))
        dynamic_min = dynamic_margin.min(dim=1).values
        margin = torch.minimum(margin, dynamic_min)

    return margin


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
    if args_cli.benchmark_profile in {"pursuit", "random"}:
        return []
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

    repeats = max(0, int(args_cli.fixed_goal_repeats))
    return [goal for goal in goals for _ in range(repeats)]


def _target_episode_count(fixed_goals: Sequence[tuple[float, float, float]]) -> int | None:
    if args_cli.benchmark_profile == "pursuit":
        planned = max(0, int(args_cli.tests_per_difficulty)) * len(DIFFICULTY_ORDER)
        if args_cli.num_episodes is not None:
            return min(int(args_cli.num_episodes), planned)
        return planned
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
    min_margins = _finite_values([result.min_safety_margin for result in results])
    time_to_target = [float(result.duration_s) for result in results if result.success]
    safety_terminated = sum(1 for result in results if result.safety_terminated)
    success = sum(1 for result in results if result.success)
    collisions = sum(1 for result in results if result.collided)

    by_scenario: dict[str, dict[str, Any]] = {}
    for kind, group in _ordered_result_groups(results):
        if not group:
            continue
        group_final = [float(result.final_pos_error) for result in group if result.final_pos_error is not None]
        group_time_to_target = [float(result.duration_s) for result in group if result.success]
        group_margins = _finite_values([result.min_safety_margin for result in group])
        group_collisions = sum(1 for result in group if result.collided)
        by_scenario[kind] = {
            "episodes": len(group),
            "success_percent": 100.0 * sum(1 for result in group if result.success) / max(1, len(group)),
            "collision_percent": 100.0 * group_collisions / max(1, len(group)),
            "safety_termination_percent": (
                100.0 * sum(1 for result in group if result.safety_terminated) / max(1, len(group))
            ),
            "time_to_target_s": _distribution_stats(group_time_to_target),
            "final_position_error_m": _distribution_stats(group_final),
            "min_safety_margin_m": _distribution_stats(group_margins),
        }

    return {
        "episodes": total,
        "success_percent": 100.0 * success / max(1, total),
        "collision_percent": 100.0 * collisions / max(1, total),
        "safety_termination_percent": 100.0 * safety_terminated / max(1, total),
        "termination_counts": reason_counts,
        "time_to_target_s": _distribution_stats(time_to_target),
        "final_position_error_m": _distribution_stats(final_errors),
        "min_safety_margin_m": _distribution_stats(min_margins),
        "performance_table": _performance_summary_rows(results),
        "by_scenario": by_scenario,
    }


def _format_optional_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.1f} s"


def _format_optional_meters(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.2f} m"


def _print_performance_table(summary: Mapping[str, Any]) -> None:
    rows = summary.get("performance_table", [])
    if not rows:
        return
    print("")
    print(f"Algorithm: {args_cli.algorithm.upper()}")
    print(f"Seed: {args_cli.seed}")
    if args_cli.benchmark_profile == "pursuit":
        print(f"Number of scenarios per difficulty: {int(args_cli.tests_per_difficulty)}")
    print("")
    print(f"{'Difficulty':<12} {'Capture Rate':<14} {'Collision Rate':<15} {'Time to Capture':<17} {'Min Margin':<10}")
    for row in rows:
        capture = f"{float(row['capture_rate_percent']):.1f}%"
        collision = f"{float(row['collision_rate_percent']):.1f}%"
        print(
            f"{str(row['difficulty']):<12} "
            f"{capture:<14} "
            f"{collision:<15} "
            f"{_format_optional_seconds(row.get('time_to_capture_s')):<17} "
            f"{_format_optional_meters(row.get('min_margin_m')):<10}"
        )
    print("")


def _plot_pursuit_trajectory_grids(
    output_dir: Path,
    env_cfg: Any,
    results: Sequence[EpisodeResult],
    plt: Any,
    Circle: Any,
    Rectangle: Any,
) -> dict[str, str]:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    paths: dict[str, str] = {}
    arena_min = env_cfg.arena_min
    arena_max = env_cfg.arena_max
    static_radius = float(getattr(env_cfg, "pillar_radius", 0.15))
    dynamic_radius = float(getattr(env_cfg, "pursuit_dynamic_obstacle_radius", static_radius))

    for difficulty in DIFFICULTY_ORDER:
        stage_results = [result for result in results if result.scenario_kind == difficulty]
        if not stage_results:
            continue

        panel_count = max(30, len(stage_results))
        cols = 5
        rows = math.ceil(panel_count / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(18, 3.45 * rows), squeeze=False)

        for panel_idx, ax in enumerate(axes.flat):
            if panel_idx >= len(stage_results):
                ax.axis("off")
                continue

            result = stage_results[panel_idx]
            ax.add_patch(
                Rectangle(
                    (arena_min[0], arena_min[1]),
                    arena_max[0] - arena_min[0],
                    arena_max[1] - arena_min[1],
                    fill=False,
                    lw=0.9,
                    ls="--",
                    ec="#555555",
                    zorder=1,
                )
            )
            for point in result.static_obstacle_xy:
                ax.add_patch(
                    Circle(
                        point,
                        radius=static_radius,
                        facecolor="#737373",
                        edgecolor="#303030",
                        linewidth=0.6,
                        alpha=0.65,
                        zorder=3,
                    )
                )
            for dynamic_path in result.dynamic_obstacle_paths_xy:
                if not dynamic_path:
                    continue
                dynamic_x = [point[0] for point in dynamic_path]
                dynamic_y = [point[1] for point in dynamic_path]
                ax.plot(dynamic_x, dynamic_y, color="#9c4f4f", lw=0.8, ls=":", alpha=0.75, zorder=2)
                ax.add_patch(
                    Circle(
                        dynamic_path[0],
                        radius=dynamic_radius,
                        facecolor="#c76b6b",
                        edgecolor="#703838",
                        linewidth=0.6,
                        alpha=0.65,
                        zorder=4,
                    )
                )

            if result.reference_xy:
                ref_x = [point[0] for point in result.reference_xy]
                ref_y = [point[1] for point in result.reference_xy]
                ax.plot(ref_x, ref_y, color="#e68624", lw=1.25, ls="--", alpha=0.9, zorder=5)
                ax.scatter(
                    ref_x[0], ref_y[0], marker="*", s=45, color="#e68624",
                    edgecolors="black", linewidths=0.4, zorder=8,
                )
                ax.scatter(
                    ref_x[-1], ref_y[-1], marker="X", s=25, color="#e68624",
                    edgecolors="black", linewidths=0.35, zorder=8,
                )

            if result.path_xy:
                path_x = [point[0] for point in result.path_xy]
                path_y = [point[1] for point in result.path_xy]
                ax.plot(path_x, path_y, color="#1769aa", lw=1.5, alpha=0.95, zorder=6)
                ax.scatter(
                    path_x[0], path_y[0], marker="o", s=24, color="#1769aa",
                    edgecolors="white", linewidths=0.6, zorder=9,
                )
                ax.scatter(path_x[-1], path_y[-1], marker="x", s=30, color="#0b3558", linewidths=1.4, zorder=9)
                progress_idx = np.linspace(0, len(path_x) - 1, 6, dtype=int)[1:-1]
                ax.scatter(
                    np.asarray(path_x)[progress_idx],
                    np.asarray(path_y)[progress_idx],
                    s=8,
                    color="#74add1",
                    edgecolors="none",
                    zorder=7,
                )

            if result.success:
                status_color = "#24733f"
            elif result.collided:
                status_color = "#a12d2d"
            else:
                status_color = "#8a6518"
            final_error = "n/a" if result.final_pos_error is None else f"{result.final_pos_error:.2f} m"
            ax.set_title(
                f"{result.scenario_label} | {result.path_kind}\n"
                f"{result.done_label}, {result.duration_s:.1f} s | final err {final_error}",
                fontsize=8,
                color=status_color,
                pad=3,
            )
            ax.set_xlim(float(arena_min[0]) - 0.08, float(arena_max[0]) + 0.08)
            ax.set_ylim(float(arena_min[1]) - 0.08, float(arena_max[1]) + 0.08)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, color="#d9d9d9", linewidth=0.45, alpha=0.75)
            ax.tick_params(labelsize=6, length=2)
            ax.set_xlabel("x [m]", fontsize=7)
            ax.set_ylabel("y [m]", fontsize=7)

        legend_handles = [
            Line2D([0], [0], color="#1769aa", lw=1.7, marker="o", markersize=4, label="Drone trajectory"),
            Line2D([0], [0], color="#e68624", lw=1.4, ls="--", marker="*", markersize=7, label="Evader trajectory"),
            Patch(facecolor="#737373", edgecolor="#303030", alpha=0.65, label="Static obstacle"),
            Line2D(
                [0], [0], color="#9c4f4f", lw=1.0, ls=":", marker="o",
                markersize=5, label="Dynamic obstacle path/start",
            ),
            Line2D([0], [0], color="#74add1", lw=0, marker="o", markersize=3, label="Drone progress (20%)"),
        ]
        fig.suptitle(
            f"{difficulty} stage trajectories ({len(stage_results)} scenarios)",
            fontsize=15,
            fontweight="bold",
            y=0.995,
        )
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.977),
            ncols=5,
            frameon=False,
            fontsize=9,
        )
        fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.945), h_pad=1.1, w_pad=0.8)
        stage_path = output_dir / f"trajectories_{difficulty.lower()}.png"
        fig.savefig(stage_path, dpi=180, facecolor="white")
        plt.close(fig)
        paths[f"trajectories_{difficulty.lower()}"] = str(stage_path)

    return paths


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

    if args_cli.benchmark_profile == "pursuit":
        paths.update(_plot_pursuit_trajectory_grids(output_dir, env_cfg, results, plt, Circle, Rectangle))
    else:
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

        cmap = plt.get_cmap("tab20", max(1, len(results)))
        for idx, result in enumerate(results):
            if len(result.path_xy) < 2:
                continue
            xs = [p[0] for p in result.path_xy]
            ys = [p[1] for p in result.path_xy]
            color = cmap(idx)
            ax.plot(xs, ys, color=color, lw=1.5, alpha=0.9, label=result.scenario_label if idx < 12 else None)
            ax.scatter(xs[0], ys[0], color=color, marker="o", s=18)
            ax.scatter(xs[-1], ys[-1], color=color, marker="x", s=30)
            if len(result.reference_xy) >= 2:
                ref_xs = [p[0] for p in result.reference_xy]
                ref_ys = [p[1] for p in result.reference_xy]
                ax.plot(ref_xs, ref_ys, color=color, lw=1.0, alpha=0.45, ls="--")
                ax.scatter(ref_xs[0], ref_ys[0], color=color, marker="*", s=110, edgecolors="black", zorder=30)
            else:
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

    grouped = [(label, group) for label, group in _ordered_result_groups(results) if group]
    labels = [label for label, _group in grouped]
    if labels:
        groups = [group for _label, group in grouped]
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

        def draw_boxplot(ax, data):
            try:
                return ax.boxplot(data, tick_labels=labels, **box_style)
            except TypeError:
                return ax.boxplot(data, labels=labels, **box_style)

        draw_boxplot(axes[0], [data if data else [np.nan] for data in time_data])
        axes[0].set_title("Time to target")
        axes[0].set_ylabel("seconds")

        draw_boxplot(axes[1], [data if data else [np.nan] for data in final_error_data])
        axes[1].set_title("Final position error")
        axes[1].set_ylabel("meters")

        draw_boxplot(axes[2], success_data)
        axes[2].set_title("Success")
        axes[2].set_ylabel("0/1")
        axes[2].set_ylim(-0.05, 1.05)

        total = max(1, len(results))
        collision_pct = 100.0 * sum(1 for result in results if result.collided) / total
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

        rows = _performance_summary_rows(results)
        if rows:
            labels = [str(row["difficulty"]) for row in rows]
            capture = [float(row["capture_rate_percent"]) for row in rows]
            collision = [float(row["collision_rate_percent"]) for row in rows]
            colors = ["#2c7fb8", "#fdae61", "#d7191c", "#7b3294", "#008837"][: len(labels)]

            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            ax.bar(labels, capture, color=colors)
            ax.set_title("Capture rate")
            ax.set_ylabel("episodes captured [%]")
            ax.set_ylim(0.0, 100.0)
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            capture_path = output_dir / "capture_rate_by_difficulty.png"
            fig.savefig(capture_path, dpi=180)
            plt.close(fig)
            paths["capture_rate_bar"] = str(capture_path)

            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            ax.bar(labels, collision, color=colors)
            ax.set_title("Collision rate")
            ax.set_ylabel("episodes with collision [%]")
            ax.set_ylim(0.0, 100.0)
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            collision_path = output_dir / "collision_rate_by_difficulty.png"
            fig.savefig(collision_path, dpi=180)
            plt.close(fig)
            paths["collision_rate_bar"] = str(collision_path)

            fig, ax = plt.subplots(figsize=(5.4, 4.8))
            for label, x, y, color in zip(labels, collision, capture, colors):
                ax.scatter(x, y, s=90, color=color, edgecolors="black", linewidths=0.7)
                ax.annotate(label, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=9)
            ax.set_title("Safety-performance")
            ax.set_xlabel("collision rate [%]")
            ax.set_ylabel("capture rate [%]")
            ax.set_xlim(-2.0, 102.0)
            ax.set_ylim(-2.0, 102.0)
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            scatter_path = output_dir / "safety_performance_scatter.png"
            fig.savefig(scatter_path, dpi=180)
            plt.close(fig)
            paths["safety_performance_scatter"] = str(scatter_path)

            margin_data = [
                _finite_values([result.min_safety_margin for result in group]) for _label, group in grouped
            ]
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            draw_boxplot(ax, [data if data else [np.nan] for data in margin_data])
            ax.axhline(0.0, color="#9e2a2b", lw=1.0, ls="--")
            ax.set_title("Minimum safety margin")
            ax.set_ylabel("meters")
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            margin_path = output_dir / "minimum_safety_margin_boxplot.png"
            fig.savefig(margin_path, dpi=180)
            plt.close(fig)
            paths["minimum_safety_margin_boxplot"] = str(margin_path)

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
    if checkpoint is None and args_cli.artifact:
        checkpoint = _download_wandb_artifact(args_cli.artifact, args_cli.artifact_file)
    if checkpoint is not None:
        checkpoint_path = _resolve_checkpoint_path(checkpoint)
        if checkpoint_path is None:
            raise ValueError("Could not resolve checkpoint path.")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        checkpoint = str(checkpoint_path)

    ppo_provenance = None
    if args_cli.policy_mode == "rl" and args_cli.algorithm == "ppo" and checkpoint is not None:
        agent_cfg_path, env_cfg_path = _require_ppo_training_context(checkpoint)
        ppo_provenance = _ppo_checkpoint_provenance(checkpoint, agent_cfg_path, env_cfg_path)
    trained_agent_cfg = _agent_cfg_for_checkpoint(agent_cfg, checkpoint)
    trained_env_cfg = _env_cfg_for_checkpoint(checkpoint)
    _validate_checkpoint_algorithm(trained_agent_cfg, checkpoint)
    _validate_task_env_contract(trained_env_cfg)
    saved_seed = trained_env_cfg.get("seed", trained_agent_cfg.get("seed"))
    training_seed = int(saved_seed) if saved_seed is not None else None
    if args_cli.seed is None and training_seed is not None:
        args_cli.seed = training_seed
        print(f"[INFO] Using training seed for benchmark: {args_cli.seed}")

    env_cfg.scene.num_envs = int(args_cli.num_envs or env_cfg.scene.num_envs)
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    _apply_env_overrides_from_agent_cfg(env_cfg, trained_agent_cfg)
    _apply_env_overrides_from_training_env_cfg(env_cfg, trained_env_cfg)
    env_cfg.domain_randomization.enable = False
    training_evader_speed_range = tuple(getattr(env_cfg, "pursuit_evader_speed_range", (0.6, 1.0)))
    benchmark_evader_speed = None
    if args_cli.benchmark_profile == "pursuit":
        env_cfg.enable_pursuit_evasion_curriculum = True
        env_cfg.enable_walls = True
        env_cfg.enable_pillars = True
        env_cfg.pursuit_max_static_obstacles = max(5, int(getattr(env_cfg, "pursuit_max_static_obstacles", 5)))
        env_cfg.pursuit_max_dynamic_obstacles = max(3, int(getattr(env_cfg, "pursuit_max_dynamic_obstacles", 3)))
        env_cfg.pursuit_scenario_attempts = max(300, int(getattr(env_cfg, "pursuit_scenario_attempts", 300)))
        env_cfg.ref_update_interval_s = 0.0
        training_evader_speed_hi = max(0.0, float(max(training_evader_speed_range)))
        benchmark_evader_speed = (
            training_evader_speed_hi if args_cli.evader_speed is None else float(args_cli.evader_speed)
        )
        if benchmark_evader_speed < 0.0:
            raise ValueError("--evader-speed must be non-negative.")
        if benchmark_evader_speed > training_evader_speed_hi + 1e-6:
            raise ValueError(
                f"--evader-speed={benchmark_evader_speed:g} exceeds the training maximum "
                f"of {training_evader_speed_hi:g} m/s."
            )
        print(
            "[INFO] Pursuit evader speed: "
            f"{benchmark_evader_speed:g} m/s "
            f"(training range {min(training_evader_speed_range):g}-{max(training_evader_speed_range):g} m/s)."
        )
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
    if ppo_provenance is not None:
        _validate_ppo_runtime_contract(base_env, trained_agent_cfg, ppo_provenance)
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
        elif checkpoint:
            policy = PolicyRunner.from_checkpoint(
                checkpoint,
                args_cli.actor_cfg,
                base_env=base_env,
                agent_cfg_data=trained_agent_cfg,
                device=str(device),
                allow_observation_adapter=args_cli.allow_observation_adapter,
            )
        else:
            print("[WARN] No checkpoint provided; falling back to random actions.")
            args_cli.policy_mode = "random"

    print("[INFO] Resetting benchmark environment.", flush=True)
    obs, _ = env.reset()
    print("[INFO] Environment reset complete.", flush=True)
    action_dim = _action_dim_from_env(base_env)

    scenario_manager = ScenarioManager(
        base_env,
        fixed_goals=fixed_goals,
        pursuit=args_cli.benchmark_profile == "pursuit",
        tests_per_difficulty=int(args_cli.tests_per_difficulty),
        evader_speed=benchmark_evader_speed,
    )
    all_env_ids = list(range(int(base_env.num_envs)))
    assigned_env_ids = scenario_manager.assign(all_env_ids)
    obs = _refresh_observations_after_reference_write(base_env, obs)
    recorder = RolloutRecorder(base_env, scenario_manager)
    recorder.start(assigned_env_ids)
    if args_cli.benchmark_profile == "pursuit":
        print(
            "[INFO] Benchmark schedule: "
            f"{int(args_cli.tests_per_difficulty)} scenarios per difficulty, "
            f"profile={args_cli.benchmark_profile}, target_episodes={target_episodes}, "
            f"terminate_on_safety_violation={base_env.cfg.terminate_on_safety_violation}, "
            f"terminate_on_success={base_env.cfg.terminate_on_success}"
        )
    else:
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
                    assigned_env_ids = scenario_manager.assign([env_id])
                    obs = _refresh_observations_after_reference_write(base_env, obs)
                    recorder.start(assigned_env_ids)
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
    _print_performance_table(summary)
    trajectory_summary = scenario_manager.trajectory_summary()
    if trajectory_summary is not None:
        print(
            "[INFO] Evader trajectory validation: "
            f"speed={trajectory_summary['requested_speed_mps']:.3f} m/s, "
            f"min_distance={trajectory_summary['min_executed_distance_m']:.3f} m, "
            f"min_distance_ratio={trajectory_summary['min_distance_ratio']:.3f}, "
            f"max_speed={trajectory_summary['max_executed_speed_mps']:.3f} m/s, "
            f"fallbacks={trajectory_summary['fallback_count']}."
        )

    metrics = {
        "task": args_cli.task,
        "algorithm": args_cli.algorithm,
        "control_mode": base_env.cfg.control_mode,
        "policy_mode": args_cli.policy_mode,
        "checkpoint": checkpoint,
        "ppo_training_provenance": ppo_provenance,
        "training_seed": training_seed,
        "benchmark_seed": args_cli.seed,
        "effective_benchmark_contract": {
            "observation_dim": _obs_dim_from_env(base_env),
            "action_dim": _action_dim_from_env(base_env),
            "episode_length_s": float(base_env.cfg.episode_length_s),
            "sim_dt": float(base_env.sim.cfg.dt),
            "step_dt": float(base_env.step_dt),
            "decimation": int(base_env.cfg.decimation),
            "control_mode": str(base_env.cfg.control_mode),
            "obstacle_observation_mode": str(base_env.cfg.obstacle_observation_mode),
            "ray_caster_observation_mode": str(base_env.cfg.ray_caster_observation_mode),
            "ray_caster_observation_data": str(base_env.cfg.ray_caster_observation_data),
            "ray_caster_num_rays": int(base_env.cfg.ray_caster_num_rays),
            "pursuit_evader_speed_range": list(base_env.cfg.pursuit_evader_speed_range),
            "benchmark_evader_speed": benchmark_evader_speed,
            "pursuit_max_static_obstacles": int(base_env.cfg.pursuit_max_static_obstacles),
            "pursuit_max_dynamic_obstacles": int(base_env.cfg.pursuit_max_dynamic_obstacles),
            "domain_randomization": bool(base_env.cfg.domain_randomization.enable),
            "terminate_on_safety_violation": bool(base_env.cfg.terminate_on_safety_violation),
            "terminate_on_success": bool(base_env.cfg.terminate_on_success),
            "terminate_on_out_of_boundaries": bool(base_env.cfg.terminate_on_out_of_boundaries),
        },
        "benchmark_profile": args_cli.benchmark_profile,
        "tests_per_difficulty": int(args_cli.tests_per_difficulty)
        if args_cli.benchmark_profile == "pursuit"
        else None,
        "num_steps": total_steps,
        "target_episodes": target_episodes,
        "fixed_goals": [list(goal) for goal in fixed_goals],
        "training_pursuit_evader_speed_range": list(training_evader_speed_range),
        "benchmark_evader_speed": benchmark_evader_speed,
        "evader_trajectory_validation": trajectory_summary,
        "pursuit_evader_speed_range": list(getattr(base_env.cfg, "pursuit_evader_speed_range", (0.0, 0.0))),
        "pursuit_dynamic_max_speed": float(getattr(base_env.cfg, "pursuit_dynamic_max_speed", 0.0)),
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
