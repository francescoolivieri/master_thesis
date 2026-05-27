#!/usr/bin/env python3
"""Plot DG-PPO critic value heatmaps over a fixed XY state slice.

Example commands
----------------
Default run from a skrl training folder (uses latest ``agent_*.pt`` found under
``checkpoints/`` and auto-loads ``params/agent.yaml``):

    cd /Midgard/home/fraoli/master_thesis
    ../run_isaac_sim.sh python scripts/benchmark/dgppo_critic_heatmap.py \
      --checkpoint logs/skrl/training/dgppo/<run_folder>

Run a specific checkpoint file:

    cd /Midgard/home/fraoli/master_thesis
    ../run_isaac_sim.sh python scripts/benchmark/dgppo_critic_heatmap.py \
      --checkpoint logs/skrl/training/dgppo/<run_folder>/checkpoints/agent_100000.pt

Force old runs to use top-k ray observations in the heatmap env:

    cd /Midgard/home/fraoli/master_thesis
    ../run_isaac_sim.sh python scripts/benchmark/dgppo_critic_heatmap.py \
      --checkpoint logs/skrl/training/dgppo/<run_folder> \
      --ray-caster-observation-mode top_k_hits \
      --ray-caster-top-k-hits 8

Plot a single Vh safety head instead of the default reduced Vh view:

    cd /Midgard/home/fraoli/master_thesis
    ../run_isaac_sim.sh python scripts/benchmark/dgppo_critic_heatmap.py \
      --checkpoint logs/skrl/training/dgppo/<run_folder> \
      --vh-reduction head \
      --vh-head 1
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate trained DG-PPO critics on an XY state grid.")
parser.add_argument("--task", type=str, default="PosTracking-v0", help="Gym task used by the checkpoint.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to a DG-PPO checkpoint (.pt), a checkpoints/ directory, or a skrl run directory.",
)
parser.add_argument("--artifact", type=str, default=None, help="WandB artifact containing the DG-PPO checkpoint.")
parser.add_argument("--artifact-file", type=str, default=None, help="Specific file inside the WandB artifact.")
parser.add_argument(
    "--trained-agent-cfg",
    type=Path,
    default=None,
    help="Agent YAML used to train the checkpoint. Defaults to params/agent.yaml next to the checkpoint.",
)
parser.add_argument("--num-envs", type=int, default=1, help="Number of IsaacLab envs to instantiate for metadata.")
parser.add_argument("--grid-size", type=int, default=121, help="Number of grid samples per XY axis.")
parser.add_argument("--batch-size", type=int, default=4096, help="Number of grid states evaluated per critic batch.")
parser.add_argument("--x-min", type=float, default=None, help="Minimum x of the heatmap. Defaults to arena_min[0].")
parser.add_argument("--x-max", type=float, default=None, help="Maximum x of the heatmap. Defaults to arena_max[0].")
parser.add_argument("--y-min", type=float, default=None, help="Minimum y of the heatmap. Defaults to arena_min[1].")
parser.add_argument("--y-max", type=float, default=None, help="Maximum y of the heatmap. Defaults to arena_max[1].")
parser.add_argument("--z", type=float, default=None, help="Fixed altitude. Defaults to the midpoint of ref_pos limits.")
parser.add_argument(
    "--goal",
    type=str,
    default=None,
    help="Fixed goal as x,y or x,y,z. Defaults to the midpoint of ref_pos limits.",
)
parser.add_argument("--velocity", type=str, default="0,0,0", help="Fixed linear velocity as vx,vy,vz.")
parser.add_argument("--yaw", type=float, default=0.0, help="Fixed yaw angle in radians when yaw observations are enabled.")
parser.add_argument(
    "--obstacle-source",
    choices=["analytic", "none"],
    default="analytic",
    help="How to synthesize obstacle observations for the grid.",
)
parser.add_argument(
    "--ray-caster-observation-mode",
    choices=["ray_ordered_hits", "top_k_hits"],
    default=None,
    help="Override env ray-caster observation mode for older runs before building the critic input layout.",
)
parser.add_argument(
    "--ray-caster-top-k-hits",
    type=int,
    default=None,
    help="Override env ray_caster_top_k_hits. Useful together with --ray-caster-observation-mode=top_k_hits.",
)
parser.add_argument(
    "--vh-reduction",
    choices=["max", "mean", "head"],
    default="max",
    help="How to reduce multi-head Vh values for the main Vh heatmap.",
)
parser.add_argument("--vh-head", type=int, default=0, help="Vh head index used when --vh-reduction=head.")
parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap name.")
parser.add_argument(
    "--color-percentiles",
    type=str,
    default="1,99",
    help="Displayed color limits as low,high percentiles. Use --full-color-range to disable.",
)
parser.add_argument(
    "--full-color-range",
    action="store_true",
    help="Use each heatmap's full min/max range instead of percentile-clipped color limits.",
)
parser.add_argument(
    "--contours",
    type=int,
    default=12,
    help="Number of faint contour levels overlaid on each heatmap. Set 0 to disable.",
)
parser.add_argument("--log-dir", type=Path, default=Path("logs/pos_tracking/critic_heatmap"), help="Output directory.")
parser.add_argument("--exp-id", type=str, default=None, help="Optional experiment id appended to --log-dir.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment metadata instance.")

AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)

args_cli, hydra_args = parser.parse_known_args()

# hydra_task_config requires args to be passed via sys.argv
import sys

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Imports requiring Isaac/Kit initialization
# -----------------------------------------------------------------------------
import gymnasium as gym
import numpy as np
import torch
from isaaclab_tasks.utils.hydra import hydra_task_config

from source.isaac_pursuit_evasion.dgppo.dgppo_config import DGPPOAgentCfg
from source.isaac_pursuit_evasion.dgppo.dgppo_models import DGPPOValueNet
from source.isaac_pursuit_evasion.dgppo.utils import (
    NUM_TYPE_INDICATORS,
    build_graph_data,
)

# Ensure tasks are registered with Gym.
import source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pos_tracking  # noqa: F401


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _as_int_tuple(values: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if values is None:
        return default
    if isinstance(values, int):
        return (int(values),)
    return tuple(int(v) for v in values)


def _parse_vector(raw: str, *, length: int, name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in raw.split(","))
    except ValueError as exc:
        raise ValueError(f"Invalid --{name} '{raw}': expected comma-separated floats.") from exc
    if len(values) != length:
        raise ValueError(f"Invalid --{name} '{raw}': expected {length} values.")
    return values


def _parse_goal(raw: str | None, env_cfg: Any) -> tuple[float, float, float]:
    default = tuple(
        float((lo + hi) * 0.5)
        for lo, hi in zip(getattr(env_cfg, "ref_pos_min"), getattr(env_cfg, "ref_pos_max"))
    )
    if raw is None:
        return default
    try:
        values = [float(part.strip()) for part in raw.split(",")]
    except ValueError as exc:
        raise ValueError(f"Invalid --goal '{raw}': expected x,y or x,y,z.") from exc
    if len(values) == 2:
        values.append(default[2])
    if len(values) != 3:
        raise ValueError(f"Invalid --goal '{raw}': expected x,y or x,y,z.")
    return (values[0], values[1], values[2])


def _color_percentiles() -> tuple[float, float]:
    values = _parse_vector(args_cli.color_percentiles, length=2, name="color-percentiles")
    low, high = values
    if not (0.0 <= low < high <= 100.0):
        raise ValueError("--color-percentiles must satisfy 0 <= low < high <= 100.")
    return float(low), float(high)


def _apply_env_overrides_from_agent_cfg(env_cfg: Any, agent_cfg: Any) -> None:
    if not isinstance(agent_cfg, Mapping):
        return
    env_overrides = agent_cfg.get("env", agent_cfg.get("environment", None))
    if not isinstance(env_overrides, Mapping):
        return
    for key, value in env_overrides.items():
        if str(key).startswith("_"):
            continue
        if not hasattr(env_cfg, key):
            print(f"[WARN] Ignoring agent env override '{key}': env config has no such attribute.")
            continue
        setattr(env_cfg, key, value)


def _apply_cli_env_overrides(env_cfg: Any) -> None:
    if args_cli.ray_caster_observation_mode is not None:
        env_cfg.ray_caster_observation_mode = str(args_cli.ray_caster_observation_mode)
    if args_cli.ray_caster_top_k_hits is not None:
        value = int(args_cli.ray_caster_top_k_hits)
        if value < 0:
            raise ValueError("--ray-caster-top-k-hits must be non-negative.")
        env_cfg.ray_caster_top_k_hits = value


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise ImportError("PyYAML is required to parse agent YAML configs.") from exc
    with path.expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
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
    candidate = Path(path).expanduser()
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


def _checkpoint_agent_yaml(checkpoint: str | Path | None) -> Path | None:
    checkpoint_path = _resolve_checkpoint_path(checkpoint)
    if checkpoint_path is None:
        return None
    candidate = checkpoint_path.parent.parent / "params" / "agent.yaml"
    return candidate if candidate.exists() else None


def _agent_cfg_for_checkpoint(default_cfg: Mapping[str, Any], checkpoint: str | Path | None) -> dict[str, Any]:
    if args_cli.trained_agent_cfg is not None:
        cfg_path = args_cli.trained_agent_cfg.expanduser()
        print(f"[INFO] Loading trained agent config: {cfg_path}")
        return _load_yaml_mapping(cfg_path)

    cfg_path = _checkpoint_agent_yaml(checkpoint)
    if cfg_path is not None:
        print(f"[INFO] Loading trained agent config next to checkpoint: {cfg_path}")
        return _load_yaml_mapping(cfg_path)

    return dict(default_cfg)


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


def _strip_prefix(state_dict: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    token = f"{prefix}."
    return {key[len(token) :]: value for key, value in state_dict.items() if str(key).startswith(token)}


def _looks_like_dgppo_value_state(state_dict: Mapping[str, Any]) -> bool:
    prefixes = ("gnn.", "head.", "rnn.", "net.")
    return any(str(key).startswith(prefixes) for key in state_dict.keys())


def _state_dict_from_value(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, torch.nn.Module):
        return value.state_dict()
    if isinstance(value, Mapping):
        return value
    return None


def _extract_dgppo_value_state_dict(payload: Any, name: str) -> Mapping[str, Any] | None:
    if isinstance(payload, torch.nn.Module):
        return payload.state_dict()
    if not isinstance(payload, Mapping):
        return None

    direct = _state_dict_from_value(payload.get(name))
    if direct is not None:
        return direct

    for container_key in ("models", "model", "modules", "checkpoint_modules", "state_dict", "model_state_dict"):
        container = payload.get(container_key)
        if not isinstance(container, Mapping):
            continue
        direct = _state_dict_from_value(container.get(name))
        if direct is not None:
            return direct
        for prefix in (
            name,
            f"models.{name}",
            f"model.{name}",
            f"modules.{name}",
            f"checkpoint_modules.{name}",
        ):
            filtered = _strip_prefix(container, prefix)
            if filtered:
                return filtered

    for prefix in (name, f"models.{name}", f"model.{name}", f"modules.{name}", f"checkpoint_modules.{name}"):
        filtered = _strip_prefix(payload, prefix)
        if filtered:
            return filtered

    if _looks_like_dgppo_value_state(payload):
        return payload
    return None


def _make_critics(agent_cfg: DGPPOAgentCfg, base_env: Any, device: str | torch.device) -> tuple[DGPPOValueNet, DGPPOValueNet]:
    layout = base_env.graph_obs_layout
    graph_state_dim = int(layout["state_dim"])
    n_agents = int(getattr(base_env, "num_agents", 1))
    n_constraints = int(getattr(base_env, "n_constraints", 1))
    node_dim = graph_state_dim + NUM_TYPE_INDICATORS
    edge_dim = graph_state_dim

    gnn_cfg = agent_cfg.gnn
    rnn_cfg = agent_cfg.rnn
    model_cfg = agent_cfg.model
    critic_kwargs = dict(
        node_dim=node_dim,
        edge_dim=edge_dim,
        n_agents=n_agents,
        gnn_out_dim=int(gnn_cfg.get("critic_out_dim", gnn_cfg.get("out_dim", 64))),
        gnn_msg_dim=int(gnn_cfg.get("msg_dim", 32)),
        gnn_heads=int(gnn_cfg.get("n_heads", 3)),
        mlp_hid=_as_int_tuple(model_cfg.get("critic_mlp_hid"), (128, 64)),
        use_rnn=bool(agent_cfg.use_rnn),
        rnn_cell=str(rnn_cfg.get("cell", "gru")),
        rnn_hidden=int(rnn_cfg.get("hidden", 64)),
        rnn_layers=int(rnn_cfg.get("layers", 1)),
        device=device,
    )
    vl = DGPPOValueNet(
        **critic_kwargs,
        gnn_layers=int(gnn_cfg.get("vl_layers", 1)),
        n_out=1,
        decompose=False,
    )
    vh = DGPPOValueNet(
        **critic_kwargs,
        gnn_layers=int(gnn_cfg.get("vh_layers", 1)),
        n_out=n_constraints,
        decompose=True,
    )
    return vl.to(device).eval(), vh.to(device).eval()


def _load_critics(
    checkpoint: str | Path,
    agent_cfg: DGPPOAgentCfg,
    base_env: Any,
    device: str | torch.device,
) -> tuple[DGPPOValueNet, DGPPOValueNet]:
    payload = torch.load(str(_resolve_path(checkpoint)), map_location="cpu")
    vl, vh = _make_critics(agent_cfg, base_env, device)
    for name, model in (("Vl", vl), ("Vh", vh)):
        state_dict = _extract_dgppo_value_state_dict(payload, name)
        if state_dict is None:
            raise ValueError(f"Unable to locate DG-PPO {name} weights in checkpoint: {checkpoint}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[WARN] {name} checkpoint load mismatch (missing={missing}, unexpected={unexpected}).")
    return vl, vh


@dataclass(frozen=True)
class GridSpec:
    xs: np.ndarray
    ys: np.ndarray
    goal: tuple[float, float, float]
    z: float
    velocity: tuple[float, float, float]
    yaw: float

    @property
    def points_xy(self) -> np.ndarray:
        mesh_x, mesh_y = np.meshgrid(self.xs, self.ys, indexing="xy")
        return np.stack((mesh_x.reshape(-1), mesh_y.reshape(-1)), axis=-1)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.ys.shape[0]), int(self.xs.shape[0]))


def _make_grid_spec(env_cfg: Any) -> GridSpec:
    arena_min = getattr(env_cfg, "arena_min")
    arena_max = getattr(env_cfg, "arena_max")
    x_min = float(arena_min[0]) if args_cli.x_min is None else float(args_cli.x_min)
    x_max = float(arena_max[0]) if args_cli.x_max is None else float(args_cli.x_max)
    y_min = float(arena_min[1]) if args_cli.y_min is None else float(args_cli.y_min)
    y_max = float(arena_max[1]) if args_cli.y_max is None else float(args_cli.y_max)
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Heatmap bounds must satisfy min < max for both x and y.")
    grid_size = int(args_cli.grid_size)
    if grid_size < 2:
        raise ValueError("--grid-size must be at least 2.")

    goal = _parse_goal(args_cli.goal, env_cfg)
    z = float(args_cli.z) if args_cli.z is not None else goal[2]
    velocity = _parse_vector(args_cli.velocity, length=3, name="velocity")
    return GridSpec(
        xs=np.linspace(x_min, x_max, grid_size, dtype=np.float32),
        ys=np.linspace(y_min, y_max, grid_size, dtype=np.float32),
        goal=goal,
        z=z,
        velocity=velocity,
        yaw=float(args_cli.yaw),
    )


def _analytic_obstacle_xy(base_env: Any, positions: torch.Tensor, yaw: float) -> torch.Tensor:
    cfg = base_env.cfg
    layout = base_env.graph_obs_layout
    n_obstacles = int(layout.get("n_obstacles", 0))
    if n_obstacles == 0 or args_cli.obstacle_source == "none":
        return positions.new_empty(positions.shape[0], 0, 2)

    mode = str(getattr(cfg, "obstacle_observation_mode", "none"))
    if mode == "pillars":
        pillars = getattr(base_env, "_pillar_positions_xy", positions.new_zeros(0, 2))
        if pillars.numel() == 0:
            return positions.new_zeros(positions.shape[0], n_obstacles, 2)
        pillars = pillars.to(device=positions.device, dtype=positions.dtype)
        if pillars.shape[0] >= n_obstacles:
            return pillars[:n_obstacles].unsqueeze(0).expand(positions.shape[0], -1, -1)
        pad = pillars[-1:].expand(n_obstacles - pillars.shape[0], -1)
        return torch.cat((pillars, pad), dim=0).unsqueeze(0).expand(positions.shape[0], -1, -1)

    if mode != "ray_caster":
        return positions.new_empty(positions.shape[0], 0, 2)

    ray_xy = _analytic_ray_hits_xy(base_env, positions, yaw)
    if str(getattr(cfg, "ray_caster_observation_mode", "ray_ordered_hits")) == "top_k_hits":
        want = max(1, int(getattr(cfg, "ray_caster_top_k_hits", n_obstacles)))
        dist = torch.linalg.vector_norm(ray_xy - positions[:, None, :2], dim=-1)
        miss_dist = float(getattr(cfg, "ray_caster_max_distance", 0.0)) + 999.0
        dist = torch.where(dist > miss_dist, torch.full_like(dist, float("inf")), dist)
        _, order = torch.topk(dist, k=min(want, ray_xy.shape[1]), dim=1, largest=False)
        ray_xy = torch.gather(ray_xy, 1, order.unsqueeze(-1).expand(-1, -1, 2))
    return ray_xy[:, :n_obstacles]


def _analytic_ray_hits_xy(base_env: Any, positions: torch.Tensor, yaw: float) -> torch.Tensor:
    cfg = base_env.cfg
    num_rays = max(1, int(getattr(cfg, "ray_caster_num_rays", 1)))
    max_dist = float(getattr(cfg, "ray_caster_max_distance", 8.0))
    miss_dist = max_dist + 1_000.0
    fov_min, fov_max = getattr(cfg, "ray_caster_horizontal_fov_range", (-180.0, 180.0))
    fov_span = float(fov_max) - float(fov_min)
    full_circle = abs(abs(fov_span) - 360.0) < 1e-6
    if full_circle:
        angles = float(fov_min) + torch.arange(num_rays, device=positions.device, dtype=positions.dtype) * (
            fov_span / num_rays
        )
    else:
        angles = torch.linspace(float(fov_min), float(fov_max), num_rays, device=positions.device, dtype=positions.dtype)
    angles = torch.deg2rad(angles) + positions.new_tensor(float(yaw))
    dirs = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    origins = positions[:, :2]
    n = origins.shape[0]
    distances = positions.new_full((n, num_rays), miss_dist)

    if bool(getattr(cfg, "enable_walls", False)):
        distances = torch.minimum(distances, _ray_box_distances(origins, dirs, cfg, max_dist, miss_dist))

    pillars = getattr(base_env, "_pillar_positions_xy", None)
    if bool(getattr(cfg, "enable_pillars", False)) and pillars is not None and pillars.numel() > 0:
        pillars = pillars.to(device=positions.device, dtype=positions.dtype)
        radius = float(getattr(cfg, "pillar_radius", 0.0))
        distances = torch.minimum(distances, _ray_circle_distances(origins, dirs, pillars, radius, max_dist, miss_dist))

    inside = torch.zeros(n, dtype=torch.bool, device=positions.device)
    if hasattr(base_env, "_agent_center_inside_ray_obstacle_mask"):
        inside = base_env._agent_center_inside_ray_obstacle_mask(positions)
    distances = torch.where(inside[:, None], torch.zeros_like(distances), distances)

    return origins[:, None, :] + distances.unsqueeze(-1) * dirs.unsqueeze(0)


def _ray_box_distances(
    origins: torch.Tensor,
    dirs: torch.Tensor,
    cfg: Any,
    max_dist: float,
    miss_dist: float,
) -> torch.Tensor:
    arena_min = getattr(cfg, "arena_min")
    arena_max = getattr(cfg, "arena_max")
    x_min = float(arena_min[0])
    x_max = float(arena_max[0])
    y_min = float(arena_min[1])
    y_max = float(arena_max[1])

    out = origins.new_full((origins.shape[0], dirs.shape[0]), miss_dist)
    ox = origins[:, 0:1]
    oy = origins[:, 1:2]
    dx = dirs[:, 0].unsqueeze(0)
    dy = dirs[:, 1].unsqueeze(0)
    eps = 1e-6

    for x_face in (x_min, x_max):
        t = torch.where(dx.abs() > eps, (x_face - ox) / dx, torch.full_like(dx, miss_dist))
        y_hit = oy + t * dy
        valid = (t >= 0.0) & (t <= max_dist) & (y_hit >= y_min) & (y_hit <= y_max)
        out = torch.minimum(out, torch.where(valid, t, torch.full_like(t, miss_dist)))

    for y_face in (y_min, y_max):
        t = torch.where(dy.abs() > eps, (y_face - oy) / dy, torch.full_like(dy, miss_dist))
        x_hit = ox + t * dx
        valid = (t >= 0.0) & (t <= max_dist) & (x_hit >= x_min) & (x_hit <= x_max)
        out = torch.minimum(out, torch.where(valid, t, torch.full_like(t, miss_dist)))

    return out


def _ray_circle_distances(
    origins: torch.Tensor,
    dirs: torch.Tensor,
    centers: torch.Tensor,
    radius: float,
    max_dist: float,
    miss_dist: float,
) -> torch.Tensor:
    rel = origins[:, None, None, :] - centers[None, None, :, :]
    d = dirs[None, :, None, :]
    b = 2.0 * torch.sum(rel * d, dim=-1)
    c = torch.sum(rel * rel, dim=-1) - float(radius) ** 2
    disc = b * b - 4.0 * c
    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
    t1 = (-b - sqrt_disc) * 0.5
    t2 = (-b + sqrt_disc) * 0.5
    valid1 = (disc >= 0.0) & (t1 >= 0.0) & (t1 <= max_dist)
    valid2 = (disc >= 0.0) & (t2 >= 0.0) & (t2 <= max_dist)
    t = torch.where(valid1, t1, torch.where(valid2, t2, torch.full_like(t1, miss_dist)))
    return t.min(dim=-1).values


def _build_model_input(
    base_env: Any,
    agent_cfg: DGPPOAgentCfg,
    positions_xy: np.ndarray,
    grid: GridSpec,
    device: torch.device,
) -> Any:
    layout = base_env.graph_obs_layout
    state_dim = int(layout["state_dim"])
    n_agents = int(getattr(base_env, "num_agents", 1))
    if n_agents != 1:
        raise ValueError(f"Heatmap state synthesis currently supports one agent, got n_agents={n_agents}.")

    n = int(positions_xy.shape[0])
    agent_state = torch.zeros(n, n_agents, state_dim, device=device, dtype=torch.float32)
    agent_state[:, 0, 0:2] = torch.as_tensor(positions_xy, device=device, dtype=torch.float32)
    agent_state[:, 0, 2] = float(grid.z)
    agent_state[:, 0, 3:6] = torch.tensor(grid.velocity, device=device, dtype=torch.float32)
    if state_dim >= 8:
        agent_state[:, 0, 6] = float(np.sin(grid.yaw))
        agent_state[:, 0, 7] = float(np.cos(grid.yaw))

    goal_state = torch.zeros_like(agent_state)
    goal_state[:, 0, 0:3] = torch.tensor(grid.goal, device=device, dtype=torch.float32)
    obs_state = torch.zeros(n, int(layout.get("n_obstacles", 0)), state_dim, device=device, dtype=torch.float32)
    obstacle_xy = _analytic_obstacle_xy(base_env, agent_state[:, 0, :3], grid.yaw)
    if obstacle_xy.numel() > 0:
        obs_state[:, :, 0:2] = obstacle_xy

    return build_graph_data(
        agent_state=agent_state,
        goal_state=goal_state,
        obs_state=obs_state,
        obs_radius=float(agent_cfg.obs_radius),
    )


def _evaluate_critics(
    vl: DGPPOValueNet,
    vh: DGPPOValueNet,
    base_env: Any,
    agent_cfg: DGPPOAgentCfg,
    grid: GridSpec,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_xy = grid.points_xy
    n_points = int(points_xy.shape[0])
    batch_size = max(1, int(args_cli.batch_size))
    vl_values: list[torch.Tensor] = []
    vh_heads: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, n_points, batch_size):
            stop = min(n_points, start + batch_size)
            graph = _build_model_input(
                base_env,
                agent_cfg,
                points_xy[start:stop],
                grid,
                device,
            )
            vl_state = vl.rnn.initialize_carry(stop - start, device=device) if vl.rnn is not None else None
            vh_state = (
                vh.rnn.initialize_carry((stop - start) * int(getattr(base_env, "num_agents", 1)), device=device)
                if vh.rnn is not None
                else None
            )
            vl_out, _ = vl(graph, vl_state)
            vh_out, _ = vh(graph, vh_state)
            vl_values.append(vl_out.reshape(stop - start, -1)[:, 0].detach().cpu())
            vh_heads.append(vh_out.reshape(stop - start, -1).detach().cpu())

    vl_grid = torch.cat(vl_values).numpy().reshape(grid.shape)
    vh_all = torch.cat(vh_heads).numpy().reshape(*grid.shape, -1)
    vh_grid = _reduce_vh(vh_all)
    return vl_grid, vh_grid, vh_all


def _reduce_vh(vh_heads: np.ndarray) -> np.ndarray:
    if args_cli.vh_reduction == "max":
        return np.max(vh_heads, axis=-1)
    if args_cli.vh_reduction == "mean":
        return np.mean(vh_heads, axis=-1)
    head = int(args_cli.vh_head)
    if head < 0 or head >= vh_heads.shape[-1]:
        raise ValueError(f"--vh-head must be in [0, {vh_heads.shape[-1] - 1}], got {head}.")
    return vh_heads[..., head]


def _display_color_limits(values: np.ndarray) -> tuple[float | None, float | None, str]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None, "neither"

    raw_min = float(np.min(finite))
    raw_max = float(np.max(finite))
    if args_cli.full_color_range:
        return raw_min, raw_max, "neither"

    low_pct, high_pct = _color_percentiles()
    vmin = float(np.percentile(finite, low_pct))
    vmax = float(np.percentile(finite, high_pct))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        return raw_min, raw_max, "neither"

    below = raw_min < vmin
    above = raw_max > vmax
    if below and above:
        extend = "both"
    elif below:
        extend = "min"
    elif above:
        extend = "max"
    else:
        extend = "neither"
    return vmin, vmax, extend


def _plot_heatmap(
    output_dir: Path,
    base_env: Any,
    grid: GridSpec,
    values: np.ndarray,
    *,
    filename: str,
    title: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    fig, ax = plt.subplots(figsize=(8, 7))
    vmin, vmax, extend = _display_color_limits(values)
    image = ax.imshow(
        values,
        extent=[float(grid.xs[0]), float(grid.xs[-1]), float(grid.ys[0]), float(grid.ys[-1])],
        origin="lower",
        aspect="equal",
        cmap=args_cli.cmap,
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(image, ax=ax, shrink=0.86, extend=extend)

    contour_levels = max(0, int(args_cli.contours))
    if contour_levels > 0 and vmin is not None and vmax is not None and vmin < vmax:
        levels = np.linspace(vmin, vmax, contour_levels + 2)[1:-1]
        ax.contour(
            grid.xs,
            grid.ys,
            values,
            levels=levels,
            colors="white",
            linewidths=0.35,
            alpha=0.28,
        )

    cfg = base_env.cfg
    arena_min = getattr(cfg, "arena_min")
    arena_max = getattr(cfg, "arena_max")
    ax.add_patch(
        Rectangle(
            (float(arena_min[0]), float(arena_min[1])),
            float(arena_max[0]) - float(arena_min[0]),
            float(arena_max[1]) - float(arena_min[1]),
            fill=False,
            lw=1.6,
            ls="--",
            ec="black",
        )
    )
    if bool(getattr(cfg, "enable_pillars", False)):
        for px, py in getattr(cfg, "pillar_positions_xy", ()):
            ax.add_patch(
                Circle(
                    (float(px), float(py)),
                    radius=float(getattr(cfg, "pillar_radius", 0.0)),
                    color="dimgray",
                    alpha=0.35,
                    ec="black",
                    lw=1.0,
                )
            )
    ax.scatter([grid.goal[0]], [grid.goal[1]], marker="*", s=190, c="white", edgecolors="black", linewidths=1.0)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_outputs(
    output_dir: Path,
    base_env: Any,
    grid: GridSpec,
    vl_grid: np.ndarray,
    vh_grid: np.ndarray,
    vh_heads: np.ndarray,
    checkpoint: str,
    dgppo_cfg: DGPPOAgentCfg,
    trained_agent_cfg: Mapping[str, Any] | None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    vl_path = _plot_heatmap(output_dir, base_env, grid, vl_grid, filename="critic_vl_heatmap.png", title="DG-PPO Vl")
    vh_title = f"DG-PPO Vh ({args_cli.vh_reduction}"
    if args_cli.vh_reduction == "head":
        vh_title += f" {int(args_cli.vh_head)}"
    vh_title += ")"
    vh_path = _plot_heatmap(output_dir, base_env, grid, vh_grid, filename="critic_vh_heatmap.png", title=vh_title)

    arrays_path = output_dir / "critic_heatmap_values.npz"
    np.savez_compressed(
        arrays_path,
        xs=grid.xs,
        ys=grid.ys,
        vl=vl_grid,
        vh=vh_grid,
        vh_heads=vh_heads,
        goal=np.asarray(grid.goal, dtype=np.float32),
        z=np.asarray([grid.z], dtype=np.float32),
        velocity=np.asarray(grid.velocity, dtype=np.float32),
        yaw=np.asarray([grid.yaw], dtype=np.float32),
    )

    cfg = base_env.cfg
    env_snapshot = {
        "control_mode": str(getattr(cfg, "control_mode", "")),
        "obstacle_observation_mode": str(getattr(cfg, "obstacle_observation_mode", "")),
        "ray_caster_observation_mode": str(getattr(cfg, "ray_caster_observation_mode", "")),
        "graph_obs_layout": dict(base_env.graph_obs_layout),
        "enable_walls": bool(getattr(cfg, "enable_walls", False)),
        "wall_thickness": float(getattr(cfg, "wall_thickness", 0.0)),
        "wall_extra_margin": float(getattr(cfg, "wall_extra_margin", 0.0)),
        "enable_pillars": bool(getattr(cfg, "enable_pillars", False)),
        "pillar_positions_xy": [list(pos) for pos in getattr(cfg, "pillar_positions_xy", ())],
        "pillar_radius": float(getattr(cfg, "pillar_radius", 0.0)),
        "drone_collision_radius": float(getattr(cfg, "drone_collision_radius", 0.0)),
        "arena_min": [float(v) for v in getattr(cfg, "arena_min", ())],
        "arena_max": [float(v) for v in getattr(cfg, "arena_max", ())],
        "ray_caster_num_rays": int(getattr(cfg, "ray_caster_num_rays", 0)),
        "ray_caster_top_k_hits": int(getattr(cfg, "ray_caster_top_k_hits", 0)),
        "ray_caster_max_distance": float(getattr(cfg, "ray_caster_max_distance", 0.0)),
        "ray_caster_horizontal_fov_range": [float(v) for v in getattr(cfg, "ray_caster_horizontal_fov_range", ())],
        "ray_caster_offset": [float(v) for v in getattr(cfg, "ray_caster_offset", ())],
    }

    metadata = {
        "task": args_cli.task,
        "checkpoint": checkpoint,
        "grid_size": int(args_cli.grid_size),
        "batch_size": int(args_cli.batch_size),
        "bounds": {
            "x": [float(grid.xs[0]), float(grid.xs[-1])],
            "y": [float(grid.ys[0]), float(grid.ys[-1])],
        },
        "goal": list(grid.goal),
        "z": float(grid.z),
        "velocity": list(grid.velocity),
        "yaw": float(grid.yaw),
        "obstacle_source": args_cli.obstacle_source,
        "cli_env_overrides": {
            "ray_caster_observation_mode": args_cli.ray_caster_observation_mode,
            "ray_caster_top_k_hits": (
                None if args_cli.ray_caster_top_k_hits is None else int(args_cli.ray_caster_top_k_hits)
            ),
        },
        "seed": None if args_cli.seed is None else int(args_cli.seed),
        "num_envs": int(args_cli.num_envs),
        "color": {
            "full_range": bool(args_cli.full_color_range),
            "percentiles": list(_color_percentiles()),
            "contours": int(args_cli.contours),
            "cmap": str(args_cli.cmap),
        },
        "dgppo": {
            "obs_radius": float(dgppo_cfg.obs_radius),
            "use_rnn": bool(dgppo_cfg.use_rnn),
            "rnn": dict(dgppo_cfg.rnn),
            "gnn": dict(dgppo_cfg.gnn),
            "model": dict(dgppo_cfg.model),
        },
        "trained_agent_cfg": None if trained_agent_cfg is None else dict(trained_agent_cfg),
        "vh_reduction": args_cli.vh_reduction,
        "vh_head": int(args_cli.vh_head),
        "vh_heads": int(vh_heads.shape[-1]),
        "env": env_snapshot,
        "outputs": {
            "vl_heatmap": str(vl_path),
            "vh_heatmap": str(vh_path),
            "arrays": str(arrays_path),
        },
    }
    metadata_path = output_dir / "critic_heatmap_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "vl": str(vl_path),
        "vh": str(vh_path),
        "arrays": str(arrays_path),
        "metadata": str(metadata_path),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
@hydra_task_config(args_cli.task, "skrl_dgppo_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    checkpoint = args_cli.checkpoint
    if checkpoint is None and args_cli.artifact:
        checkpoint = _download_wandb_artifact(args_cli.artifact, args_cli.artifact_file)
    if checkpoint is None:
        raise ValueError("Provide --checkpoint or --artifact for a trained DG-PPO run.")
    checkpoint_path = _resolve_checkpoint_path(checkpoint)
    if checkpoint_path is None:
        raise ValueError("Could not resolve the DG-PPO checkpoint path.")
    checkpoint = str(checkpoint_path)

    trained_agent_cfg = _agent_cfg_for_checkpoint(agent_cfg, checkpoint)
    env_cfg.scene.num_envs = int(args_cli.num_envs or env_cfg.scene.num_envs)
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    _apply_env_overrides_from_agent_cfg(env_cfg, trained_agent_cfg)
    _apply_cli_env_overrides(env_cfg)
    env_cfg.domain_randomization.enable = False
    env_cfg.debug_vis = False
    env_cfg.debug_visualizer = False
    env_cfg.enable_cameras = False
    if args_cli.seed is not None:
        env_cfg.seed = int(args_cli.seed)

    safe_task = args_cli.task.replace("/", "-")
    exp_id = args_cli.exp_id or f"{safe_task}-dgppo-critics-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir = _resolve_path(args_cli.log_dir) / exp_id

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    device = torch.device(base_env.device)

    try:
        cfg_data = trained_agent_cfg.get("agent", trained_agent_cfg) if isinstance(trained_agent_cfg, Mapping) else {}
        dgppo_cfg = DGPPOAgentCfg.from_dict(cfg_data)
        vl, vh = _load_critics(checkpoint, dgppo_cfg, base_env, device)
        grid = _make_grid_spec(env_cfg)
        vl_grid, vh_grid, vh_heads = _evaluate_critics(vl, vh, base_env, dgppo_cfg, grid, device)
        paths = _write_outputs(
            output_dir,
            base_env,
            grid,
            vl_grid,
            vh_grid,
            vh_heads,
            checkpoint,
            dgppo_cfg,
            trained_agent_cfg if isinstance(trained_agent_cfg, Mapping) else None,
        )
    finally:
        env.close()

    print(f"[INFO] Vl heatmap: {paths['vl']}")
    print(f"[INFO] Vh heatmap: {paths['vh']}")
    print(f"[INFO] Raw arrays: {paths['arrays']}")
    print(f"[INFO] Metadata: {paths['metadata']}")


if __name__ == "__main__":
    main()
    simulation_app.close()
