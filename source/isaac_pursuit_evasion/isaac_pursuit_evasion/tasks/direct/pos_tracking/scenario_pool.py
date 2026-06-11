"""Offline scenario-pool generation and cache utilities."""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch


SCENARIO_POOL_SCHEMA_VERSION = 5
SCENARIO_POOL_SEED = 1729
SCENARIO_POOL_DIR = Path(__file__).resolve().parents[6] / "data" / "scenario_pools"

POOL_KEYS = {
    "evader_xy",
    "pursuer_candidates_xy",
    "pursuer_candidate_count",
    "static_xy",
    "static_active",
    "dynamic_waypoints",
    "dynamic_active",
    "path_type",
}

POOL_CONFIG_NAMES = (
    "arena_min",
    "arena_max",
    "arena_margin",
    "episode_length_s",
    "decimation",
    "pillar_radius",
    "drone_collision_radius",
    "pursuit_max_static_obstacles",
    "pursuit_max_dynamic_obstacles",
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
    "pursuit_dynamic_obstacle_radius",
    "pursuit_dynamic_obstacle_height",
    "pursuit_obstacle_clearance",
    "pursuit_dynamic_max_speed",
    "pursuit_pursuer_wall_clearance",
    "pursuit_pursuer_min_evader_distance",
    "pursuit_scenario_pool_size",
)

DYNAMIC_POOL_CONFIG_NAMES = {
    "pursuit_max_dynamic_obstacles",
    "pursuit_dynamic_rail_length_range",
    "pursuit_dynamic_obstacle_radius",
    "pursuit_dynamic_obstacle_height",
    "pursuit_dynamic_max_speed",
}


def enabled_pool_phases(cfg: Any) -> list[int]:
    phases = [
        phase
        for phase, fraction in enumerate(cfg.pursuit_curriculum_phase_fractions, start=1)
        if float(fraction) > 0.0
    ]
    return phases or [1]


def path_layout(cfg: Any, device: torch.device | str = "cpu") -> tuple[float, int, torch.Tensor]:
    step_dt = float(cfg.sim.dt) * int(cfg.decimation)
    path_steps = int(math.ceil(float(cfg.episode_length_s) / step_dt)) + 1
    waypoint_dt = max(step_dt, float(cfg.pursuit_path_waypoint_dt))
    stride = max(1, int(round(waypoint_dt / max(step_dt, 1e-6))))
    last_step = max(0, path_steps - 1)
    count = max(2, int(math.ceil(last_step / stride)) + 1)
    steps = torch.arange(count, device=device, dtype=torch.long) * stride
    steps[-1] = last_step
    return step_dt, path_steps, steps


def max_evader_speed(cfg: Any) -> float:
    speed_range = getattr(cfg, "pursuit_evader_speed_range", None)
    if speed_range is None:
        return max(0.0, float(getattr(cfg, "pursuit_evader_speed", 0.0)))
    return max(0.0, float(max(speed_range)))


def canonical_path_length(cfg: Any, waypoint_steps: torch.Tensor, step_dt: float) -> float:
    dt_steps = (waypoint_steps[1:] - waypoint_steps[:-1]).to(torch.float32)
    distance = torch.sum(dt_steps) * float(step_dt) * max_evader_speed(cfg)
    return float(distance.item()) + float(cfg.pursuit_grid_cell_size)


def canonical_path_point_count(cfg: Any, waypoint_steps: torch.Tensor, step_dt: float) -> int:
    resolution = max(0.025, 0.5 * float(cfg.pursuit_grid_cell_size))
    return max(2, int(math.ceil(canonical_path_length(cfg, waypoint_steps, step_dt) / resolution)) + 1)


def scenario_pool_config(
    cfg: Any,
    waypoint_steps: torch.Tensor | None = None,
    step_dt: float | None = None,
) -> dict[str, object]:
    if waypoint_steps is None or step_dt is None:
        step_dt, _, waypoint_steps = path_layout(cfg)
    config: dict[str, object] = {
        "schema_version": SCENARIO_POOL_SCHEMA_VERSION,
        "seed": SCENARIO_POOL_SEED,
        "phases": enabled_pool_phases(cfg),
        "path_waypoint_count": int(waypoint_steps.numel()),
        "path_waypoint_steps": waypoint_steps.detach().cpu().tolist(),
        "canonical_path_points": canonical_path_point_count(cfg, waypoint_steps, step_dt),
        "step_dt": float(step_dt),
    }
    dynamic_phase_enabled = any(phase >= 4 for phase in config["phases"])
    for name in POOL_CONFIG_NAMES:
        value = None if name in DYNAMIC_POOL_CONFIG_NAMES and not dynamic_phase_enabled else getattr(cfg, name)
        if isinstance(value, tuple):
            value = list(value)
        config[name] = value
    return config


def scenario_pool_cache_path(
    cfg: Any,
    waypoint_steps: torch.Tensor | None = None,
    step_dt: float | None = None,
) -> Path:
    config_json = json.dumps(
        scenario_pool_config(cfg, waypoint_steps, step_dt),
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:20]
    return SCENARIO_POOL_DIR / f"grid_pool_v{SCENARIO_POOL_SCHEMA_VERSION}_{digest}.pt"


def load_scenario_pools(
    cfg: Any,
    waypoint_steps: torch.Tensor,
    step_dt: float,
) -> dict[int, dict[str, torch.Tensor]]:
    cache_path = scenario_pool_cache_path(cfg, waypoint_steps, step_dt)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Pursuit scenario pool cache is missing: {cache_path}\n"
            "Generate it before training with:\n"
            f"  python scripts/generate_scenario_pool.py --pool-size {int(cfg.pursuit_scenario_pool_size)} --headless"
        )

    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"Could not load pursuit scenario pool cache {cache_path}: {exc}") from exc

    expected_config = scenario_pool_config(cfg, waypoint_steps, step_dt)
    if int(payload.get("schema_version", -1)) != SCENARIO_POOL_SCHEMA_VERSION:
        raise RuntimeError(f"Scenario pool cache has the wrong schema version: {cache_path}")
    if payload.get("config") != expected_config:
        raise RuntimeError(f"Scenario pool cache does not match the resolved environment config: {cache_path}")

    pools = {int(phase): pool for phase, pool in payload["pools"].items()}
    expected_size = max(1, int(cfg.pursuit_scenario_pool_size))
    expected_points = canonical_path_point_count(cfg, waypoint_steps, step_dt)
    if sorted(pools) != enabled_pool_phases(cfg):
        raise RuntimeError(f"Scenario pool cache has the wrong phase set: {cache_path}")

    for phase, pool in pools.items():
        if set(pool) != POOL_KEYS:
            raise RuntimeError(f"Scenario pool phase {phase} has an invalid tensor schema: {cache_path}")
        if int(pool["path_type"].shape[0]) != expected_size:
            raise RuntimeError(f"Scenario pool phase {phase} has the wrong row count: {cache_path}")
        if tuple(pool["evader_xy"].shape[1:]) != (expected_points, 2):
            raise RuntimeError(f"Scenario pool phase {phase} has the wrong path shape: {cache_path}")
        if bool(torch.any(pool["pursuer_candidate_count"] <= 0).item()):
            raise RuntimeError(f"Scenario pool phase {phase} contains no-spawn rows: {cache_path}")

        target_dynamic = max(0, int(cfg.pursuit_max_dynamic_obstacles))
        pool_dynamic = int(pool["dynamic_active"].shape[1])
        if pool_dynamic > target_dynamic:
            raise RuntimeError(f"Scenario pool phase {phase} has too many dynamic slots: {cache_path}")
        if pool_dynamic < target_dynamic:
            missing = target_dynamic - pool_dynamic
            pool["dynamic_waypoints"] = torch.cat(
                (
                    pool["dynamic_waypoints"],
                    torch.zeros(
                        expected_size,
                        pool["dynamic_waypoints"].shape[1],
                        missing,
                        3,
                        dtype=pool["dynamic_waypoints"].dtype,
                    ),
                ),
                dim=2,
            )
            pool["dynamic_active"] = torch.cat(
                (
                    pool["dynamic_active"],
                    torch.zeros(expected_size, missing, dtype=torch.bool),
                ),
                dim=1,
            )

    size_bytes = sum(
        tensor.numel() * tensor.element_size()
        for pool in pools.values()
        for tensor in pool.values()
    )
    print(
        f"[INFO] Loaded pursuit scenario pools: phases={sorted(pools)}, "
        f"rows={expected_size} each, device=cpu, memory={size_bytes / (1024.0 * 1024.0):.1f} MiB"
    )
    return pools


def point_free(
    cfg: Any,
    pos: torch.Tensor,
    static_xy: torch.Tensor,
    static_active: torch.Tensor,
    dynamic_pos: torch.Tensor,
    dynamic_active: torch.Tensor,
) -> bool:
    if bool(static_active.any().item()):
        distance = torch.linalg.vector_norm(static_xy - pos[:2], dim=-1)
        safe = float(cfg.pillar_radius + cfg.drone_collision_radius + cfg.pursuit_obstacle_clearance)
        if bool(torch.any((distance <= safe) & static_active).item()):
            return False
    if bool(dynamic_active.any().item()):
        dynamic_now = dynamic_pos[0] if dynamic_pos.ndim == 3 else dynamic_pos
        distance = torch.linalg.vector_norm(dynamic_now[:, :2] - pos[:2], dim=-1)
        safe = float(cfg.pursuit_dynamic_obstacle_radius + cfg.drone_collision_radius)
        if bool(torch.any((distance <= safe) & dynamic_active).item()):
            return False
    return True


class ScenarioPoolBuilder:
    """Generate exact-phase grid scenarios without constructing an Isaac environment."""

    def __init__(self, cfg: Any, device: torch.device | str) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.step_dt, self.path_steps, self.waypoint_steps = path_layout(cfg, self.device)
        self.waypoint_count = int(self.waypoint_steps.numel())
        self.max_static = max(1, int(cfg.pursuit_max_static_obstacles)) if cfg.enable_pillars else 0
        self.max_dynamic = max(0, int(cfg.pursuit_max_dynamic_obstacles))
        self.pool_max_dynamic = (
            self.max_dynamic if any(phase >= 4 for phase in enabled_pool_phases(cfg)) else 0
        )
        self.arena_min_safe = torch.tensor(cfg.arena_min, device=self.device, dtype=torch.float32)
        self.arena_max_safe = torch.tensor(cfg.arena_max, device=self.device, dtype=torch.float32)
        margin = float(cfg.arena_margin)
        self.arena_min_safe += margin
        self.arena_max_safe -= margin

    def build_and_save(self) -> Path:
        pools = self.build_pools()
        cache_path = scenario_pool_cache_path(self.cfg, self.waypoint_steps, self.step_dt)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        config = scenario_pool_config(self.cfg, self.waypoint_steps, self.step_dt)
        torch.save(
            {
                "schema_version": SCENARIO_POOL_SCHEMA_VERSION,
                "config": config,
                "pools": {str(phase): pool for phase, pool in pools.items()},
            },
            temporary,
        )
        temporary.replace(cache_path)

        manifest_path = cache_path.with_suffix(".json")
        manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
        manifest_tmp.write_text(
            json.dumps(
                {
                    "scenario_file": cache_path.name,
                    "schema_version": SCENARIO_POOL_SCHEMA_VERSION,
                    "rows_per_phase": int(self.cfg.pursuit_scenario_pool_size),
                    "phases": sorted(pools),
                    "config": config,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        manifest_tmp.replace(manifest_path)
        return cache_path

    def build_pools(self) -> dict[int, dict[str, torch.Tensor]]:
        pool_size = max(1, int(self.cfg.pursuit_scenario_pool_size))
        attempts_per_row = max(1, int(self.cfg.pursuit_scenario_attempts))
        phases = enabled_pool_phases(self.cfg)
        pools: dict[int, dict[str, torch.Tensor]] = {}
        start_time = time.perf_counter()
        print(
            f"[INFO] Generating pursuit scenario pools: phases={phases}, "
            f"rows={pool_size} each, seed={SCENARIO_POOL_SEED}"
        )

        fork_devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(SCENARIO_POOL_SEED)
            for phase in phases:
                rows: dict[str, list[torch.Tensor]] = {name: [] for name in POOL_KEYS}
                attempts = 0
                max_attempts = pool_size * attempts_per_row
                report_stride = max(1, pool_size // 10)
                while len(rows["path_type"]) < pool_size and attempts < max_attempts:
                    attempts += 1
                    scenario = self.sample_scenario(phase, evader_speed=max_evader_speed(self.cfg))
                    if scenario is None:
                        continue
                    candidates, candidate_count = self.pursuer_edge_candidates(scenario)
                    canonical_xy = self.canonical_polyline(scenario["evader_polyline"])
                    if candidate_count == 0 or canonical_xy is None:
                        continue

                    rows["evader_xy"].append(canonical_xy)
                    rows["pursuer_candidates_xy"].append(candidates)
                    rows["pursuer_candidate_count"].append(
                        torch.tensor(candidate_count, device=self.device, dtype=torch.long)
                    )
                    rows["static_xy"].append(scenario["static_xy"].clone())
                    rows["static_active"].append(scenario["static_active"].clone())
                    rows["dynamic_waypoints"].append(
                        scenario["dynamic_waypoints"][:, : self.pool_max_dynamic].clone()
                    )
                    rows["dynamic_active"].append(
                        scenario["dynamic_active"][: self.pool_max_dynamic].clone()
                    )
                    rows["path_type"].append(
                        torch.tensor(int(scenario["path_type"]), device=self.device, dtype=torch.long)
                    )

                    accepted = len(rows["path_type"])
                    if accepted % report_stride == 0 or accepted == pool_size:
                        print(
                            f"[INFO] Scenario pool phase {phase}: {accepted}/{pool_size} "
                            f"accepted from {attempts} attempts"
                        )

                if len(rows["path_type"]) != pool_size:
                    raise RuntimeError(
                        f"Could not build exact phase-{phase} scenario pool: accepted "
                        f"{len(rows['path_type'])}/{pool_size} after {attempts} attempts."
                    )
                pools[phase] = {name: torch.stack(values).cpu() for name, values in rows.items()}

        print(f"[INFO] Generated pursuit scenario pools in {time.perf_counter() - start_time:.1f}s")
        return pools

    def sample_scenario(
        self,
        phase: int,
        *,
        evader_speed: float | None = None,
    ) -> dict[str, torch.Tensor | int | float] | None:
        n_static, n_dynamic = self.phase_obstacle_counts(phase)
        grid = self.make_grid()
        if grid is None:
            return None

        pursuer_start = self.sample_pursuer_start(grid)
        if pursuer_start is None:
            return None
        pursuer_mask = self.mark_square(grid, pursuer_start[:2], float(self.cfg.pursuit_pursuer_occupied_side))
        grid["occupied"] |= pursuer_mask

        dynamic = self.sample_dynamic_obstacles(grid, n_dynamic)
        if dynamic is None:
            return None
        dynamic_waypoints, dynamic_active = dynamic

        static = self.sample_static_obstacles(grid, n_static)
        if static is None:
            return None
        static_xy, static_active = static

        evader_cell = self.sample_evader_cell(grid, pursuer_start[:2])
        if evader_cell is None:
            return None
        evader_z = float(self.sample_evader_z())
        pursuer_start[2] = evader_z

        if int(phase) == 1:
            evader_polyline = self.grid_cell_xy(grid, evader_cell).view(1, 2)
            evader_waypoints = self.static_evader_waypoints(grid, evader_cell, evader_z)
            path_type = -2
        else:
            speed = self.sample_evader_speed() if evader_speed is None else float(evader_speed)
            evader_polyline = self.sample_evader_polyline(grid, evader_cell, pursuer_mask, speed)
            if evader_polyline is None:
                return None
            evader_waypoints = self.polyline_waypoints(
                evader_polyline,
                self.evader_step_distances(speed),
                evader_z,
            )
            path_type = 0

        if not point_free(
            self.cfg,
            pursuer_start,
            static_xy,
            static_active,
            dynamic_waypoints,
            dynamic_active,
        ):
            return None

        velocity = self.first_waypoint_velocity(evader_waypoints)
        yaw = torch.atan2(velocity[1], velocity[0])
        return {
            "phase": int(phase),
            "path_type": int(path_type),
            "evader_polyline": evader_polyline,
            "evader_waypoints": evader_waypoints,
            "evader_yaw": float(yaw.item()),
            "pursuer_start": pursuer_start,
            "static_xy": static_xy,
            "static_active": static_active,
            "dynamic_waypoints": dynamic_waypoints,
            "dynamic_active": dynamic_active,
            "fallback": False,
        }

    def sample_with_fallback(self, phase: int, attempts: int) -> dict[str, Any]:
        for _ in range(max(1, attempts)):
            scenario = self.sample_scenario(phase)
            if scenario is not None:
                return scenario
        scenario = self.fallback_scenario(phase)
        scenario["fallback"] = True
        return scenario

    def fallback_scenario(self, phase: int) -> dict[str, Any]:
        lo = self.arena_min_safe[:2] + float(self.cfg.pursuit_evader_wall_clearance)
        hi = self.arena_max_safe[:2] - float(self.cfg.pursuit_evader_wall_clearance)
        center = 0.5 * (lo + hi)
        length = min(1.6, max(0.5, float(hi[0] - lo[0]) * 0.35))
        t = torch.linspace(0.0, 1.0, self.waypoint_count, device=self.device)
        waypoints = torch.zeros(self.waypoint_count, 3, device=self.device)
        waypoints[:, :2] = center.view(1, 2)
        waypoints[:, 0] = center[0] + (t - 0.5) * length
        waypoints[:, 2] = 0.5 * (self.arena_min_safe[2] + self.arena_max_safe[2])
        dynamic = self.inactive_dynamic_pos(self.max_dynamic).view(1, -1, 3).repeat(
            self.waypoint_count, 1, 1
        )
        velocity = self.first_waypoint_velocity(waypoints)
        return {
            "phase": int(phase),
            "path_type": -1,
            "evader_polyline": waypoints[:, :2],
            "evader_waypoints": waypoints,
            "evader_yaw": float(torch.atan2(velocity[1], velocity[0]).item()),
            "pursuer_start": self.fallback_pursuer_start(waypoints[0]),
            "static_xy": self.inactive_obstacle_xy(self.max_static),
            "static_active": torch.zeros(self.max_static, dtype=torch.bool, device=self.device),
            "dynamic_waypoints": dynamic,
            "dynamic_active": torch.zeros(self.max_dynamic, dtype=torch.bool, device=self.device),
            "fallback": True,
        }

    def canonical_polyline(self, xy: torch.Tensor) -> torch.Tensor | None:
        point_count = canonical_path_point_count(self.cfg, self.waypoint_steps, self.step_dt)
        if xy.shape[0] <= 1:
            return xy[:1].repeat(point_count, 1)
        seg_len = torch.linalg.vector_norm(xy[1:] - xy[:-1], dim=-1).clamp_min(1e-6)
        cumulative = torch.cat((torch.zeros(1, device=self.device), torch.cumsum(seg_len, dim=0)))
        length = canonical_path_length(self.cfg, self.waypoint_steps, self.step_dt)
        if float(cumulative[-1].item()) + 1e-5 < length:
            return None
        target = torch.linspace(0.0, length, point_count, device=self.device)
        seg = torch.searchsorted(cumulative[1:], target).clamp(max=seg_len.shape[0] - 1)
        tau = ((target - cumulative[seg]) / seg_len[seg]).view(-1, 1)
        return xy[seg] * (1.0 - tau) + xy[seg + 1] * tau

    def pursuer_edge_candidates(self, scenario: dict[str, Any]) -> tuple[torch.Tensor, int]:
        grid = self.make_grid()
        if grid is None:
            return torch.zeros(0, 2, device=self.device), 0
        rows = int(grid["ys"].shape[0])
        cols = int(grid["xs"].shape[0])
        top = torch.stack(
            (torch.zeros(cols, device=self.device, dtype=torch.long), torch.arange(cols, device=self.device)),
            dim=-1,
        )
        bottom = top.clone()
        bottom[:, 0] = rows - 1
        side_rows = torch.arange(1, max(1, rows - 1), device=self.device)
        left = torch.stack((side_rows, torch.zeros_like(side_rows)), dim=-1)
        right = left.clone()
        right[:, 1] = cols - 1
        cells = torch.cat((top, bottom, left, right), dim=0)
        xy = self.grid_cells_xy(grid, cells)

        clearance = float(self.cfg.pursuit_pursuer_wall_clearance)
        lo = self.arena_min_safe[:2] + clearance
        hi = self.arena_max_safe[:2] - clearance
        valid = torch.all((xy >= lo) & (xy <= hi), dim=-1)
        valid &= torch.linalg.vector_norm(xy - scenario["evader_waypoints"][0, :2], dim=-1) >= float(
            self.cfg.pursuit_pursuer_min_evader_distance
        )

        static_active = scenario["static_active"]
        if bool(static_active.any().item()):
            safe = float(
                self.cfg.pillar_radius
                + self.cfg.drone_collision_radius
                + self.cfg.pursuit_obstacle_clearance
            )
            distance = torch.linalg.vector_norm(xy[:, None] - scenario["static_xy"][None], dim=-1)
            valid &= ~torch.any((distance <= safe) & static_active.view(1, -1), dim=1)

        dynamic_active = scenario["dynamic_active"]
        if bool(dynamic_active.any().item()):
            safe = float(self.cfg.pursuit_dynamic_obstacle_radius + self.cfg.drone_collision_radius)
            dynamic_xy = scenario["dynamic_waypoints"][0, :, :2]
            distance = torch.linalg.vector_norm(xy[:, None] - dynamic_xy[None], dim=-1)
            valid &= ~torch.any((distance <= safe) & dynamic_active.view(1, -1), dim=1)

        selected = xy[valid]
        packed = torch.zeros_like(xy)
        packed[: selected.shape[0]] = selected
        return packed, int(selected.shape[0])

    def phase_obstacle_counts(self, phase: int) -> tuple[int, int]:
        if phase == 1 or phase == 2:
            choices = [(count, 0) for count in range(1, 3)]
        elif phase == 3:
            choices = [(count, 0) for count in range(2, 6)]
        else:
            choices = [
                (total - dynamic, dynamic)
                for total in range(4, 7)
                for dynamic in range(1, 4)
            ]
        choices = [
            pair for pair in choices
            if 0 <= pair[0] <= self.max_static and 0 <= pair[1] <= self.max_dynamic
        ]
        if not choices:
            return 0, 0
        index = int(torch.randint(0, len(choices), (1,), device=self.device).item())
        return choices[index]

    def make_grid(self) -> dict[str, torch.Tensor | float] | None:
        cell = max(0.05, float(self.cfg.pursuit_grid_cell_size))
        margin = max(0.0, float(self.cfg.pursuit_grid_wall_margin))
        lo = self.arena_min_safe[:2] + margin
        hi = self.arena_max_safe[:2] - margin
        if bool(torch.any(hi <= lo).item()):
            return None
        xs = torch.arange(float(lo[0]), float(hi[0]) + 0.5 * cell, cell, device=self.device)
        ys = torch.arange(float(lo[1]), float(hi[1]) + 0.5 * cell, cell, device=self.device)
        xs = xs[xs <= float(hi[0]) + 1e-6]
        ys = ys[ys <= float(hi[1]) + 1e-6]
        if xs.numel() < 3 or ys.numel() < 3:
            return None
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        occupied = torch.zeros((ys.numel(), xs.numel()), dtype=torch.bool, device=self.device)
        return {"xs": xs, "ys": ys, "xx": xx, "yy": yy, "occupied": occupied, "cell_size": cell}

    def sample_pursuer_start(self, grid: dict[str, Any]) -> torch.Tensor | None:
        rows = int(grid["ys"].shape[0])
        cols = int(grid["xs"].shape[0])
        if rows == 0 or cols == 0:
            return None
        side = int(torch.randint(0, 4, (1,), device=self.device).item())
        if side == 0:
            cell = (int(torch.randint(0, rows, (1,), device=self.device).item()), 0)
        elif side == 1:
            cell = (int(torch.randint(0, rows, (1,), device=self.device).item()), cols - 1)
        elif side == 2:
            cell = (0, int(torch.randint(0, cols, (1,), device=self.device).item()))
        else:
            cell = (rows - 1, int(torch.randint(0, cols, (1,), device=self.device).item()))
        pos = torch.zeros(3, device=self.device)
        pos[:2] = self.grid_cell_xy(grid, cell)
        pos[2] = 0.5 * (self.arena_min_safe[2] + self.arena_max_safe[2])
        return pos

    def sample_dynamic_obstacles(
        self,
        grid: dict[str, Any],
        count: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        waypoints = self.inactive_dynamic_pos(self.max_dynamic).view(1, self.max_dynamic, 3).repeat(
            self.waypoint_count, 1, 1
        )
        active = torch.zeros(self.max_dynamic, dtype=torch.bool, device=self.device)
        for slot in range(min(max(0, count), self.max_dynamic)):
            rail = self.sample_dynamic_rail(grid)
            if rail is None:
                return None
            start_xy, end_xy, mask = rail
            waypoints[:, slot] = self.dynamic_rail_waypoints(start_xy, end_xy)
            active[slot] = True
            grid["occupied"] |= mask
        return waypoints, active

    def sample_dynamic_rail(
        self,
        grid: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        length_range = tuple(self.cfg.pursuit_dynamic_rail_length_range)
        length_lo = max(0.1, float(min(length_range)))
        length_hi = max(length_lo, float(max(length_range)))
        directions = ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0))
        lo = torch.stack((grid["xs"][0], grid["ys"][0]))
        hi = torch.stack((grid["xs"][-1], grid["ys"][-1]))
        for _ in range(96):
            free = self.grid_free_cells(grid)
            if free.numel() == 0:
                return None
            pick = int(torch.randint(0, free.shape[0], (1,), device=self.device).item())
            center = self.grid_cell_xy(grid, (int(free[pick, 0]), int(free[pick, 1])))
            direction = torch.tensor(
                directions[int(torch.randint(0, len(directions), (1,), device=self.device).item())],
                device=self.device,
            )
            direction /= torch.linalg.vector_norm(direction).clamp_min(1e-6)
            length = float(torch.empty((), device=self.device).uniform_(length_lo, length_hi).item())
            start_xy = center - 0.5 * length * direction
            end_xy = center + 0.5 * length * direction
            if not bool(torch.all((start_xy >= lo) & (start_xy <= hi) & (end_xy >= lo) & (end_xy <= hi)).item()):
                continue
            mask = self.segment_mask(grid, start_xy, end_xy, self.dynamic_grid_radius())
            if not bool(torch.any(grid["occupied"] & mask).item()):
                return start_xy, end_xy, mask
        return None

    def dynamic_rail_waypoints(self, start_xy: torch.Tensor, end_xy: torch.Tensor) -> torch.Tensor:
        path = torch.zeros(self.waypoint_count, 3, device=self.device)
        length = torch.linalg.vector_norm(end_xy - start_xy).clamp_min(1e-6)
        speed = max(0.0, float(self.cfg.pursuit_dynamic_max_speed))
        if speed <= 1e-6:
            alpha = torch.zeros(self.waypoint_count, device=self.device)
        else:
            t = self.waypoint_steps.to(torch.float32) * self.step_dt
            phase = torch.remainder(t * speed / length, 2.0)
            alpha = torch.where(phase <= 1.0, phase, 2.0 - phase)
        path[:, :2] = start_xy * (1.0 - alpha[:, None]) + end_xy * alpha[:, None]
        path[:, 2] = self.dynamic_center_z()
        return path

    def sample_static_obstacles(
        self,
        grid: dict[str, Any],
        count: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        xy = self.inactive_obstacle_xy(self.max_static)
        active = torch.zeros(self.max_static, dtype=torch.bool, device=self.device)
        radius = self.static_grid_radius()
        safe_static = 2.0 * float(self.cfg.pillar_radius) + float(self.cfg.pursuit_obstacle_clearance)
        for slot in range(min(max(0, count), self.max_static)):
            free = self.grid_free_cells(grid)
            if free.numel() == 0:
                return None
            placed = False
            for pick in torch.randperm(free.shape[0], device=self.device).tolist():
                candidate = self.grid_cell_xy(grid, (int(free[pick, 0]), int(free[pick, 1])))
                if slot > 0 and bool(
                    torch.any(torch.linalg.vector_norm(xy[:slot] - candidate, dim=-1) <= safe_static).item()
                ):
                    continue
                mask = self.disc_mask(grid, candidate, radius)
                if bool(torch.any(grid["occupied"] & mask).item()):
                    continue
                xy[slot] = candidate
                active[slot] = True
                grid["occupied"] |= mask
                placed = True
                break
            if not placed:
                return None
        return xy, active

    def sample_evader_cell(
        self,
        grid: dict[str, Any],
        pursuer_xy: torch.Tensor,
    ) -> tuple[int, int] | None:
        free = self.grid_free_cells(grid)
        if free.numel() == 0:
            return None
        xy = self.grid_cells_xy(grid, free)
        valid = torch.linalg.vector_norm(xy - pursuer_xy, dim=-1) >= float(
            self.cfg.pursuit_pursuer_min_evader_distance
        )
        candidates = free[valid]
        if candidates.numel() == 0:
            return None
        pick = int(torch.randint(0, candidates.shape[0], (1,), device=self.device).item())
        return int(candidates[pick, 0]), int(candidates[pick, 1])

    def static_evader_waypoints(self, grid: dict[str, Any], cell: tuple[int, int], z: float) -> torch.Tensor:
        waypoints = torch.zeros(self.waypoint_count, 3, device=self.device)
        waypoints[:, :2] = self.grid_cell_xy(grid, cell)
        waypoints[:, 2] = z
        return waypoints

    def sample_evader_polyline(
        self,
        grid: dict[str, Any],
        start_cell: tuple[int, int],
        pursuer_mask: torch.Tensor,
        speed: float,
    ) -> torch.Tensor | None:
        if speed <= 0.0:
            return self.grid_cell_xy(grid, start_cell).view(1, 2)
        required_distance = float(torch.sum(self.evader_step_distances(speed)).item())
        needed = required_distance + float(grid["cell_size"])
        occupied = grid["occupied"].clone()
        current = start_cell
        polyline = self.grid_cell_xy(grid, current).view(1, 2)
        distance = 0.0
        smooth = bool(self.cfg.pursuit_smooth_evader_path)

        for path_id in range(24):
            if distance >= needed:
                break
            next_polyline = None
            attempts = max(1, int(self.cfg.pursuit_smooth_evader_goal_attempts)) if path_id and smooth else 1
            for _ in range(attempts):
                incoming = None
                if path_id and polyline.shape[0] > 1 and smooth:
                    incoming = torch.nn.functional.normalize(polyline[-1] - polyline[-2], dim=0)
                path = self.sample_goal_path(grid, occupied, current, incoming)
                if path is None:
                    continue
                path_xy = self.grid_cells_to_xy(grid, path)
                next_polyline = (
                    torch.cat((polyline, path_xy[1:]), dim=0)
                    if path_id == 0 or not smooth
                    else self.blend_goal_junction(grid, polyline, path_xy)
                )
                if next_polyline is not None:
                    break
            if next_polyline is None:
                return None
            polyline = next_polyline
            distance = self.polyline_length(polyline)
            current = path[-1]
            if path_id == 0:
                occupied = occupied.clone()
                occupied[pursuer_mask] = False

        if distance < required_distance:
            return None
        if bool(self.cfg.pursuit_smooth_evader_validate) and not self.path_free(grid, polyline):
            return None
        return polyline

    def sample_goal_path(
        self,
        grid: dict[str, Any],
        occupied: torch.Tensor,
        start_cell: tuple[int, int],
        incoming: torch.Tensor | None,
    ) -> list[tuple[int, int]] | None:
        free = torch.nonzero(~occupied, as_tuple=False)
        if free.numel() == 0:
            return None
        start_xy = self.grid_cell_xy(grid, start_cell)
        xy = self.grid_cells_xy(grid, free)
        valid = torch.linalg.vector_norm(xy - start_xy, dim=-1) >= float(self.cfg.pursuit_evader_goal_min_distance)
        if bool(self.cfg.pursuit_smooth_evader_path):
            margin = max(0.0, float(self.cfg.pursuit_smooth_evader_goal_blend_distance))
            lo = torch.stack((grid["xs"][0], grid["ys"][0])) + margin
            hi = torch.stack((grid["xs"][-1], grid["ys"][-1])) - margin
            valid &= torch.all((xy >= lo) & (xy <= hi), dim=-1)
        if incoming is not None:
            direction = torch.nn.functional.normalize(xy - start_xy, dim=-1)
            valid &= torch.sum(direction * incoming, dim=-1) >= float(
                self.cfg.pursuit_smooth_evader_goal_sample_min_alignment
            )
        candidates = free[valid]
        for pick in torch.randperm(candidates.shape[0], device=self.device)[:64].tolist():
            goal = (int(candidates[pick, 0]), int(candidates[pick, 1]))
            path = self.astar(occupied, start_cell, goal)
            if path is not None and len(path) > 1:
                return path
        return None

    def polyline_waypoints(self, xy: torch.Tensor, step_dist: torch.Tensor, z: float) -> torch.Tensor:
        waypoints = torch.zeros(self.waypoint_count, 3, device=self.device)
        if xy.shape[0] <= 1:
            waypoints[:, :2] = xy[:1]
            waypoints[:, 2] = z
            return waypoints
        seg_len = torch.linalg.vector_norm(xy[1:] - xy[:-1], dim=-1).clamp_min(1e-6)
        cumulative = torch.cat((torch.zeros(1, device=self.device), torch.cumsum(seg_len, dim=0)))
        target = torch.cat((torch.zeros(1, device=self.device), torch.cumsum(step_dist, dim=0)))
        target = target.clamp(max=float(cumulative[-1].item()))
        seg = torch.searchsorted(cumulative[1:], target).clamp(max=seg_len.shape[0] - 1)
        tau = ((target - cumulative[seg]) / seg_len[seg]).view(-1, 1)
        waypoints[:, :2] = xy[seg] * (1.0 - tau) + xy[seg + 1] * tau
        waypoints[:, 2] = z
        return waypoints

    def blend_goal_junction(
        self,
        grid: dict[str, Any],
        previous: torch.Tensor,
        following: torch.Tensor,
    ) -> torch.Tensor | None:
        if previous.shape[0] <= 1 or following.shape[0] <= 1:
            return None
        blend = max(0.0, float(self.cfg.pursuit_smooth_evader_goal_blend_distance))
        blend = min(blend, 0.45 * self.polyline_length(previous), 0.45 * self.polyline_length(following))
        if blend <= 1e-6:
            return None
        before, entry = self.polyline_prefix(previous, self.polyline_length(previous) - blend)
        exit_point, after = self.polyline_suffix(following, blend)
        incoming = torch.nn.functional.normalize(previous[-1] - entry, dim=0)
        outgoing = torch.nn.functional.normalize(exit_point - following[0], dim=0)
        if float(torch.dot(incoming, outgoing).item()) < float(self.cfg.pursuit_smooth_evader_goal_min_alignment):
            return None
        control_in = entry + 0.5 * blend * torch.nn.functional.normalize(before[-1] - before[-2], dim=0)
        control_out = exit_point - 0.5 * blend * torch.nn.functional.normalize(after[1] - after[0], dim=0)
        resolution = max(2, int(self.cfg.pursuit_smooth_evader_resolution))
        t = torch.linspace(0.0, 1.0, resolution + 1, device=self.device).view(-1, 1)
        curve = self.cubic_bezier(entry, control_in, control_out, exit_point, t)
        if bool(self.cfg.pursuit_smooth_evader_validate) and not self.path_free(grid, curve):
            return None
        return torch.cat((before[:-1], curve, after[1:]), dim=0)

    def astar(
        self,
        occupied: torch.Tensor,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]] | None:
        occ = occupied.detach().cpu().tolist()
        rows = len(occ)
        cols = len(occ[0]) if rows else 0
        if not rows or occ[start[0]][start[1]] or occ[goal[0]][goal[1]]:
            return None
        moves = (
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
        )
        heuristic = lambda cell: math.hypot(float(cell[0] - goal[0]), float(cell[1] - goal[1]))
        heap = [(heuristic(start), 0, start)]
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        cost = {start: 0.0}
        closed: set[tuple[int, int]] = set()
        push_id = 1
        while heap:
            _, _, cell = heapq.heappop(heap)
            if cell in closed:
                continue
            if cell == goal:
                path = [cell]
                while cell in parent:
                    cell = parent[cell]
                    path.append(cell)
                return list(reversed(path))
            closed.add(cell)
            row, col = cell
            for dr, dc, move_cost in moves:
                nr, nc = row + dr, col + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols or occ[nr][nc]:
                    continue
                if dr and dc and (occ[row][nc] or occ[nr][col]):
                    continue
                new_cost = cost[cell] + move_cost
                nxt = (nr, nc)
                if new_cost >= cost.get(nxt, float("inf")):
                    continue
                cost[nxt] = new_cost
                parent[nxt] = cell
                heapq.heappush(heap, (new_cost + heuristic(nxt), push_id, nxt))
                push_id += 1
        return None

    def path_free(self, grid: dict[str, Any], xy: torch.Tensor) -> bool:
        if xy.shape[0] <= 1:
            points = xy
        else:
            cell = max(1e-6, float(grid["cell_size"]))
            seg_len = torch.linalg.vector_norm(xy[1:] - xy[:-1], dim=-1)
            samples = max(2, int(torch.ceil(torch.max(seg_len) / (0.5 * cell)).item()) + 1)
            t = torch.linspace(0.0, 1.0, samples, device=self.device)
            points = (xy[:-1, None] * (1.0 - t[None, :, None]) + xy[1:, None] * t[None, :, None]).reshape(-1, 2)
        cell = max(1e-6, float(grid["cell_size"]))
        cols = torch.round((points[:, 0] - grid["xs"][0]) / cell).to(torch.long)
        rows = torch.round((points[:, 1] - grid["ys"][0]) / cell).to(torch.long)
        in_bounds = (rows >= 0) & (rows < grid["ys"].shape[0]) & (cols >= 0) & (cols < grid["xs"].shape[0])
        return bool(torch.all(in_bounds).item()) and not bool(torch.any(grid["occupied"][rows, cols]).item())

    def sample_evader_speed(self) -> float:
        speed_range = tuple(self.cfg.pursuit_evader_speed_range)
        lo, hi = max(0.0, float(min(speed_range))), max(0.0, float(max(speed_range)))
        return lo if hi <= lo else float(torch.empty((), device=self.device).uniform_(lo, hi).item())

    def sample_evader_z(self) -> torch.Tensor:
        clearance = float(self.cfg.pursuit_evader_wall_clearance)
        lo = self.arena_min_safe[2] + clearance
        hi = self.arena_max_safe[2] - clearance
        if float(hi) <= float(lo):
            lo, hi = self.arena_min_safe[2], self.arena_max_safe[2]
        return lo + (hi - lo) * torch.rand((), device=self.device)

    def evader_step_distances(self, speed: float) -> torch.Tensor:
        dt_steps = (self.waypoint_steps[1:] - self.waypoint_steps[:-1]).to(torch.float32)
        return dt_steps * self.step_dt * float(speed)

    def first_waypoint_velocity(self, waypoints: torch.Tensor) -> torch.Tensor:
        velocity = torch.zeros_like(waypoints)
        dt = (self.waypoint_steps[1:] - self.waypoint_steps[:-1]).to(waypoints.dtype) * self.step_dt
        velocity[:-1] = (waypoints[1:] - waypoints[:-1]) / dt[:, None].clamp_min(self.step_dt)
        velocity[-1] = velocity[-2]
        moving = torch.nonzero(torch.linalg.vector_norm(velocity[:, :2], dim=-1) > 0.05).flatten()
        return velocity[int(moving[0].item()) if moving.numel() else 0]

    def fallback_pursuer_start(self, evader_start: torch.Tensor) -> torch.Tensor:
        lo = self.arena_min_safe + float(self.cfg.pursuit_pursuer_wall_clearance)
        hi = self.arena_max_safe - float(self.cfg.pursuit_pursuer_wall_clearance)
        wanted = max(float(self.cfg.pursuit_pursuer_min_evader_distance), 0.9)
        for ox, oy in ((-wanted, 0.0), (0.0, -wanted), (wanted, 0.0), (0.0, wanted)):
            pos = torch.clamp(evader_start + torch.tensor((ox, oy, 0.0), device=self.device), min=lo, max=hi)
            if float(torch.linalg.vector_norm(pos[:2] - evader_start[:2]).item()) >= float(
                self.cfg.pursuit_pursuer_min_evader_distance
            ):
                return pos
        pos = evader_start.clone()
        pos[:2] = 0.5 * (lo[:2] + hi[:2])
        return pos

    def inactive_obstacle_xy(self, slots: int) -> torch.Tensor:
        idx = torch.arange(max(0, slots), device=self.device, dtype=torch.float32)
        x = self.arena_min_safe[0] - 6.0 - 0.35 * idx
        y = torch.full_like(x, float(self.arena_min_safe[1]) - 6.0)
        return torch.stack((x, y), dim=-1)

    def inactive_dynamic_pos(self, slots: int) -> torch.Tensor:
        xy = self.inactive_obstacle_xy(slots)
        z = torch.full((xy.shape[0], 1), self.dynamic_center_z(), device=self.device)
        return torch.cat((xy, z), dim=-1)

    def dynamic_center_z(self) -> float:
        return float(self.cfg.arena_min[2] + 0.5 * self.cfg.pursuit_dynamic_obstacle_height)

    def static_grid_radius(self) -> float:
        return float(self.cfg.pillar_radius + self.cfg.pursuit_evader_radius + self.cfg.pursuit_obstacle_clearance)

    def dynamic_grid_radius(self) -> float:
        return float(
            self.cfg.pursuit_dynamic_obstacle_radius
            + self.cfg.pursuit_evader_radius
            + self.cfg.pursuit_obstacle_clearance
        )

    @staticmethod
    def grid_free_cells(grid: dict[str, Any]) -> torch.Tensor:
        return torch.nonzero(~grid["occupied"], as_tuple=False)

    @staticmethod
    def grid_cell_xy(grid: dict[str, Any], cell: tuple[int, int]) -> torch.Tensor:
        return torch.stack((grid["xs"][cell[1]], grid["ys"][cell[0]]))

    @staticmethod
    def grid_cells_xy(grid: dict[str, Any], cells: torch.Tensor) -> torch.Tensor:
        return torch.stack((grid["xs"][cells[:, 1]], grid["ys"][cells[:, 0]]), dim=-1)

    def grid_cells_to_xy(self, grid: dict[str, Any], cells: list[tuple[int, int]]) -> torch.Tensor:
        rows = torch.tensor([cell[0] for cell in cells], device=self.device)
        cols = torch.tensor([cell[1] for cell in cells], device=self.device)
        return torch.stack((grid["xs"][cols], grid["ys"][rows]), dim=-1)

    @staticmethod
    def mark_square(grid: dict[str, Any], center: torch.Tensor, side: float) -> torch.Tensor:
        half = 0.5 * max(0.0, side)
        return (torch.abs(grid["xx"] - center[0]) <= half) & (torch.abs(grid["yy"] - center[1]) <= half)

    @staticmethod
    def disc_mask(grid: dict[str, Any], center: torch.Tensor, radius: float) -> torch.Tensor:
        dx, dy = grid["xx"] - center[0], grid["yy"] - center[1]
        return dx * dx + dy * dy <= radius * radius

    @staticmethod
    def segment_mask(
        grid: dict[str, Any],
        start_xy: torch.Tensor,
        end_xy: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        points = torch.stack((grid["xx"], grid["yy"]), dim=-1)
        ab = end_xy - start_xy
        t = torch.clamp(
            torch.sum((points - start_xy) * ab, dim=-1) / torch.sum(ab * ab).clamp_min(1e-6),
            0.0,
            1.0,
        )
        closest = start_xy + t[..., None] * ab
        return torch.linalg.vector_norm(points - closest, dim=-1) <= radius

    @staticmethod
    def polyline_length(xy: torch.Tensor) -> float:
        return float(torch.sum(torch.linalg.vector_norm(xy[1:] - xy[:-1], dim=-1)).item())

    @staticmethod
    def polyline_prefix(xy: torch.Tensor, distance: float) -> tuple[torch.Tensor, torch.Tensor]:
        seg_len = torch.linalg.vector_norm(xy[1:] - xy[:-1], dim=-1).clamp_min(1e-6)
        cumulative = torch.cat((torch.zeros(1, device=xy.device), torch.cumsum(seg_len, dim=0)))
        target = torch.as_tensor(distance, device=xy.device).clamp(0.0, cumulative[-1])
        seg = int(torch.searchsorted(cumulative[1:], target).clamp(max=seg_len.shape[0] - 1).item())
        tau = (target - cumulative[seg]) / seg_len[seg]
        point = xy[seg] * (1.0 - tau) + xy[seg + 1] * tau
        before = xy[: seg + 1] if float(tau.item()) <= 1e-6 else torch.cat((xy[: seg + 1], point[None]))
        return before, point

    @staticmethod
    def polyline_suffix(xy: torch.Tensor, distance: float) -> tuple[torch.Tensor, torch.Tensor]:
        seg_len = torch.linalg.vector_norm(xy[1:] - xy[:-1], dim=-1).clamp_min(1e-6)
        cumulative = torch.cat((torch.zeros(1, device=xy.device), torch.cumsum(seg_len, dim=0)))
        target = torch.as_tensor(distance, device=xy.device).clamp(0.0, cumulative[-1])
        seg = int(torch.searchsorted(cumulative[1:], target).clamp(max=seg_len.shape[0] - 1).item())
        tau = (target - cumulative[seg]) / seg_len[seg]
        point = xy[seg] * (1.0 - tau) + xy[seg + 1] * tau
        after = xy[seg + 1 :] if float(tau.item()) >= 1.0 - 1e-6 else torch.cat((point[None], xy[seg + 1 :]))
        return point, after

    @staticmethod
    def cubic_bezier(
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        u = 1.0 - t
        return u**3 * a + 3.0 * u * u * t * b + 3.0 * u * t * t * c + t**3 * d
