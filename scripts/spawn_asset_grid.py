#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn a grid of quadrotor assets to profile load/startup time.

.. code-block:: bash

    ./isaaclab.sh -p scripts/spawn_asset_grid.py --asset crazyflie_brushless --num-envs 512 --headless
"""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Spawn a grid of quadrotor assets.")
parser.add_argument(
    "--asset",
    type=str,
    default="crazyflie_brushless",
    choices=["crazyflie_brushless", "vaporx5", "crazyflie_brushed"],
    help="Which asset to spawn.",
)
parser.add_argument("--num-envs", type=int, default=512, help="Number of prims to spawn.")
parser.add_argument("--spacing", type=float, default=2.0, help="Grid spacing between prims.")
parser.add_argument("--root-path", type=str, default="/World/Robots", help="Root prim path prefix.")
parser.add_argument(
    "--collision-approximation",
    type=str,
    default="default",
    choices=["default", "convex_hull", "convex_decomposition", "triangle_mesh_simplification"],
    help="Override mesh collision approximation for all robots.",
)
parser.add_argument(
    "--disable-collision-all",
    action="store_true",
    help="Disable collision on all meshes under each robot (brutal test).",
)
parser.add_argument(
    "--collision-log-stats",
    action="store_true",
    help="Log mesh/collision counts for a sample robot.",
)
parser.add_argument(
    "--force-uninstanceable",
    action="store_true",
    help="Make robot prims uninstanceable before collision edits (needed for instanceable assets).",
)
parser.add_argument(
    "--copy-from-source",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Clone Xform prims as copies instead of references.",
)
parser.add_argument(
    "--replicate-physics",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Ask the cloner to replicate physics when cloning Xform prims.",
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=0,
    help="If >0, step this many frames then exit. If 0, keep running until the app closes.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni
from isaacsim.core.cloner import GridCloner

import isaaclab.sim as sim_utils
from isaaclab.sim import schemas as sim_schemas
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext


def _resolve_robot_cfg(asset_name: str):
    if asset_name == "crazyflie_brushless":
        from source.isaac_pursuit_evasion.assets.crazyflie_brushless import CrazyflieBrushlessPursuer, CrazyflieBrushlessEvader

        # return CrazyflieBrushlessPursuer
        return CrazyflieBrushlessEvader
    
    if asset_name == "vaporx5":
        from source.isaac_pursuit_evasion.assets.vaporX5 import VaporX5

        return VaporX5
    if asset_name == "crazyflie_brushed":
        from isaaclab_assets import CRAZYFLIE_CFG

        return CRAZYFLIE_CFG
    raise ValueError(f"Unsupported asset '{asset_name}'.")


def _format_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:.2f} ms"


def _collect_collision_stats(prim_path: str) -> tuple[int, int, int]:
    from isaaclab.sim.utils import get_all_matching_child_prims
    from pxr import UsdGeom, UsdPhysics

    prims = get_all_matching_child_prims(prim_path)
    mesh_prims = [prim for prim in prims if prim.IsA(UsdGeom.Mesh)]
    collision_meshes = [prim for prim in mesh_prims if UsdPhysics.CollisionAPI(prim)]
    return len(prims), len(mesh_prims), len(collision_meshes)


def _log_collision_stats(sample_path: str, label: str, total_envs: int) -> None:
    try:
        prim_count, mesh_count, collision_count = _collect_collision_stats(sample_path)
    except Exception as exc:
        print(f"[WARN] Collision stats unavailable ({label}): {exc}")
        return
    print(
        f"[INFO] Collision stats ({label}, sample 1/{total_envs}): "
        f"prims={prim_count}, meshes={mesh_count}, collision_meshes={collision_count}"
    )


def _apply_collision_approximation(robot_paths: list[str], mode: str) -> int:
    if mode == "default":
        return 0

    if mode == "convex_hull":
        mesh_cfg = sim_schemas.ConvexHullPropertiesCfg()
    elif mode == "convex_decomposition":
        mesh_cfg = sim_schemas.ConvexDecompositionPropertiesCfg()
    elif mode == "triangle_mesh_simplification":
        mesh_cfg = sim_schemas.TriangleMeshSimplificationPropertiesCfg()
    else:
        raise ValueError(f"Unsupported collision approximation '{mode}'.")

    from isaaclab.sim.utils import get_all_matching_child_prims
    from pxr import UsdGeom, UsdPhysics

    total_meshes = 0
    for path in robot_paths:
        collision_meshes = get_all_matching_child_prims(
            path,
            predicate=lambda prim: prim.IsA(UsdGeom.Mesh) and UsdPhysics.CollisionAPI(prim),
        )
        for prim in collision_meshes:
            sim_schemas.define_mesh_collision_properties(prim.GetPath().pathString, mesh_cfg)
        total_meshes += len(collision_meshes)
    return total_meshes


def _disable_collisions(robot_paths: list[str]) -> None:
    cfg = sim_schemas.CollisionPropertiesCfg(collision_enabled=False)
    for path in robot_paths:
        sim_schemas.modify_collision_properties(path, cfg)


def main() -> None:
    if args_cli.num_envs < 1:
        raise ValueError("--num-envs must be >= 1.")

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([4.0, 3.0, 2.5], [0.0, 0.0, 1.0])

    # Basic scene
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    root_path = args_cli.root_path.rstrip("/")
    cloner = GridCloner(spacing=args_cli.spacing)
    target_paths = cloner.generate_paths(root_path, args_cli.num_envs)

    stage = omni.usd.get_context().get_stage()
    stage.DefinePrim(target_paths[0], "Xform")

    t_clone_start = time.perf_counter()
    cloner.clone(
        source_prim_path=target_paths[0],
        prim_paths=target_paths,
        replicate_physics=args_cli.replicate_physics,
        copy_from_source=args_cli.copy_from_source,
    )
    t_clone_end = time.perf_counter()

    robot_cfg = _resolve_robot_cfg(args_cli.asset).replace(prim_path=f"{root_path}_.*/Robot")
    t_robot_start = time.perf_counter()
    robot = Articulation(robot_cfg)  # noqa: F841
    t_robot_end = time.perf_counter()

    robot_paths = [f"{path}/Robot" for path in target_paths]
    if args_cli.force_uninstanceable:
        from isaaclab.sim.utils import make_uninstanceable

        t_uninst_start = time.perf_counter()
        for path in robot_paths:
            make_uninstanceable(path)
        t_uninst_end = time.perf_counter()
        print(
            f"[INFO] Made {len(robot_paths)} robots uninstanceable in "
            f"{_format_ms(t_uninst_end - t_uninst_start)}"
        )

    if args_cli.collision_log_stats:
        _log_collision_stats(robot_paths[0], "before", len(robot_paths))

    if args_cli.disable_collision_all:
        t_disable_start = time.perf_counter()
        _disable_collisions(robot_paths)
        t_disable_end = time.perf_counter()
        print(
            f"[INFO] Collision disabled for {len(robot_paths)} robots in "
            f"{_format_ms(t_disable_end - t_disable_start)}"
        )
        if args_cli.collision_log_stats:
            _log_collision_stats(robot_paths[0], "after_disable", len(robot_paths))

    if args_cli.collision_approximation != "default":
        t_approx_start = time.perf_counter()
        meshes_updated = _apply_collision_approximation(robot_paths, args_cli.collision_approximation)
        t_approx_end = time.perf_counter()
        print(
            f"[INFO] Collision approximation '{args_cli.collision_approximation}' applied to "
            f"{meshes_updated} meshes in {_format_ms(t_approx_end - t_approx_start)}"
        )
        if args_cli.collision_log_stats:
            _log_collision_stats(robot_paths[0], "after_approx", len(robot_paths))

    t_reset_start = time.perf_counter()
    sim.reset()
    t_reset_end = time.perf_counter()

    t_step_start = time.perf_counter()
    sim.step()
    t_step_end = time.perf_counter()

    print(
        "[INFO] Spawn timings: "
        f"clone_xforms={_format_ms(t_clone_end - t_clone_start)}, "
        f"articulation_init={_format_ms(t_robot_end - t_robot_start)}, "
        f"sim_reset={_format_ms(t_reset_end - t_reset_start)}, "
        f"first_step={_format_ms(t_step_end - t_step_start)}"
    )
    print(
        "[INFO] Settings: "
        f"asset={args_cli.asset}, "
        f"num_envs={args_cli.num_envs}, "
        f"spacing={args_cli.spacing}, "
        f"copy_from_source={args_cli.copy_from_source}, "
        f"replicate_physics={args_cli.replicate_physics}"
    )

    sim_dt = sim.get_physics_dt()
    steps_remaining = args_cli.max_steps
    while simulation_app.is_running():
        sim.step()
        robot.update(sim_dt)
        if steps_remaining > 0:
            steps_remaining -= 1
            if steps_remaining == 0:
                break

    simulation_app.close()


if __name__ == "__main__":
    main()
