#!/usr/bin/env python3
"""Estimate drone collision radius from spawned geometry (AABB)."""
from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Estimate XY drone collision radius from world AABB.")
parser.add_argument("--task", type=str, default="PosTracking-RL-velocity-v0", help="Gym task name.")
parser.add_argument("--num-envs", type=int, default=1, help="Number of envs to instantiate (use 1 for this tool).")
parser.add_argument(
    "--env-index",
    type=int,
    default=0,
    help="Environment index to inspect (prim path /World/envs/env_<idx>/Robot).",
)

AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli, hydra_args = parser.parse_known_args()

# hydra_task_config expects command-line overrides in sys.argv.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_tasks.utils.hydra import hydra_task_config
import omni.usd
import source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pos_tracking  # noqa: F401

try:
    from pxr import Usd, UsdGeom
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Module 'pxr' is unavailable in this Python environment. "
        "Run this script with IsaacLab's launcher, e.g. "
        "'/home/francesco/isaaclab/IsaacLab/isaaclab.sh -p scripts/estimate_drone_radius.py --headless'."
    ) from exc


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, _agent_cfg) -> None:
    env_cfg.scene.num_envs = max(1, int(args_cli.num_envs))
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    try:
        env.reset()

        env_index = int(args_cli.env_index)
        robot_path = f"/World/envs/env_{env_index}/Robot"

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(robot_path)
        if not prim.IsValid():
            raise RuntimeError(f"Robot prim not found at '{robot_path}'. Try --env-index 0 and --num-envs 1.")

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
        world_bound = bbox_cache.ComputeWorldBound(prim)
        box = world_bound.GetBox()

        bbox_min = box.GetMin()
        bbox_max = box.GetMax()
        dx = float(bbox_max[0] - bbox_min[0])
        dy = float(bbox_max[1] - bbox_min[1])
        dz = float(bbox_max[2] - bbox_min[2])

        radius_half_max = 0.5 * max(dx, dy)
        radius_half_diag = 0.5 * math.sqrt(dx * dx + dy * dy)

        print(f"[INFO] Robot prim: {robot_path}")
        print(f"[INFO] AABB size (m): dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")
        print(f"[INFO] Suggested drone_collision_radius (half max XY): {radius_half_max:.6f}")
        print(f"[INFO] Conservative radius (half XY diagonal):   {radius_half_diag:.6f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
