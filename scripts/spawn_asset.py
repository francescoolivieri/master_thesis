#!/usr/bin/env python3
"""Spawn the Crazyflie Brushless pursuer/evader and spin the propeller joints."""

from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher


def _find_prop_joints(drone) -> list[int]:
    import re

    joint_ids, joint_names = drone.find_joints(
        ["revolute_prop_.*"],
        preserve_order=True,
    )
    if not joint_ids:
        joint_ids, joint_names = drone.find_joints(".*prop.*", preserve_order=True)
    if not joint_ids:
        print("[WARN] No propeller joints matched. Falling back to all joints.")
        joint_names = drone.joint_names
        joint_ids = list(range(len(joint_names)))
    indexed = []
    for joint_id, joint_name in zip(joint_ids, joint_names):
        match = re.search(r"(\d+)$", joint_name)
        if match:
            indexed.append((int(match.group(1)), joint_id, joint_name))
    if indexed:
        indexed.sort(key=lambda item: item[0])
        joint_ids = [item[1] for item in indexed]
        joint_names = [item[2] for item in indexed]
    print(f"[INFO] Propeller joints: {joint_names}")
    return joint_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Spawn Crazyflie Brushless pursuer/evader and spin props.")
    parser.add_argument("--drone_height", type=float, default=2.0, help="Spawn height (m).")
    parser.add_argument("--prop-omega", type=float, default=200.0, help="Propeller angular velocity (rad/s).")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.sensors import Camera
    from isaaclab.sim import SimulationCfg
    from source.isaac_pursuit_evasion.assets.crazyflie_brushless import (
        CrazyflieBrushlessEvader,
        CrazyflieBrushlessPursuer,
        fpv_camera_cfg,
    )

    sim_cfg = SimulationCfg(dt=0.01, device=args.device, render_interval=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[4.0, 2.5, 2.0], target=[0.0, 0.0, 1.0])

    ground_cfg = sim_utils.GroundPlaneCfg(color=(0.2, 0.2, 0.2))
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
    light_cfg.func("/World/Light", light_cfg)

    pursuer_cfg = CrazyflieBrushlessPursuer.replace(prim_path="/World/Pursuer")
    evader_cfg = CrazyflieBrushlessEvader.replace(prim_path="/World/Evader")
    pursuer = Articulation(pursuer_cfg) 
    evader = Articulation(evader_cfg)
    camera = Camera(fpv_camera_cfg(prim_path="/World/Pursuer"))

    sim.reset()
    pursuer.reset()
    evader.reset()

    pursuer_state = pursuer.data.default_root_state.clone()
    evader_state = evader.data.default_root_state.clone()
    pursuer_state[:, :2] = torch.tensor([-0.6, 0.0], device=args.device)
    evader_state[:, :2] = torch.tensor([0.6, 0.0], device=args.device)
    pursuer_state[:, 2] = args.drone_height
    evader_state[:, 2] = args.drone_height
    pursuer.write_root_pose_to_sim(pursuer_state[:, :7])
    pursuer.write_root_velocity_to_sim(pursuer_state[:, 7:])
    evader.write_root_pose_to_sim(evader_state[:, :7])
    evader.write_root_velocity_to_sim(evader_state[:, 7:])

    pursuer_prop_joint_ids = _find_prop_joints(pursuer)
    evader_prop_joint_ids = _find_prop_joints(evader)
    pursuer_omega = torch.full((1, len(pursuer_prop_joint_ids)), args.prop_omega, device=args.device)
    evader_omega = torch.full((1, len(evader_prop_joint_ids)), args.prop_omega, device=args.device)
    if len(pursuer_prop_joint_ids) > 1:
        pursuer_omega[:, 0::2] *= -1.0
    if len(evader_prop_joint_ids) > 1:
        evader_omega[:, 0::2] *= -1.0

    while simulation_app.is_running():
        pursuer.write_root_pose_to_sim(pursuer_state[:, :7])
        pursuer.write_root_velocity_to_sim(pursuer_state[:, 7:])
        evader.write_root_pose_to_sim(evader_state[:, :7])
        evader.write_root_velocity_to_sim(evader_state[:, 7:])
        pursuer.write_joint_velocity_to_sim(pursuer_omega, joint_ids=pursuer_prop_joint_ids)
        evader.write_joint_velocity_to_sim(evader_omega, joint_ids=evader_prop_joint_ids)
        sim.step()
        dt = sim.get_physics_dt()
        pursuer.update(dt)
        evader.update(dt)
        camera.update(dt, force_recompute=False)

    simulation_app.close()


if __name__ == "__main__":
    main()
