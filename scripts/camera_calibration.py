#!/usr/bin/env python3
"""Interactive FPV camera calibration and visualization for the VaporX5 drone.

Features:
- Spawn a single VaporX5 with either a pinhole or fisheye camera attached.
- Keyboard teleop (W/S forward/back, A/D strafe, arrow up/down for vertical, arrow left/right for yaw rate).
- Simple hover stabilization via Lee position controller.
- Frustum visualization (debug-draw pyramid) to inspect configured FOV.
- Scene with colored props for visual reference.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import carb
import omni
import torch

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="FPV camera calibration / visualization for VaporX5.")
parser.add_argument("--hover-height", type=float, default=2.0, help="Initial hover height (m).")
parser.add_argument("--enable-frustum", action="store_true", help="Draw frustum pyramid for the active camera.")
parser.add_argument("--scene-props", type=int, default=12, help="Number of colored props to scatter in front.")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationCfg
from source.isaac_pursuit_evasion.assets.vaporX5 import (
    VaporX5,
    fpv_camera_cfg,
    fpv_camera_center_line,
    transform_camera_line,
)
from source.isaac_pursuit_evasion.controllers.lee_controller import LeePositionController
from source.isaac_pursuit_evasion.dynamics.propellers import Drone_cfg, Propellers
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_mul
from source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pursuit_evasion.tools.frustum_viz import FrustumVisualizer

# ---------------------------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------------------------- #
def _spawn_scene(sim, cam_cfg: CameraCfg, num_props: int):
    """Spawn terrain, lights, drone, camera, and props."""
    # lights and plane
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
    light.func("/World/Light", light)
    ground = sim_utils.GroundPlaneCfg(color=(0.2, 0.2, 0.2))
    ground.func("/World/ground", ground, translation=(0.0, 0.0, 0.0))

    # drone
    drone_cfg: ArticulationCfg = VaporX5.replace(prim_path="/World/Drone")
    drone = Articulation(drone_cfg)

    # camera
    camera = Camera(cam_cfg)
    # optional tiny visual marker for the camera frame
    cam_vis_cfg = sim_utils.CuboidCfg(
        size=(0.02, 0.02, 0.02),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)),
    )
    cam_vis_cfg.func(f"{cam_cfg.prim_path}/visual", cam_vis_cfg)

    # props: colored cubes in a grid
    for idx in range(num_props):
        x = 3.0 + 0.6 * (idx % 6)
        y = -1.5 + 0.6 * (idx // 6)
        color = (0.2 + 0.13 * idx, 0.4 + 0.07 * idx, 0.9 - 0.05 * idx)
        cube_cfg = sim_utils.CuboidCfg(
            size=(0.25, 0.25, 0.25),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )
        cube_cfg.func(f"/World/Props/prop_{idx}", cube_cfg, translation=(x, y, 0.125))

    return drone, camera


class KeyboardTeleop:
    """Simple keyboard interface mapping keys to velocity + yaw-rate commands."""

    def __init__(self, device: str):
        self.device = device
        self.cmd = torch.zeros(1, 4, device=device)  # vx, vy, vz, yaw_rate
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)

    def _on_event(self, event):
        if event.type not in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_RELEASE):
            return
        val = 1.0 if event.type == carb.input.KeyboardEventType.KEY_PRESS else 0.0
        name = event.input.name.upper()
        mapping = {
            "W": (0, 1.0),
            "S": (0, -1.0),
            "A": (1, 1.0),
            "D": (1, -1.0),
            "UP": (2, 1.0),
            "DOWN": (2, -1.0),
            "LEFT": (3, 0.6),
            "RIGHT": (3, -0.6),
        }
        if name in mapping:
            idx, sign = mapping[name]
            self.cmd[0, idx] = sign * val

    def command(self) -> torch.Tensor:
        return self.cmd.clone()


# ---------------------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------------------- #
def main():
    device = args.device
    sim_cfg = SimulationCfg(dt=0.01, device=device, render_interval=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[6.0, 0.0, 3.0], target=[0.0, 0.0, 1.0])

    cam_cfg = fpv_camera_cfg()
    drone, camera = _spawn_scene(sim, cam_cfg, num_props=args.scene_props)

    sim.reset()

    # Articulation data is available after reset
    drone.reset()
    init_state = drone.data.default_root_state.clone()
    init_state[0, 2] = args.hover_height
    drone.write_root_pose_to_sim(init_state[:, :7])
    drone.write_root_velocity_to_sim(init_state[:, 7:])

    drone_body_id = drone.find_bodies("body")[0]
    drone_cfg = Drone_cfg("vaporX5", device=device)
    propellers = Propellers(1, drone_cfg, sim_cfg.dt, use=True, device=device)
    lee = LeePositionController(1, drone_cfg, device=device)

    teleop = KeyboardTeleop(device=device)
    frustum = FrustumVisualizer(args.enable_frustum, cam_cfg, device=device)
    cam_origin, cam_line = fpv_camera_center_line(length=5.0, device=device)
    offset_pos = torch.tensor(cam_cfg.offset.pos, device=device, dtype=torch.float32).view(1, 3)
    offset_quat = torch.tensor(cam_cfg.offset.rot, device=device, dtype=torch.float32).view(1, 4)

    target_pos = init_state[:, :3].clone().to(device)
    target_yaw = torch.zeros(1, 1, device=device)

    print("[INFO] Controls: W/S fwd/back, A/D strafe, Up/Down ascend/descend, Left/Right yaw, ESC to quit.")

    while simulation_app.is_running():
        # commands
        cmd = teleop.command()
        dt = sim.get_physics_dt()
        # body-frame forward aligned with world +X at yaw = 0
        yaw = target_yaw.item()
        R_yaw = torch.tensor(
            [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        )
        lin_cmd_world = (R_yaw @ cmd[0, :3].view(3, 1)).squeeze(-1)
        target_pos += lin_cmd_world.view(1, 3) * dt
        target_yaw += cmd[0, 3].view(1, 1) * dt

        root_state = drone.data.root_state_w
        omega_ref, _, _ = lee(root_state, 
                              target_pos=target_pos, 
                              target_yaw=target_yaw, 
                              target_yaw_rate=cmd[:, 3:4]
                              )
        propellers.compute_omega(omega_ref)
        vel_body = drone.data.root_lin_vel_b
        state_stub = torch.zeros(1, 6, device=device)
        state_stub[:, 3:6] = vel_body
        thrust, moment = propellers.compute_force_and_torque(state_stub)
        drone.set_external_force_and_torque(thrust, moment, body_ids=drone_body_id)
        drone.write_data_to_sim()

        sim.step()
        drone.update(dt)
        camera.update(dt, force_recompute=False)

        # frustum
        body_quat = drone.data.root_quat_w
        cam_pos = drone.data.root_pos_w + quat_apply(body_quat, offset_pos)
        cam_quat = quat_mul(body_quat, offset_quat)
        frustum.draw(cam_pos[0], cam_quat[0])
        start_w, end_w, _, _ = transform_camera_line(cam_origin, cam_line, drone.data.root_pos_w, drone.data.root_quat_w)
        if frustum.debug:
            frustum.debug.draw_lines(
                [start_w[0].detach().cpu().tolist()],
                [end_w[0].detach().cpu().tolist()],
                [(0.0, 0.0, 0.0, 1.0)],
                [2.5],
            )

    simulation_app.close()


if __name__ == "__main__":
    main()
