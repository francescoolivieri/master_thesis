#!/usr/bin/env python3
"""Capture one screenshot of the pos-tracking arena (training-equivalent setup)."""
from __future__ import annotations

from pathlib import Path

from isaaclab.app import AppLauncher

TASK_NAME = "PosTracking-RL-velocity-v0"
NUM_ENVS = 1
WARMUP_STEPS = 20
RENDER_ATTEMPTS = 40
MIN_BRIGHTNESS = 0.03
OUTPUT_PATH = Path("logs/pos_tracking/arena_posttraining.png")

# Keep script usage simple: no custom CLI arguments.
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

# Workaround for omni.replicator overscan bug where dataWindow values can be None.
import carb as _carb

_carb.settings.get_settings().set("/rtx/dataWindow/fitOutputToDataWindow", True)

import gymnasium as gym
import torch
from isaaclab.sensors.camera.utils import save_images_to_file
import source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pos_tracking  # noqa: F401
from source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pos_tracking.pos_tracking_env_cfg import (
    pos_tracking_velocity_cfg,
)


def _to_hwc_float(frame) -> torch.Tensor:
    image = torch.as_tensor(frame)
    if image.dim() == 4:
        image = image[0]
    if image.dim() != 3:
        raise RuntimeError(f"Unexpected frame shape: {tuple(image.shape)}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = image.permute(1, 2, 0)
    image = image.to(torch.float32)
    if image.max() > 1.0:
        image = image / 255.0
    image = image.clamp(0.0, 1.0)
    if image.shape[-1] == 4:
        image = image[..., :3]
    elif image.shape[-1] == 1:
        image = image.repeat(1, 1, 3)
    elif image.shape[-1] != 3:
        raise RuntimeError(f"Unexpected channel count in frame: {image.shape[-1]}")
    return image


def main() -> None:
    env_cfg = pos_tracking_velocity_cfg(num_envs=NUM_ENVS)
    env_cfg.enable_cameras = True
    env_cfg.debug_vis = True
    env_cfg.debug_visualizer = True

    env = gym.make(TASK_NAME, cfg=env_cfg, render_mode="rgb_array")
    try:
        env.reset()
        base_env = env.unwrapped
        action_dim = int(getattr(base_env.cfg, "action_space", 0))

        action = None
        if action_dim > 0:
            action = torch.zeros((NUM_ENVS, action_dim), device=base_env.device)
            for _ in range(WARMUP_STEPS):
                env.step(action)

        image = None
        brightness = 0.0
        for _ in range(RENDER_ATTEMPTS):
            frame = env.render()
            image = _to_hwc_float(frame)
            brightness = float(image.mean().item())
            if brightness >= MIN_BRIGHTNESS:
                break
            if action is not None:
                env.step(action)

        if image is None:
            raise RuntimeError("No frame captured from viewport render.")
        if brightness < MIN_BRIGHTNESS:
            print(f"[WARN] Dark frame captured (brightness={brightness:.4f}).")

        out_path = OUTPUT_PATH.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_images_to_file(image.unsqueeze(0).cpu(), str(out_path))
        print(f"[INFO] Saved screenshot to: {out_path} (brightness={brightness:.4f})")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
