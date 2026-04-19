#!/usr/bin/env python3
"""Capture a single arena screenshot from a task without training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Capture a single screenshot from an Isaac Lab task.")
parser.add_argument("--task", type=str, default="PosTracking-RL-velocity-v0", help="Gym task name.")
parser.add_argument("--num-envs", type=int, default=1, help="Number of environments to instantiate.")
parser.add_argument("--steps", type=int, default=2, help="Warmup steps before capturing.")
parser.add_argument("--output", type=Path, default=Path("logs/pos_tracking/arena_screenshot.png"), help="Output PNG path.")

AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli, hydra_args = parser.parse_known_args()

# hydra_task_config expects command-line overrides in sys.argv.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab.sensors.camera.utils import save_images_to_file
from isaaclab_tasks.utils.hydra import hydra_task_config
import source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pos_tracking  # noqa: F401


def _to_hwc_float(frame) -> torch.Tensor:
    """Convert rendered frame to [H, W, C] float tensor in [0, 1]."""
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
    return image.clamp(0.0, 1.0)


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, _agent_cfg) -> None:
    env_cfg.scene.num_envs = int(max(1, args_cli.num_envs))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    try:
        obs, _ = env.reset()
        del obs

        action_dim = int(getattr(env.unwrapped.cfg, "action_space", 0))
        if action_dim > 0:
            action = torch.zeros((env_cfg.scene.num_envs, action_dim), device=env.unwrapped.device)
            for _ in range(max(0, int(args_cli.steps))):
                env.step(action)

        frame = env.render()
        image = _to_hwc_float(frame).unsqueeze(0).cpu()

        output_path = args_cli.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_images_to_file(image, str(output_path))
        print(f"[INFO] Saved screenshot to: {output_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
