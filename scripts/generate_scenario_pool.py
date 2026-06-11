#!/usr/bin/env python3
"""Generate pursuit-evasion scenario pools without constructing an environment."""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Generate cached position-tracking pursuit scenarios.")
parser.add_argument("--task", type=str, default="PosTracking-RL-velocity-v0", help="Gym task name.")
parser.add_argument("--pool-size", type=int, default=5000, help="Exact scenarios generated for each enabled phase.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, device="cpu")
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaac_pursuit_evasion.tasks.direct.pos_tracking.pos_tracking_env_cfg import (
    pos_tracking_rates_cfg,
    pos_tracking_velocity_cfg,
)
from isaac_pursuit_evasion.tasks.direct.pos_tracking.scenario_pool import ScenarioPoolBuilder


def main() -> None:
    cfg_factory = pos_tracking_rates_cfg if "rates" in args.task.lower() else pos_tracking_velocity_cfg
    cfg = cfg_factory(num_envs=1)
    cfg.enable_pursuit_evasion_curriculum = True
    cfg.enable_walls = True
    cfg.enable_pillars = True
    cfg.pursuit_scenario_pool_size = max(1, int(args.pool_size))
    cfg.sim.device = args.device

    cache_path = ScenarioPoolBuilder(cfg, args.device).build_and_save()
    print(f"[INFO] Scenario pool cache ready: {cache_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
