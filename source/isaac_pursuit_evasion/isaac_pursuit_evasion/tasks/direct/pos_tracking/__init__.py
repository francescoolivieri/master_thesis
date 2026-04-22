"""Task environments for single-drone position tracking."""

import gymnasium as gym

from . import agents


gym.register(
    id="PosTracking-v0",
    entry_point=f"{__name__}.pos_tracking_env:PosTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos_tracking_env_cfg:pos_tracking_velocity_cfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="PosTracking-RL-velocity-v0",
    entry_point=f"{__name__}.pos_tracking_env:PosTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos_tracking_env_cfg:pos_tracking_velocity_cfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="PosTracking-RL-rates-v0",
    entry_point=f"{__name__}.pos_tracking_env:PosTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos_tracking_env_cfg:pos_tracking_rates_cfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# DGPPO variant of PosTracking-RL-rates-v0: same env class & cfg; only the
# agent config differs. The DGPPO agent reads graph / constraint data via
# the side-channel methods on PosTrackingEnv.
gym.register(
    id="PosTracking-RL-rates-DGPPO-v0",
    entry_point=f"{__name__}.pos_tracking_env:PosTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pos_tracking_env_cfg:pos_tracking_rates_cfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_dgppo_cfg_entry_point": f"{agents.__name__}:skrl_dgppo_cfg.yaml",
    },
)
