# Isaac Pursuit Evasion Package Instructions

Scope: everything under `source/isaac_pursuit_evasion/`.

## Package Map

- `isaac_pursuit_evasion/tasks/direct/pos_tracking/`: IsaacLab DirectRLEnv task for Crazyflie position tracking.
- `isaac_pursuit_evasion/tasks/direct/pos_tracking/agents/`: Hydra/skrl agent configs, including `dgppo_cfg.yaml` and `skrl_ppo_cfg.yaml`.
- `dgppo/`: in-progress PyTorch/skrl DG-PPO implementation and parity suite.
- `controllers/`: Crazyflie controllers and RL action wrappers.
- `dynamics/`: propeller and drone dynamics helpers.
- `assets/`: Crazyflie asset definitions.
- `deployment/`: policy loading and deployment helpers.

## IsaacLab Conventions

- Preserve IsaacLab entrypoint registration and config patterns. Task configs live beside their environment implementation.
- Keep simulator tensors on `self.device`; avoid CPU round-trips in environment step, reward, observation, and reset code.
- Prefer vectorized Torch operations over per-environment Python loops in IsaacLab runtime paths.
- Keep observation layout changes explicit. If changing the flat policy observation, update any matching layout metadata and DG-PPO graph parsing code.
- Avoid importing heavy Isaac/Omniverse modules in files that should remain lightweight utilities.

## Training Configs

- Keep `agents/dgppo_cfg.yaml` aligned with `DGPPOAgent.load_dgppo_hyperparameters()` and `DGPPORunner.__init__()`.
- When changing task IDs, config entrypoints, or environment registration, also update README/docs references.

## Validation

- For task/environment changes, prefer a short headless run with small `--num_envs`, IsaacLab/IsaacSim environment can be activated using the "s_isaac" alias.
- For config-only changes, validate YAML syntax and check that referenced config keys are consumed correctly by code.
