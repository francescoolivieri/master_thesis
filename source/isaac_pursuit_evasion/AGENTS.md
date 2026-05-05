# Isaac Pursuit Evasion Package Instructions

Scope: everything under `source/isaac_pursuit_evasion/`.

## Package Map

- `isaac_pursuit_evasion/tasks/direct/pos_tracking/`: IsaacLab DirectRLEnv task for Crazyflie position tracking.
- `isaac_pursuit_evasion/tasks/direct/pos_tracking/agents/`: Hydra/skrl agent configs, including `dgppo_cfg.yaml`.
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
- Before documenting or running DG-PPO training, confirm that the task registration exposes a DG-PPO Hydra agent entrypoint; the config file may exist before the CLI wiring is complete.
- Preserve user-facing CLI override compatibility in `scripts/skrl/train.py`.
- When changing task IDs, config entrypoints, or environment registration, also update README/docs references.

## Validation

- For task/environment changes, prefer a short headless run with small `--num_envs` if IsaacLab is available.
- For controller or dynamics changes, add a focused tensor-level sanity check where practical before running the simulator.
- For config-only changes, validate YAML syntax and check that referenced config keys are consumed by code.
