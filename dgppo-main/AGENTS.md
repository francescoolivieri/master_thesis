# JAX DG-PPO Reference Instructions

Scope: the upstream JAX reference implementation under `dgppo-main/`.

This directory is primarily a source reference for the PyTorch/skrl port in `source/isaac_pursuit_evasion/dgppo/`.

## Rules

- Do not modify reference algorithm files unless the user explicitly asks for changes in `dgppo-main`.
- Prefer reading these files to understand intended behavior, tensor shapes, schedules, and algorithm structure.
- Keep existing artifacts and logs as generated/reference data. Do not delete or rewrite them without an explicit cleanup request.
- Treat parity fixtures as optional reference material, not as the default validation strategy for the PyTorch port.
- If exporting new fixtures or reports, keep outputs under `dgppo-main/parity_artifacts/` and record the command used.
- Keep notes and helper code simple, direct, and easy to read.

## Useful Entry Points

- `dgppo/algo/dgppo.py`: main DGPPO algorithm.
- `dgppo/algo/utils.py`: Dec-EFOCP GAE and lower-level update helpers.
- `dgppo/algo/module/`: policy and value network modules.
- `dgppo/nn/`: GNN, MLP, and RNN modules.
- `dgppo/trainer/`: rollout and trainer abstractions.
- `dgppo/parity/`: optional deterministic fixtures, comparison reports, and checkpoint manifest.
- `parity_checks.py`: optional CLI for exporting and comparing parity fixtures.
