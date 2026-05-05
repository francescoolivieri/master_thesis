# JAX DG-PPO Reference Instructions

Scope: the upstream JAX reference implementation under `dgppo-main/`.

This directory is primarily a source reference for the PyTorch/skrl port in `source/isaac_pursuit_evasion/dgppo/`.

## Rules

- Do not modify reference algorithm files unless the user explicitly asks for changes in `dgppo-main`.
- Prefer reading these files to understand intended behavior, tensor shapes, schedules, and parity checkpoints.
- Keep parity artifacts and logs as generated/reference data. Do not delete or rewrite them without an explicit cleanup request.
- If exporting new fixtures, keep outputs under `dgppo-main/parity_artifacts/` and record the command used.

## Useful Entry Points

- `dgppo/algo/dgppo.py`: main DGPPO algorithm.
- `dgppo/algo/utils.py`: Dec-EFOCP GAE and lower-level update helpers.
- `dgppo/algo/module/`: policy and value network modules.
- `dgppo/nn/`: GNN, MLP, and RNN modules.
- `dgppo/trainer/`: rollout and trainer abstractions.
- `dgppo/parity/`: deterministic fixtures, comparison reports, and checkpoint manifest.
- `parity_checks.py`: CLI for exporting and comparing parity fixtures.
