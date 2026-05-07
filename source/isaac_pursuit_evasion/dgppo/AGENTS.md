# DG-PPO Port Instructions

Scope: the PyTorch/skrl DG-PPO port in this directory along with the parity suite.

## Source Of Truth

Use these JAX reference files when changing behavior:

- `dgppo-main/dgppo/algo/dgppo.py`: DGPPO update loop, deterministic rollout use, CBF advantage, schedules.
- `dgppo-main/dgppo/algo/informarl.py` and `dgppo-main/dgppo/algo/informarl_lagr.py`: inherited policy/value update structure.
- `dgppo-main/dgppo/algo/module/policy.py`: squashed Gaussian policy behavior.
- `dgppo-main/dgppo/algo/module/value.py`: centralized/decentralized value heads.
- `dgppo-main/dgppo/nn/gnn.py`, `dgppo-main/dgppo/nn/mlp.py`, `dgppo-main/dgppo/nn/rnn.py`: neural network building blocks.
- `dgppo-main/dgppo/algo/utils.py`: Dec-EFOCP GAE and kernel utilities.
- `dgppo-main/dgppo/parity/` and `dgppo-main/parity_artifacts`: fixture and tolerance tooling.

## Porting Priorities

- Preserve algorithm semantics before refactoring for elegance.
- Keep tensor shape names explicit in comments and variable names. Existing code uses `B`/`T`/`A`/`NH` and `bTah_*` style names; continue that convention.
- Match the reference update structure: stochastic rollout for policy/Vl, deterministic rollout for Vh targets, CBF-derived advantages for policy updates.
- Keep the deterministic/stochastic environment split clear. Current code requires an even `num_envs >= 2`.
- Treat RNN support as a first-class path even when `use_rnn: false` is the current config default. Avoid changes that make RNN parity harder.
- Preserve runtime episode boundaries. Rollout memory must keep `terminated | truncated` masks,
  GAE must not bootstrap through done transitions, and recurrent carries must be reset for done envs.
- Do not silently change sign conventions: rewards are converted to costs as `l = -reward`, and final policy advantages are negated before PPO.
- Keep graph construction compatible with the IsaacLab observation layout in `pos_tracking_env.py`.
- Keep constraint costs compatible with DG-PPO's `[E, A, H]` convention. If a task has obstacles
  or safety heads, do not fall back to zero costs unless the task truly has no active constraints.

## Implementation Notes

- `dgppo_runner.py` adapts the IsaacLab/skrl environment to `DGPPOAgent`.
- `dgppo_agent.py` owns rollout collection, cost extraction, done handling, bootstrapping, PPO updates, and skrl compatibility.
- `dgppo_memory.py` stores stochastic and deterministic rollout splits, including done masks.
- `dgppo_models.py` owns policy/value modules and the squashed Gaussian distribution.
- `utils.py` owns graph data structures, GNN layers, done-aware GAE, CBF advantage, and PPO surrogate helpers.

## Validation

- For parity-sensitive changes, run `source/isaac_pursuit_evasion/dgppo/parity_suite` first; these tests must call production DG-PPO code (no copied test-only PPO/GAE/loss logic).
- Use `dgppo-main/parity_artifacts` as the default oracle fixtures. Export fresh fixtures with `dgppo-main/parity_checks.py` when JAX reference behavior changes or when adding new coverage (for example `num_envs = 6`).
- Treat `num_envs=6` single-update parity and the `drift_num_envs6` multi-update replay as the current executable multi-env gates. The drift artifact must include per-update replay inputs/checkpoints, not just scalar summaries.
- For runtime integration changes, run the shortest practical DG-PPO headless smoke test through `scripts/skrl/train.py` after parity checks pass.
- Add focused tests for PyTorch/IsaacLab runtime contracts that are not represented in JAX parity
  fixtures, such as done-mask bootstrapping or task-specific cost extraction.
- If a full IsaacLab or JAX dependency stack is unavailable, document exactly which validation could not run.
