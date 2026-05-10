# DG-PPO Debugging Report

Last updated: 2026-05-10

This branch contains narrow DG-PPO live-training fixes for the IsaacLab port. It is intended as a migration note for
copying changes back to the production branch, not as a full run diary.

## Current Status

The user-reported recurrent update bug is confirmed and fixed.

During rollout, live policy and Vl recurrent carries are reset after `terminated | truncated`. Before this update, PPO
minibatch evaluation restarted each chunk from the stored chunk-start carry but then scanned through the whole chunk
without applying mid-chunk done resets. If an environment finished at step `r`, rollout action `r + 1` was sampled from
a reset carry, while update log-prob evaluation recomputed it from the pre-reset carried state. That can produce rare
large `old_logp` / `log_prob` mismatches with low average clip fraction but extreme `ratio_max`.

The fix keeps truncated-BPTT scanning, but applies the rollout `bT_done` mask after evaluating each recurrent step and
before feeding the carry to the next step.

## Changes To Import

Applied later on 2026-05-10:

- `source/isaac_pursuit_evasion/dgppo/dgppo_agent.py`
  - Calls `compute_dec_ocp_gae(..., bootstrap_on_truncated=False)` for both stochastic and deterministic targets.
    This treats IsaacLab truncations as rollout boundaries unless/until terminal-observation bootstrap is wired in.
  - Passes `view["bT_done"]` into `compute_cbf_advantages(...)` so CBF finite differences do not cross into reset
    episode values.

- `source/isaac_pursuit_evasion/dgppo/utils.py`
  - Adds optional `bT_done` to `compute_cbf_advantages(...)`.
  - On done transitions, uses `Vh_t` as the finite-difference endpoint instead of `Vh_{t+1}`. This preserves the
    current-state CBF penalty while preventing reset-state contamination.

- `source/isaac_pursuit_evasion/dgppo/parity_suite/test_kernel_math_parity.py`
  - Adds coverage for explicit truncation-boundary masking.
  - Adds coverage that CBF advantages do not use reset-episode `Vh_{t+1}` on done transitions.

Applied on 2026-05-10:

- `source/isaac_pursuit_evasion/dgppo/dgppo_agent.py`
  - Passes `batch.done_mask` into recurrent policy loss evaluation.
  - Passes `batch.done_mask` into recurrent Vl loss evaluation.

- `source/isaac_pursuit_evasion/dgppo/update_helpers.py`
  - Adds `done_mask` to `UpdateGraphBatch`, populated from memory's `bT_done`.
  - Threads `done_mask` into `compute_rollout_policy_loss`, `compute_value_losses`, and `compute_rollout_vl_loss`.
  - Threads chunked masks into `_evaluate_policy_chunks` and `_evaluate_vl_chunks`.
  - Adds out-of-place carry masking helpers:
    `_reset_policy_carry_after_done(...)` and `_reset_env_carry_after_done(...)`.
  - Resets recurrent carries after the current step's log-prob/value is emitted, matching rollout timing.

- `source/isaac_pursuit_evasion/dgppo/parity_suite/test_kernel_math_parity.py`
  - Adds a regression test that done-mask carry resets zero the finished env slot, preserve unfinished slots, and do not
    mutate the original autograd tensor in place.

Earlier fixes already present on this branch:

- Live rollout resets policy and Vl carries on `terminated | truncated`.
- Rollout memory stores stochastic/deterministic `terminated`, `truncated`, and `done` masks.
- GAE/target recursion masks true terminations while keeping truncation bootstrap semantics separate.
- The position-tracking task publishes signed DG-PPO costs, with an agent-side fallback adapter.
- Rollout memory stores incoming policy and Vl carries so recurrent update chunks can start from the real live context.

## Important Line References

Line numbers below are from this debugging branch after the 2026-05-10 patch:

- `dgppo_agent.py`
  - `compute_dec_ocp_gae(..., bootstrap_on_truncated=False)`: around lines 615 and 629.
  - `compute_cbf_advantages(..., bT_done=view["bT_done"])`: around line 645.
  - `compute_rollout_policy_loss(..., done_mask=batch.done_mask)`: around line 769.
  - `compute_value_losses(..., done_mask=batch.done_mask)`: around line 804.

- `utils.py`
  - `compute_cbf_advantages(..., bT_done=...)`: around line 281.
  - CBF done-boundary finite-difference masking: around lines 302-306.

- `update_helpers.py`
  - `UpdateGraphBatch.done_mask`: around line 33.
  - `build_update_graph_batch` reads `view["bT_done"]`: around lines 88 and 101.
  - `compute_rollout_policy_loss(..., done_mask=...)`: around lines 205 and 238.
  - `compute_value_losses(..., done_mask=...)`: around lines 326 and 345.
  - `compute_rollout_vl_loss(..., done_mask=...)`: around lines 397 and 410.
  - Policy scan reset after each recurrent step: around line 561.
  - Vl scan reset after each recurrent step: around line 597.
  - Out-of-place reset helpers: around lines 601 and 622.

- `test_kernel_math_parity.py`
  - Truncation-boundary target regression: around line 104.
  - CBF done-boundary regression: around line 117.
  - Recurrent update reset regression: around line 154.

## Validation

Run in this plain local shell:

- `python3 -m compileall -q source/isaac_pursuit_evasion/dgppo` passed.
- Focused pytest command did not execute the tests because the module skipped on missing runtime imports in this
  non-IsaacLab environment:
  `PYTHONPATH=$PWD/source/isaac_pursuit_evasion:$PYTHONPATH python3 -m pytest -q
  source/isaac_pursuit_evasion/dgppo/parity_suite/test_kernel_math_parity.py`
- `pre-commit run --files ...` could not run because this shell's `pre-commit` launcher cannot import the
  `pre_commit` Python package.

Recommended validation inside the IsaacLab environment:

- Focused pytest:
  `PYTHONPATH=$PWD/source/isaac_pursuit_evasion:$PYTHONPATH python -m pytest -q
  source/isaac_pursuit_evasion/dgppo/parity_suite/test_kernel_math_parity.py`
- A tiny headless DG-PPO smoke run, then inspect the first minibatch `ratio_max` / log-prob mismatch diagnostics.

## Open Items

- Re-run a real DG-PPO training window and confirm first-minibatch `ratio_max` no longer spikes on post-done samples.
- Check CBF diagnostics after the boundary fix; `cbf_deriv_max` should no longer spike solely because a done transition
  is followed by a reset observation.
- If importing into production manually, also import the earlier memory/done-mask and incoming-RNN-state fixes; this
  patch depends on `bT_done`, `bTa_rnn_states`, and `bT_vl_rnn_states` already existing in the rollout views.
