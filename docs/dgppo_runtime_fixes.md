# DG-PPO Runtime Fixes Report

This note documents the DG-PPO issues found during the IsaacLab integration
debugging pass, what symptoms they can cause, and how they were fixed. The main
lesson is that the DG-PPO parity tests can pass while the live IsaacLab training
loop still has runtime integration bugs.

## Context

The training symptom was unstable position-tracking reward. In test runs, the
position reward mostly oscillated instead of showing a clear positive trend.
The DG-PPO parity suite passed, so the first suspicion was not the isolated math
kernels, but the connection between the algorithm and the IsaacLab environment.

The fixes touched the DG-PPO rollout/update path and the position-tracking task:

- `source/isaac_pursuit_evasion/dgppo/dgppo_agent.py`
- `source/isaac_pursuit_evasion/dgppo/dgppo_memory.py`
- `source/isaac_pursuit_evasion/dgppo/utils.py`
- `source/isaac_pursuit_evasion/isaac_pursuit_evasion/tasks/direct/pos_tracking/pos_tracking_env.py`
- `source/isaac_pursuit_evasion/dgppo/parity_suite/test_kernel_math_parity.py`

## Problem 1: GAE Crossed Episode Boundaries

DG-PPO was computing `Ql` and `Qh` targets as if every rollout was one
continuous trajectory. In IsaacLab, individual environments can finish inside a
rollout because of success, crash, out-of-bounds, pillar collision, invalid
state, or timeout. After that, IsaacLab resets that environment and starts a new
episode.

Before the fix, `compute_dec_ocp_gae(...)` did not know which transitions ended
an episode. This allowed value targets from the new episode after reset to
bootstrap backwards into the previous episode. That corrupts the learning target:
the state after reset is not the future of the state before reset.

This issue is easy to miss in parity tests because the exported JAX fixtures are
fixed rollout tensors and do not exercise IsaacLab reset behavior.

## Fix 1: Store Done Masks and Cut Bootstrap at Done

`DGPPORolloutMemory` now stores done masks for both rollout splits:

- `stc_dones`
- `det_dones`

`DGPPOAgent.record_transition(...)` records:

```python
dones = terminated | truncated
```

and passes the split masks into memory. The memory view exposes them as:

```python
"bT_dones": dones.transpose(0, 1).to(dtype=torch.bool)
```

`compute_dec_ocp_gae(...)` now accepts an optional `T_done` argument. When a
transition is done, future bootstrap values are masked out for that transition.
When `T_done=None`, the function behaves as before, preserving compatibility
with existing parity fixtures.

## Problem 2: RNN State Continued Across Resets

DG-PPO currently uses recurrent state when `use_rnn: true`. The policy RNN and
the low-level value RNN (`Vl`) were carried forward from step to step, but they
were not reset when an IsaacLab environment finished an episode.

That means one env could crash or succeed, reset to a new state and target, and
still keep hidden memory from the previous episode. The model would then see a
fresh episode through stale recurrent state. This can create noisy, misleading
gradients and unstable behavior.

Again, this is a runtime integration problem. The parity suite replays tensors;
it does not simulate an environment resetting in the middle of training.

## Fix 2: Reset RNN Carries for Done Envs

`DGPPOAgent` now has a helper:

```python
_reset_rnn_states_for_done(dones)
```

After the transition is recorded, this helper zeros recurrent state for all envs
where `terminated | truncated` is true. It handles both recurrent layouts used
by the implementation:

- policy state: `[L, E * A, C, H]`
- `Vl` state: `[L, E, C, H]`

This keeps recurrent memory local to one episode.

## Problem 3: Constraint Costs Were Always Zero

DG-PPO relies on constraint costs for the safety critic `Vh` and for CBF-based
advantages. The agent tried to read costs from:

```python
infos["costs"]
```

or:

```python
infos["cost"]
```

The position-tracking environment did not provide these costs. The DG-PPO agent
therefore silently used zeros for all constraints.

This is especially damaging because `n_constraints` was set from the number of
pillars, so the model had safety heads but no meaningful safety signal. `Vh` and
the CBF advantage path were effectively being trained on fake zero-cost data.

## Fix 3: Add Signed Pillar-Clearance Costs

The position-tracking environment now exposes:

```python
compute_constraint_costs(...)
```

The output shape is:

```python
[num_envs, 1, n_pillars]
```

which matches DG-PPO's `[E, A, H]` convention for environment, agent, and
constraint head.

For each pillar, the signed cost is approximately:

```python
pillar_collision_radius - distance_to_pillar_xy
```

Interpretation:

- positive cost means unsafe or colliding
- negative cost means safe clearance

The cost is clamped to `[-1, 1]`. If the drone is outside the active pillar
height range, the cost is forced negative so that the pillar is not treated as
an active collision constraint.

The environment stores the latest costs in:

```python
self._last_constraint_costs
self.extras["costs"]
```

The DG-PPO agent first tries to call `base_env.compute_constraint_costs(...)`
directly from the current graph state. If that is unavailable, it falls back to
`infos["costs"]` / `infos["cost"]`, and only then to zeros.

## Problem 4: No Test Covered Done-Boundary GAE

The existing parity tests correctly checked the JAX math fixtures, but they did
not cover the IsaacLab-specific done-mask case. Therefore, the old code could
pass parity while still bootstrapping through resets during real training.

## Fix 4: Add a Runtime-Correctness Test

A focused test was added:

```python
test_dec_ocp_gae_respects_episode_boundaries
```

This test verifies that `compute_dec_ocp_gae(...)` does not bootstrap through a
done transition. It is not a JAX parity test; it is a PyTorch/IsaacLab integration
correctness test.

## Validation

The DG-PPO parity suite was rerun:

```bash
PYTHONPATH=$PWD/source/isaac_pursuit_evasion:$PYTHONPATH \
python -m pytest -q source/isaac_pursuit_evasion/dgppo/parity_suite
```

Result:

```text
30 passed
```

A tiny IsaacLab smoke run was also executed:

```bash
python scripts/skrl/train.py \
  --task PosTracking-RL-velocity-v0 \
  --num_envs 10 \
  --algorithm DGPPO \
  --headless \
  --total_frames 160
```

It completed successfully. This does not prove learning quality, but it confirms
that the modified rollout, done-mask, constraint-cost, and update paths execute
inside IsaacLab.

Pre-commit was run on the touched files with the local broken license hook
skipped:

```bash
SKIP=insert-license pre-commit run --files <touched-files>
```

The normal formatting and lint checks passed. The license hook was skipped
because `.github/LICENSE_HEADER.txt` is missing locally.

## Expected Impact

These fixes remove three sources of corrupted training signal:

- value targets no longer leak across unrelated episodes
- recurrent state no longer leaks across resets
- `Vh` and the CBF path now receive real pillar-clearance costs

This does not guarantee immediately smooth reward curves. Hyperparameters,
observation scaling, reward design, controller behavior, number of environments,
and total training time can still matter a lot. However, these were structural
runtime bugs, so fixing them should make DG-PPO training much more meaningful.

## Things To Watch Next

Useful metrics to inspect in future runs:

- position reward trend and variance
- termination reason distribution
- `DGPPO/safe_rate`
- `DGPPO/adv_raw_mean`
- `DGPPO/loss_value_h`
- pillar collision rate
- whether recurrent and non-recurrent DG-PPO behave differently

If rewards still oscillate after these fixes, the next likely suspects are
observation normalization, action scale, reward shaping, PPO minibatch/update
settings, and whether the task is too termination-heavy early in training.
