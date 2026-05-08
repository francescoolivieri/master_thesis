# DG-PPO Live Debugging Report

Date: 2026-05-08

This branch started diagnostic-first. It now includes the first narrow fixes for the confirmed live-integration bugs,
with code regions marked by `DGPPO DEBUG FIX START/END` comments so they can be lifted into production deliberately.

## Fixes Added On 2026-05-08

Implemented fixes, in the order requested:

1. RNN state reset on episode end.
   - `source/isaac_pursuit_evasion/dgppo/dgppo_agent.py`: after storing each transition, DG-PPO now zeros policy and
     Vl recurrent state for `terminated | truncated`, matching skrl PPO_RNN live-carry behavior.
   - `source/isaac_pursuit_evasion/dgppo/utils.py`: added reusable RNN carry zeroing helpers for policy
     `[L, E*A, C, H]` and centralized value `[L, E, C, H]` carries.

2. Rollout mask storage for both split rollouts.
   - `source/isaac_pursuit_evasion/dgppo/dgppo_memory.py`: stochastic and deterministic memory now stores
     `terminated` and `truncated` masks and exposes `bT_terminated`, `bT_truncated`, and `bT_done` in `as_bTah_view`.

3. Masked target recursion.
   - `source/isaac_pursuit_evasion/dgppo/utils.py`: `compute_dec_ocp_gae` now accepts episode-boundary masks and stops
     recursive bootstrapping after true terminations. Truncation masks are passed through and stored; live training keeps
     `bootstrap_on_truncated=True` so time-limit semantics remain separate from true failures.
   - `source/isaac_pursuit_evasion/dgppo/dgppo_agent.py`: both stochastic and deterministic DG-PPO target calls pass
     their own masks.

4. Real/adapted safety costs.
   - `source/isaac_pursuit_evasion/isaac_pursuit_evasion/tasks/direct/pos_tracking/pos_tracking_env.py`: the task now
     publishes signed DG-PPO costs in `infos["costs"]`: one arena-boundary head plus one head per pillar. Positive means
     unsafe, negative means safe, following the JAX DG-PPO convention.
     Note: this changes the default position-tracking Vh output width from 2 pillar heads to 3 heads
     (`arena_bounds`, `pillar_0`, `pillar_1`), so old 2-head DG-PPO checkpoints need migration or retraining.
   - `source/isaac_pursuit_evasion/dgppo/dgppo_agent.py`: if an env does not provide costs, DG-PPO derives the same
     signed arena/pillar costs from graph states as an adapter fallback.
   - `source/isaac_pursuit_evasion/dgppo/utils.py`: added the shared position-tracking safety cost helper and cost-head
     alignment helper.

5. Focused regression tests.
   - `source/isaac_pursuit_evasion/dgppo/parity_suite/test_kernel_math_parity.py`: added tests for true-termination
     masking, truncation bootstrap separation, RNN carry zeroing, and signed safety costs.
   - `source/isaac_pursuit_evasion/dgppo/parity_suite/test_memory_and_adapter_parity.py`: added tests for split
     terminated/truncated mask storage.

6. Recurrent update-window state alignment.
   - `source/isaac_pursuit_evasion/dgppo/dgppo_agent.py`: DG-PPO now snapshots incoming Vl carries alongside the
     existing incoming policy carries before each live rollout step.
   - `source/isaac_pursuit_evasion/dgppo/dgppo_memory.py`: rollout memory stores per-step centralized Vl carries as
     `bT_vl_rnn_states`.
   - `source/isaac_pursuit_evasion/dgppo/update_helpers.py`: recurrent policy and Vl minibatch evaluation now starts
     each rollout chunk from the stored incoming carry at that chunk's first timestep. This preserves the existing
     zero-start default for JAX reset-at-rollout parity helpers, while live IsaacLab continuous windows use their real
     mid-episode recurrent context.
   - `source/isaac_pursuit_evasion/dgppo/parity_suite/test_memory_and_adapter_parity.py`: added a storage regression
     for policy and Vl recurrent carries.

No hyperparameters were tuned.

Validation run locally:

- `PYTHONPATH=$PWD/source/isaac_pursuit_evasion:$PYTHONPATH python -m pytest -q source/isaac_pursuit_evasion/dgppo/parity_suite`
  passed: 33 tests.
- `python scripts/skrl/train.py --task PosTracking-RL-velocity-v0 --num_envs 4 --algorithm DGPPO --headless --total_frames 64`
  completed one tiny 16-timestep smoke run. Its debug JSONL reported `n_constraints: 3`, `infos_keys: ["costs"]`,
  `costs.shape: [4, 1, 3]`, and stored mask fields in `update_start`.
- `python3 -m compileall -q source/isaac_pursuit_evasion/dgppo` passed after the recurrent update-window fix.
- `PYTHONPATH=$PWD/source/isaac_pursuit_evasion:$PYTHONPATH python3 -m pytest -q source/isaac_pursuit_evasion/dgppo/parity_suite`
  could not execute the parity tests in this plain local shell; pytest reported `1 skipped`, consistent with the
  IsaacLab/torch/skrl environment not being active.

## Instrumentation Added

DG-PPO now writes structured JSONL events to:

```text
logs/skrl/training/dgppo/<run_name>/dgppo_debug/events.jsonl
```

It also mirrors selected scalars through skrl `track_data`, so they should appear in TensorBoard and in the existing
W&B bridge when W&B is enabled.

Main event types:

- `setup`: resolved skrl/trainer config, graph observation layout, split env ids, RNN shapes, model flags.
- `transition`: observation/action/reward/value/cost stats, done/truncated rates, done reasons, reward components,
  observation consistency against IsaacLab env internals, deterministic-vs-stochastic split drift, RNN carry norms.
- `bootstrap`: final rollout bootstraps, including current Vh bootstrap vs reference-style RNN bootstrap.
- `update_start`: rollout boundary tensors seen by skrl, GAE/CBF target stats, advantage stats, stochastic/deterministic
  rollout stats.
- `minibatch`: PPO ratio/log-prob/entropy, losses, value predictions, and gradient norms per minibatch.
- `update_end`: aggregate update losses, clip fraction, learning rates.

The defaults live in the DG-PPO task YAML under `agent.debug`:
`source/isaac_pursuit_evasion/isaac_pursuit_evasion/tasks/direct/pos_tracking/agents/dgppo_cfg.yaml`.

## Run Analysis: `reference_debug`

Analyzed file:

```text
logs/skrl/training/dgppo/reference_debug/dgppo_debug/events.jsonl
```

Run setup:

- 500 envs, split into 250 deterministic envs and 250 stochastic envs.
- `use_rnn: true` for policy, Vl, and Vh.
- `rollouts: 16`, `learning_epochs: 4`, `mini_batches: 8`, so a complete update should emit 32 minibatch events.
- The run was configured for 3000 trainer timesteps and crashed at timestep 2319.

Event counts:

- `setup`: 1
- `transition`: 2253
- `bootstrap`: 145
- `update_start`: 145
- `minibatch`: 4623
- `update_end`: 144

Crash boundary:

- Update 145 started at timestep 2319.
- Update 145 emitted only 15 minibatches instead of the expected 32.
- The last event is `minibatch` for update 145, epoch 1, minibatch 6.
- No JSON event records the exception itself, so the crash occurred during or immediately after the next minibatch or
  update operation. The debug stream rules out a rollout-side crash.

### Confirmed Finding 1: RNN state leaks across episode boundaries

This is now the strongest confirmed issue.

Across transition events:

- `transition.data.rnn.done_new_nonzero_rate`: mean 0.996, median 1.0, max 1.0.
- `transition.data.anomaly_reasons` contains `rnn_state_nonzero_after_done` in 2244 of 2253 transition events.
- Done-env RNN norm mean is about 6.04, with max about 7.12.

Across update windows:

- `update_start.data.rollout_boundaries.rnn_done_new_norms.mean`: mean about 6.02.
- `update_start.data.rollout_boundaries.rnn_done_new_norms.max`: mean about 6.33, max about 7.38.

This means that when IsaacLab resets an env, DG-PPO usually carries a nonzero hidden state into the new episode. skrl's
own recurrent PPO explicitly zeros RNN state on `terminated | truncated`; before the fixes above, this DG-PPO
integration did not.

Recommended first fix after this logging branch:

- Reset `_policy_rnn_state` for finished env-agent slots after storing the transition.
- Reset `_vl_rnn_state` for finished env slots.
- Decide whether Vh should use the reset policy carry or the stored incoming rollout carry for target construction,
  then add a focused regression test around this boundary.

Status: the first two bullets are implemented in the 2026-05-08 fixes above. The Vh bootstrap-path comparison remains
diagnostic-only.

### Confirmed Finding 2: Episode boundaries are frequent and ignored by DG-PPO targets

Across update windows:

- Rollout done rate mean: 0.02045.
- Rollout terminated rate mean: 0.02045.
- Rollout truncated rate is essentially zero, except one timeout-level event.

Aggregated done reasons from transition/update logs:

- Out of bounds (`3`): 16634 env-step done records.
- Pillar collision (`6`): 7089 env-step done records.
- Success (`1`): 2 env-step done records.
- Timeout (`4`): 1 env-step done record.

So most resets are true terminations, not time-limit truncations. Before the fixes above, DG-PPO memory did not store
done masks and `compute_dec_ocp_gae` was unmasked, so targets could cross from a terminal state into the next reset
episode. This is not a minor edge case: roughly 2 percent of env slots per rollout window are affected on average.

Recommended second fix:

- Store `terminated` and `truncated` in `DGPPORolloutMemory` for both deterministic and stochastic splits.
- Mask recursive target/advantage computation across true terminations.
- Treat time-limit truncation separately if time-limit bootstrap is desired, matching skrl PPO semantics.

Status: implemented for storage and true-termination masking. Truncation masks are stored and passed through, with
time-limit bootstrap intentionally left enabled rather than folded into true termination behavior.

### Confirmed Finding 3: Constraint costs are exactly zero

Across all 2253 transition events:

- `transition.data.tensors.costs.max`: 0.0 for every event.
- `transition.data.tensors.costs.mean`: 0.0 for every event.
- `transition.data.env.infos_keys`: always empty.

This confirmed that the debug-run env was not supplying `infos["costs"]` or `infos["cost"]`. DG-PPO was therefore
training the safety pathway without a real environment cost signal, even though terminations showed many out-of-bounds
and pillar collisions.

This may be the second most important behavior bug after RNN/done handling. The CBF/Vh machinery is active, but its
input costs are all zero. That makes safety learning poorly grounded in the task's actual failure modes.

Recommended fix:

- Define a cost tensor in the env or adapter. At minimum, expose positive costs for pillar collision/out-of-bounds and
  possibly smooth signed-distance/clearance costs for obstacles and arena bounds.
- Log `infos["costs"]` explicitly after adding it.

Status: implemented as signed arena-boundary and per-pillar costs in the env, plus an agent-side adapter fallback.

### Finding 4: Deterministic/stochastic split drift exists early, then narrows

Across all transitions:

- Stochastic reward mean: -0.009.
- Deterministic reward mean: -0.223.
- Stochastic done rate mean: 0.0139.
- Deterministic done rate mean: 0.0282.
- Stochastic action norm mean: 1.60.
- Deterministic action norm mean: 1.09.

By update quartile:

- Updates 1-36: stochastic reward mean -0.062, deterministic reward mean -0.592.
- Updates 37-72: stochastic reward mean -0.041, deterministic reward mean -0.213.
- Updates 73-108: stochastic reward mean 0.015, deterministic reward mean -0.076.
- Updates 109-145: stochastic reward mean 0.075, deterministic reward mean 0.028.

The split distributions are noticeably different early in the run. They become closer later, but the deterministic half
has more terminations overall. This does not look like the primary crash cause, but it is a real integration difference
from the JAX reference's separate deterministic rollout.

### Finding 5: Bootstrap mismatch is present but smaller than RNN leakage

Across 145 bootstraps:

- `bootstrap.data.vh_boot_abs_diff.mean`: mean about 0.0030, max about 0.0073.
- `bootstrap.data.vh_boot_abs_diff.max`: mean about 0.0395, max about 0.133.

This confirms that the live Vh bootstrap path and the reference-style RNN bootstrap are not identical. The mismatch is
not huge in this run, but it is systematic enough to fix after done/RNN reset and cost handling.

### Finding 6: Value learning is noisy but not obviously the crash trigger

Across minibatches:

- Policy loss mean: 0.538, max 1.94.
- Vl loss mean: 2.26, max 11.42.
- Vh loss mean: 0.000070, max 0.0172. This is consistent with zero costs.
- Policy grad norm mean: 0.765, max 3.18.
- Vl grad norm mean: 37.1, max 358.5.
- Vh grad norm mean: 0.015, max 0.751.
- Ratio max mean: 1.83, max 16.46, although ratio mean stays near 1.0.
- Clip fraction mean: 0.050, max 0.289.

The incomplete final update does not show an obvious blow-up in the logged minibatches. That points more toward an
unlogged exception/OOM/runtime issue during the update than a scalar divergence visible before the crash. Capturing
stderr or adding exception logging around `_update_minibatch` would be useful if this crash repeats.

### Diagnostic caveat: observation consistency fields

The current `observation_consistency` diagnostics compare the stored pre-step `observations` against post-step live env
internals, because `record_transition` runs after `env.step`. Large diffs in those fields are therefore not valid
evidence of an observation layout bug. They should either be ignored for this run or changed to compare
`next_observations` against post-step env internals in a later instrumentation pass.

## Run Analysis: `new_reference_debug`

Analyzed file:

```text
logs/skrl/training/dgppo/new_reference_debug/dgppo_debug/events.jsonl
```

Run setup and outcome:

- Same 500-env split as `reference_debug`: 250 deterministic envs and 250 stochastic envs.
- `n_constraints: 3`, matching the new `arena_bounds`, `pillar_0`, and `pillar_1` cost heads.
- The run reached the configured 3000 trainer timesteps. The last event is a transition at timestep 2999.
- Event counts: `setup`: 1, `transition`: 2692, `bootstrap`: 187, `update_start`: 187, `minibatch`: 5984,
  `update_end`: 187.
- Every update emitted the expected 32 minibatches (`learning_epochs: 4`, `mini_batches: 8`) and an `update_end`.
  The previous incomplete-update crash signature did not repeat.

### Fix Check 1: RNN state reset worked

Across all 2692 transition events:

- `transition.data.rnn.done_new_nonzero_rate`: mean 0.0, median 0.0, max 0.0.
- Sum of `done_new_nonzero_count`: 0.
- `transition.data.anomaly_reasons`: empty for the whole run.

Across update windows:

- `update_start.data.rollout_boundaries.rnn_done_new_norms` is present in 185 of 187 updates. The two absent windows
  had no done envs in the rollout.
- For the 185 done-containing updates, `rnn_done_new_norms.mean` and `rnn_done_new_norms.max` are both exactly 0.0.

This confirms the policy/Vl carry reset after `terminated | truncated` is active in the live run. The strongest
`reference_debug` bug is fixed.

### Fix Check 2: Termination/truncation masks are stored and consumed

Every `update_start` reports:

- `rollout_boundaries.mask_status`: `DGPPO memory/update consumes true-termination masks; truncation masks are stored.`
- `rollout_boundaries.done`, `terminated`, and `truncated` tensors present with shape `[16, 500]`.

Observed boundary rates:

- Done true rate: mean 0.00692, median 0.00587, max 0.03613.
- Terminated true rate: mean 0.00678, median 0.00575, max 0.03613.
- Truncated true rate: mean 0.000136, median 0.0, max 0.00163.

The split rollout tensors are also present in every update. Total true counts across update windows:

- Stochastic rollout: 4078 terminations and 133 truncations.
- Deterministic rollout: 6065 terminations and 70 truncations.

So the masks are not merely allocated; they cover real episode boundaries and are flowing into the target/advantage
computation path. Time-limit truncations remain rare but are now distinguishable from true terminations.

### Fix Check 3: `infos["costs"]` is present and nonzero

Across transition events:

- `env.infos_keys` contains `costs` in all 2692 transitions.
- `transition.data.tensors.costs.shape` is `[500, 1, 3]` in every transition.
- `transition.data.tensors.costs.finite_rate` is 1.0 throughout.
- Cost mean is negative throughout, with run mean -0.544.
- Cost max is positive in 2622 of 2692 transition events; observed max is 0.101.
- Cost min is negative in every transition and often saturated at -1.0.

Across update windows, both split rollouts report finite nonzero `costs_h` with the same 3-head semantics. Vh prediction
and target tensors also use the 3-head shape. This confirms that the safety pathway is no longer training on all-zero
costs.

### Remaining instability

The run is healthier than `reference_debug`, but there are still signs to investigate before any hyperparameter tuning:

- No NaNs or incomplete updates are visible in the JSONL stream.
- Minibatch losses are lower overall than the old run: policy loss mean 0.128, Vh loss mean 0.000211, Vl loss mean
  0.656.
- Vl gradients are much smaller than before but still large: mean 15.7, p99 43.3, max 70.7.
- Policy gradients are usually modest, but there is one spike to 204.7 at update 165.
- PPO ratio mean stays near 1.0 overall, but rare ratio outliers remain. The largest ratio max is 976.6 at update 88,
  followed by 507.8 and 280.5 in the same update. Another spike reaches 162.0 at update 165.
- Clip fraction does not explode despite those outliers: mean 0.068, p99 0.208, max 0.333.
- Stochastic log-probs grow very large by the second half of the run. Minibatch `log_prob.max` reaches 24.51, and the
  final-quarter mean of `log_prob.max` is about 23.6.
- Vl predictions and Ql targets drift upward over training. By update quartiles, stochastic `value_l.mean` moves from
  0.31 to 3.92 to 9.61 to 12.76, while `ql.mean` moves from -0.07 to 2.71 to 7.85 to 10.88. `vl_boot.max` reaches
  20.23.
- Vh remains comparatively stable: minibatch Vh prediction mean is about -0.096, Vh grad norm mean is 0.014, and Vh
  loss stays small.

Interpretation: the original live-integration failures are fixed, and the run no longer crashes early, but there is
still a policy/value-scale issue. The rare ratio spikes and high positive log-probs point toward a remaining policy
distribution or action-squashing/log-prob stability question. The steadily rising Vl/Ql scale is a separate thing to
inspect before treating this as a tuning problem.

### Bootstrap mismatch remains

The reference-style Vh bootstrap comparison is still nonzero:

- `vh_boot_abs_diff.mean`: mean 0.00324, median 0.00276, max 0.02795.
- `vh_boot_abs_diff.max`: mean 0.0566, median 0.0400, max 0.2845.

This is not the dominant failure mode in the new run, but it remains a systematic RNN-specific mismatch. The max
mismatch is larger than in `reference_debug`, likely because the run survives longer and the recurrent states grow to
larger norms.

### Cost-head sanity check

Nothing in this run suggests a shape or finiteness bug from adding `arena_bounds` plus pillar cost heads:

- Setup reports `n_constraints: 3`.
- Transition costs, rollout costs, Vh predictions, and Vh losses all use 3-head tensors.
- Costs are finite, signed, and nonzero.
- CBF quantities are finite. `safe_rate` rises from about 0.46 in the first update quartile to about 0.94 in the final
  quartile, while mean CBF penalty remains small.

The only caveat is semantic, not a clear bug: costs are mostly negative/safe and often clipped at -1.0, with positive
violations relatively small. That may be exactly the intended signed-distance convention, but if safety learning still
looks weak later, inspect per-head cost distributions before changing hyperparameters.

### Split-drift status

The deterministic/stochastic split drift is still present but less suspicious than the old RNN/mask/cost failures:

- Stochastic done rate mean: 0.00627; deterministic done rate mean: 0.00913.
- Stochastic reward mean: 0.165; deterministic reward mean: 0.112.
- Stochastic action norm mean: 1.48; deterministic action norm mean: 0.70.
- Position error is close between splits: stochastic mean 1.146, deterministic mean 1.154.

This remains worth tracking because Vh targets depend on the deterministic half, but it does not look like a new
arena/pillar cost-head bug.

## Run Analysis: provided `events.jsonl`

Analyzed file:

```text
events.jsonl
```

Run setup and outcome:

- GPU-cluster run with 20,000 envs, split into 10,000 deterministic envs and 10,000 stochastic envs.
- `n_constraints: 3`, matching `arena_bounds`, `pillar_0`, and `pillar_1`.
- The JSONL stream contains timesteps 0 through 2319. Event counts: `setup`: 1, `transition`: 2293, `bootstrap`: 145,
  `update_start`: 145, `minibatch`: 4640, `update_end`: 145.
- Every observed update emitted the expected 32 minibatches and an `update_end`. There is no incomplete-update or NaN
  signature in this file.

Fix checks:

- RNN episode-end reset still looks fixed. `transition.data.rnn.done_new_nonzero_rate` is zero throughout and
  `anomaly_reasons` is empty.
- `infos["costs"]` is present from the first transition. Costs are finite, signed, and nonzero; `costs.max` is positive
  in unsafe samples, with observed max about 0.235.
- Termination/truncation masks are present in every update and carry real events. Mean rollout termination rate is about
  0.00147 and mean truncation rate is about 0.00109 in the update windows.

Performance degradation:

- Mean reward improves in the second quartile and then degrades: 0.231 -> 0.331 -> 0.308 -> 0.215.
- Mean position error follows the inverse trend: 1.061 -> 0.799 -> 0.854 -> 1.079.
- Action norm rises steadily and is nearly saturated late: 0.553 -> 1.113 -> 1.497 -> 1.684.
- Stochastic old log-prob mean rises from -1.98 to 9.86 by the final quartile. Minibatch `log_prob.max` reaches about
  24.4, and the action tensor is already at `abs_max: 1.0` in most update windows.
- Vl scale drifts strongly upward. Stochastic `value_l.mean` moves 1.04 -> 2.83 -> 9.52 -> 17.21, while Ql mean moves
  -0.38 -> 0.55 -> 7.15 -> 15.07.
- Vh remains comparatively stable: Qh/Vh means stay around -0.24 and Vh losses remain small.

Most likely integration issue found from this run:

- The live rollout stores old log-probs and Vl values using recurrent carries from long, continuous IsaacLab episodes.
  Before the new recurrent update-window fix, the minibatch update recomputed recurrent policy log-probs and Vl
  predictions from zero state at the start of every 16-step rollout window.
- That zero-start behavior matches the JAX reset-at-rollout fixture semantics, but it is wrong for skrl live windows
  that usually start mid-episode. It can explain both symptoms in this file: PPO ratios/log-probs become distorted by a
  hidden-state mismatch, and Vl is trained/evaluated under a different recurrent context than the values used to build
  targets.

Status: fixed in the current code by storing incoming Vl carries and by evaluating recurrent policy/Vl chunks from
stored chunk-start carries. This needs a new cluster run to confirm whether late reward degradation and action
saturation are reduced.

## Updated Issue Register

### 1. RNN carries are not reset on episode end

Status: fixed by the 2026-05-08 code changes and confirmed in `new_reference_debug`.

skrl's recurrent PPO path resets recurrent states for environments where `terminated | truncated` is true. Before the
fix, DG-PPO kept `_policy_rnn_state` and `_vl_rnn_state` running through IsaacLab automatic resets. The new run reports
zero done-env RNN carry norms across all done-containing updates.

Evidence to inspect:

- `transition.data.rnn.done_new_nonzero_rate`
- `transition.data.anomaly_reasons` containing `rnn_state_nonzero_after_done`
- `update_start.data.rollout_boundaries.rnn_done_new_norms`

If this regresses, done-heavy periods should again show nonzero RNN carry norms on finished envs, followed by degrading
rewards/position error.

### 2. Episode boundaries were observed but not consumed by DG-PPO targets

Likelihood: confirmed high in `reference_debug`; fixed in the 2026-05-08 code changes.

`DGPPOAgent.record_transition` receives `terminated` and `truncated`. Before the fix, `DGPPORolloutMemory` did not
store them and `compute_dec_ocp_gae` had no done mask. Live IsaacLab rollouts could therefore bootstrap across
environment resets. PPO's skrl implementation masks `terminated` in GAE and has explicit time-limit bootstrapping
support.

Evidence to inspect:

- `update_start.data.rollout_boundaries.done.true_rate`
- `update_start.data.rollout_boundaries.warning`
- correlation between nonzero done/truncated rates and advantage, clip fraction, and losses.

`new_reference_debug` confirms the split rollout masks and boundary tensors are present on every update and that the
masked target path is the active computation path.

### 3. RNN bootstrap path may not match the JAX reference

Likelihood: confirmed medium in `reference_debug`, especially with `use_rnn: true`.

The live bootstrap currently evaluates Vh on `next_graph` with the already-advanced live policy carry. The reference
computes the final safety value by applying the deterministic policy to the final next graph from the stored incoming
rollout RNN state, then evaluating Vh with that resulting carry. The new logs compute this reference-style value only
for comparison and keep the current behavior unchanged.

Evidence to inspect:

- `bootstrap.data.vh_boot_abs_diff.mean`
- `bootstrap.data.vh_boot_abs_diff.max`
- `DGPPO Debug/bootstrap_vh_abs_diff_mean`

If the diff is persistently nonzero or spikes near episode boundaries, this is a strong RNN-specific target mismatch.

### 4. The live skrl rollout semantics differ from the JAX reference rollout

Likelihood: medium.

The JAX reference collects fixed-horizon rollouts from environment reset with fresh initial RNN state. The skrl
integration collects short continuous windows (`rollouts: 16`) from long IsaacLab episodes (`episode_length_s: 10`,
policy rate 50 Hz). Episode-boundary recurrent reset is now implemented, but rollout-window semantics still differ
from the reference.

Evidence to inspect:

- `transition.data.memory_cursor`
- `transition.data.env.episode_length_buf`
- `update_start.data.rollout_boundaries.episode_length_buf`
- relationship between episode age, done rate, and advantage/value drift.

This may be an integration semantics issue rather than a hyperparameter issue.

### 5. Recurrent policy/Vl update windows started from zero mid-episode

Status: fixed in the current code after analyzing the provided `events.jsonl`; needs a new cluster run for live
confirmation.

The live skrl integration stores old log-probs and Vl values from continuous episodes, using the incoming recurrent
state at each environment step. Before this fix, recurrent minibatch evaluation rebuilt policy log-probs and Vl
predictions by initializing the RNN state to zero at the start of every rollout chunk. That was compatible with the JAX
fixture's reset-at-rollout assumption, but not with live IsaacLab windows that usually begin mid-episode.

Why this matters:

- PPO ratios compare new log-probs against old log-probs produced from a different hidden state.
- Vl losses compare predictions from a zero-start hidden state against Ql targets built from live Vl values and
  bootstraps produced from the real carry.
- The provided `events.jsonl` shows exactly the expected symptom cluster: high positive log-probs, rare ratio outliers,
  rising action saturation, and strongly drifting Vl/Ql scale while Vh remains stable.

Fix:

- Store incoming centralized Vl carries in memory as `bT_vl_rnn_states`.
- Evaluate recurrent policy chunks from stored incoming policy carries at each chunk's first timestep.
- Evaluate recurrent Vl chunks from stored incoming Vl carries at each chunk's first timestep.

Evidence to inspect on the next run:

- `minibatch.data.ratio.max` and `minibatch.data.log_prob.max`
- `transition.data.tensors.action_norm` and action `abs_max`
- `transition.data.tensors.value_l.mean`
- `update_start.data.targets_and_advantages.ql.mean`
- reward/position-error quartiles after timestep 1000.

### 6. Deterministic and stochastic halves may drift into different state distributions

Likelihood: confirmed medium in `reference_debug` and still present in `new_reference_debug`.

The port uses half the live IsaacLab envs for deterministic actions and half for stochastic actions. The JAX reference
uses a separate deterministic rollout for Vh targets. The split adaptation may be valid, but if deterministic envs
become easier/harder or reset at different rates than stochastic envs, Vh targets can be trained on a different state
distribution than the policy update.

Evidence to inspect:

- `transition.data.split.stc_reward` vs `det_reward`
- `transition.data.split.stc_pos_error` vs `det_pos_error`
- `transition.data.done.stc_any` vs `det_any`
- `update_start.data.stochastic_rollout` vs `deterministic_rollout`

### 7. Costs were zero unless the env supplied `infos["costs"]`

Status: fixed by the 2026-05-08 code changes and confirmed in `new_reference_debug`.

The position-tracking env exposes pillar collisions and done reasons. Before the fix, the DG-PPO agent only read costs
from `infos["costs"]` or `infos["cost"]`; if those keys were absent, it logged and trained with zero constraints. This
does not explain every degradation mode, but it meant the "safety" part of DG-PPO could be inactive in this task. The
new run has `infos["costs"]` in every transition and finite signed costs with shape `[500, 1, 3]`.

Evidence to inspect:

- `transition.data.tensors.costs`
- `transition.data.env.infos_keys`
- `update_start.data.stochastic_rollout.costs_h`
- `DGPPO/costs_max` if added later from the JSONL-derived analysis.

## skrl Integration Notes

Checked against skrl develop/2.1 docs and source:

- PPO/PPO_RNN docs: https://skrl.readthedocs.io/en/develop/api/agents/ppo.html
- Recurrent PPO source: https://github.com/Toni-SM/skrl/blob/develop/skrl/agents/torch/ppo/ppo_rnn.py
- Trainer source: https://github.com/Toni-SM/skrl/blob/develop/skrl/trainers/torch/base.py
- PyPI currently lists 2.0.0 stable: https://pypi.org/project/skrl/

Relevant differences to keep in mind:

- skrl `Trainer.train()` passes `terminated` and `truncated` into `record_transition`, then for vectorized envs carries
  `next_observations` forward. IsaacLab wrappers usually return post-reset observations for finished envs.
- skrl `PPO_RNN.record_transition()` resets recurrent states where `terminated | truncated` is true.
- skrl PPO stores termination tensors in memory and uses them in GAE. DG-PPO now stores those tensors and masks
  true-termination bootstraps.
- skrl eval expects `outputs["mean_actions"]`; DG-PPO currently returns `mean_action`. This is probably evaluation-only,
  not the training degradation, but it is an integration mismatch to fix later.

## Suggested First Analysis Pass

After a cluster run, inspect:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("logs/skrl/training/dgppo/<run_name>/dgppo_debug/events.jsonl")
counts = {}
for line in path.open():
    event = json.loads(line)["event"]
    counts[event] = counts.get(event, 0) + 1
print(counts)
PY
```

Then plot or aggregate these fields:

- RNN leakage: `transition.data.rnn.done_new_nonzero_rate`
- Boundary pressure: `update_start.data.rollout_boundaries.done.true_rate`
- Bootstrap mismatch: `bootstrap.data.vh_boot_abs_diff.mean`
- Split drift: `transition.data.split.*`
- Update instability: `minibatch.data.ratio`, `clip_frac`, gradient norms, value predictions.

After these fixes, the next cluster run should confirm that done-env RNN norms stay at zero, rollout masks are present
in `update_start`, and `transition.data.tensors.costs` is no longer identically zero.
