# DG-PPO Parity Suite

Run the PyTorch-side parity checks from the repository root in the IsaacSim environment:

```bash
s_isaac
PYTHONPATH=$PWD/source/isaac_pursuit_evasion:$PYTHONPATH \
  python -m pytest -q source/isaac_pursuit_evasion/dgppo/parity_suite
```

For the focused unfinished-contract gate:

```bash
s_isaac
PYTHONPATH=$PWD/source/isaac_pursuit_evasion:$PYTHONPATH \
  python -m pytest -q source/isaac_pursuit_evasion/dgppo/parity_suite/test_unfinished_parity_contract.py
```

The JAX fixture exporter expects the `dgppo` conda environment:

```bash
conda activate dgppo
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/matplotlib \
  python dgppo-main/parity_checks.py export-update \
  --output-dir dgppo-main/parity_artifacts/update_num_envs6 \
  --n-env-train 3 --batch-size 128 --rnn-step 16 --force-cpu
```

Refresh the small fixture with the same exporter when deterministic replay inputs are needed:

```bash
conda activate dgppo
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/matplotlib \
  python dgppo-main/parity_checks.py export-update \
  --output-dir dgppo-main/parity_artifacts/update_small \
  --n-env-train 1 --batch-size 128 --rnn-step 16 --force-cpu
```

Export a multi-update drift trace on the same multi-env split as the
`num_envs=6` PyTorch fixture:

```bash
conda activate dgppo
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/matplotlib \
  python dgppo-main/parity_checks.py export-drift \
  --output-dir dgppo-main/parity_artifacts/drift_num_envs6 \
  --n-env-train 3 --batch-size 128 --rnn-step 16 \
  --n-drift-updates 4 --force-cpu
```

## Fixture Inventory

- Present: `dgppo-main/parity_artifacts/kernel/kernel_fixtures.npz`
- Present: `dgppo-main/parity_artifacts/update_small/update_fixture.npz` (`n_env_train=1`, PyTorch `num_envs=2`)
- Present: `dgppo-main/parity_artifacts/update_num_envs6/update_fixture.npz`
  (`n_env_train=3`, PyTorch `num_envs=6`)
- Present: `dgppo-main/parity_artifacts/drift_num_envs6/drift_trace.npz`
  (`n_env_train=3`, PyTorch `num_envs=6`, `n_drift_updates=4`)

## Coverage

- Strict checks call the PyTorch DG-PPO model/math/update helper code in `dgppo_models.py`, `utils.py`,
  `dgppo_memory.py`, and `update_helpers.py`; tests may adapt fixture leaves into tensors, but they do not replace
  PPO, GAE, CBF, distribution, value-network, memory, or update-helper behavior with duplicate implementations.
- Actor distribution parity: uses `checkpoints/actor/rollout/{mean,std,mode,log_prob}`.
- Fixed-noise sampled action parity: use
  `checkpoints/actor/rollout/{fixed_noise,fixed_noise_action,fixed_noise_log_prob}`.
- Vl/Vh forward parity: use `checkpoints/update/value/{bT_Vl,bTp1_Vl,bTah_Vh,bTp1ah_Vh,bTah_Vh_det,bTp1ah_Vh_det}`.
- Full pre-update loss parity: strict for policy ratio/surrogate, GAE/CBF advantages,
  `checkpoints/update/loss/Vl_global`, and `checkpoints/update/loss/Vh_det_global` loss math.
  Entropy-scaled `policy_loss` is strict using the exported entropy scalar; production entropy-sample parity still
  needs deterministic entropy-noise export.
- Optimizer-step fixture contract: params before/after, logged grad norms, param deltas, optimizer metadata, and
  pre-update `optax.apply_if_finite(optax.adam)` state are exported for policy, Vl, and Vh. The executable optimizer
  contract is strict for losses and global grad norms, then checks bounded first-step PyTorch/Optax delta drift. This is
  intentional: the JAX policy loss includes a one-sample TFP tanh-entropy estimate whose sample seed is generated
  through NumPy inside the distribution, and the fixture does not export per-leaf clipped gradients or entropy samples.
- Deterministic replay loop parity: uses exported `inputs/det_rollout/*` raw graphs/actions/rewards/costs/RNN
  states plus deterministic Vh targets/loss.
- Multi-env single-update coverage is executable for `num_envs=6` through the parametrized PyTorch tests.
- Multi-update drift replay uses `drift_trace.npz` per-update replay inputs/checkpoints and infers `num_envs` and
  update count from the artifact metadata. The policy entropy-gradient path remains a scalar-loss contract because
  the JAX TFP entropy sample is not exported as a deterministic gradient input.

There are no expected xfails for the current GRU fixtures. LSTM-specific paths may still xfail only when a fixture
explicitly requests unsupported LSTM/gate-order behavior.
