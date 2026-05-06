# DG-PPO Update Timing Report

Date: 2026-05-06

This report summarizes the timing profile of the PyTorch/skrl DG-PPO update phase in
`source/isaac_pursuit_evasion/dgppo/dgppo_agent.py`. The profiled update includes return/advantage
computation, rollout graph preparation, policy/value losses, and optimizer steps after one 32-step rollout.

Profiling was enabled with:

```bash
DGPPO_PROFILE_UPDATE=1
DGPPO_PROFILE_UPDATE_SYNC_CUDA=1
```

The baseline training command used one update on `PosTracking-RL-velocity-v0`:

```bash
python scripts/skrl/train.py \
  --task PosTracking-RL-velocity-v0 \
  --num_envs 500 \
  --algorithm DGPPO \
  --headless \
  --total_frames 17000 \
  agent.agent.experiment.wandb=false \
  agent.trainer.disable_progressbar=true
```

Validation before profiling: `29 passed` in `source/isaac_pursuit_evasion/dgppo/parity_suite`.

Hardware observed by Isaac Sim:

- GPU: NVIDIA GeForce GTX 1650 Mobile, 4.3 GB
- CPU: Intel Core i7-10510U, 4 cores / 8 threads
- RAM: 15.8 GB
- Isaac Sim warned that the CPU governor was `powersave`

## Results

The representative baseline is `num_envs=500`, split into 250 stochastic and 250 deterministic envs, with
`rollouts=32`, `learning_epochs=4`, `mini_batches=8`, and `rnn_step=16`.

| Run | Total update | Mean minibatch | CUDA peak allocated | Notes |
| --- | ---: | ---: | ---: | --- |
| 500 envs, `rnn_step=16` | **14.303 s** | **442.5 ms** | **70.5 MB** | Baseline |
| 500 envs, `rnn_step=32` | **13.464 s** | **416.4 ms** | **70.5 MB** | Slightly faster, about 5.9% |
| 100 envs, `rnn_step=16` | **13.813 s** | **427.3 ms** | **30.9 MB** | Similar time, less memory |
| 500 envs, no explicit CUDA sync | **14.855 s** | **459.6 ms** | **70.5 MB** | Confirms sync is not the bottleneck |

Main baseline phase breakdown:

| Phase | Total ms | Mean per call | Share |
| --- | ---: | ---: | ---: |
| Full update | 14302.8 | 14302.8 | 100.0% |
| Value losses, including Vl/Vh forward scans | 4331.9 | 135.4 | 30.3% |
| Policy loss, including recurrent scan | 3216.4 | 100.5 | 22.5% |
| Policy optimizer/backward | 2972.0 | 92.9 | 20.8% |
| Vl optimizer/backward | 1850.6 | 57.8 | 12.9% |
| Vh optimizer/backward | 1740.2 | 54.4 | 12.2% |
| Minibatch batch-building | 41.7 | 1.3 | 0.3% |
| GAE + CBF + graph build + reset | about 35 | n/a | below 0.3% |

The meaningful result is simple: one update costs about **13.5-14.3 seconds** on this machine, and nearly all
of that time is inside the 32 minibatch updates.

## Findings

The update bottleneck is not Isaac Sim, rollout graph construction, memory views, GAE, or CBF advantage math.
Those parts are tiny compared with the PPO minibatch loop. In the 500-env baseline, graph construction is about
2.5 ms total, CBF advantage computation is about 0.5 ms, and both GAE passes together are about 30 ms.

The expensive forward path is the recurrent scan. In the baseline, the policy recurrent scan costs about
3.20 s per update, while Vl and Vh recurrent scans add about 4.30 s combined. With 32 minibatches, the code
repeatedly walks rollout timesteps in Python inside:

- `compute_rollout_policy_loss`
- `compute_rollout_vl_loss`
- `compute_rollout_vh_loss`

Backward and optimizer steps are also a major part of the cost. In the baseline, policy/Vl/Vh optimizer phases
sum to about **6.56 s**, roughly **46%** of total update time.

Increasing env count from 100 to 500 mostly increases memory, not update wall time. The 100-env run used less
than half the CUDA memory, but total update time only changed from 13.81 s to 14.30 s. This points to fixed
per-minibatch/per-timestep overhead and many small GPU launches rather than raw tensor size as the practical
bottleneck.

Setting `rnn_step=rollouts=32` helped, but only modestly: update time improved from 14.30 s to 13.46 s. This is
consistent with the main bottleneck still being the same repeated recurrent/graph evaluation structure.

## Optimization Direction

The highest-impact direction is to reduce the number of Python-level recurrent timestep evaluations and small
GPU launches.

Recommended next steps:

- Vectorize or batch recurrent rollout evaluation so policy, Vl, and Vh consume the full rollout sequence more
  directly, instead of repeatedly calling `rollout_graph_timestep` inside Python loops.
- Reduce update multiplication while iterating: fewer `learning_epochs`, fewer `mini_batches`, or larger
  minibatches. The current default performs 32 policy backward passes and 64 critic backward passes per update.
- Investigate whether Vl and Vh losses can be accumulated or updated in a more fused way without changing DG-PPO
  semantics.
- Use `torch.profiler` or Nsight on one update after the coarse profiler, focusing on the repeated graph select,
  GNN, GRU, and optimizer kernels.
- Set the CPU governor away from `powersave` before timing runs, because this workload appears sensitive to
  Python loop and kernel-launch overhead.

Bottom line: the current PyTorch port is update-loop limited. The best optimization target is the recurrent
minibatch scan/backward structure, not the simulation or the non-neural DG-PPO math.
