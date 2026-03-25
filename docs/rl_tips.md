# RL Tips

Short, practical tips for training stable policies.

## Start simple
- Disable yaw tracking until position tracking is stable.
- Keep `ref_update_interval_s = 0.0` to track a fixed target per episode.

## Observation scaling
- The environment outputs **raw values** (no normalization in env).
- Use skrl’s `RunningStandardScaler` (already enabled in `skrl_ppo_cfg.yaml`).

## Action modes. Check the differences here: https://arxiv.org/abs/2202.10796. I also explain them a bit more in pos_tracking.md
- `RL_velocity`: easier to learn for beginners.
- `RL_rates`: closer to low‑level control, more sensitive to tuning.

## Debugging
- Use small `--num_envs` (e.g., 32) to iterate quickly.
- Turn on debug markers: `debug_visualizer=True` in config.
- Use benchmark scripts to validate reward components and termination rates.

## Logging
- Use `--log-episodes` and `--log-actions` in benchmarks to analyze trajectories.
- W&B logs include termination rates + reward components during training.

