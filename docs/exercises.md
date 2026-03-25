# Exercises

These are short, structured exercises for students. Check all the plots in w&b and try to understand if they make sense based on the implemented environment.

## 1) Fixed‑point hover
- Set `ref_update_interval_s = 0.0`.
- Train `RL_velocity` for 1–2M frames.
- Evaluate with benchmark and report mean position error.

## 2) Enable yaw tracking
- Turn on `flag_yaw_tracking` in config.
- Add yaw error to observations and rewards (already implemented).
- Compare success rate vs. position‑only policy.

## 3) RL_rates control
- Train `PosTracking-RL-rates-v0`.
- Compare convergence and stability vs RL_velocity.
