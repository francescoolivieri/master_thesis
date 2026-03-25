# Crazyflie Position Tracking (Student Guide)

## Overview
This environment trains a single **crazyflie_brushless** drone to track a reference position (and optional yaw) using RL.
Two action interfaces are supported:

- **RL_velocity**: policy outputs target world-frame linear velocity + yaw rate.
- **RL_rates**: policy outputs body-rate targets + thrust command.

The environment is designed to be short and readable for beginners while matching the controller conventions in
`crazy_controller.py` and `rl_controllers.py`.

---

## Control Modes (What the actions mean)

### RL_velocity
Policy outputs:
```
[a_vx, a_vy, a_vz, a_yawrate] in [-1, 1]
```
Mapping:
```
v_target = [a_vx, a_vy, a_vz] * vel_scale
yaw_rate_target = a_yawrate * pi
```
These targets go into the Crazyflie PID with `command_level="velocity"`.

### RL_rates
Policy outputs:
```
[a_p, a_q, a_r, a_thrust] in [-1, 1]
```
Mapping:
```
rates_target = [a_p, a_q, a_r] * pi
thrust_cmd = ((a_thrust + 1)/2) * weight * thrust_to_weight / thrust_cmd_scale
```
These targets go into the Crazyflie PID with `command_level="body_rate"`.

---

## Training (skrl PPO)

Velocity control (default):
```
python scripts/skrl/train.py --task PosTracking-RL-velocity-v0 --num_envs 512 --headless
```

Body-rate control:
```
python scripts/skrl/train.py --task PosTracking-RL-rates-v0 --num_envs 512 --headless
```

To change yaw tracking or reference update behavior, edit:
```
source/isaac_pursuit_evasion/isaac_pursuit_evasion/tasks/direct/pos_tracking/pos_tracking_env_cfg.py
```

---

## Benchmarking

### RL policy evaluation
```
python scripts/benchmark/bench_pos_tracking.py \
  --task PosTracking-RL-velocity-v0 \
  --checkpoint /path/to/policy.pt \
  --num-envs 32 --num-steps 2000 \
  --log-episodes --log-actions
```

### Baseline (controller-only)
```
python scripts/benchmark/bench_pos_tracking.py \
  --task PosTracking-RL-velocity-v0 \
  --policy-mode baseline \
  --num-envs 32 --num-steps 2000
```

### Enable yaw tracking in benchmark
```
python scripts/benchmark/bench_pos_tracking.py --yaw-tracking ...
```

---