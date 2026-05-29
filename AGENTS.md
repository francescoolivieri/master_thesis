# Repository Instructions

This repository is an Isaac Sim / IsaacLab project for Crazyflie Brushless reinforcement-learning experiments. The main package is `source/isaac_pursuit_evasion`; `dgppo-main` is the upstream JAX DG-PPO reference used for the ongoing PyTorch/skrl port.

## Project Map

- `scripts/skrl/train.py`: main IsaacLab training entrypoint.
- `scripts/benchmark/`: benchmark and evaluation scripts.
- `scripts/analyze*.py`: run-output analysis helpers.
- `docs/`: human-facing project notes and task guides.
- `source/isaac_pursuit_evasion/`: local IsaacLab extension package.
- `source/isaac_pursuit_evasion/dgppo/`: in-progress PyTorch/skrl DG-PPO port.
- `source/isaac_pursuit_evasion/isaac_pursuit_evasion/tasks/direct/pos_tracking/`: Crazyflie position-tracking task and agent configs.
- `dgppo-main/`: JAX DG-PPO reference implementation.

## Working Rules

- Keep changes narrow. Do not remove small comments present to help user readability of the code.
- Prefer short, simple, clean code in the spirit of antirez/Redis style: readable control flow, good names, and straightforward intuition over clever abstractions.
- Avoid over-complicated helpers, frameworky indirection, and speculative generality. Add abstractions only when they make the code easier to read now.
- Do not edit generated run output, W&B output, checkpoints, `__pycache__`, or large `.npz`/`.pkl` artifacts unless the user explicitly asks.
- Treat existing uncommitted changes as user work. Do not restore deleted or modified files unless asked.

## Development Environment

This project expects an IsaacLab/Isaac Sim environment. Many commands need that environment activated and may be unavailable in a plain Python shell.

Example training commands:

- Train PPO:
  `python scripts/skrl/train.py --task PosTracking-RL-velocity-v0 --num_envs 100 --total_frames 20000000 --headless`
- Train DG-PPO:
  `python scripts/skrl/train.py --task PosTracking-RL-velocity-v0 --algorithm DGPPO --num_envs 512 --headless`

If IsaacLab imports fail, first report that the IsaacLab environment appears inactive rather than rewriting imports.

## Validation

- For documentation-only changes, no simulator validation is required.
- For Python changes outside Isaac-dependent runtime paths, prefer a targeted import or unit-style smoke test when possible.
- For IsaacLab environment or training changes, run the smallest practical headless smoke test and state if the local environment prevents it.
- For important DG-PPO algorithm changes, compare with the JAX reference where it helps, then design focused checks around the behavior at risk: tensor shapes, kernels, rollout/update math, numerical sanity, and short varied scenarios.
- Do not rely on a standing parity suite as the default gate. Prefer stronger, situation-specific tests or temporary improvised scripts for the question being checked.
- To verify behaviour of the agent in isaaclab in certain settings, do testing scripts and runs in headless with few agents. Important is to delete the manufactured scripts afterwards and report the tests/findings to the user.

## DG-PPO Port Goal

The near-term goal is to port the JAX DG-PPO algorithm in `dgppo-main/dgppo/algo/dgppo.py` and related modules into the IsaacLab project as PyTorch code under `source/isaac_pursuit_evasion/dgppo/`. 
