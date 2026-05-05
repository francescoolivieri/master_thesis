# Repository Instructions

This repository is an Isaac Sim / IsaacLab project for Crazyflie Brushless reinforcement-learning experiments. The main package is `source/isaac_pursuit_evasion`; `dgppo-main` is the upstream JAX DG-PPO reference used for the ongoing PyTorch/skrl port.

## Project Map

- `scripts/skrl/train.py`: main IsaacLab training entrypoint.
- `scripts/benchmark/`: benchmark and evaluation scripts.
- `scripts/analyze*.py`: run-output analysis helpers.
- `docs/`: human-facing project notes and task guides.
- `source/isaac_pursuit_evasion/`: local IsaacLab extension package.
- `source/isaac_pursuit_evasion/dgppo/`: in-progress PyTorch/skrl DG-PPO port and parity suite.
- `source/isaac_pursuit_evasion/isaac_pursuit_evasion/tasks/direct/pos_tracking/`: Crazyflie position-tracking task and agent configs.
- `dgppo-main/`: JAX DG-PPO reference implementation and parity fixture tooling.

## Working Rules

- Keep changes narrow and preserve the existing IsaacLab project layout.
- Do not edit generated run output, W&B output, checkpoints, `__pycache__`, or large `.npz`/`.pkl` artifacts unless the user explicitly asks.
- Treat existing uncommitted changes as user work. Do not restore deleted or modified files unless asked.
- Prefer `rg` / `rg --files` for repository search.
- Use Python 3.10+ syntax and keep line length compatible with the repository's Black configuration: 120 columns.
- Use ASCII for new text unless an existing file clearly uses non-ASCII terminology.
- Keep documentation concise and link to the source of truth rather than duplicating long explanations.

## Development Environment

This project expects an IsaacLab/Isaac Sim environment. Many commands need that environment activated and may be unavailable in a plain Python shell.

Common commands:

- Train PPO:
  `python scripts/skrl/train.py --task PosTracking-RL-rates-v0 --num_envs 2048 --total_frames 20000000 --algorithm PPO --headless`
- Train DG-PPO after confirming its Hydra agent entrypoint is registered:
  `python scripts/skrl/train.py --task PosTracking-RL-rates-v0 --agent <dgppo_cfg_entry_point> --num_envs 512 --headless`
- List registered environments:
  `python scripts/list_envs.py`
- Run pre-commit on touched files when making Python/YAML/TOML changes:
  `pre-commit run --files <paths>`

If IsaacLab imports fail, first report that the IsaacLab environment appears inactive rather than rewriting imports.

## Validation

- For documentation-only changes, no simulator validation is required.
- For Python changes outside Isaac-dependent runtime paths, prefer a targeted import or unit-style smoke test when possible.
- For IsaacLab environment or training changes, run the smallest practical headless smoke test and state if the local environment prevents it.
- For DG-PPO algorithm changes, prefer parity checks or focused tensor-shape/kernel checks before long training runs.

## DG-PPO Port Goal

The near-term goal is to port the JAX DG-PPO algorithm in `dgppo-main/dgppo/algo/dgppo.py` and related modules into the IsaacLab project as PyTorch code under `source/isaac_pursuit_evasion/dgppo/`. Preserve algorithmic semantics first; performance cleanup comes after parity and shape correctness.
