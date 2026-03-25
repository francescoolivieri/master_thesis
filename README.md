# IsaacDronesRPL — Crazyflie Brushless RL

## Overview
This repository is a student‑friendly Isaac Sim / IsaacLab project for training and benchmarking **Crazyflie Brushless** RL policies. It includes a single‑drone **position tracking** task, pursuit‑evasion tasks, benchmark scripts, and deployment helpers.

### Key entrypoints
- **Training (skrl):** `scripts/skrl/train.py`
- **Benchmarking:** `scripts/benchmark/bench_pos_tracking.py`
- **Analysis:** `scripts/analyze_stats.py` *(I haven't tested it)*
- **Tasks:** `source/isaac_pursuit_evasion/isaac_pursuit_evasion/tasks/direct/pos_tracking`

### Docs
- RL tips: `docs/rl_tips.md`
- Common pitfalls: `docs/common_pitfalls.md`
- Exercises: `docs/exercises.md`
- Pos‑tracking guide: `docs/pos_tracking.md`

### Video explaining isaac
https://youtu.be/w4OLf4D4N4g

---

## Steps to get it running (Isaac Sim + IsaacLab + VSCode) in local installation (Steps to use the cluster will come soon)

### 1) Install Isaac Sim (and close IsaacLab if it’s open)
Use the **Isaac Sim pre-built binaries** method from the official IsaacLab docs:
- https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html

Basically: download Isaac Sim and move it into your `$HOME` directory.

### 2) Create an IsaacLab virtual environment
Go to your **IsaacLab** folder and follow the installation instructions inside a venv (same link as above).

I usually use **uv** because it installs fast, but **conda** should work too.

Roughly (uv flow):
- Create the env:
  - `./isaaclab.sh --uv my_env`
- Activate it, then install:
  - `./isaaclab.sh --install`

After this, you should be able to run IsaacLab examples, e.g.:
```bash
python scripts/tutorials/00_sim/create_empty.py
```

### 3) Install git-lfs (you need it for the drone models)
```bash
sudo apt update
sudo apt install -y git-lfs
git lfs install
```

### 4) Go to your cloned project repo (this repo)
Important: use the **same virtual environment** you created for IsaacLab (that’s the one you run from).

Example:
- If your repo is `$HOME/Repos/Thesis`, you should still activate the IsaacLab venv, e.g.:
  - `source $HOME/Repos/IsaacLab/env_x/bin/activate` *(uv keeps the venv inside IsaacLab)*
- With conda it’s usually easier to activate.

### 5) VSCode integration (important)
You need VSCode to know where IsaacLab is so imports resolve properly.

If your `.vscode` folder isn’t there, follow:
- https://isaac-sim.github.io/IsaacLab/main/source/overview/developer-guide/vs_code.html#setup-vs-code

Main idea:
1. Open the directory in VSCode
2. Run VSCode Tasks:
   - `Ctrl+Shift+P` → **Tasks: Run Task** → run `setup_python_env`

Then, in **this repo**, edit `.vscode/settings.json` and add these to `python.analysis.extraPaths`:
```json
[
  "${workspaceFolder}/../IsaacLab/source/isaaclab",
  "${workspaceFolder}/../IsaacLab/source/isaaclab_tasks",
  "${workspaceFolder}/../IsaacLab/source/isaaclab_rl",
  "${workspaceFolder}/../IsaacLab/source/isaaclab_assets",
  "${workspaceFolder}/../IsaacLab/source/isaaclab_mimic"
]
```

This makes VSCode understand IsaacLab paths, and things show up nicely in "green".

### 6) Create a W&B account (optional but recommended)
- https://wandb.ai/site/

Being a student gives you a professional license. You can create your own project and update the W&B params in `skrl_ppo_cfg.yaml`.


### 7) Codex or Claude code (optional but recommended)
- For coding, in the beginning it's better if you do it yourself. But later, I would recommend using either codex or claude code, which are already integrated in vscode. They can be handy for fixing bugs and implement things which technically are not difficult, but takes one time.

---

## Quickstart

### Train position tracking (RL velocity)
```bash
python scripts/skrl/train.py --task PosTracking-RL-velocity-v0 --num_envs 1024 --headless
```

Expected outputs:
- Training logs and checkpoints under `training/`
- W&B run *(if enabled in `skrl_ppo_cfg.yaml`)*

### VSCode launch.json tip (makes life easier)
You can run this stuff very easily by creating an entry in `launch.json`. Then when you click debug, you can choose which configuration.

For instance, try this:
```json
{
  "name": "Train Pos Tracking PPO",
  "type": "debugpy",
  "request": "launch",
  "program": "${workspaceFolder}/scripts/skrl/train.py",
  "console": "integratedTerminal",
  "args": [
    // "--task", "PosTracking-RL-velocity-v0",
    "--task", "PosTracking-RL-rates-v0",
    "--num_envs", "2048",
    "--total_frames", "20000000",
    "--algorithm", "PPO",
    "--seed", "0",
    "--headless",
    "--video",
    "--video-interval-frames", "5000000",
    "--video_length", "500",
    // "--domain-randomization",
    // "--wandb-name", "pos_tracking_rl_velocity",
    "--wandb-name", "pos_tracking_rl_rate"
  ]
}
```

If you uncomment the velocity task and comment the rates task, then it trains velocity.

---

### Benchmark a trained policy (W&B artifact) *(I haven't tested this)*
```bash
python scripts/benchmark/bench_pos_tracking.py \
  --task PosTracking-RL-velocity-v0 \
  --policy-mode rl \
  --artifact entity/project/pos_tracking_rl_velocity:latest \
  --num-envs 32 --num-steps 2000 \
  --log-episodes --log-actions
```