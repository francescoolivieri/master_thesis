# Common Pitfalls

## 1) Pylance missing imports
Fix `.vscode/settings.json` using the extraPaths snippet from `docs/setup.md`.

## 2) CUDA out of memory
- Reduce `--num_envs`
- Disable cameras (`--disable-cameras`) or run `--headless`

## 3) W&B artifact not found
Make sure the artifact path is `entity/project/artifact_name:alias`.
Example:
```
entity/project/pos_tracking_rl_velocity:latest
```

## 4) Float vs double errors
If you see dtype mismatch errors, ensure observations are float32. The deployment and benchmark loaders already cast to float32.

## 5) No movement / immediate crashes
- Check arena bounds and collision altitude.
- Try baseline controller benchmark (`--policy-mode baseline`).

## 6) Termination rates look wrong
- Termination rates are logged per step and normalized by the number of terminations for that step.
- Use benchmark logs (`metrics.json`) for episode‑level rates.

## 7) launch.json changes don’t stick
`launch.json` is generated. Edit `.vscode/tools/launch.template.json` and re‑run:
```
python .vscode/tools/setup_vscode.py --isaac_path /path/to/isaacsim
```
