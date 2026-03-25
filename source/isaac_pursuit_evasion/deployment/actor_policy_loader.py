"""Lightweight actor-only policy loader for deployment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from source.isaac_pursuit_evasion.deployment.skrl_scaler import load_skrl_scalers


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}
_CFG_DIR = Path(__file__).parent / "cfg"


@dataclass
class ActorPolicyConfig:
    obs_dim: int
    action_dim: int
    hidden_layers: Sequence[int]
    activation: str = "elu"
    log_std_init: float = 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActorPolicyConfig":
        return cls(
            obs_dim=int(data["obs_dim"]),
            action_dim=int(data["action_dim"]),
            hidden_layers=[int(x) for x in data.get("hidden_layers", ())],
            activation=str(data.get("activation", "elu")).lower(),
            log_std_init=float(data.get("log_std_init", 0.0)),
        )


class SimpleGaussianActor(nn.Module):
    """Minimal MLP actor matching skrl Gaussian model naming (net_container + log_std_parameter)."""

    def __init__(self, cfg: ActorPolicyConfig) -> None:
        super().__init__()
        act_cls = _ACTIVATIONS.get(cfg.activation, nn.ELU)
        layers: list[nn.Module] = []
        in_dim = cfg.obs_dim
        for hidden in cfg.hidden_layers:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(act_cls())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, cfg.action_dim))
        self.net_container = nn.Sequential(*layers)
        self.log_std_parameter = nn.Parameter(
            torch.full((cfg.action_dim,), float(cfg.log_std_init), dtype=torch.float32)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net_container(obs.to(dtype=torch.float32))

    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        mean = self.forward(obs)
        if deterministic:
            return mean
        std = torch.exp(self.log_std_parameter).expand_as(mean)
        return mean + std * torch.randn_like(mean)


def _strip_prefix(state_dict: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    token = f"{prefix}."
    return {key[len(token) :]: value for key, value in state_dict.items() if key.startswith(token)}


def _extract_policy_state_dict(payload: Any) -> Mapping[str, Any] | None:
    if isinstance(payload, nn.Module):
        return payload.state_dict()
    if not isinstance(payload, Mapping):
        return None

    if "policy" in payload and isinstance(payload["policy"], Mapping):
        return payload["policy"]
    for container_key in ("models", "model", "model_state_dict", "state_dict"):
        container = payload.get(container_key)
        if isinstance(container, Mapping):
            if "policy" in container and isinstance(container["policy"], Mapping):
                return container["policy"]
            for prefix in ("policy", "models.policy", "model.policy"):
                filtered = _strip_prefix(container, prefix)
                if filtered:
                    return filtered

    for prefix in ("policy", "models.policy", "model.policy"):
        filtered = _strip_prefix(payload, prefix)
        if filtered:
            return filtered

    if any(key.startswith("net_container.") for key in payload.keys()):
        return payload
    return None


def _resolve_cfg_path(path: str | Path | None, default_name: str) -> Path:
    if path is None:
        return _CFG_DIR / default_name
    candidate = Path(path)
    if candidate.exists():
        return candidate
    if not candidate.is_absolute() and candidate.parent == Path("."):
        fallback = _CFG_DIR / candidate.name
        if fallback.exists():
            return fallback
    return candidate


def load_actor_policy_config(path: str | Path | None = None) -> ActorPolicyConfig:
    path = _resolve_cfg_path(path, "actor_tracker_cfg.yml")
    if not path.exists():
        raise FileNotFoundError(f"Actor policy config not found at {path}")
    data: dict[str, Any]
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise ImportError("PyYAML is required to parse actor policy YAML configs.") from exc
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        import json

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    return ActorPolicyConfig.from_dict(data)


def load_actor_from_checkpoint(
    checkpoint: str | Path,
    cfg: ActorPolicyConfig,
    *,
    device: str | torch.device = "cpu",
    strict: bool = False,
) -> SimpleGaussianActor:
    checkpoint = str(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    state_dict = _extract_policy_state_dict(payload)
    if state_dict is None:
        raise ValueError(f"Unable to locate policy weights in checkpoint: {checkpoint}")

    actor = SimpleGaussianActor(cfg)
    missing, unexpected = actor.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        print(f"[WARN] Actor checkpoint load mismatch (missing={missing}, unexpected={unexpected}).")
    obs_scaler, value_scaler = load_skrl_scalers(payload)
    actor.obs_scaler = obs_scaler
    actor.value_scaler = value_scaler
    actor.to(device)
    actor.eval()
    return actor


def _download_wandb_artifact(
    artifact: str,
    artifact_file: str | None = None,
    local_dir: str | Path | None = None,
) -> str:
    try:
        import wandb  # type: ignore
    except Exception as exc:
        raise ImportError("wandb is required to download artifacts.") from exc

    api = wandb.Api()
    artifact_obj = api.artifact(artifact)
    download_dir = Path(artifact_obj.download(root=str(local_dir)) if local_dir else artifact_obj.download())
    if artifact_file:
        candidate = download_dir / artifact_file
        if candidate.exists():
            return str(candidate)
    pt_files = sorted(download_dir.rglob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt checkpoints found in artifact {artifact}")
    return str(pt_files[-1])


def load_actor_from_wandb(
    artifact: str,
    *,
    artifact_file: str | None = None,
    local_dir: str | Path | None = None,
    cfg: ActorPolicyConfig | None = None,
    device: str | torch.device = "cpu",
) -> SimpleGaussianActor:
    cfg = cfg or load_actor_policy_config()
    checkpoint = _download_wandb_artifact(artifact, artifact_file, local_dir)
    return load_actor_from_checkpoint(checkpoint, cfg, device=device)


def load_tracker_cf_rate_actor(
    *,
    device: str | torch.device = "cpu",
    artifact: str = "kthxulg/ppo_baseline/pretrain_tracker_cf_rate_DR_NoNorm:latest",
    artifact_file: str | None = None,
    cfg_path: str | Path | None = None,
) -> SimpleGaussianActor:
    cfg = load_actor_policy_config(cfg_path)
    return load_actor_from_wandb(
        artifact,
        artifact_file=artifact_file,
        cfg=cfg,
        device=device,
    )


class ActorPolicyCallable:
    """Adapter to use a SimpleGaussianActor with RL wrappers (TensorDict in, Tensor out)."""

    def __init__(self, actor: SimpleGaussianActor, device: str | torch.device = "cpu") -> None:
        self.actor = actor
        self.device = torch.device(device)
        self.actor.to(self.device)
        self.actor.eval()

    def __call__(self, td) -> torch.Tensor:
        obs = td.get("observation")
        if obs is None:
            raise KeyError("Expected 'observation' key in TensorDict for actor policy execution.")
        obs = obs.to(self.device)
        with torch.no_grad():
            return self.actor.act(obs, deterministic=True)
