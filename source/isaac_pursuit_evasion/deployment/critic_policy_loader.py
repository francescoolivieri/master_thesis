"""Lightweight critic-only policy loader for deployment."""
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
class CriticPolicyConfig:
    obs_dim: int
    output_dim: int = 1
    hidden_layers: Sequence[int] = ()
    activation: str = "elu"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CriticPolicyConfig":
        return cls(
            obs_dim=int(data["obs_dim"]),
            output_dim=int(data.get("output_dim", 1)),
            hidden_layers=[int(x) for x in data.get("hidden_layers", ())],
            activation=str(data.get("activation", "elu")).lower(),
        )


class SimpleValueCritic(nn.Module):
    """Minimal MLP critic matching skrl deterministic model naming (net_container)."""

    def __init__(self, cfg: CriticPolicyConfig) -> None:
        super().__init__()
        act_cls = _ACTIVATIONS.get(cfg.activation, nn.ELU)
        layers: list[nn.Module] = []
        in_dim = cfg.obs_dim
        for hidden in cfg.hidden_layers:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(act_cls())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, cfg.output_dim))
        self.net_container = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net_container(obs)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)


def _strip_prefix(state_dict: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    token = f"{prefix}."
    return {key[len(token) :]: value for key, value in state_dict.items() if key.startswith(token)}


def _extract_value_state_dict(payload: Any) -> Mapping[str, Any] | None:
    if isinstance(payload, nn.Module):
        return payload.state_dict()
    if not isinstance(payload, Mapping):
        return None

    for key in ("value", "critic"):
        if key in payload and isinstance(payload[key], Mapping):
            return payload[key]
    for container_key in ("models", "model", "model_state_dict", "state_dict"):
        container = payload.get(container_key)
        if isinstance(container, Mapping):
            for key in ("value", "critic"):
                if key in container and isinstance(container[key], Mapping):
                    return container[key]
            for prefix in ("value", "critic", "models.value", "models.critic", "model.value", "model.critic"):
                filtered = _strip_prefix(container, prefix)
                if filtered:
                    return filtered

    for prefix in ("value", "critic", "models.value", "models.critic", "model.value", "model.critic"):
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


def load_critic_policy_config(path: str | Path | None = None) -> CriticPolicyConfig:
    path = _resolve_cfg_path(path, "critic_tracker_cfg.yml")
    if not path.exists():
        raise FileNotFoundError(f"Critic policy config not found at {path}")
    data: dict[str, Any]
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise ImportError("PyYAML is required to parse critic policy YAML configs.") from exc
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        import json

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    return CriticPolicyConfig.from_dict(data)


def load_critic_from_checkpoint(
    checkpoint: str | Path,
    cfg: CriticPolicyConfig,
    *,
    device: str | torch.device = "cpu",
    strict: bool = False,
) -> SimpleValueCritic:
    checkpoint = str(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu")
    state_dict = _extract_value_state_dict(payload)
    if state_dict is None:
        raise ValueError(f"Unable to locate critic weights in checkpoint: {checkpoint}")

    critic = SimpleValueCritic(cfg)
    missing, unexpected = critic.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        print(f"[WARN] Critic checkpoint load mismatch (missing={missing}, unexpected={unexpected}).")
    obs_scaler, value_scaler = load_skrl_scalers(payload)
    critic.obs_scaler = obs_scaler
    critic.value_scaler = value_scaler
    critic.to(device)
    critic.eval()
    return critic


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


def load_critic_from_wandb(
    artifact: str,
    *,
    artifact_file: str | None = None,
    local_dir: str | Path | None = None,
    cfg: CriticPolicyConfig | None = None,
    device: str | torch.device = "cpu",
) -> SimpleValueCritic:
    cfg = cfg or load_critic_policy_config()
    checkpoint = _download_wandb_artifact(artifact, artifact_file, local_dir)
    return load_critic_from_checkpoint(checkpoint, cfg, device=device)


def load_tracker_cf_rate_critic(
    *,
    device: str | torch.device = "cpu",
    artifact: str = "kthxulg/ppo_baseline/pretrain_tracker_cf_rate_DR_NoNorm:latest",
    artifact_file: str | None = None,
    cfg_path: str | Path | None = None,
) -> SimpleValueCritic:
    cfg = load_critic_policy_config(cfg_path)
    return load_critic_from_wandb(
        artifact,
        artifact_file=artifact_file,
        cfg=cfg,
        device=device,
    )


class CriticPolicyCallable:
    """Adapter to use a SimpleValueCritic with RL wrappers (TensorDict in, Tensor out)."""

    def __init__(
        self,
        critic: SimpleValueCritic,
        device: str | torch.device = "cpu",
        obs_key: str = "state",
    ) -> None:
        self.critic = critic
        self.device = torch.device(device)
        self.obs_key = obs_key
        self.critic.to(self.device)
        self.critic.eval()

    def __call__(self, td) -> torch.Tensor:
        obs = td.get(self.obs_key)
        if obs is None:
            raise KeyError(f"Expected '{self.obs_key}' key in TensorDict for critic evaluation.")
        obs = obs.to(self.device)
        with torch.no_grad():
            return self.critic.value(obs)
