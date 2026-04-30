"""Utilities for extracting skrl running scaler stats from checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass
class SkrlScaler:
    mean: torch.Tensor | None
    std: torch.Tensor | None


def _coerce_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    try:
        return torch.as_tensor(value, dtype=torch.float32)
    except Exception:
        return None


def _extract_mean_std(state: Mapping[str, Any] | None) -> SkrlScaler | None:
    if not state:
        return None
    mean = None
    std = None
    for key in ("mean", "running_mean", "_mean"):
        if key in state:
            mean = _coerce_tensor(state[key])
            break
    for key in ("std", "running_std", "_std"):
        if key in state:
            std = _coerce_tensor(state[key])
            break
    if std is None:
        for key in ("var", "running_var", "running_variance", "variance", "_var"):
            if key in state:
                var = _coerce_tensor(state[key])
                if var is not None:
                    std = torch.sqrt(torch.clamp(var, min=0.0))
                break
    if mean is None and std is None:
        return None
    return SkrlScaler(mean=mean, std=std)


def _state_dict_from_candidate(candidate: Any) -> Mapping[str, Any] | None:
    if candidate is None:
        return None
    if isinstance(candidate, Mapping):
        if "state_dict" in candidate and isinstance(candidate["state_dict"], Mapping):
            return candidate["state_dict"]
        return candidate
    if hasattr(candidate, "state_dict"):
        return candidate.state_dict()
    return None


def _extract_preprocessor_state(payload: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    candidates = [
        payload.get(key),
        payload.get(f"{key}_state_dict"),
    ]
    preprocessors = payload.get("preprocessors")
    if isinstance(preprocessors, Mapping):
        candidates.append(preprocessors.get(key))
    agent = payload.get("agent")
    if isinstance(agent, Mapping):
        candidates.append(agent.get(key))
    agent_state = payload.get("agent_state_dict")
    if isinstance(agent_state, Mapping):
        candidates.append(agent_state.get(key))
    for candidate in candidates:
        state = _state_dict_from_candidate(candidate)
        if state:
            return state
    prefix = f"{key}."
    filtered = {k[len(prefix) :]: v for k, v in payload.items() if isinstance(k, str) and k.startswith(prefix)}
    if filtered:
        return filtered
    if isinstance(agent_state, Mapping):
        filtered = {
            k[len(prefix) :]: v for k, v in agent_state.items() if isinstance(k, str) and k.startswith(prefix)
        }
        if filtered:
            return filtered
    return None


def load_skrl_scalers(payload: Any) -> tuple[SkrlScaler | None, SkrlScaler | None]:
    if not isinstance(payload, Mapping):
        return None, None
    obs_state = _extract_preprocessor_state(payload, "observation_preprocessor")
    if obs_state is None:
        obs_state = _extract_preprocessor_state(payload, "state_preprocessor")
    value_state = _extract_preprocessor_state(payload, "value_preprocessor")
    return _extract_mean_std(obs_state), _extract_mean_std(value_state)
