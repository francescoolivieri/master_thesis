"""Minimal policy loader + command mapper for Crazyflie position tracking."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from source.isaac_pursuit_evasion.deployment.actor_policy_loader import (
    SimpleGaussianActor,
    load_actor_from_checkpoint,
    load_actor_policy_config,
)
from source.isaac_pursuit_evasion.dynamics.propellers import Drone_cfg


@dataclass
class PosTrackingPolicyConfig:
    control_mode: str = "RL_velocity"
    vel_scale: Sequence[float] = (5.0, 5.0, 5.0)
    yaw_rate_scale: float = math.pi
    thrust_to_weight: float = 3.15
    drone_name: str = "crazyflie_brushless"


class PosTrackingPolicy:
    def __init__(
        self,
        actor: SimpleGaussianActor,
        cfg: PosTrackingPolicyConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.actor = actor
        self.cfg = cfg or PosTrackingPolicyConfig()
        self.device = torch.device(device)
        self.actor.to(self.device)
        self.actor.eval()
        self.obs_scaler = getattr(self.actor, "obs_scaler", None)

        self._vel_scale = torch.as_tensor(self.cfg.vel_scale, device=self.device, dtype=torch.float32)
        self._yaw_rate_scale = float(self.cfg.yaw_rate_scale)
        self._thrust_to_weight = float(self.cfg.thrust_to_weight)

        self._drone_cfg = Drone_cfg(self.cfg.drone_name, device=self.device)
        self._weight = float(self._drone_cfg.mass) * 9.81
        self._thrust_cmd_max = float(getattr(self._drone_cfg, "thrust_cmd_max", 65535.0))
        self._thrust_cmd_scale = self._compute_thrust_cmd_scale(self._drone_cfg, self._thrust_cmd_max)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        *,
        actor_cfg: str | None = None,
        policy_cfg: PosTrackingPolicyConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> "PosTrackingPolicy":
        cfg = load_actor_policy_config(actor_cfg)
        actor = load_actor_from_checkpoint(checkpoint, cfg, device=device)
        return cls(actor, cfg=policy_cfg, device=device)

    def _scale_observation(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_scaler and self.obs_scaler.mean is not None and self.obs_scaler.std is not None:
            mean = self.obs_scaler.mean.to(self.device)
            std = self.obs_scaler.std.to(self.device)
            return (obs - mean) / (std + 1e-6)
        return obs

    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        obs = obs.to(self.device, dtype=torch.float32)
        obs = self._scale_observation(obs)
        with torch.no_grad():
            action = self.actor.act(obs, deterministic=deterministic)
        return action.clamp(-1.0, 1.0)

    def command(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        action = self.act(obs, deterministic=deterministic)
        mode = self.cfg.control_mode
        if mode == "RL_velocity":
            target_vel = action[..., :3] * self._vel_scale
            yaw_rate = action[..., 3:4] if action.shape[-1] > 3 else torch.zeros(
                (target_vel.shape[0], 1), device=self.device, dtype=target_vel.dtype
            )
            yaw_rate = yaw_rate * self._yaw_rate_scale
            return torch.cat((target_vel, yaw_rate), dim=-1)
        if mode == "RL_rates":
            target_rates = action[..., :3] * math.pi
            thrust_norm = action[..., 3:4] if action.shape[-1] > 3 else torch.zeros(
                (target_rates.shape[0], 1), device=self.device, dtype=target_rates.dtype
            )
            thrust_norm = thrust_norm.clamp(-1.0, 1.0)
            thrust = ((thrust_norm + 1.0) / 2.0) * self._weight * self._thrust_to_weight
            thrust_cmd = thrust / self._thrust_cmd_scale
            thrust_cmd = torch.clamp(thrust_cmd, 0.0, self._thrust_cmd_max)
            return torch.cat((target_rates, thrust_cmd), dim=-1)
        raise ValueError(f"Unsupported control_mode '{mode}'.")

    @staticmethod
    def _compute_thrust_cmd_scale(drone_cfg: Drone_cfg, thrust_cmd_max: float) -> float:
        k_eta = float(getattr(drone_cfg, "k_eta", 0.0))
        omega_max = float(getattr(drone_cfg, "motor_speed_max", getattr(drone_cfg, "omega_max", 0.0)))
        if k_eta > 0.0 and omega_max > 0.0 and thrust_cmd_max > 0.0:
            thrust_max = 4.0 * k_eta * omega_max**2
            return thrust_max / thrust_cmd_max
        return 1.0
