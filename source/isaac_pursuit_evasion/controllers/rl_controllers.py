from __future__ import annotations

from typing import Callable, Optional, Union

import torch
from tensordict import TensorDictBase

from source.isaac_pursuit_evasion.controllers.crazy_controller import build_crazyflie_pid
from source.isaac_pursuit_evasion.dynamics.propellers import Drone_cfg

from isaaclab.utils import math as math_utils

TensorDictLike = TensorDictBase
PolicyOutput = Union[torch.Tensor, TensorDictBase]
PolicyCallable = Callable[[TensorDictBase], PolicyOutput]


def _ensure_tensordict_device(td: TensorDictBase, device: torch.device) -> TensorDictBase:
    """Move a TensorDict to the expected device if necessary."""
    if td.device == device:
        return td
    return td.to(device)


class RLVelocityController:
    """Runs an RL policy that outputs velocity targets (and yaw-rate) for the cascaded controller."""

    def __init__(
        self,
        num_envs: int,
        drone_cfg: Drone_cfg,
        policy: PolicyCallable,
        dt: float,
        device: str = "cuda",
        action_key: str = "action",
        root_state_key: str = "root_state",
        vel_scale: Optional[torch.Tensor] = None,
        yaw_rate_scale: Optional[Union[float, torch.Tensor]] = None,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.policy = policy
        self.action_key = action_key
        self.root_state_key = root_state_key
        self.dt = float(dt)

        if vel_scale is None:
            self.vel_scale = torch.tensor([5.0, 5.0, 5.0], device=self.device, dtype=torch.float32)
        else:
            self.vel_scale = torch.as_tensor(vel_scale, device=self.device, dtype=torch.float32)

        if yaw_rate_scale is None:
            self.yaw_rate_scale = torch.pi
        else:
            self.yaw_rate_scale = torch.as_tensor(yaw_rate_scale, device=self.device, dtype=torch.float32)

        # Try to place the policy on the same device (if it is a torch.nn.Module)
        if hasattr(policy, "to"):
            policy.to(self.device)
        if hasattr(policy, "eval"):
            policy.eval()

    def __call__(self, td: TensorDictBase) -> torch.Tensor:
        td = _ensure_tensordict_device(td, self.device)
        actions = self._run_policy(td)

        target_vel = actions[..., :3]
        if self.vel_scale is not None:
            target_vel = target_vel * self.vel_scale

        yaw_rate = actions[..., 3:4] if actions.shape[-1] > 3 else torch.zeros(
            (target_vel.shape[0], 1), device=self.device, dtype=target_vel.dtype
        )
        yaw_rate = yaw_rate * self.yaw_rate_scale

        return torch.cat((target_vel, yaw_rate), dim=-1)

    def _run_policy(self, td: TensorDictBase) -> torch.Tensor:
        with torch.no_grad():
            output = self.policy(td)

        if isinstance(output, TensorDictBase):
            if self.action_key in output.keys(include_nested=True, leaves_only=True):
                return output.get(self.action_key)
            if "action" in output.keys(include_nested=True, leaves_only=True):
                return output.get("action")
            raise KeyError(f"TensorDict output from policy is missing '{self.action_key}' key.")
        if isinstance(output, torch.Tensor):
            return output

        raise TypeError(f"Unsupported policy output type: {type(output)}")


class RLBodyRatesController:
    """Runs an RL policy that outputs body-rate targets and thrust commands."""

    def __init__(
        self,
        num_envs: int,
        drone_cfg: Drone_cfg,
        policy: PolicyCallable,
        dt: float,
        device: str = "cuda",
        action_key: str = "action",
        root_state_key: str = "root_state",
        body_rate_key: str = "body_rate",
        thrust_scale: Optional[float] = None,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.policy = policy
        self.action_key = action_key
        self.root_state_key = root_state_key
        self.body_rate_key = body_rate_key
        self.dt = float(dt)
        self.thrust_to_weight = float(thrust_scale) if thrust_scale is not None else 3.15
        weight = float(drone_cfg.mass) * 9.81
        self.weight = torch.tensor(weight, device=self.device, dtype=torch.float32)
        self.thrust_cmd_max = float(getattr(drone_cfg, "thrust_cmd_max", 65535.0))
        self.thrust_cmd_scale = self._compute_thrust_cmd_scale(drone_cfg, self.thrust_cmd_max)

        if hasattr(policy, "to"):
            policy.to(self.device)
        if hasattr(policy, "eval"):
            policy.eval()

    def __call__(self, td: TensorDictBase) -> torch.Tensor:
        td = _ensure_tensordict_device(td, self.device)
        actions = self._run_policy(td)

        target_rates = actions[..., :3] * torch.pi  # scale to rad/s
        thrust_norm = actions[..., 3:4] if actions.shape[-1] > 3 else torch.zeros(
            (target_rates.shape[0], 1), device=self.device, dtype=target_rates.dtype
        )
        thrust_norm = thrust_norm.clamp(-1.0, 1.0)
        thrust = ((thrust_norm + 1.0) / 2.0) * self.weight * self.thrust_to_weight
        thrust_cmd = thrust / self.thrust_cmd_scale
        thrust_cmd = torch.clamp(thrust_cmd, 0.0, self.thrust_cmd_max)

        return torch.cat((target_rates, thrust_cmd), dim=-1)

    def _run_policy(self, td: TensorDictBase) -> torch.Tensor:
        with torch.no_grad():
            output = self.policy(td)

        if isinstance(output, TensorDictBase):
            if self.action_key in output.keys(include_nested=True, leaves_only=True):
                return output.get(self.action_key)
            if "action" in output.keys(include_nested=True, leaves_only=True):
                return output.get("action")
            raise KeyError(f"TensorDict output from policy is missing '{self.action_key}' key.")
        if isinstance(output, torch.Tensor):
            return output

        raise TypeError(f"Unsupported policy output type: {type(output)}")

    @staticmethod
    def _compute_thrust_cmd_scale(drone_cfg: Drone_cfg, thrust_cmd_max: float) -> float:
        k_eta = float(getattr(drone_cfg, "k_eta", 0.0))
        omega_max = float(getattr(drone_cfg, "motor_speed_max", getattr(drone_cfg, "omega_max", 0.0)))
        if k_eta > 0.0 and omega_max > 0.0 and thrust_cmd_max > 0.0:
            thrust_max = 4.0 * k_eta * omega_max**2
            return thrust_max / thrust_cmd_max
        return 1.0


class CrazyflieRLVelocityWrapper:
    """Crazyflie wrapper that maps RL velocity commands to thrust/moment."""

    def __init__(
        self,
        num_envs: int,
        drone_cfg: Drone_cfg,
        policy: PolicyCallable,
        dt: float,
        pid_dt: float | None = None,
        device: str = "cuda",
        action_key: str = "action",
        root_state_key: str = "root_state",
        vel_scale: Optional[torch.Tensor] = None,
        yaw_rate_scale: Optional[Union[float, torch.Tensor]] = None,
        pid_params: Optional[dict] = None,
    ) -> None:
        self.device = torch.device(device)
        self.root_state_key = root_state_key
        self.controller = RLVelocityController(
            num_envs=num_envs,
            drone_cfg=drone_cfg,
            policy=policy,
            dt=dt,
            device=device,
            action_key=action_key,
            root_state_key=root_state_key,
            vel_scale=vel_scale,
            yaw_rate_scale=yaw_rate_scale,
        )
        pid_dt = dt if pid_dt is None else pid_dt
        self.pid = build_crazyflie_pid(num_envs, drone_cfg, pid_dt, self.device, pid_params)

    def __call__(self, td: TensorDictBase) -> tuple[torch.Tensor, torch.Tensor]:
        td = _ensure_tensordict_device(td, self.device)
        cmd = self.command(td)
        root_state = td.get(self.root_state_key)
        wrench = self.wrench_from_command(root_state, cmd)
        return cmd, wrench

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        self.pid.reset(env_ids)

    def command(self, td: TensorDictBase) -> torch.Tensor:
        td = _ensure_tensordict_device(td, self.device)
        return self.controller(td)

    def wrench_from_command(self, root_state: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        thrust, moment = self.pid(
            root_state=root_state,
            target_vel=cmd[:, :3],
            target_yaw_rate=cmd[:, 3:4],
            command_level="velocity",
        )
        return torch.cat((thrust, moment), dim=-1)


class CrazyflieRLBodyRatesWrapper:
    """Crazyflie wrapper that maps RL body-rate commands to thrust/moment."""

    def __init__(
        self,
        num_envs: int,
        drone_cfg: Drone_cfg,
        policy: PolicyCallable,
        dt: float,
        pid_dt: float | None = None,
        device: str = "cuda",
        action_key: str = "action",
        root_state_key: str = "root_state",
        body_rate_key: str = "body_rate",
        thrust_scale: Optional[float] = None,
        pid_params: Optional[dict] = None,
    ) -> None:
        self.device = torch.device(device)
        self.root_state_key = root_state_key
        self.controller = RLBodyRatesController(
            num_envs=num_envs,
            drone_cfg=drone_cfg,
            policy=policy,
            dt=dt,
            device=device,
            action_key=action_key,
            root_state_key=root_state_key,
            body_rate_key=body_rate_key,
            thrust_scale=thrust_scale,
        )
        pid_dt = dt if pid_dt is None else pid_dt
        self.pid = build_crazyflie_pid(num_envs, drone_cfg, pid_dt, self.device, pid_params)

    def __call__(self, td: TensorDictBase) -> tuple[torch.Tensor, torch.Tensor]:
        td = _ensure_tensordict_device(td, self.device)
        cmd = self.command(td)
        root_state = td.get(self.root_state_key)
        wrench = self.wrench_from_command(root_state, cmd)
        return cmd, wrench

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        self.pid.reset(env_ids)

    def command(self, td: TensorDictBase) -> torch.Tensor:
        td = _ensure_tensordict_device(td, self.device)
        return self.controller(td)

    def wrench_from_command(self, root_state: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        thrust, moment = self.pid(
            root_state=root_state,
            target_body_rates=cmd[:, :3],
            thrust_cmd=cmd[:, 3:4],
            command_level="body_rate",
        )
        return torch.cat((thrust, moment), dim=-1)
