from __future__ import annotations

from typing import Optional, Tuple

import math
import torch

from isaaclab.utils import math as math_utils

from ..dynamics.propellers import Drone_cfg


DEG2RAD = math.pi / 180.0

DEFAULT_GAINS = {
    "pos": {"kp": [2.0, 2.0, 2.0], "ki": [0.0, 0.0, 0.5], "kd": [0.0, 0.0, 0.0], "kff": [0.0, 0.0, 0.0]},
    "vel": {"kp": [25.0, 25.0, 25.0], "ki": [1.0, 1.0, 15.0], "kd": [0.0, 0.0, 0.0], "kff": [0.0, 0.0, 0.0]},
    "att": {"kp": [6.0, 6.0, 6.0], "ki": [3.0, 3.0, 1.0], "kd": [0.0, 0.0, 0.35], "kff": [0.0, 0.0, 0.0]},
    "rate": {"kp": [200.0, 200.0, 120.0], "ki": [400.0, 400.0, 16.7], "kd": [2.5, 2.5, 0.0], "kff": [0.0, 0.0, 0.0]},
}

DEFAULT_LIMITS = {
    "att_integral": [20.0 * DEG2RAD, 20.0 * DEG2RAD, 360.0 * DEG2RAD],
    "rate_integral": [33.3 * DEG2RAD, 33.3 * DEG2RAD, 166.7 * DEG2RAD],
    "pos_vel_max": [1.0, 1.0, 1.0],
    "roll_max": 20.0 * DEG2RAD,
    "pitch_max": 20.0 * DEG2RAD,
    "yaw_max_delta": 0.0,
    "thrust_base": 30000.0,
    "thrust_min": 20000.0,
    "thrust_cmd_max": 65535.0,
}


def _expand_to(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if tensor.dim() < reference.dim():
        for _ in range(reference.dim() - tensor.dim()):
            tensor = tensor.unsqueeze(0)
    return tensor.expand(reference.shape)


def _as_tensor(value, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _as_column(value, reference: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = _as_tensor(value, device, dtype)
    if tensor.dim() == 0:
        tensor = tensor.view(1, 1)
    elif tensor.dim() == 1:
        tensor = tensor.view(-1, 1)
    elif tensor.dim() == 2 and tensor.shape[-1] == 1:
        pass
    else:
        raise ValueError("Expected a scalar, (N,), or (N,1) tensor.")
    ref_batch = reference.shape[0] if reference.dim() > 0 else 1
    if tensor.shape[0] not in (1, ref_batch):
        raise ValueError("Batch dimension mismatch for yaw inputs.")
    if tensor.shape[0] == 1 and ref_batch != 1:
        tensor = tensor.expand(ref_batch, 1)
    return tensor


def _wrap_angle_rad(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def build_crazyflie_pid(
    num_envs: int,
    drone_cfg: Drone_cfg,
    dt: float,
    device: torch.device | str,
    pid_params: Optional[dict] = None,
) -> "CrazyfliePIDController":
    device = torch.device(device)
    pid = CrazyfliePIDController(
        dt=dt,
        drone_cfg=drone_cfg,
        num_envs=num_envs,
        device=str(device),
        params=pid_params or {},
    )
    inertia_tensor = getattr(drone_cfg, "inertia_tensor", None)
    if inertia_tensor is not None:
        inertia_tensor = torch.as_tensor(inertia_tensor, device=device, dtype=torch.float32)
        if inertia_tensor.dim() == 2:
            inertia_tensor = inertia_tensor.unsqueeze(0)
        inertia_tensor = inertia_tensor.repeat(num_envs, 1, 1)
    pid.set_physical_params(drone_cfg.mass, inertia_tensor)
    return pid


class PID:
    def __init__(
        self,
        kp: torch.Tensor,
        ki: torch.Tensor,
        kd: torch.Tensor,
        kff: Optional[torch.Tensor],
        dt: float,
        integral_limit: Optional[torch.Tensor] = None,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.kff = kff
        self.dt = float(dt)
        self.integral_limit = integral_limit
        if self.integral_limit is None:
            self.integral_limit = torch.full_like(self.ki, float("inf"))
        self._ki_safe = torch.where(self.ki.abs() > 1e-6, self.ki, torch.ones_like(self.ki))
        self._integral = None
        self._prev_error = None

    def _ensure_state(self, error: torch.Tensor) -> None:
        if self._integral is None or self._integral.shape != error.shape:
            self._integral = torch.zeros_like(error)
            self._prev_error = torch.zeros_like(error)

    def reset(self, mask: Optional[torch.Tensor] = None) -> None:
        if self._integral is None:
            return
        if mask is None:
            self._integral.zero_()
            self._prev_error.zero_()
            return
        self._integral[mask] = 0.0
        self._prev_error[mask] = 0.0

    def update_error(self, error: torch.Tensor, feedforward: Optional[torch.Tensor] = None) -> torch.Tensor:
        self._ensure_state(error)
        self._integral = self._integral + error * self.dt

        i_term = self.ki * self._integral
        i_term = torch.clamp(i_term, -self.integral_limit, self.integral_limit)
        self._integral = i_term / self._ki_safe

        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        if feedforward is not None and self.kff is not None:
            output = output + self.kff * feedforward
        return output


class CrazyfliePIDController:
    """Cascaded Crazyflie-style PID controller with explicit loop scheduling.

    All angular quantities are in radians.
    """

    def __init__(
        self,
        dt: float,
        drone_cfg: Optional[Drone_cfg] = None,
        num_envs: Optional[int] = None,
        device: str = "cuda",
        params: Optional[dict] = None,
    ) -> None:
        self.dt = float(dt)
        self.device = torch.device(device)
        self.drone_cfg = drone_cfg
        self._num_envs = num_envs

        mass_value = getattr(drone_cfg, "mass", 1.0)
        inertia_value = getattr(drone_cfg, "inertia", [1.0, 1.0, 1.0])
        self.mass = _as_tensor(mass_value, self.device, torch.float32).view(())
        self.inertia = _as_tensor(inertia_value, self.device, torch.float32).view(-1)
        self._inertia_tensor = None

        params = params or {}
        self.sim_rate_hz = float(params.get("sim_rate_hz", 1.0 / self.dt))
        self.posvel_rate_hz = float(params.get("pid_posvel_loop_rate_hz", 100.0))
        self.att_rate_hz = float(params.get("pid_loop_rate_hz", 500.0))
        self.posvel_decimation = max(1, int(round(self.sim_rate_hz / self.posvel_rate_hz)))
        self.att_decimation = max(1, int(round(self.sim_rate_hz / self.att_rate_hz)))
        self.posvel_dt = self.dt * self.posvel_decimation
        self.att_dt = self.dt * self.att_decimation

        pos_kp = self._resolve_axis_param(params, "pos_kp", ("xKp", "yKp", "zKp"), DEFAULT_GAINS["pos"]["kp"])
        pos_ki = self._resolve_axis_param(params, "pos_ki", ("xKi", "yKi", "zKi"), DEFAULT_GAINS["pos"]["ki"])
        pos_kd = self._resolve_axis_param(params, "pos_kd", ("xKd", "yKd", "zKd"), DEFAULT_GAINS["pos"]["kd"])
        pos_kff = self._resolve_axis_param(params, "pos_kff", ("xKff", "yKff", "zKff"), DEFAULT_GAINS["pos"]["kff"])

        vel_kp = self._resolve_axis_param(params, "vel_kp", ("vxKp", "vyKp", "vzKp"), DEFAULT_GAINS["vel"]["kp"])
        vel_ki = self._resolve_axis_param(params, "vel_ki", ("vxKi", "vyKi", "vzKi"), DEFAULT_GAINS["vel"]["ki"])
        vel_kd = self._resolve_axis_param(params, "vel_kd", ("vxKd", "vyKd", "vzKd"), DEFAULT_GAINS["vel"]["kd"])
        vel_kff = self._resolve_axis_param(params, "vel_kff", ("vxKff", "vyKff", "vzKff"), DEFAULT_GAINS["vel"]["kff"])
        vel_angle_scale = torch.as_tensor([DEG2RAD, DEG2RAD, 1.0], device=self.device, dtype=torch.float32)
        vel_kp = vel_kp * vel_angle_scale
        vel_ki = vel_ki * vel_angle_scale
        vel_kd = vel_kd * vel_angle_scale
        vel_kff = vel_kff * vel_angle_scale

        att_kp = self._resolve_axis_param(params, "att_kp", ("rollKp", "pitchKp", "yawKp"), DEFAULT_GAINS["att"]["kp"])
        att_ki = self._resolve_axis_param(params, "att_ki", ("rollKi", "pitchKi", "yawKi"), DEFAULT_GAINS["att"]["ki"])
        att_kd = self._resolve_axis_param(params, "att_kd", ("rollKd", "pitchKd", "yawKd"), DEFAULT_GAINS["att"]["kd"])
        att_kff = self._resolve_axis_param(params, "att_kff", ("rollKff", "pitchKff", "yawKff"), DEFAULT_GAINS["att"]["kff"])

        rate_kp = self._resolve_axis_param(params, "rate_kp", ("rollRateKp", "pitchRateKp", "yawRateKp"), DEFAULT_GAINS["rate"]["kp"])
        rate_ki = self._resolve_axis_param(params, "rate_ki", ("rollRateKi", "pitchRateKi", "yawRateKi"), DEFAULT_GAINS["rate"]["ki"])
        rate_kd = self._resolve_axis_param(params, "rate_kd", ("rollRateKd", "pitchRateKd", "yawRateKd"), DEFAULT_GAINS["rate"]["kd"])
        rate_kff = self._resolve_axis_param(params, "rate_kff", ("rollRateKff", "pitchRateKff", "yawRateKff"), DEFAULT_GAINS["rate"]["kff"])

        att_integral_limit = self._resolve_angle_vector(
            params, "att_integral_limit", "att_integral_limit_deg", DEFAULT_LIMITS["att_integral"]
        )
        rate_integral_limit = self._resolve_angle_vector(
            params, "rate_integral_limit", "rate_integral_limit_deg", DEFAULT_LIMITS["rate_integral"]
        )

        self.vel_max = torch.as_tensor(
            params.get("pos_vel_max", DEFAULT_LIMITS["pos_vel_max"]), device=self.device, dtype=torch.float32
        )

        self.roll_limit = self._resolve_angle_scalar(
            params,
            ("vel_roll_max", "roll_max"),
            ("vel_roll_max_deg", "roll_max_deg"),
            DEFAULT_LIMITS["roll_max"],
        )
        self.pitch_limit = self._resolve_angle_scalar(
            params,
            ("vel_pitch_max", "pitch_max"),
            ("vel_pitch_max_deg", "pitch_max_deg"),
            DEFAULT_LIMITS["pitch_max"],
        )
        self.yaw_max_delta = self._resolve_angle_scalar(
            params,
            ("yaw_max_delta",),
            ("yaw_max_delta_deg",),
            DEFAULT_LIMITS["yaw_max_delta"],
        )

        thrust_cmd_max = float(params.get("thrust_cmd_max", DEFAULT_LIMITS["thrust_cmd_max"]))
        thrust_cmd_scale = params.get("thrust_cmd_scale", None)
        if thrust_cmd_scale is None and self.drone_cfg is not None:
            k_eta = float(getattr(self.drone_cfg, "k_eta", 0.0))
            omega_max = float(getattr(self.drone_cfg, "motor_speed_max", getattr(self.drone_cfg, "omega_max", 0.0)))
            if k_eta > 0.0 and omega_max > 0.0 and thrust_cmd_max > 0.0:
                thrust_max = float(getattr(self.drone_cfg, "thrust_max", 0.0)) or 4.0 * k_eta * omega_max**2
                thrust_cmd_scale = thrust_max / thrust_cmd_max
        self.thrust_cmd_scale = float(thrust_cmd_scale) if thrust_cmd_scale is not None else 1.0
        self.vel_thrust_scale = float(params.get("vel_thrust_scale", params.get("thrust_scale", 1000.0)))

        self._thrust_base_from_params = "thrust_base_cmd" in params or "thrustBase" in params
        thrust_base_cmd = params.get("thrust_base_cmd", params.get("thrustBase", None))
        if thrust_base_cmd is None:
            thrust_base_cmd = (self.mass.item() * 9.81) / max(self.thrust_cmd_scale, 1e-6)
        self.thrust_base_cmd = float(thrust_base_cmd)
        self.thrust_min_cmd = float(params.get("thrust_min", params.get("thrustMin", DEFAULT_LIMITS["thrust_min"])))
        self.thrust_max_cmd = float(params.get("thrust_max", thrust_cmd_max))

        self.pos_pid = PID(pos_kp, pos_ki, pos_kd, pos_kff, dt=self.posvel_dt)
        self.vel_pid = PID(vel_kp, vel_ki, vel_kd, vel_kff, dt=self.posvel_dt)
        self.att_pid = PID(att_kp, att_ki, att_kd, att_kff, dt=self.att_dt, integral_limit=att_integral_limit)

        self.rate_kp = rate_kp
        self.rate_ki = rate_ki
        self.rate_kd = rate_kd
        self.rate_kff = rate_kff
        self._rate_integral_limit = rate_integral_limit
        if self._rate_integral_limit is None:
            self._rate_integral_limit = torch.full_like(self.rate_kp, float("inf"))

        self._step_count = 0
        self._vel_sp = None
        self._att_sp = None
        self._rate_sp = None
        self._thrust_cmd = None
        self._yaw_sp = None
        self._rate_integral = None
        self._prev_rate_meas = None

        self._command_handlers = {
            "position": self._cmd_position,
            "velocity": self._cmd_velocity,
            "attitude": self._cmd_attitude,
            "body_rate": self._cmd_body_rate,
        }

    def _resolve_axis_param(
        self,
        params: dict,
        new_key: str,
        legacy_keys: Tuple[str, str, str],
        default: list,
    ) -> torch.Tensor:
        if new_key in params:
            values = params[new_key]
        elif any(key in params for key in legacy_keys):
            values = [params.get(key, default[idx]) for idx, key in enumerate(legacy_keys)]
        else:
            values = default
        return torch.as_tensor(values, device=self.device, dtype=torch.float32)

    def _resolve_angle_scalar(
        self,
        params: dict,
        rad_keys: Tuple[str, ...],
        deg_keys: Tuple[str, ...],
        default: float,
    ) -> float:
        for key in rad_keys:
            if key in params:
                return float(params[key])
        for key in deg_keys:
            if key in params:
                return float(params[key]) * DEG2RAD
        return float(default)

    def _resolve_angle_vector(
        self,
        params: dict,
        rad_key: str,
        deg_key: str,
        default: list,
    ) -> torch.Tensor:
        if rad_key in params:
            values = params[rad_key]
            return torch.as_tensor(values, device=self.device, dtype=torch.float32)
        if deg_key in params:
            values = torch.as_tensor(params[deg_key], device=self.device, dtype=torch.float32)
            return values * DEG2RAD
        return torch.as_tensor(default, device=self.device, dtype=torch.float32)

    def set_physical_params(self, mass: Optional[torch.Tensor] = None, inertia_tensor: Optional[torch.Tensor] = None) -> None:
        if mass is not None:
            self.mass = _as_tensor(mass, self.device, torch.float32).view(())
            if not self._thrust_base_from_params:
                self.thrust_base_cmd = (self.mass.item() * 9.81) / max(self.thrust_cmd_scale, 1e-6)
        if inertia_tensor is not None:
            inertia_tensor = _as_tensor(inertia_tensor, self.device, torch.float32)
            if inertia_tensor.dim() == 2:
                inertia_tensor = inertia_tensor.unsqueeze(0)
            if inertia_tensor.dim() != 3 or inertia_tensor.shape[-2:] != (3, 3):
                raise ValueError("inertia_tensor must be (3,3) or (N,3,3).")
            self._inertia_tensor = inertia_tensor
            self.inertia = torch.diagonal(inertia_tensor, dim1=-2, dim2=-1).mean(dim=0)

    def set_inertia_tensor(self, inertia_tensor: torch.Tensor) -> None:
        self.set_physical_params(inertia_tensor=inertia_tensor)

    def set_rate_gains(
        self,
        rate_kp: Optional[torch.Tensor] = None,
        rate_ki: Optional[torch.Tensor] = None,
        rate_kd: Optional[torch.Tensor] = None,
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        def _ensure_batched(gains: torch.Tensor) -> torch.Tensor:
            if gains.dim() == 1 and self._num_envs is not None:
                return gains.view(1, -1).repeat(self._num_envs, 1)
            return gains

        if env_ids is None:
            if rate_kp is not None:
                self.rate_kp = _as_tensor(rate_kp, self.device, torch.float32)
            if rate_ki is not None:
                self.rate_ki = _as_tensor(rate_ki, self.device, torch.float32)
            if rate_kd is not None:
                self.rate_kd = _as_tensor(rate_kd, self.device, torch.float32)
            return

        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        if rate_kp is not None:
            self.rate_kp = _ensure_batched(self.rate_kp)
            self.rate_kp[env_ids] = _as_tensor(rate_kp, self.device, torch.float32)
        if rate_ki is not None:
            self.rate_ki = _ensure_batched(self.rate_ki)
            self.rate_ki[env_ids] = _as_tensor(rate_ki, self.device, torch.float32)
        if rate_kd is not None:
            self.rate_kd = _ensure_batched(self.rate_kd)
            self.rate_kd[env_ids] = _as_tensor(rate_kd, self.device, torch.float32)

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if env_ids is None:
            self.pos_pid.reset()
            self.vel_pid.reset()
            self.att_pid.reset()
            self._rate_integral = None
            self._prev_rate_meas = None
            self._vel_sp = None
            self._att_sp = None
            self._rate_sp = None
            self._thrust_cmd = None
            self._yaw_sp = None
            self._step_count = 0
            return

        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        self.pos_pid.reset(mask=env_ids)
        self.vel_pid.reset(mask=env_ids)
        self.att_pid.reset(mask=env_ids)
        if self._rate_integral is not None:
            self._rate_integral[env_ids] = 0.0
        if self._prev_rate_meas is not None:
            self._prev_rate_meas[env_ids] = 0.0
        if self._vel_sp is not None:
            self._vel_sp[env_ids] = 0.0
        if self._att_sp is not None:
            self._att_sp[env_ids] = 0.0
        if self._rate_sp is not None:
            self._rate_sp[env_ids] = 0.0
        if self._thrust_cmd is not None:
            self._thrust_cmd[env_ids] = self.thrust_base_cmd
        if self._yaw_sp is not None:
            self._yaw_sp[env_ids] = float("nan")

    def __call__(
        self,
        root_state: torch.Tensor,
        target_pos: Optional[torch.Tensor] = None,
        target_vel: Optional[torch.Tensor] = None,
        target_attitude: Optional[torch.Tensor] = None,
        target_body_rates: Optional[torch.Tensor] = None,
        target_yaw: Optional[torch.Tensor] = None,
        target_yaw_rate: Optional[torch.Tensor] = None,
        thrust_cmd: Optional[torch.Tensor] = None,
        *,
        command_level: str,
        body_rates_in_body_frame: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if root_state.dim() == 1:
            root_state = root_state.unsqueeze(0)

        pos, quat, lin_vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        pos = pos.to(device=self.device, dtype=torch.float32)
        quat = quat.to(device=self.device, dtype=torch.float32)
        lin_vel = lin_vel.to(device=self.device, dtype=torch.float32)
        ang_vel = ang_vel.to(device=self.device, dtype=torch.float32)

        if not body_rates_in_body_frame:
            ang_vel = math_utils.quat_apply_inverse(quat, ang_vel)

        command_level = str(command_level).lower()
        if command_level not in self._command_handlers:
            raise ValueError(f"Unsupported command_level '{command_level}'.")

        update_posvel = (self._step_count % self.posvel_decimation) == 0
        update_att = (self._step_count % self.att_decimation) == 0
        self._step_count += 1

        batch = pos.shape[0]
        self._ensure_buffers(batch)

        target_pos = pos if target_pos is None else _expand_to(_as_tensor(target_pos, self.device, pos.dtype), pos)
        target_vel = torch.zeros_like(lin_vel) if target_vel is None else _expand_to(
            _as_tensor(target_vel, self.device, lin_vel.dtype), lin_vel
        )

        euler = math_utils.euler_xyz_from_quat(quat)
        att_actual = torch.stack(euler, dim=-1)
        yaw_actual = att_actual[..., 2:3]
        if target_yaw is not None:
            target_yaw = _as_column(target_yaw, yaw_actual, self.device, yaw_actual.dtype)
        if target_yaw_rate is not None:
            target_yaw_rate = _as_column(target_yaw_rate, yaw_actual, self.device, yaw_actual.dtype)

        yaw_sp = self._update_yaw_setpoint(yaw_actual, target_yaw, target_yaw_rate)

        if command_level == "velocity" and update_posvel:
            self._vel_sp = target_vel

        handler = self._command_handlers[command_level]
        handler(
            pos=pos,
            lin_vel=lin_vel,
            ang_vel=ang_vel,
            att_actual=att_actual,
            yaw_actual=yaw_actual,
            yaw_sp=yaw_sp,
            target_pos=target_pos,
            target_vel=target_vel,
            target_attitude=target_attitude,
            target_body_rates=target_body_rates,
            thrust_cmd=thrust_cmd,
            update_posvel=update_posvel,
            update_att=update_att,
        )

        moment = self._rate_pid_to_moment(self._rate_sp, ang_vel)
        thrust = self._thrust_cmd * self.thrust_cmd_scale

        return thrust, moment

    def _ensure_buffers(self, batch: int) -> None:
        if self._vel_sp is None or self._vel_sp.shape[0] != batch:
            self._vel_sp = torch.zeros((batch, 3), device=self.device)
        if self._att_sp is None or self._att_sp.shape[0] != batch:
            self._att_sp = torch.zeros((batch, 3), device=self.device)
        if self._rate_sp is None or self._rate_sp.shape[0] != batch:
            self._rate_sp = torch.zeros((batch, 3), device=self.device)
        if self._thrust_cmd is None or self._thrust_cmd.shape[0] != batch:
            self._thrust_cmd = torch.full((batch, 1), self.thrust_base_cmd, device=self.device)

    def _cmd_position(self, **ctx) -> None:
        if ctx["update_posvel"]:
            vel_sp = self.pos_pid.update_error(ctx["target_pos"] - ctx["pos"])
            vel_sp = torch.clamp(vel_sp, -self.vel_max, self.vel_max)
            self._vel_sp = vel_sp
        self._cmd_velocity(**ctx)

    def _cmd_velocity(self, **ctx) -> None:
        yaw_sp = ctx["yaw_sp"]
        yaw_sp_scalar = yaw_sp.squeeze(-1)
        self._att_sp[:, 2] = yaw_sp_scalar
        
        if ctx["update_posvel"]:
            yaw_actual_scalar = ctx["yaw_actual"].squeeze(-1)
            c = torch.cos(yaw_actual_scalar)
            s = torch.sin(yaw_actual_scalar)
            lin_vel = ctx["lin_vel"]

            vel_body_x = c * lin_vel[:, 0] + s * lin_vel[:, 1]
            vel_body_y = -s * lin_vel[:, 0] + c * lin_vel[:, 1]
            vel_sp_body_x = c * self._vel_sp[:, 0] + s * self._vel_sp[:, 1]
            vel_sp_body_y = -s * self._vel_sp[:, 0] + c * self._vel_sp[:, 1]

            vel_error = torch.stack(
                [
                    vel_sp_body_x - vel_body_x,
                    vel_sp_body_y - vel_body_y,
                    self._vel_sp[:, 2] - lin_vel[:, 2],
                ],
                dim=-1,
            )

            vel_out = self.vel_pid.update_error(vel_error)
            pitch_cmd = (vel_out[:, 0]).clamp(-self.pitch_limit, self.pitch_limit)
            roll_cmd = (-vel_out[:, 1]).clamp(-self.roll_limit, self.roll_limit)
            thrust_cmd_out = self.thrust_base_cmd + vel_out[:, 2] * self.vel_thrust_scale
            thrust_cmd_out = torch.clamp(thrust_cmd_out, self.thrust_min_cmd, self.thrust_max_cmd)

            self._att_sp = torch.stack((roll_cmd, pitch_cmd, yaw_sp_scalar), dim=-1)
            self._thrust_cmd = thrust_cmd_out.unsqueeze(-1)

        if ctx["update_att"]:
            self._update_rate_from_attitude(ctx["att_actual"], ctx["update_att"])

    def _cmd_attitude(self, **ctx) -> None:
        yaw_sp = ctx["yaw_sp"]
 
        if ctx["update_att"]:
            target_att = ctx["target_attitude"]
            att_des = ctx["att_actual"] if target_att is None else _expand_to(
                _as_tensor(target_att, self.device, ctx["ang_vel"].dtype), ctx["ang_vel"]
            )
            att_des = att_des.clone()
            att_des[..., 2:3] = yaw_sp
            self._att_sp = att_des

        if ctx["thrust_cmd"] is not None:
            self._thrust_cmd = _as_tensor(ctx["thrust_cmd"], self.device, torch.float32).view(-1, 1)

        self._update_rate_from_attitude(ctx["att_actual"], ctx["update_att"])

    def _cmd_body_rate(self, **ctx) -> None:
        if ctx["target_body_rates"] is None:
            self._rate_sp = torch.zeros_like(ctx["ang_vel"])
        else:
            rate_sp = _expand_to(
                _as_tensor(ctx["target_body_rates"], self.device, ctx["ang_vel"].dtype),
                ctx["ang_vel"],
            )
            self._rate_sp = rate_sp

        if ctx["thrust_cmd"] is not None:
            self._thrust_cmd = _as_tensor(ctx["thrust_cmd"], self.device, torch.float32).view(-1, 1)

    def _update_rate_from_attitude(self, att_actual: torch.Tensor, update_att: bool) -> None:
        if update_att:
            att_error = _wrap_angle_rad(self._att_sp - att_actual)
            self._rate_sp = self.att_pid.update_error(att_error)

    def _rate_pid_to_moment(self, rate_sp: torch.Tensor, rate_meas: torch.Tensor) -> torch.Tensor:
        if self._rate_integral is None or self._rate_integral.shape != rate_sp.shape:
            self._rate_integral = torch.zeros_like(rate_sp)
        if self._prev_rate_meas is None or self._prev_rate_meas.shape != rate_meas.shape:
            self._prev_rate_meas = rate_meas.clone()

        rate_error = rate_sp - rate_meas
        self._rate_integral = self._rate_integral + rate_error * self.dt

        limits = _expand_to(self._rate_integral_limit, self._rate_integral)
        self._rate_integral = torch.clamp(self._rate_integral, -limits, limits)

        rate_meas_dot = (rate_meas - self._prev_rate_meas) / self.dt
        self._prev_rate_meas = rate_meas.clone()

        omega_dot = (
            self.rate_kp * rate_error
            + self.rate_ki * self._rate_integral
            - self.rate_kd * rate_meas_dot
        )
        if self._inertia_tensor is not None:
            moment = torch.bmm(self._inertia_tensor, omega_dot.unsqueeze(-1)).squeeze(-1)
        else:
            moment = self.inertia.view(1, 3) * omega_dot
        return moment

    def _update_yaw_setpoint(
        self,
        yaw_actual: torch.Tensor,
        yaw_target_rad: Optional[torch.Tensor],
        yaw_rate_rad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self._yaw_sp is None or self._yaw_sp.shape != yaw_actual.shape:
            self._yaw_sp = yaw_actual.clone()

        self._yaw_sp = torch.where(torch.isnan(self._yaw_sp), yaw_actual.clone(), self._yaw_sp)

        if yaw_rate_rad is not None:
            self._yaw_sp = _wrap_angle_rad(self._yaw_sp + yaw_rate_rad * self.dt)

        elif yaw_target_rad is not None:
            self._yaw_sp = _wrap_angle_rad(yaw_target_rad)

        if self.yaw_max_delta > 0.0:
            delta = _wrap_angle_rad(self._yaw_sp - yaw_actual)
            delta = torch.clamp(delta, -self.yaw_max_delta, self.yaw_max_delta)
            self._yaw_sp = _wrap_angle_rad(yaw_actual + delta)
        return self._yaw_sp
