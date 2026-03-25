import math
import os.path as osp

import torch
import yaml


class Drone_cfg:
    def __init__(self, drone_params, device="cuda"):
        """Initialize Crazyflie brushless parameters."""
        self.device = device

        if isinstance(drone_params, str):
            cfg_path = osp.join(osp.dirname(__file__), "cfg", f"{drone_params}.yaml")
            with open(cfg_path, "r") as f:
                drone_params = yaml.safe_load(f)

        self.name = drone_params.get("name", "unknown")
        self.model = drone_params.get("model", "cf_brushless")

        arm_length_value = drone_params.get("arm_length", None)
        front_arm_length_value = drone_params.get("front_arm_length", arm_length_value)
        back_arm_length_value = drone_params.get("back_arm_length", arm_length_value)
        if arm_length_value is None:
            arm_length_value = 0.5 * (front_arm_length_value + back_arm_length_value)
        else:
            if front_arm_length_value is None:
                front_arm_length_value = arm_length_value
            if back_arm_length_value is None:
                back_arm_length_value = arm_length_value

        default_angle = math.pi / 4.0
        if "arm_angle_deg" in drone_params:
            default_angle = math.radians(float(drone_params["arm_angle_deg"]))
        front_arm_angle_value = drone_params.get("front_arm_angle", None)
        back_arm_angle_value = drone_params.get("back_arm_angle", None)
        if front_arm_angle_value is None and "front_arm_angle_deg" in drone_params:
            front_arm_angle_value = math.radians(float(drone_params["front_arm_angle_deg"]))
        if back_arm_angle_value is None and "back_arm_angle_deg" in drone_params:
            back_arm_angle_value = math.radians(float(drone_params["back_arm_angle_deg"]))
        if front_arm_angle_value is None:
            front_arm_angle_value = default_angle
        if back_arm_angle_value is None:
            back_arm_angle_value = default_angle

        self.arm_length = torch.tensor(arm_length_value, device=device, dtype=torch.float32)
        self.front_arm_length = torch.tensor(front_arm_length_value, device=device, dtype=torch.float32)
        self.back_arm_length = torch.tensor(back_arm_length_value, device=device, dtype=torch.float32)
        self.front_arm_angle = torch.tensor(front_arm_angle_value, device=device, dtype=torch.float32)
        self.back_arm_angle = torch.tensor(back_arm_angle_value, device=device, dtype=torch.float32)

        tau_m_value = drone_params.get("tau_m", drone_params.get("tau", 0.01))
        self.tau_m = torch.tensor(tau_m_value, device=device, dtype=torch.float32)
        self.tau = torch.full((4,), float(self.tau_m), device=device, dtype=torch.float32)

        k_eta_value = drone_params.get("k_eta", drone_params.get("thrust_coeff", 0.0))
        k_m_value = drone_params.get("k_m", drone_params.get("prop_drag_coeff", 0.0))
        self.k_eta = torch.tensor(k_eta_value, device=device, dtype=torch.float32)
        self.k_m = torch.tensor(k_m_value, device=device, dtype=torch.float32)
        self.thrust_coeff = torch.tensor(
            drone_params.get("thrust_coeff", float(self.k_eta)), device=device, dtype=torch.float32
        )
        self.prop_drag_coeff = torch.tensor(
            drone_params.get("prop_drag_coeff", float(self.k_m)), device=device, dtype=torch.float32
        )
        self.rotor_inertia = torch.tensor(drone_params.get("rotor_inertia", 0.0), device=device, dtype=torch.float32)

        self.motor_speed_min = torch.tensor(drone_params.get("motor_speed_min", 0.0), device=device, dtype=torch.float32)
        motor_speed_max_value = drone_params.get("motor_speed_max", drone_params.get("omega_max", 0.0))
        self.motor_speed_max = torch.tensor(motor_speed_max_value, device=device, dtype=torch.float32)
        self.omega_max = self.motor_speed_max.clone()

        self.k_aero_xy = torch.tensor(drone_params.get("k_aero_xy", 0.0), device=device, dtype=torch.float32)
        self.k_aero_z = torch.tensor(drone_params.get("k_aero_z", 0.0), device=device, dtype=torch.float32)

        mass_ref = drone_params.get("mass_ref", drone_params.get("mass", None))
        inertia_ref = drone_params.get("inertia_ref", drone_params.get("inertia", None))
        self.mass_ref = (
            torch.tensor(mass_ref, device=device, dtype=torch.float32) if mass_ref is not None else None
        )
        self.inertia_ref = (
            torch.tensor(inertia_ref, device=device, dtype=torch.float32) if inertia_ref is not None else None
        )

        mass_value = drone_params.get("mass", mass_ref)
        if mass_value is None:
            mass_value = 0.0
        self.mass = torch.tensor(mass_value, device=device, dtype=torch.float32)

        inertia_value = drone_params.get("inertia", inertia_ref)
        if inertia_value is None:
            inertia_value = [0.0, 0.0, 0.0]
        self.inertia = torch.tensor(inertia_value, device=device, dtype=torch.float32)
        self.inertia_tensor = torch.diag(self.inertia)

        self._rotor_positions = None
        self._rotor_directions = None
        self.f_to_TM = None
        self.TM_to_f = None
        self._build_mixer()

    def _build_mixer(self) -> None:
        if float(self.k_eta) <= 0.0:
            raise ValueError("k_eta must be positive for Crazyflie brushless mixing.")

        arm_length = float(self.arm_length)
        if arm_length <= 0.0:
            arm_length = float(0.5 * (self.front_arm_length + self.back_arm_length))
        r2o2 = math.sqrt(2.0) / 2.0
        rotor_positions = torch.tensor(
            [
                [r2o2, -r2o2, 0.0],
                [-r2o2, -r2o2, 0.0],
                [-r2o2, r2o2, 0.0],
                [r2o2, r2o2, 0.0],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        self._rotor_positions = arm_length * rotor_positions
        self._rotor_directions = torch.tensor([1.0, -1.0, 1.0, -1.0], device=self.device, dtype=torch.float32)

        k_ratio = self.k_m / self.k_eta.clamp_min(1e-12)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=torch.float32)
        torque_arms = torch.stack(
            [torch.linalg.cross(self._rotor_positions[i], z_axis)[:2] for i in range(4)],
            dim=1,
        )

        self.f_to_TM = torch.cat(
            [
                torch.ones((1, 4), device=self.device, dtype=torch.float32),
                torque_arms,
                (k_ratio * self._rotor_directions).view(1, 4),
            ],
            dim=0,
        )
        self.TM_to_f = torch.linalg.inv(self.f_to_TM)

    def get_mixer(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.f_to_TM is None or self.TM_to_f is None:
            self._build_mixer()
        return self.f_to_TM, self.TM_to_f

    def set_physical_params(self, mass: torch.Tensor | float, inertia: torch.Tensor | list) -> None:
        mass_tensor = torch.as_tensor(mass, device=self.device, dtype=torch.float32).view(())
        inertia_tensor = torch.as_tensor(inertia, device=self.device, dtype=torch.float32)
        if inertia_tensor.numel() == 9:
            inertia_matrix = inertia_tensor.view(3, 3)
            inertia_diag = torch.diagonal(inertia_matrix)
        elif inertia_tensor.numel() == 3:
            inertia_diag = inertia_tensor.view(3)
            inertia_matrix = torch.diag(inertia_diag)
        else:
            raise ValueError("Inertia must be a 3-vector or 3x3 matrix.")

        self.mass = mass_tensor
        self.inertia = inertia_diag
        self.inertia_tensor = inertia_matrix

        if self.mass_ref is not None:
            rel = torch.abs((mass_tensor - self.mass_ref) / self.mass_ref.clamp_min(1e-6))
            if float(rel) > 0.3:
                print(
                    f"[WARN] Crazyflie mass mismatch: USD={mass_tensor.item():.5f} kg, "
                    f"ref={self.mass_ref.item():.5f} kg"
                )
            else:
                print(
                    f"[INFO] Crazyflie mass check: USD={mass_tensor.item():.5f} kg, "
                    f"ref={self.mass_ref.item():.5f} kg"
                )
        if self.inertia_ref is not None:
            inertia_ref = self.inertia_ref.to(self.device)
            rel = torch.abs((inertia_diag - inertia_ref) / inertia_ref.clamp_min(1e-9))
            if float(rel.max()) > 0.3:
                print(f"[WARN] Crazyflie inertia mismatch: USD={inertia_diag.tolist()}, ref={inertia_ref.tolist()}")
            else:
                print(f"[INFO] Crazyflie inertia check: USD={inertia_diag.tolist()}, ref={inertia_ref.tolist()}")


class Propellers:
    def __init__(self, num_envs, drone_cfg: Drone_cfg, dt, use, device="cuda"):
        """Crazyflie brushless motor and aerodynamics model."""
        self.num_envs = num_envs
        self.use = use
        self.device = device
        self.drone_cfg = drone_cfg

        self.dt = float(dt)
        self.omega = torch.zeros((num_envs, 4), device=self.device, dtype=torch.float32)
        self.omega_rate = torch.zeros((num_envs, 4), device=self.device, dtype=torch.float32)
        self.forces = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float32)
        self.torques = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float32)

        self.k_eta = torch.full((num_envs, 1), float(drone_cfg.k_eta), device=self.device, dtype=torch.float32)
        self.k_m = torch.full((num_envs, 1), float(drone_cfg.k_m), device=self.device, dtype=torch.float32)
        self._base_f_to_TM, self._base_TM_to_f = drone_cfg.get_mixer()
        self._torque_arms = self._base_f_to_TM[1:3].to(self.device, dtype=torch.float32)
        rotor_dirs = getattr(drone_cfg, "_rotor_directions", None)
        if rotor_dirs is None:
            nominal_k_ratio = float(drone_cfg.k_m) / max(float(drone_cfg.k_eta), 1e-12)
            rotor_dirs = self._base_f_to_TM[3] / max(nominal_k_ratio, 1e-12)
        self._rotor_directions = torch.as_tensor(rotor_dirs, device=self.device, dtype=torch.float32).view(4)
        self.f_to_TM = self._base_f_to_TM.to(self.device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1, 1)
        self.TM_to_f = self._base_TM_to_f.to(self.device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1, 1)
        self.motor_speed_min = float(drone_cfg.motor_speed_min)
        self.motor_speed_max = float(drone_cfg.motor_speed_max)
        self.tau_m = torch.full((num_envs, 1), float(drone_cfg.tau_m), device=self.device, dtype=torch.float32)
        self.K_aero = torch.tensor(
            [float(drone_cfg.k_aero_xy), float(drone_cfg.k_aero_xy), float(drone_cfg.k_aero_z)],
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3).repeat(num_envs, 1)
        self._update_mixer(torch.arange(num_envs, device=self.device))

    def compute_motor_speeds_from_wrench(self, wrench_des: torch.Tensor) -> torch.Tensor:
        if self.TM_to_f.dim() == 2:
            f_des = torch.matmul(wrench_des, self.TM_to_f.t())
        else:
            f_des = torch.bmm(wrench_des.unsqueeze(1), self.TM_to_f.transpose(1, 2)).squeeze(1)
        motor_speed_squared = (f_des / self.k_eta).clamp_min(0.0)
        motor_speeds_des = torch.sqrt(motor_speed_squared)
        return motor_speeds_des.clamp(self.motor_speed_min, self.motor_speed_max)

    def compute_omega(self, omega_ref: torch.Tensor) -> torch.Tensor:
        if not self.use:
            self.omega = omega_ref
            return self.omega

        self.omega_rate = (omega_ref - self.omega) / self.tau_m
        self.omega = self.omega + self.dt * self.omega_rate
        self.omega = self.omega.clamp(self.motor_speed_min, self.motor_speed_max)
        return self.omega

    def compute_force_and_torque(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute forces/torques using body-frame linear velocity in state[:, 3:6]."""
        vel_body = state[:, 3:6]
        omega = self.omega
        omega_sq = omega**2

        motor_forces = self.k_eta * omega_sq
        if self.f_to_TM.dim() == 2:
            wrench = torch.matmul(motor_forces, self.f_to_TM.t())
        else:
            wrench = torch.bmm(motor_forces.unsqueeze(1), self.f_to_TM.transpose(1, 2)).squeeze(1)
        theta_dot = torch.sum(omega, dim=1, keepdim=True)
        drag = -theta_dot * self.K_aero * vel_body

        total_force = drag
        total_force[:, 2] += wrench[:, 0]
        total_torque = wrench[:, 1:]

        self.forces = total_force.unsqueeze(1)
        self.torques = total_torque.unsqueeze(1)
        return self.forces, self.torques

    def reset(self, env_ids):
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.to(dtype=torch.long, device=self.device)
        self.omega[env_ids] = torch.zeros(4, device=self.device, dtype=torch.float32).expand(len(env_ids), -1)
        self.omega_rate[env_ids] = 0.0

    def set_params(
        self,
        env_ids: torch.Tensor,
        k_eta: torch.Tensor | None = None,
        k_m: torch.Tensor | None = None,
        tau_m: torch.Tensor | None = None,
        k_aero: torch.Tensor | None = None,
    ) -> None:
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        if k_eta is not None:
            self.k_eta[env_ids] = k_eta.to(device=self.device, dtype=torch.float32).view(-1, 1)
        if k_m is not None:
            self.k_m[env_ids] = k_m.to(device=self.device, dtype=torch.float32).view(-1, 1)
        if tau_m is not None:
            self.tau_m[env_ids] = tau_m.to(device=self.device, dtype=torch.float32).view(-1, 1)
        if k_aero is not None:
            k_aero = k_aero.to(device=self.device, dtype=torch.float32)
            if k_aero.dim() == 1:
                k_aero = k_aero.view(-1, 1).repeat(1, 3)
            self.K_aero[env_ids] = k_aero
        if k_eta is not None or k_m is not None:
            self._update_mixer(env_ids)

    def _update_mixer(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        k_ratio = (self.k_m[env_ids] / self.k_eta[env_ids].clamp_min(1e-12)).view(-1, 1)
        self.f_to_TM[env_ids, 0, :] = 1.0
        self.f_to_TM[env_ids, 1:3, :] = self._torque_arms
        self.f_to_TM[env_ids, 3, :] = k_ratio * self._rotor_directions
        self.TM_to_f[env_ids] = torch.linalg.inv(self.f_to_TM[env_ids])
