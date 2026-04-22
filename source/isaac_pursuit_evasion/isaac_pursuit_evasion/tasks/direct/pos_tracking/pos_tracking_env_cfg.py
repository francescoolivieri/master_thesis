"""Configuration for the Crazyflie position-tracking environment."""
from __future__ import annotations

import math
from typing import Literal

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class DomainRandomizationCfg:
    """Domain randomization configuration."""

    enable: bool = False
    debug_checks: bool = True

    # Uniform scale range for mass/inertia/k_eta/k_m/tau.
    scale_min: float = 0.85
    scale_max: float = 1.15

    randomize_mass: bool = True
    randomize_inertia: bool = True
    randomize_k_eta: bool = True
    randomize_k_m: bool = True
    randomize_tau: bool = True
    randomize_k_aero: bool = True
    randomize_rate_gains: bool = True

    # Aerodynamics scaling.
    k_aero_xy_min_scale: float = 0.5
    k_aero_xy_max_scale: float = 2.0
    k_aero_z_min_scale: float = 0.5
    k_aero_z_max_scale: float = 2.0

    # Rate gains scaling.
    rate_kp_min_scale: float = 0.85
    rate_kp_max_scale: float = 1.15
    rate_ki_min_scale: float = 0.85
    rate_ki_max_scale: float = 1.15
    rate_kd_min_scale: float = 0.7
    rate_kd_max_scale: float = 1.2


@configclass
class PosTrackingEnvCfg(DirectRLEnvCfg):
    """Configuration for the position tracking environment."""

    # Simulation settings
    episode_length_s = 10.0
    sim_frequency = 500
    policy_rate_hz = 50
    pid_loop_rate_hz = 500
    pid_posvel_loop_rate_hz = 100
    decimation = sim_frequency // policy_rate_hz
    action_space = 4
    observation_space = 0  # computed in env
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / sim_frequency, render_interval=decimation)
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        debug_vis=False,
    )

    # Arena bounds [min, max] for [x, y, z]
    arena_min = (-2.5, -2.0, 0.0) 
    arena_max = (2.5, 2.0, 2.0)   
    collision_altitude: float = 0.2
    arena_margin: float = 0.0

    enable_walls: bool = True
    wall_thickness: float = 0.05
    wall_extra_margin: float = 0.5

    enable_pillars: bool = True
    pillar_positions_xy: tuple[tuple[float, float], ...] = ((-0.7, 0.0), (0.7, 0.0))
    pillar_radius: float = 0.18
    pillar_height: float = 1.8
    drone_collision_radius: float = 0.09

    env_spacing = max(
        (arena_max[0] - arena_min[0]) + 2 * wall_extra_margin + 2 * wall_thickness,
        (arena_max[1] - arena_min[1]) + 2 * wall_extra_margin + 2 * wall_thickness,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=env_spacing, replicate_physics=True)

    # Robot configuration
    robot = None  # lazily populated
    drone_name: str = "crazyflie_brushless"

    # Control interface
    control_mode: Literal["RL_velocity", "RL_rates"] = "RL_velocity"
    vel_scale = (5.0, 5.0, 5.0)
    yaw_rate_scale: float = math.pi
    thrust_to_weight: float = 3.15
    use_position_controller: bool = False

    # Task flags
    flag_yaw_tracking: bool = False
    flag_penalize_linvel: bool = False
    flag_action_smoothness_penalty: bool = False
    enable_obstacle_observations: bool = True

    # Camera flags
    enable_cameras: bool = False
    camera_tiled_max_megapixels: float = 1.0
    save_camera_images: bool = False
    camera_image_dir: str = "logs/pos_tracking/camera"
    camera_overlay_text: bool = False

    # Debug visualization
    debug_vis: bool = True
    debug_visualizer: bool = True

    # Domain randomization
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()

    # Reference sampling (local coordinates)
    ref_pos_min = (-1.5, -1.0, 0.3)
    ref_pos_max = (1.5, 1.0, 1.5)
    ref_yaw_range = (-math.pi, math.pi)
    ref_update_interval_s: float = 0.0  # 0 means static target per episode
    reference_obstacle_clearance: float = 0.2

    # Rewards (positive weights; signs applied in env)
    reward_pos: float = 1.0
    reward_pos_scale: float = 1.5
    reward_yaw: float = 0.3
    reward_body_rates: float = 0.001
    reward_lin_vel: float = 0.02
    reward_action_smoothness: float = 0.02
    reward_crash: float = 10.0
    reward_out_of_bounds: float = 10.0
    reward_pillar_collision: float = 10.0

    # Success criteria
    pos_tolerance: float = 0.15
    yaw_tolerance: float = 0.25
    success_hold_time_s: float = 1.0 # prev: 0.5
    terminate_on_success: bool = True

    total_timesteps = episode_length_s * sim_frequency


def pos_tracking_cfg(
    num_envs: int = 512,
    control_mode: str = "RL_velocity",
    domain_randomization: bool | None = None,
) -> PosTrackingEnvCfg:
    """Base configuration helper for position-tracking tasks."""
    cfg = PosTrackingEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.control_mode = "RL_rates" if control_mode == "RL_rates" else "RL_velocity"
    if domain_randomization is not None:
        cfg.domain_randomization.enable = bool(domain_randomization)
    return cfg


def pos_tracking_velocity_cfg(num_envs: int = 512, domain_randomization: bool | None = None) -> PosTrackingEnvCfg:
    """Position-tracking config for RL_velocity control."""
    return pos_tracking_cfg(num_envs=num_envs, control_mode="RL_velocity", domain_randomization=domain_randomization)


def pos_tracking_rates_cfg(num_envs: int = 512, domain_randomization: bool | None = None) -> PosTrackingEnvCfg:
    """Position-tracking config for RL_rates control."""
    return pos_tracking_cfg(num_envs=num_envs, control_mode="RL_rates", domain_randomization=domain_randomization)
