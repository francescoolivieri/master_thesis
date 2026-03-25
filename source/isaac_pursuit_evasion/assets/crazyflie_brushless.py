"""Configuration for the Crazyflie Brushless pursuer/evader."""
import os.path as osp

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_mul
import torch


ASSET_DIR = osp.join(osp.dirname(__file__), "Crazyflie")

def _make_cfg(usd_name: str, prim_path: str) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=osp.join(ASSET_DIR, usd_name),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            joint_pos={
                ".*": 0.0,
            },
            joint_vel={
                ".*": 0.0,
            },
        ),
        actuators={
            "dummy": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )


CrazyflieBrushlessPursuer = _make_cfg("crazyflie_brushless_pursuer.usd", "/World/Pursuer")
CrazyflieBrushlessEvader = _make_cfg("crazyflie_brushless_evader.usd", "/World/Evader")


def fpv_camera_cfg(
    width: int = 320,  # 640
    height: int = 240,  # 480
    frequency: float = 30,
    fx: float = 92.38,
    fy: float = 92.38,
    cx: float = 160.0,
    cy: float = 120.0,
    tilt_deg: float = -0.0,
    data_types: list = ["rgb", "depth", "semantic_segmentation"],
    prim_path: str = "/World/Pursuer",
    camera_link_path: str = "body/camera_link",
    tiled: bool = False,
) -> CameraCfg:
    """Return a reusable FPV camera configuration for the Crazyflie Brushless pursuer."""
    import math

    tilt_rad = tilt_deg * 3.14159265 / 180.0
    offset_cfg = CameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),  # overwritten below
        convention="world",
    )
    half = tilt_rad * 0.5
    offset_cfg.rot = (math.cos(half), 0.0, math.sin(half), 0.0)

    intrinsic_matrix = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    spawn = PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=intrinsic_matrix,
        width=width,
        height=height,
        clipping_range=(0.02, 20.0),
        projection_type="pinhole",
        lock_camera=True,
    )

    camera_link_path = camera_link_path.strip("/")
    cfg_type = TiledCameraCfg if tiled else CameraCfg
    return cfg_type(
        prim_path=f"{prim_path}/{camera_link_path}/fpv_camera",
        update_period=1.0 / frequency,
        height=height,
        width=width,
        data_types=data_types,
        spawn=spawn,
        offset=offset_cfg,
    )


def fpv_camera_center_line(length: float = 5.0, device: str = "cuda"):
    """Camera-frame center line endpoints for the FPV camera (+X forward, ROS convention)."""
    origin = torch.zeros(3, device=device, dtype=torch.float32)
    line_end = torch.tensor([length, 0.0, 0.0], device=device, dtype=torch.float32)
    return origin, line_end


def transform_camera_line(
    origin: torch.Tensor,
    line_end: torch.Tensor,
    link_pos: torch.Tensor,
    link_quat: torch.Tensor,
    cam_cfg: CameraCfg | None = None,
):
    """Transform camera-frame line into world coordinates using the camera_link pose."""
    cam_cfg = cam_cfg if cam_cfg is not None else fpv_camera_cfg()
    link_pos = link_pos.view(-1, 3)
    link_quat = link_quat.view(-1, 4)
    batch = link_pos.shape[0]

    offset_pos = torch.tensor(cam_cfg.offset.pos, device=link_pos.device, dtype=link_pos.dtype).view(1, 3).expand(
        batch, -1
    )
    offset_quat = torch.tensor(cam_cfg.offset.rot, device=link_pos.device, dtype=link_pos.dtype).view(1, 4).expand(
        batch, -1
    )
    cam_pos_w = link_pos + quat_apply(link_quat, offset_pos)
    cam_quat_w = quat_mul(link_quat, offset_quat)
    rot = matrix_from_quat(cam_quat_w).view(-1, 3, 3)

    origin_cam = origin.view(-1, 3)
    end_cam = line_end.view(-1, 3)
    if origin_cam.shape[0] == 1:
        origin_cam = origin_cam.expand(batch, -1)
    if end_cam.shape[0] == 1:
        end_cam = end_cam.expand(batch, -1)
    start_w = torch.bmm(rot, origin_cam.unsqueeze(-1)).squeeze(-1) + cam_pos_w
    end_w = torch.bmm(rot, end_cam.unsqueeze(-1)).squeeze(-1) + cam_pos_w
    return start_w, end_w, cam_pos_w, cam_quat_w
