"""Official IsaacLab asset configuration for Liberhand.

This module is the runtime source of truth for the converted Liberhand asset.
It intentionally does not depend on ``liberhand_constants.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


LIBERHAND_DIR = Path(__file__).resolve().parent
LIBERHAND_USD_PATH = str(LIBERHAND_DIR / "usd" / "liberhand_right" / "liberhand_right.usd")

LIBERHAND_JOINT_NAMES = (
    "hand_right_f11_joint",
    "hand_right_f12_joint",
    "hand_right_f13_joint",
    "hand_right_f14_joint",
    "hand_right_f21_joint",
    "hand_right_f22_joint",
    "hand_right_f23_joint",
    "hand_right_f24_joint",
    "hand_right_f31_joint",
    "hand_right_f32_joint",
    "hand_right_f33_joint",
    "hand_right_f34_joint",
    "hand_right_f41_joint",
    "hand_right_f42_joint",
    "hand_right_f43_joint",
    "hand_right_f44_joint",
    "hand_right_f51_joint",
    "hand_right_f52_joint",
    "hand_right_f53_joint",
    "hand_right_f54_joint",
)

LIBERHAND_ACTUATED_JOINT_NAMES = (
    "hand_right_f11_joint",
    "hand_right_f12_joint",
    "hand_right_f13_joint",
    "hand_right_f21_joint",
    "hand_right_f22_joint",
    "hand_right_f23_joint",
    "hand_right_f31_joint",
    "hand_right_f32_joint",
    "hand_right_f41_joint",
    "hand_right_f42_joint",
    "hand_right_f51_joint",
    "hand_right_f52_joint",
    "hand_right_f53_joint",
)

LIBERHAND_HOME_JOINT_POSITIONS = (
    0.0,
    0.3,
    0.1,
    0.1,
    0.0,
    0.3,
    0.1,
    0.1,
    0.0,
    0.1,
    0.1,
    0.1,
    0.0,
    0.1,
    0.1,
    0.1,
    1.6,
    0.0,
    0.0,
    0.0,
)

LIBERHAND_POSITION_ACTION_OFFSET_MAP = {
    "hand_right_f11_joint": 0.0,
    "hand_right_f12_joint": 0.3,
    "hand_right_f13_joint": 0.1,
    "hand_right_f21_joint": 0.0,
    "hand_right_f22_joint": 0.3,
    "hand_right_f23_joint": 0.1,
    "hand_right_f31_joint": 0.0,
    "hand_right_f32_joint": 0.1,
    "hand_right_f41_joint": 0.0,
    "hand_right_f42_joint": 0.1,
    "hand_right_f51_joint": 1.6,
    "hand_right_f52_joint": 0.0,
    "hand_right_f53_joint": 0.0,
}

LIBERHAND_POSITION_ACTION_SCALE_MAP = {
    "hand_right_f11_joint": 0.4,
    "hand_right_f12_joint": 1.9,
    "hand_right_f13_joint": 1.65,
    "hand_right_f21_joint": 0.4,
    "hand_right_f22_joint": 1.9,
    "hand_right_f23_joint": 1.65,
    "hand_right_f31_joint": 0.4,
    "hand_right_f32_joint": 1.65,
    "hand_right_f41_joint": 0.4,
    "hand_right_f42_joint": 1.6,
    "hand_right_f51_joint": 1.6,
    "hand_right_f52_joint": 0.8,
    "hand_right_f53_joint": 1.65,
}

LIBERHAND_HAND_TRANSFORM_POS = (0.0, 0.0, 0.0)
LIBERHAND_HAND_TRANSFORM_QUAT = (0.92387953, 0.0, 0.38268343, 0.0)
# Sim2sim default root pose relative to the object frame.
# This is the measured T_hand_in_object pose for the current Liberhand sim2sim
# setup. Quaternion order follows the repository-wide wxyz convention.
LIBERHAND_HAND_OBJECT_RELATIVE_POS = (-0.037118, -0.328763, -0.161792)
LIBERHAND_HAND_OBJECT_RELATIVE_QUAT = (0.456858, 0.499834, 0.574008, 0.460393)
LIBERHAND_CAM_OBJECT_RELATIVE_POS = (-0.272372, 0.040000, 0.335367)
LIBERHAND_CAM_OBJECT_RELATIVE_QUAT = (0.360753, -0.608159, 0.608157, -0.360756)
LIBERHAND_PALM_BODY_NAME = "hand_right_palm"

LIBERHAND_GEO_BODY_NAMES = (
    "hand_right_f11",
    "hand_right_f12",
    "hand_right_f13",
    "hand_right_f14",
    "hand_right_f21",
    "hand_right_f22",
    "hand_right_f23",
    "hand_right_f24",
    "hand_right_f31",
    "hand_right_f32",
    "hand_right_f33",
    "hand_right_f34",
    "hand_right_f41",
    "hand_right_f42",
    "hand_right_f43",
    "hand_right_f44",
    "hand_right_f51",
    "hand_right_f52",
    "hand_right_f53",
    "hand_right_f54",
)

LIBERHAND_GEO_BODY_WEIGHTS = (
    1.0,
    1.0,
    1.0,
    3.0,
    1.0,
    1.0,
    1.0,
    3.0,
    1.0,
    1.0,
    1.0,
    3.0,
    1.0,
    1.0,
    1.0,
    3.0,
    1.0,
    1.0,
    1.0,
    3.0,
)
LIBERHAND_ARTICULATION_ROOT_PRIM_PATH = "/hand_root/hand_root"
LIBERHAND_CONTACT_BODY_NAMES = (
    "hand_right_f11",
    "hand_right_f12",
    "hand_right_f13",
    "hand_right_f14",
    "hand_right_f21",
    "hand_right_f22",
    "hand_right_f23",
    "hand_right_f24",
    "hand_right_f31",
    "hand_right_f32",
    "hand_right_f33",
    "hand_right_f34",
    "hand_right_f41",
    "hand_right_f42",
    "hand_right_f43",
    "hand_right_f44",
    "hand_right_f51",
    "hand_right_f52",
    "hand_right_f53",
    "hand_right_f54",
)
LIBERHAND_CONTACT_FINGER_BODY_NAMES = (
    ("hand_right_f11", "hand_right_f12", "hand_right_f13", "hand_right_f14"),
    ("hand_right_f21", "hand_right_f22", "hand_right_f23", "hand_right_f24"),
    ("hand_right_f31", "hand_right_f32", "hand_right_f33", "hand_right_f34"),
    ("hand_right_f41", "hand_right_f42", "hand_right_f43", "hand_right_f44"),
    ("hand_right_f51", "hand_right_f52", "hand_right_f53", "hand_right_f54"),
)
LIBERHAND_CONTACT_FINGERTIP_BODY_NAMES = (
    "hand_right_f14",
    "hand_right_f24",
    "hand_right_f34",
    "hand_right_f44",
    "hand_right_f54",
)

LIBERHAND_HOME_JOINT_POS_MAP = {
    name: value for name, value in zip(LIBERHAND_JOINT_NAMES, LIBERHAND_HOME_JOINT_POSITIONS, strict=True)
}

LIBERHAND_DRIVE_STIFFNESS = {
    "hand_right_f11_joint": 1.0,
    "hand_right_f12_joint": 1.0,
    "hand_right_f13_joint": 1.0,
    "hand_right_f21_joint": 1.0,
    "hand_right_f22_joint": 1.0,
    "hand_right_f23_joint": 1.0,
    "hand_right_f31_joint": 1.0,
    "hand_right_f32_joint": 1.0,
    "hand_right_f41_joint": 1.0,
    "hand_right_f42_joint": 1.0,
    "hand_right_f51_joint": 1.0,
    "hand_right_f52_joint": 1.0,
    "hand_right_f53_joint": 1.0,
}

LIBERHAND_DRIVE_DAMPING = {
    "hand_right_f11_joint": 0.1,
    "hand_right_f12_joint": 0.1,
    "hand_right_f13_joint": 0.1,
    "hand_right_f21_joint": 0.1,
    "hand_right_f22_joint": 0.1,
    "hand_right_f23_joint": 0.1,
    "hand_right_f31_joint": 0.1,
    "hand_right_f32_joint": 0.1,
    "hand_right_f41_joint": 0.1,
    "hand_right_f42_joint": 0.1,
    "hand_right_f51_joint": 0.1,
    "hand_right_f52_joint": 0.1,
    "hand_right_f53_joint": 0.1,
}

LIBERHAND_DRIVE_MAX_FORCE = {
    "hand_right_f11_joint": 1.0,
    "hand_right_f12_joint": 1.0,
    "hand_right_f13_joint": 1.0,
    "hand_right_f21_joint": 1.0,
    "hand_right_f22_joint": 1.0,
    "hand_right_f23_joint": 1.0,
    "hand_right_f31_joint": 1.0,
    "hand_right_f32_joint": 1.0,
    "hand_right_f41_joint": 1.0,
    "hand_right_f42_joint": 1.0,
    "hand_right_f51_joint": 1.0,
    "hand_right_f52_joint": 1.0,
    "hand_right_f53_joint": 1.0,
}


LIBERHAND_CFG = ArticulationCfg(
    articulation_root_prim_path=LIBERHAND_ARTICULATION_ROOT_PRIM_PATH,
    spawn=sim_utils.UsdFileCfg(
        usd_path=LIBERHAND_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=True,
            angular_damping=0.001,
            linear_damping=0.001,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0,
            max_depenetration_velocity=10.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=False,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # 中文说明：这里仅是资产 spawn/default pose，不代表任务 reset 后的 step-0 观测语义。
        pos=(0.0, 0.0, 0.3),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=LIBERHAND_HOME_JOINT_POS_MAP,
    ),
    actuators={
        "hand": ImplicitActuatorCfg(
            joint_names_expr=list(LIBERHAND_ACTUATED_JOINT_NAMES),
            effort_limit_sim=LIBERHAND_DRIVE_MAX_FORCE,
            velocity_limit_sim=100.0,
            stiffness=LIBERHAND_DRIVE_STIFFNESS,
            damping=LIBERHAND_DRIVE_DAMPING,
            armature=0.0002,
            friction=0.0,
        ),
    },
    soft_joint_pos_limit_factor=0.98,
)


__all__ = [
    "LIBERHAND_CFG",
    "LIBERHAND_USD_PATH",
    "LIBERHAND_ARTICULATION_ROOT_PRIM_PATH",
    "LIBERHAND_HAND_TRANSFORM_POS",
    "LIBERHAND_HAND_TRANSFORM_QUAT",
    "LIBERHAND_HAND_OBJECT_RELATIVE_POS",
    "LIBERHAND_HAND_OBJECT_RELATIVE_QUAT",
    "LIBERHAND_CAM_OBJECT_RELATIVE_POS",
    "LIBERHAND_CAM_OBJECT_RELATIVE_QUAT",
    "LIBERHAND_PALM_BODY_NAME",
    "LIBERHAND_JOINT_NAMES",
    "LIBERHAND_ACTUATED_JOINT_NAMES",
    "LIBERHAND_HOME_JOINT_POSITIONS",
    "LIBERHAND_HOME_JOINT_POS_MAP",
    "LIBERHAND_POSITION_ACTION_SCALE_MAP",
    "LIBERHAND_POSITION_ACTION_OFFSET_MAP",
    "LIBERHAND_CONTACT_BODY_NAMES",
    "LIBERHAND_CONTACT_FINGER_BODY_NAMES",
    "LIBERHAND_CONTACT_FINGERTIP_BODY_NAMES",
    "LIBERHAND_GEO_BODY_NAMES",
    "LIBERHAND_GEO_BODY_WEIGHTS",
    "LIBERHAND_DRIVE_STIFFNESS",
    "LIBERHAND_DRIVE_DAMPING",
    "LIBERHAND_DRIVE_MAX_FORCE",
]
