"""Runtime IsaacLab asset configuration for Inspire Hand."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


INSPIRE_HAND_DIR = Path(__file__).resolve().parent

INSPIRE_HAND_JOINT_NAMES = (
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "thumb_intermediate_joint",
    "thumb_distal_joint",
    "index_proximal_joint",
    "index_intermediate_joint",
    "middle_proximal_joint",
    "middle_intermediate_joint",
    "ring_proximal_joint",
    "ring_intermediate_joint",
    "pinky_proximal_joint",
    "pinky_intermediate_joint",
)
INSPIRE_HAND_ACTUATED_JOINT_NAMES = (
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
)
INSPIRE_HAND_HOME_JOINT_POSITIONS = (
    1.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)

INSPIRE_HAND_POSITION_ACTION_OFFSET_MAP = {
    "thumb_proximal_yaw_joint": 1.2,
    "thumb_proximal_pitch_joint": 0.0,
    "index_proximal_joint": 0.0,
    "middle_proximal_joint": 0.0,
    "ring_proximal_joint": 0.0,
    "pinky_proximal_joint": 0.0,
}
INSPIRE_HAND_POSITION_ACTION_SCALE_MAP = {
    "thumb_proximal_yaw_joint": 1.2,
    "thumb_proximal_pitch_joint": 0.6,
    "index_proximal_joint": 1.47,
    "middle_proximal_joint": 1.47,
    "ring_proximal_joint": 1.47,
    "pinky_proximal_joint": 1.47,
}

INSPIRE_HAND_TRANSFORM_POS = (0.02, 0.0, 0.08)
INSPIRE_HAND_TRANSFORM_QUAT = (0.9659258, 0.0, 0.258819, 0.0)
INSPIRE_HAND_OBJECT_RELATIVE_POS = (-0.02267948, -0.38569981, -0.25711504)
INSPIRE_HAND_OBJECT_RELATIVE_QUAT = (0.18301267, 0.18301267, 0.68301271, 0.68301271)
INSPIRE_CAM_OBJECT_RELATIVE_POS = (-0.272372, 0.040000, 0.335367)
INSPIRE_CAM_OBJECT_RELATIVE_QUAT = (0.360753, -0.608159, 0.608157, -0.360756)
INSPIRE_HAND_PALM_BODY_NAME = "hand_base_link"

INSPIRE_HAND_GEO_BODY_NAMES = (
    "thumb_proximal",
    "thumb_intermediate",
    "thumb_distal",
    "index_proximal",
    "index_intermediate",
    "middle_proximal",
    "middle_intermediate",
    "ring_proximal",
    "ring_intermediate",
    "pinky_proximal",
    "pinky_intermediate",
)
INSPIRE_HAND_GEO_BODY_WEIGHTS = (
    1.0,
    1.0,
    3.0,
    1.0,
    3.0,
    1.0,
    3.0,
    1.0,
    3.0,
    1.0,
    3.0,
)
INSPIRE_HAND_ARTICULATION_ROOT_PRIM_PATH = "/hand_root/hand_root"
INSPIRE_HAND_CONTACT_BODY_NAMES = (
    "thumb_proximal",
    "thumb_intermediate",
    "thumb_distal",
    "index_proximal",
    "index_intermediate",
    "middle_proximal",
    "middle_intermediate",
    "ring_proximal",
    "ring_intermediate",
    "pinky_proximal",
    "pinky_intermediate",
)
INSPIRE_HAND_CONTACT_FINGER_BODY_NAMES = (
    ("thumb_proximal", "thumb_intermediate", "thumb_distal"),
    ("index_proximal", "index_intermediate"),
    ("middle_proximal", "middle_intermediate"),
    ("ring_proximal", "ring_intermediate"),
    ("pinky_proximal", "pinky_intermediate"),
)
INSPIRE_HAND_CONTACT_FINGERTIP_BODY_NAMES = (
    "thumb_distal",
    "index_intermediate",
    "middle_intermediate",
    "ring_intermediate",
    "pinky_intermediate",
)

INSPIRE_HAND_HOME_JOINT_POS_MAP = {
    name: value for name, value in zip(INSPIRE_HAND_JOINT_NAMES, INSPIRE_HAND_HOME_JOINT_POSITIONS, strict=True)
}


def _inspire_usd_path(side: str) -> str:
    return str(INSPIRE_HAND_DIR / "usd" / f"inspire_hand_{side}" / f"inspire_hand_{side}.usd")


def make_inspire_hand_cfg(side: str) -> ArticulationCfg:
    side = side.lower()
    if side not in ("right", "left"):
        raise ValueError(f"Unsupported Inspire Hand side: {side!r}")

    return ArticulationCfg(
        articulation_root_prim_path=INSPIRE_HAND_ARTICULATION_ROOT_PRIM_PATH,
        spawn=sim_utils.UsdFileCfg(
            usd_path=_inspire_usd_path(side),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                enable_gyroscopic_forces=True,
                angular_damping=0.01,
                linear_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=10.0,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.0,
                stabilization_threshold=0.0,
                fix_root_link=False,
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.3),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=INSPIRE_HAND_HOME_JOINT_POS_MAP,
        ),
        actuators={
            "hand": ImplicitActuatorCfg(
                joint_names_expr=list(INSPIRE_HAND_ACTUATED_JOINT_NAMES),
                effort_limit_sim=1.0,
                velocity_limit_sim=100.0,
                stiffness=1.0,
                damping=0.1,
                armature=0.0002,
                friction=0.0,
            ),
        },
        soft_joint_pos_limit_factor=0.98,
    )


INSPIRE_HAND_RIGHT_CFG = make_inspire_hand_cfg("right")
INSPIRE_HAND_LEFT_CFG = make_inspire_hand_cfg("left")


__all__ = [
    "INSPIRE_HAND_JOINT_NAMES",
    "INSPIRE_HAND_ACTUATED_JOINT_NAMES",
    "INSPIRE_HAND_HOME_JOINT_POSITIONS",
    "INSPIRE_HAND_POSITION_ACTION_OFFSET_MAP",
    "INSPIRE_HAND_POSITION_ACTION_SCALE_MAP",
    "INSPIRE_HAND_TRANSFORM_POS",
    "INSPIRE_HAND_TRANSFORM_QUAT",
    "INSPIRE_HAND_OBJECT_RELATIVE_POS",
    "INSPIRE_HAND_OBJECT_RELATIVE_QUAT",
    "INSPIRE_CAM_OBJECT_RELATIVE_POS",
    "INSPIRE_CAM_OBJECT_RELATIVE_QUAT",
    "INSPIRE_HAND_PALM_BODY_NAME",
    "INSPIRE_HAND_GEO_BODY_NAMES",
    "INSPIRE_HAND_GEO_BODY_WEIGHTS",
    "INSPIRE_HAND_ARTICULATION_ROOT_PRIM_PATH",
    "INSPIRE_HAND_CONTACT_BODY_NAMES",
    "INSPIRE_HAND_CONTACT_FINGER_BODY_NAMES",
    "INSPIRE_HAND_CONTACT_FINGERTIP_BODY_NAMES",
    "INSPIRE_HAND_HOME_JOINT_POS_MAP",
    "INSPIRE_HAND_RIGHT_CFG",
    "INSPIRE_HAND_LEFT_CFG",
    "make_inspire_hand_cfg",
]
