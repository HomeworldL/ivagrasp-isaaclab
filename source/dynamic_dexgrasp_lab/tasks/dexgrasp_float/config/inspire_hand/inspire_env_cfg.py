"""Inspire Hand variants for the dexgrasp_float task."""

from isaaclab.utils import configclass

from dynamic_dexgrasp_lab.assets.hands.inspire_hand import (
    INSPIRE_HAND_ACTUATED_JOINT_NAMES,
    INSPIRE_HAND_ARTICULATION_ROOT_PRIM_PATH,
    INSPIRE_HAND_CONTACT_BODY_NAMES,
    INSPIRE_HAND_CONTACT_FINGER_BODY_NAMES,
    INSPIRE_HAND_CONTACT_FINGERTIP_BODY_NAMES,
    INSPIRE_HAND_GEO_BODY_NAMES,
    INSPIRE_HAND_GEO_BODY_WEIGHTS,
    INSPIRE_HAND_HOME_JOINT_POSITIONS,
    INSPIRE_HAND_HOME_JOINT_POS_MAP,
    INSPIRE_HAND_JOINT_NAMES,
    INSPIRE_CAM_OBJECT_RELATIVE_POS,
    INSPIRE_CAM_OBJECT_RELATIVE_QUAT,
    INSPIRE_HAND_OBJECT_RELATIVE_POS,
    INSPIRE_HAND_OBJECT_RELATIVE_QUAT,
    INSPIRE_HAND_PALM_BODY_NAME,
    INSPIRE_HAND_POSITION_ACTION_OFFSET_MAP,
    INSPIRE_HAND_POSITION_ACTION_SCALE_MAP,
    INSPIRE_HAND_RIGHT_CFG,
    INSPIRE_HAND_TRANSFORM_POS,
    INSPIRE_HAND_TRANSFORM_QUAT,
)
from dynamic_dexgrasp_lab.tasks.dexgrasp_float import dexgrasp_float_env_cfg


_INSPIRE_HAND_DRIVE_STIFFNESS = {joint_name: 1.0 for joint_name in INSPIRE_HAND_ACTUATED_JOINT_NAMES}
_INSPIRE_HAND_DRIVE_DAMPING = {joint_name: 0.1 for joint_name in INSPIRE_HAND_ACTUATED_JOINT_NAMES}
_INSPIRE_HAND_DRIVE_MAX_FORCE = {joint_name: 1.0 for joint_name in INSPIRE_HAND_ACTUATED_JOINT_NAMES}

INSPIRE_HAND_PROFILE = dexgrasp_float_env_cfg.DexGraspHandProfile(
    hand_name="inspire_right",
    hand_cfg=INSPIRE_HAND_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    hand_joint_names=INSPIRE_HAND_JOINT_NAMES,
    hand_actuated_joint_names=INSPIRE_HAND_ACTUATED_JOINT_NAMES,
    hand_home_joint_positions=INSPIRE_HAND_HOME_JOINT_POSITIONS,
    hand_home_joint_pos_map=INSPIRE_HAND_HOME_JOINT_POS_MAP,
    hand_position_action_offset_map=INSPIRE_HAND_POSITION_ACTION_OFFSET_MAP,
    hand_position_action_scale_map=INSPIRE_HAND_POSITION_ACTION_SCALE_MAP,
    hand_transform_pos=INSPIRE_HAND_TRANSFORM_POS,
    hand_transform_quat=INSPIRE_HAND_TRANSFORM_QUAT,
    hand_object_relative_pos=INSPIRE_HAND_OBJECT_RELATIVE_POS,
    hand_object_relative_quat=INSPIRE_HAND_OBJECT_RELATIVE_QUAT,
    hand_camera_object_relative_pos=INSPIRE_CAM_OBJECT_RELATIVE_POS,
    hand_camera_object_relative_quat=INSPIRE_CAM_OBJECT_RELATIVE_QUAT,
    hand_palm_body_name=INSPIRE_HAND_PALM_BODY_NAME,
    hand_geo_body_names=INSPIRE_HAND_GEO_BODY_NAMES,
    hand_geo_body_weights=INSPIRE_HAND_GEO_BODY_WEIGHTS,
    hand_articulation_root_prim_path=INSPIRE_HAND_ARTICULATION_ROOT_PRIM_PATH,
    hand_contact_body_names=INSPIRE_HAND_CONTACT_BODY_NAMES,
    hand_contact_finger_body_names=INSPIRE_HAND_CONTACT_FINGER_BODY_NAMES,
    hand_contact_fingertip_body_names=INSPIRE_HAND_CONTACT_FINGERTIP_BODY_NAMES,
    hand_drive_stiffness=_INSPIRE_HAND_DRIVE_STIFFNESS,
    hand_drive_damping=_INSPIRE_HAND_DRIVE_DAMPING,
    hand_drive_max_force=_INSPIRE_HAND_DRIVE_MAX_FORCE,
    hand_contact_sensor_prim_path_prefix="{ENV_REGEX_NS}/Robot/hand_root/",
)


@configclass
class InspireDexGraspFloatEnvCfg(dexgrasp_float_env_cfg.DexGraspFloatEnvCfg):
    """Floating-hand dexgrasp environment using Inspire assets."""

    hand_profile = INSPIRE_HAND_PROFILE


@configclass
class InspireDexGraspFloatEnvCfg_PLAY(InspireDexGraspFloatEnvCfg):
    """Reduced play/debug variant for Inspire."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.goal.debug_vis = True
        self.scene.num_envs = 64
        self.scene.env_spacing = 1.0
        self.observations.actor.enable_corruption = False
        self.observations.critic.enable_corruption = False


@configclass
class InspireDexGraspFloatDistillEnvCfg(InspireDexGraspFloatEnvCfg):
    """Teacher/student distillation variant for Inspire float grasp."""

    observations: dexgrasp_float_env_cfg.DistillObservationsCfg = dexgrasp_float_env_cfg.DistillObservationsCfg()
