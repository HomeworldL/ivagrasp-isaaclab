"""Liberhand variants for the dexgrasp_float task."""

from isaaclab.utils import configclass

from dynamic_dexgrasp_lab.assets.hands.liberhand import (
    LIBERHAND_ACTUATED_JOINT_NAMES,
    LIBERHAND_ARTICULATION_ROOT_PRIM_PATH,
    LIBERHAND_CFG,
    LIBERHAND_CONTACT_BODY_NAMES,
    LIBERHAND_CONTACT_FINGER_BODY_NAMES,
    LIBERHAND_CONTACT_FINGERTIP_BODY_NAMES,
    LIBERHAND_CAM_OBJECT_RELATIVE_POS,
    LIBERHAND_CAM_OBJECT_RELATIVE_QUAT,
    LIBERHAND_DRIVE_DAMPING,
    LIBERHAND_DRIVE_MAX_FORCE,
    LIBERHAND_DRIVE_STIFFNESS,
    LIBERHAND_GEO_BODY_NAMES,
    LIBERHAND_GEO_BODY_WEIGHTS,
    LIBERHAND_HAND_TRANSFORM_POS,
    LIBERHAND_HAND_TRANSFORM_QUAT,
    LIBERHAND_HAND_OBJECT_RELATIVE_POS,
    LIBERHAND_HAND_OBJECT_RELATIVE_QUAT,
    LIBERHAND_HOME_JOINT_POSITIONS,
    LIBERHAND_HOME_JOINT_POS_MAP,
    LIBERHAND_JOINT_NAMES,
    LIBERHAND_PALM_BODY_NAME,
    LIBERHAND_POSITION_ACTION_OFFSET_MAP,
    LIBERHAND_POSITION_ACTION_SCALE_MAP,
)
from dynamic_dexgrasp_lab.tasks.dexgrasp_float import dexgrasp_float_env_cfg


LIBERHAND_HAND_PROFILE = dexgrasp_float_env_cfg.DexGraspHandProfile(
    hand_name="liberhand_right",
    hand_cfg=LIBERHAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    hand_joint_names=LIBERHAND_JOINT_NAMES,
    hand_actuated_joint_names=LIBERHAND_ACTUATED_JOINT_NAMES,
    hand_home_joint_positions=LIBERHAND_HOME_JOINT_POSITIONS,
    hand_home_joint_pos_map=LIBERHAND_HOME_JOINT_POS_MAP,
    hand_position_action_offset_map=LIBERHAND_POSITION_ACTION_OFFSET_MAP,
    hand_position_action_scale_map=LIBERHAND_POSITION_ACTION_SCALE_MAP,
    hand_transform_pos=LIBERHAND_HAND_TRANSFORM_POS,
    hand_transform_quat=LIBERHAND_HAND_TRANSFORM_QUAT,
    hand_object_relative_pos=LIBERHAND_HAND_OBJECT_RELATIVE_POS,
    hand_object_relative_quat=LIBERHAND_HAND_OBJECT_RELATIVE_QUAT,
    hand_camera_object_relative_pos=LIBERHAND_CAM_OBJECT_RELATIVE_POS,
    hand_camera_object_relative_quat=LIBERHAND_CAM_OBJECT_RELATIVE_QUAT,
    hand_palm_body_name=LIBERHAND_PALM_BODY_NAME,
    hand_geo_body_names=LIBERHAND_GEO_BODY_NAMES,
    hand_geo_body_weights=LIBERHAND_GEO_BODY_WEIGHTS,
    hand_articulation_root_prim_path=LIBERHAND_ARTICULATION_ROOT_PRIM_PATH,
    hand_contact_body_names=LIBERHAND_CONTACT_BODY_NAMES,
    hand_contact_finger_body_names=LIBERHAND_CONTACT_FINGER_BODY_NAMES,
    hand_contact_fingertip_body_names=LIBERHAND_CONTACT_FINGERTIP_BODY_NAMES,
    hand_drive_stiffness=LIBERHAND_DRIVE_STIFFNESS,
    hand_drive_damping=LIBERHAND_DRIVE_DAMPING,
    hand_drive_max_force=LIBERHAND_DRIVE_MAX_FORCE,
    hand_contact_sensor_prim_path_prefix="{ENV_REGEX_NS}/Robot/hand_root/",
)


@configclass
class LiberhandDexGraspFloatEnvCfg(dexgrasp_float_env_cfg.DexGraspFloatEnvCfg):
    """Floating-hand dexgrasp environment using Liberhand assets."""

    hand_profile = LIBERHAND_HAND_PROFILE


@configclass
class LiberhandDexGraspFloatEnvCfg_PLAY(LiberhandDexGraspFloatEnvCfg):
    """Reduced play/debug variant for Liberhand."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.goal.debug_vis = True
        self.scene.num_envs = 64
        self.scene.env_spacing = 1.0
        self.observations.actor.enable_corruption = False
        self.observations.critic.enable_corruption = False


@configclass
class LiberhandDexGraspFloatDistillEnvCfg(LiberhandDexGraspFloatEnvCfg):
    """Teacher/student distillation variant for Liberhand float grasp."""

    observations: dexgrasp_float_env_cfg.DistillObservationsCfg = dexgrasp_float_env_cfg.DistillObservationsCfg()
