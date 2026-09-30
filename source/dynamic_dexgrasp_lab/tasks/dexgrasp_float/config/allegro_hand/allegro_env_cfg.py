"""Allegro-hand variants for the dexgrasp_float task."""

from isaaclab.utils import configclass

from dynamic_dexgrasp_lab.assets.hands.allegro_hand import (
    ALLEGRO_ACTUATED_JOINT_NAMES,
    ALLEGRO_ARTICULATION_ROOT_PRIM_PATH,
    ALLEGRO_CAM_OBJECT_RELATIVE_POS,
    ALLEGRO_CAM_OBJECT_RELATIVE_QUAT,
    ALLEGRO_CONTACT_BODY_NAMES,
    ALLEGRO_CONTACT_FINGER_BODY_NAMES,
    ALLEGRO_CONTACT_FINGERTIP_BODY_NAMES,
    ALLEGRO_GEO_BODY_NAMES,
    ALLEGRO_GEO_BODY_WEIGHTS,
    ALLEGRO_HAND_CFG,
    ALLEGRO_HAND_OBJECT_RELATIVE_POS,
    ALLEGRO_HAND_OBJECT_RELATIVE_QUAT,
    ALLEGRO_HAND_TRANSFORM_POS,
    ALLEGRO_HAND_TRANSFORM_QUAT,
    ALLEGRO_HOME_JOINT_POSITIONS,
    ALLEGRO_HOME_JOINT_POS_MAP,
    ALLEGRO_JOINT_NAMES,
    ALLEGRO_PALM_BODY_NAME,
    ALLEGRO_POSITION_ACTION_OFFSET_MAP,
    ALLEGRO_POSITION_ACTION_SCALE_MAP,
)
from dynamic_dexgrasp_lab.tasks.dexgrasp_float import dexgrasp_float_env_cfg

ALLEGRO_HAND_PROFILE = dexgrasp_float_env_cfg.DexGraspHandProfile(
    hand_name="allegro_right",
    hand_cfg=ALLEGRO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    hand_joint_names=ALLEGRO_JOINT_NAMES,
    hand_actuated_joint_names=ALLEGRO_ACTUATED_JOINT_NAMES,
    hand_home_joint_positions=ALLEGRO_HOME_JOINT_POSITIONS,
    hand_home_joint_pos_map=ALLEGRO_HOME_JOINT_POS_MAP,
    hand_position_action_offset_map=ALLEGRO_POSITION_ACTION_OFFSET_MAP,
    hand_position_action_scale_map=ALLEGRO_POSITION_ACTION_SCALE_MAP,
    hand_transform_pos=ALLEGRO_HAND_TRANSFORM_POS,
    hand_transform_quat=ALLEGRO_HAND_TRANSFORM_QUAT,
    hand_object_relative_pos=ALLEGRO_HAND_OBJECT_RELATIVE_POS,
    hand_object_relative_quat=ALLEGRO_HAND_OBJECT_RELATIVE_QUAT,
    hand_camera_object_relative_pos=ALLEGRO_CAM_OBJECT_RELATIVE_POS,
    hand_camera_object_relative_quat=ALLEGRO_CAM_OBJECT_RELATIVE_QUAT,
    hand_palm_body_name=ALLEGRO_PALM_BODY_NAME,
    hand_geo_body_names=ALLEGRO_GEO_BODY_NAMES,
    hand_geo_body_weights=ALLEGRO_GEO_BODY_WEIGHTS,
    hand_articulation_root_prim_path=ALLEGRO_ARTICULATION_ROOT_PRIM_PATH,
    hand_contact_body_names=ALLEGRO_CONTACT_BODY_NAMES,
    hand_contact_finger_body_names=ALLEGRO_CONTACT_FINGER_BODY_NAMES,
    hand_contact_fingertip_body_names=ALLEGRO_CONTACT_FINGERTIP_BODY_NAMES,
    hand_drive_stiffness=None,
    hand_drive_damping=None,
    hand_drive_max_force=None,
    hand_contact_sensor_prim_path_prefix="{ENV_REGEX_NS}/Robot/",
)


@configclass
class AllegroDexGraspFloatEnvCfg(dexgrasp_float_env_cfg.DexGraspFloatEnvCfg):
    """Floating-hand dexgrasp environment using Allegro assets."""

    hand_profile = ALLEGRO_HAND_PROFILE


@configclass
class AllegroDexGraspFloatEnvCfg_PLAY(AllegroDexGraspFloatEnvCfg):
    """Reduced play/debug variant for Allegro."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.goal.debug_vis = True
        self.scene.num_envs = 64
        self.scene.env_spacing = 1.0
        self.observations.actor.enable_corruption = False
        self.observations.critic.enable_corruption = False


@configclass
class AllegroDexGraspFloatDistillEnvCfg(AllegroDexGraspFloatEnvCfg):
    """Teacher/student distillation variant for Allegro float grasp."""

    observations: dexgrasp_float_env_cfg.DistillObservationsCfg = dexgrasp_float_env_cfg.DistillObservationsCfg()
