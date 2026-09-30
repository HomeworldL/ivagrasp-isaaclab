"""Inspire Hand variants for the dexgrasp_moving task."""

from isaaclab.utils import configclass

from dynamic_dexgrasp_lab.tasks.dexgrasp_float.config.inspire_hand.inspire_env_cfg import INSPIRE_HAND_PROFILE
from dynamic_dexgrasp_lab.tasks.dexgrasp_moving import dexgrasp_moving_env_cfg


@configclass
class InspireDexGraspMovingEnvCfg(dexgrasp_moving_env_cfg.DexGraspMovingEnvCfg):
    """Moving-object dexgrasp environment using Inspire assets."""

    hand_profile = INSPIRE_HAND_PROFILE


@configclass
class InspireDexGraspMovingEnvCfg_PLAY(InspireDexGraspMovingEnvCfg):
    """Reduced play/debug variant for Inspire moving-object grasp."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.goal.debug_vis = True
        self.scene.num_envs = 64
        self.scene.env_spacing = 1.0
        self.observations.actor.enable_corruption = False
        self.observations.critic.enable_corruption = False


@configclass
class InspireDexGraspMovingDistillEnvCfg(InspireDexGraspMovingEnvCfg):
    """Student distillation variant for Inspire moving-object grasp."""

    observations: dexgrasp_moving_env_cfg.DistillObservationsCfg = dexgrasp_moving_env_cfg.DistillObservationsCfg()
