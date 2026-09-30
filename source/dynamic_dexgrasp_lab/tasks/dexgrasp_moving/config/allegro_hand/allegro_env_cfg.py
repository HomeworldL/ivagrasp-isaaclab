"""Allegro-hand variants for the dexgrasp_moving task."""

from isaaclab.utils import configclass

from dynamic_dexgrasp_lab.tasks.dexgrasp_float.config.allegro_hand.allegro_env_cfg import ALLEGRO_HAND_PROFILE
from dynamic_dexgrasp_lab.tasks.dexgrasp_moving import dexgrasp_moving_env_cfg


@configclass
class AllegroDexGraspMovingEnvCfg(dexgrasp_moving_env_cfg.DexGraspMovingEnvCfg):
    """Moving-object dexgrasp environment using Allegro assets."""

    hand_profile = ALLEGRO_HAND_PROFILE


@configclass
class AllegroDexGraspMovingEnvCfg_PLAY(AllegroDexGraspMovingEnvCfg):
    """Reduced play/debug variant for Allegro moving-object grasp."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.goal.debug_vis = True
        self.scene.num_envs = 64
        self.scene.env_spacing = 1.0
        self.observations.actor.enable_corruption = False
        self.observations.critic.enable_corruption = False


@configclass
class AllegroDexGraspMovingDistillEnvCfg(AllegroDexGraspMovingEnvCfg):
    """Student distillation variant for Allegro moving-object grasp."""

    observations: dexgrasp_moving_env_cfg.DistillObservationsCfg = dexgrasp_moving_env_cfg.DistillObservationsCfg()
