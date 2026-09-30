"""Liberhand variants for the dexgrasp_moving task."""

from isaaclab.utils import configclass

from dynamic_dexgrasp_lab.tasks.dexgrasp_float.config.liberhand.liberhand_env_cfg import LIBERHAND_HAND_PROFILE
from dynamic_dexgrasp_lab.tasks.dexgrasp_moving import dexgrasp_moving_env_cfg


@configclass
class LiberhandDexGraspMovingEnvCfg(dexgrasp_moving_env_cfg.DexGraspMovingEnvCfg):
    """Moving-object dexgrasp environment using Liberhand assets."""

    hand_profile = LIBERHAND_HAND_PROFILE


@configclass
class LiberhandDexGraspMovingEnvCfg_PLAY(LiberhandDexGraspMovingEnvCfg):
    """Reduced play/debug variant for Liberhand moving-object grasp."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.goal.debug_vis = True
        self.scene.num_envs = 64
        self.scene.env_spacing = 1.0
        self.observations.actor.enable_corruption = False
        self.observations.critic.enable_corruption = False


@configclass
class LiberhandDexGraspMovingDistillEnvCfg(LiberhandDexGraspMovingEnvCfg):
    """Student distillation variant for Liberhand moving-object grasp."""

    observations: dexgrasp_moving_env_cfg.DistillObservationsCfg = dexgrasp_moving_env_cfg.DistillObservationsCfg()
