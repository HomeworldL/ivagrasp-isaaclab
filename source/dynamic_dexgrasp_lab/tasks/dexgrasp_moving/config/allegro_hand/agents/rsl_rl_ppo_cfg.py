"""RSL-RL PPO config for the Allegro dexgrasp_moving task."""

from isaaclab.utils import configclass

from dynamic_dexgrasp_lab.tasks.dexgrasp_float.config.allegro_hand.agents.rsl_rl_ppo_cfg import (
    AllegroDexGraspFloatPPORunnerCfg,
)


@configclass
class AllegroDexGraspMovingPPORunnerCfg(AllegroDexGraspFloatPPORunnerCfg):
    """PPO runner config for the Allegro dexgrasp_moving task."""

    experiment_name = "dexgrasp_moving_allegro_goal"
