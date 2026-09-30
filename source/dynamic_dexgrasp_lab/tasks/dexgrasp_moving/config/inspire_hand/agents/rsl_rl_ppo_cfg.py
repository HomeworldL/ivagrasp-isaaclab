"""RSL-RL PPO config for the Inspire dexgrasp_moving task."""

from isaaclab.utils import configclass

from dynamic_dexgrasp_lab.tasks.dexgrasp_float.config.inspire_hand.agents.rsl_rl_ppo_cfg import (
    InspireDexGraspFloatPPORunnerCfg,
)


@configclass
class InspireDexGraspMovingPPORunnerCfg(InspireDexGraspFloatPPORunnerCfg):
    """PPO runner config for the Inspire dexgrasp_moving task."""

    experiment_name = "dexgrasp_moving_inspire_goal"
