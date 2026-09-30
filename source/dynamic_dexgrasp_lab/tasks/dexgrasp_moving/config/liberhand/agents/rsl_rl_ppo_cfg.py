"""RSL-RL PPO config for the Liberhand dexgrasp_moving task."""

from isaaclab.utils import configclass

from dynamic_dexgrasp_lab.tasks.dexgrasp_float.config.liberhand.agents.rsl_rl_ppo_cfg import (
    LiberhandDexGraspFloatPPORunnerCfg,
)


@configclass
class LiberhandDexGraspMovingPPORunnerCfg(LiberhandDexGraspFloatPPORunnerCfg):
    """PPO runner config for the Liberhand dexgrasp_moving task."""

    experiment_name = "dexgrasp_moving_liberhand_goal"
