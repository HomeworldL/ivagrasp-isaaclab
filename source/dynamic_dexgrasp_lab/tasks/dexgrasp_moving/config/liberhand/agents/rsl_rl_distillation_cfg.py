"""RSL-RL distillation config for the Liberhand dexgrasp_moving task."""

from isaaclab.utils import configclass

from dynamic_dexgrasp_lab.tasks.dexgrasp_float.config.liberhand.agents.rsl_rl_distillation_cfg import (
    LiberhandDexGraspFloatDistillationRunnerCfg,
)


@configclass
class LiberhandDexGraspMovingDistillationRunnerCfg(LiberhandDexGraspFloatDistillationRunnerCfg):
    """Distillation runner config for the Liberhand moving-object student policy."""

    experiment_name = "dexgrasp_moving_liberhand_student"
