"""RSL-RL distillation config for the Inspire dexgrasp_moving task."""

from isaaclab.utils import configclass

from dynamic_dexgrasp_lab.tasks.dexgrasp_float.config.inspire_hand.agents.rsl_rl_distillation_cfg import (
    InspireDexGraspFloatDistillationRunnerCfg,
)


@configclass
class InspireDexGraspMovingDistillationRunnerCfg(InspireDexGraspFloatDistillationRunnerCfg):
    """Distillation runner config for the Inspire moving-object student policy."""

    experiment_name = "dexgrasp_moving_inspire_student"
