"""RSL-RL distillation config for the Allegro dexgrasp_moving task."""

from isaaclab.utils import configclass

from dynamic_dexgrasp_lab.tasks.dexgrasp_float.config.allegro_hand.agents.rsl_rl_distillation_cfg import (
    AllegroDexGraspFloatDistillationRunnerCfg,
)


@configclass
class AllegroDexGraspMovingDistillationRunnerCfg(AllegroDexGraspFloatDistillationRunnerCfg):
    """Distillation runner config for the Allegro moving-object student policy."""

    experiment_name = "dexgrasp_moving_allegro_student"
