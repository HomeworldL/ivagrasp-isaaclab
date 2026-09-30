"""RSL-RL agent configs for Inspire dexgrasp_moving variants."""

from .rsl_rl_distillation_cfg import InspireDexGraspMovingDistillationRunnerCfg
from .rsl_rl_ppo_cfg import InspireDexGraspMovingPPORunnerCfg

__all__ = ["InspireDexGraspMovingPPORunnerCfg", "InspireDexGraspMovingDistillationRunnerCfg"]
