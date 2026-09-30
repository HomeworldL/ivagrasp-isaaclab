"""RSL-RL agent configs for Inspire dexgrasp variants."""

from .rsl_rl_distillation_cfg import InspireDexGraspFloatDistillationRunnerCfg
from .rsl_rl_ppo_cfg import InspireDexGraspFloatPPORunnerCfg

__all__ = ["InspireDexGraspFloatPPORunnerCfg", "InspireDexGraspFloatDistillationRunnerCfg"]
