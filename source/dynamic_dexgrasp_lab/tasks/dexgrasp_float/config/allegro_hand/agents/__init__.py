"""RSL-RL agent configs for Allegro dexgrasp variants."""

from .rsl_rl_distillation_cfg import AllegroDexGraspFloatDistillationRunnerCfg
from .rsl_rl_ppo_cfg import AllegroDexGraspFloatPPORunnerCfg

__all__ = ["AllegroDexGraspFloatPPORunnerCfg", "AllegroDexGraspFloatDistillationRunnerCfg"]
