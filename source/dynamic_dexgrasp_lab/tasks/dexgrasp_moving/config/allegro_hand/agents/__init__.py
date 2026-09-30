"""RSL-RL agent configs for Allegro dexgrasp_moving variants."""

from .rsl_rl_distillation_cfg import AllegroDexGraspMovingDistillationRunnerCfg
from .rsl_rl_ppo_cfg import AllegroDexGraspMovingPPORunnerCfg

__all__ = ["AllegroDexGraspMovingPPORunnerCfg", "AllegroDexGraspMovingDistillationRunnerCfg"]
