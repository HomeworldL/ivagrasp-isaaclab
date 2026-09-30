"""RSL-RL agent configs for Liberhand dexgrasp variants."""

from .rsl_rl_distillation_cfg import LiberhandDexGraspFloatDistillationRunnerCfg
from .rsl_rl_ppo_cfg import LiberhandDexGraspFloatPPORunnerCfg

__all__ = ["LiberhandDexGraspFloatPPORunnerCfg", "LiberhandDexGraspFloatDistillationRunnerCfg"]
