"""RSL-RL agent configs for Liberhand dexgrasp_moving variants."""

from .rsl_rl_distillation_cfg import LiberhandDexGraspMovingDistillationRunnerCfg
from .rsl_rl_ppo_cfg import LiberhandDexGraspMovingPPORunnerCfg

__all__ = ["LiberhandDexGraspMovingPPORunnerCfg", "LiberhandDexGraspMovingDistillationRunnerCfg"]
