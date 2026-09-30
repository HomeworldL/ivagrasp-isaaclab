"""Liberhand variants for the dexgrasp_moving task."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-DexGrasp-Moving-Liberhand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.liberhand_env_cfg:LiberhandDexGraspMovingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiberhandDexGraspMovingPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-DexGrasp-Moving-Liberhand-Distill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.liberhand_env_cfg:LiberhandDexGraspMovingDistillEnvCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:LiberhandDexGraspMovingDistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-DexGrasp-Moving-Liberhand-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.liberhand_env_cfg:LiberhandDexGraspMovingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiberhandDexGraspMovingPPORunnerCfg",
    },
)

__all__ = ["agents"]
