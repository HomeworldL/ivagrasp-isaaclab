"""Liberhand variants for the dexgrasp_float task."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-DexGrasp-Float-Liberhand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.liberhand_env_cfg:LiberhandDexGraspFloatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiberhandDexGraspFloatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-DexGrasp-Float-Liberhand-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.liberhand_env_cfg:LiberhandDexGraspFloatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiberhandDexGraspFloatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-DexGrasp-Float-Liberhand-Distill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.liberhand_env_cfg:LiberhandDexGraspFloatDistillEnvCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:LiberhandDexGraspFloatDistillationRunnerCfg"
        ),
    },
)

__all__ = ["agents"]
