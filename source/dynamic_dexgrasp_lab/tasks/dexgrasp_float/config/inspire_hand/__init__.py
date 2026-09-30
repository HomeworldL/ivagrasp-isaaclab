"""Inspire Hand variants for the dexgrasp_float task."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-DexGrasp-Float-Inspire-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inspire_env_cfg:InspireDexGraspFloatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InspireDexGraspFloatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-DexGrasp-Float-Inspire-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inspire_env_cfg:InspireDexGraspFloatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InspireDexGraspFloatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-DexGrasp-Float-Inspire-Distill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inspire_env_cfg:InspireDexGraspFloatDistillEnvCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:InspireDexGraspFloatDistillationRunnerCfg"
        ),
    },
)

__all__ = ["agents"]
