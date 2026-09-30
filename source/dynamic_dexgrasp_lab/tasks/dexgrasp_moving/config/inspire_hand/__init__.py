"""Inspire Hand variants for the dexgrasp_moving task."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-DexGrasp-Moving-Inspire-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inspire_env_cfg:InspireDexGraspMovingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InspireDexGraspMovingPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-DexGrasp-Moving-Inspire-Distill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inspire_env_cfg:InspireDexGraspMovingDistillEnvCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:InspireDexGraspMovingDistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-DexGrasp-Moving-Inspire-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inspire_env_cfg:InspireDexGraspMovingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:InspireDexGraspMovingPPORunnerCfg",
    },
)

__all__ = ["agents"]
