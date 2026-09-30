"""Bootstrap Allegro-hand variants for the dexgrasp_float task."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-DexGrasp-Float-Allegro-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_env_cfg:AllegroDexGraspFloatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AllegroDexGraspFloatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-DexGrasp-Float-Allegro-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_env_cfg:AllegroDexGraspFloatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AllegroDexGraspFloatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-DexGrasp-Float-Allegro-Distill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_env_cfg:AllegroDexGraspFloatDistillEnvCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:AllegroDexGraspFloatDistillationRunnerCfg"
        ),
    },
)

__all__ = ["agents"]
