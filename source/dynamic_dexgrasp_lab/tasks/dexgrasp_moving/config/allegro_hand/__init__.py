"""Bootstrap Allegro-hand variants for the dexgrasp_moving task."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-DexGrasp-Moving-Allegro-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_env_cfg:AllegroDexGraspMovingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AllegroDexGraspMovingPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-DexGrasp-Moving-Allegro-Distill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_env_cfg:AllegroDexGraspMovingDistillEnvCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:AllegroDexGraspMovingDistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-DexGrasp-Moving-Allegro-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_env_cfg:AllegroDexGraspMovingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AllegroDexGraspMovingPPORunnerCfg",
    },
)

__all__ = ["agents"]
