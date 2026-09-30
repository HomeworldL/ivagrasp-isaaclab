"""RSL-RL distillation config for the Liberhand dexgrasp_float task."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)


@configclass
class LiberhandDexGraspFloatDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Distillation runner config for the Liberhand student policy."""

    num_steps_per_env = 32
    max_iterations = 10000
    save_interval = 100
    experiment_name = "dexgrasp_float_liberhand_student"
    run_name = ""
    clip_actions = 1.0
    obs_groups = {"student": ["student"], "teacher": ["teacher"]}
    student = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.8),
    )
    teacher = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.0),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=3.0e-4,
        gradient_length=16,
        max_grad_norm=1.0,
        loss_type="mse",
    )
