"""RSL-RL PPO config for the Inspire dexgrasp_float task."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class InspireDexGraspFloatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for the Inspire dexgrasp_float task."""

    num_steps_per_env = 10
    max_iterations = 10000
    save_interval = 100
    experiment_name = "dexgrasp_float_inspire_goal"
    run_name = ""
    clip_actions = 1.0
    obs_groups = {"actor": ["actor"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.97,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )
