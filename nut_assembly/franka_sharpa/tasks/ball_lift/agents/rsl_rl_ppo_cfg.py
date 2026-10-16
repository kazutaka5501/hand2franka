from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class FrankaSharpaBallLiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Same PPO settings as Dexsuite Kuka-Allegro."""

    num_steps_per_env = 32
    obs_groups = {"policy": ["policy", "proprio", "perception"], "critic": ["policy", "proprio", "perception"]}
    max_iterations = 15000
    save_interval = 250
    experiment_name = "franka_sharpa_ball_lift"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
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
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class FrankaSharpaBallLiftV1PPORunnerCfg(FrankaSharpaBallLiftPPORunnerCfg):
    """v1 has no point-cloud observation group."""

    obs_groups = {"policy": ["policy", "proprio"], "critic": ["policy", "proprio"]}
    experiment_name = "franka_sharpa_ball_lift_v1"


##
# v2: one environment, the agent config picks the observation groups of the policy (see ball_lift_v2_env_cfg)
##

_V2_CRITIC = ["proprio", "task", "tactile", "gt"]


@configclass
class V2FullTouchPPORunnerCfg(FrankaSharpaBallLiftPPORunnerCfg):
    obs_groups = {"policy": ["proprio", "task", "tactile", "gt"], "critic": _V2_CRITIC}
    experiment_name = "franka_sharpa_ball_lift_v2_full_touch"


@configclass
class V2FullNoTouchPPORunnerCfg(FrankaSharpaBallLiftPPORunnerCfg):
    obs_groups = {"policy": ["proprio", "task", "gt"], "critic": _V2_CRITIC}
    experiment_name = "franka_sharpa_ball_lift_v2_full_notouch"


@configclass
class V2BlindTouchPPORunnerCfg(FrankaSharpaBallLiftPPORunnerCfg):
    obs_groups = {"policy": ["proprio", "task", "tactile", "first_look"], "critic": _V2_CRITIC}
    experiment_name = "franka_sharpa_ball_lift_v2_blind_touch"


@configclass
class V2BlindNoTouchPPORunnerCfg(FrankaSharpaBallLiftPPORunnerCfg):
    obs_groups = {"policy": ["proprio", "task", "first_look"], "critic": _V2_CRITIC}
    experiment_name = "franka_sharpa_ball_lift_v2_blind_notouch"
