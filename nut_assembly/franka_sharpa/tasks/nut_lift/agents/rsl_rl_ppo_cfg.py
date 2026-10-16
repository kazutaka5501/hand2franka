from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from franka_sharpa.tasks.ball_lift.agents.rsl_rl_ppo_cfg import FrankaSharpaBallLiftV1PPORunnerCfg


@configclass
class NutLiftPPORunnerCfg(FrankaSharpaBallLiftV1PPORunnerCfg):
    """Ball v1 PPO settings: policy and critic see the task and proprio groups."""

    experiment_name = "franka_sharpa_nut_lift"


@configclass
class NutLiftDensePPORunnerCfg(NutLiftPPORunnerCfg):
    experiment_name = "franka_sharpa_nut_lift_dense"


@configclass
class NutLiftV1PPORunnerCfg(NutLiftPPORunnerCfg):
    experiment_name = "franka_sharpa_nut_lift_v1"


@configclass
class NutPlayPPORunnerCfg(NutLiftPPORunnerCfg):
    """Continues a trained policy: smaller steps, so the critic catches up with the new rewards before the grasp is lost."""

    experiment_name = "franka_sharpa_nut_play"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )


@configclass
class NutScrewPPORunnerCfg(NutPlayPPORunnerCfg):
    experiment_name = "franka_sharpa_nut_screw"
