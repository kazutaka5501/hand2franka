from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents.rsl_rl_ppo_cfg import LiftCubePPORunnerCfg


@configclass
class GripperNutLiftPPORunnerCfg(LiftCubePPORunnerCfg):
    """Isaac Lab's Franka cube-lift PPO settings unchanged."""

    experiment_name = "franka_gripper_nut_lift"
