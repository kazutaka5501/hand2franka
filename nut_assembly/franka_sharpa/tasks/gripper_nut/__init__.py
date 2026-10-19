"""Franka Panda with its standard parallel gripper on the measured hex nut and M38x6 bolt.

    Franka-Gripper-Nut-Lift-v0     Isaac Lab's manager-based Franka lift with the nut (rsl_rl, scripts/train.py)
    Franka-Gripper-Nut-Thread-v0   Isaac Lab's Factory NutThread with the nut and bolt (rl_games,
                                   scripts/train_rl_games.py)
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Franka-Gripper-Nut-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_env_cfg:FrankaGripperNutLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GripperNutLiftPPORunnerCfg",
    },
)

gym.register(
    id="Franka-Gripper-Nut-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_env_cfg:FrankaGripperNutLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GripperNutLiftPPORunnerCfg",
    },
)

gym.register(
    id="Franka-Gripper-Nut-Thread-v0",
    entry_point="isaaclab_tasks.direct.factory.factory_env:FactoryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.thread_env_cfg:FactoryTaskNutThreadM38Cfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)
