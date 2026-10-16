"""Franka + Sharpa Wave ball grasp-and-lift environments."""

import gymnasium as gym

from . import agents

gym.register(
    id="Franka-Sharpa-Ball-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_lift_env_cfg:FrankaSharpaBallLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaSharpaBallLiftPPORunnerCfg",
    },
)

gym.register(
    id="Franka-Sharpa-Ball-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_lift_env_cfg:FrankaSharpaBallLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaSharpaBallLiftPPORunnerCfg",
    },
)

# v1: ball-specific observations and rewards
gym.register(
    id="Franka-Sharpa-Ball-Lift-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_lift_v1_env_cfg:FrankaSharpaBallLiftV1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaSharpaBallLiftV1PPORunnerCfg",
    },
)

gym.register(
    id="Franka-Sharpa-Ball-Lift-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ball_lift_v1_env_cfg:FrankaSharpaBallLiftV1EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaSharpaBallLiftV1PPORunnerCfg",
    },
)

# v2: tactile observation groups, choose the agent with --agent rsl_rl_<variant>_cfg_entry_point
_V2_AGENTS = {
    f"rsl_rl_{variant}_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{cls}"
    for variant, cls in {
        "full_touch": "V2FullTouchPPORunnerCfg",
        "full_notouch": "V2FullNoTouchPPORunnerCfg",
        "blind_touch": "V2BlindTouchPPORunnerCfg",
        "blind_notouch": "V2BlindNoTouchPPORunnerCfg",
    }.items()
}

gym.register(
    id="Franka-Sharpa-Ball-Lift-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.ball_lift_v2_env_cfg:FrankaSharpaBallLiftV2EnvCfg", **_V2_AGENTS},
)

gym.register(
    id="Franka-Sharpa-Ball-Lift-Play-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.ball_lift_v2_env_cfg:FrankaSharpaBallLiftV2EnvCfg_PLAY", **_V2_AGENTS},
)
