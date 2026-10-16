"""Franka + Sharpa Wave picking up the 3D-printed hex nut and bringing it to a commanded position."""

import gymnasium as gym

from . import agents

_ENVS = {
    # SimToolReal progress rewards
    "Franka-Sharpa-Nut-Lift-v0": ("NutLiftEnvCfg", "NutLiftPPORunnerCfg"),
    "Franka-Sharpa-Nut-Lift-Play-v0": ("NutLiftEnvCfg_PLAY", "NutLiftPPORunnerCfg"),
    # SimToolReal progress rewards without Dexsuite's action penalties (v0 kept them and learned to keep still)
    "Franka-Sharpa-Nut-Lift-v1": ("NutLiftV1EnvCfg", "NutLiftV1PPORunnerCfg"),
    "Franka-Sharpa-Nut-Lift-Play-v1": ("NutLiftV1EnvCfg_PLAY", "NutLiftV1PPORunnerCfg"),
    # stage 1 of the assembly plan: 6D goals, warm-started from the dense lift policy
    "Franka-Sharpa-Nut-Play-v0": ("NutPlayEnvCfg", "NutPlayPPORunnerCfg"),
    "Franka-Sharpa-Nut-Play-Play-v0": ("NutPlayEnvCfg_PLAY", "NutPlayPPORunnerCfg"),
    # stage 2: screw the nut onto the bolt, continuing from stage 1
    "Franka-Sharpa-Nut-Screw-v0": ("NutScrewEnvCfg", "NutScrewPPORunnerCfg"),
    "Franka-Sharpa-Nut-Screw-Play-v0": ("NutScrewEnvCfg_PLAY", "NutScrewPPORunnerCfg"),
    # ball v1 dense rewards, for comparison
    "Franka-Sharpa-Nut-Lift-Dense-v0": ("NutLiftDenseEnvCfg", "NutLiftDensePPORunnerCfg"),
    "Franka-Sharpa-Nut-Lift-Dense-Play-v0": ("NutLiftDenseEnvCfg_PLAY", "NutLiftDensePPORunnerCfg"),
}

for task_id, (env_cfg, agent_cfg) in _ENVS.items():
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.nut_lift_env_cfg:{env_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{agent_cfg}",
        },
    )
