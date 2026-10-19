"""Evaluate the Franka-gripper nut policies and record trajectories for ``scripts/render_gripper.py``.

    python scripts/evaluate_gripper.py --headless --task thread --checkpoint logs/rl_games/FactoryNutM38/<run>/nn/<ckpt>.pth
    python scripts/evaluate_gripper.py --headless --task lift --checkpoint logs/rsl_rl/franka_gripper_nut_lift/<run>/model_<n>.pt

thread  Factory's nut-threading task with the M38 nut and bolt (rl_games policy). Success is Factory's own criterion
        (nut down the thread past its success height and centred), latched over the episode.
lift    pick the nut up from the table and bring it to the commanded position (rsl_rl policy). Success: within 5 cm
        of the goal at the end, and at least 5 cm above the table.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["thread", "lift"], required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--record_envs", type=int, default=8)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--seed", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import math

import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import franka_sharpa.tasks  # noqa: F401
from franka_sharpa import nut_geometry, thread_geometry

TASKS = {"thread": "Franka-Gripper-Nut-Thread-v0", "lift": "Franka-Gripper-Nut-Lift-Play-v0"}
NUT_FRAME_TO_CENTRE = thread_geometry.BOLT_HEAD_HEIGHT + 0.5 * nut_geometry.THICKNESS  # Factory nut frame


def run_thread(task):
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
    gym_env = gym.make(task, cfg=env_cfg)
    rl_device = agent_cfg["params"]["config"]["device"]
    env = RlGamesVecEnvWrapper(gym_env, rl_device, agent_cfg["params"]["env"].get("clip_observations", math.inf),
                               agent_cfg["params"]["env"].get("clip_actions", math.inf),
                               agent_cfg["params"]["env"].get("obs_groups"),
                               agent_cfg["params"]["env"].get("concate_obs_groups", True))
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args.checkpoint
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(args.checkpoint)
    agent.reset()
    base = env.unwrapped
    robot, nut, bolt = base._robot, base._held_asset, base._fixed_asset

    obs = env.reset()
    obs = obs["obs"] if isinstance(obs, dict) else obs
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()
    rec = slice(0, min(args.record_envs, base.num_envs))
    traj = {"joint_pos": [], "nut_pos": [], "nut_quat": []}
    succeeded = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    bolt_pos = (bolt.data.root_pos_w - base.scene.env_origins)[rec].cpu().numpy()
    bolt_quat = bolt.data.root_quat_w[rec].cpu().numpy()
    for _ in range(int(base.max_episode_length) - 1):
        traj["joint_pos"].append(robot.data.joint_pos[rec].cpu().numpy())
        traj["nut_pos"].append((nut.data.root_pos_w - base.scene.env_origins)[rec].cpu().numpy())
        traj["nut_quat"].append(nut.data.root_quat_w[rec].cpu().numpy())
        with torch.inference_mode():
            actions = agent.get_action(agent.obs_to_torch(obs), is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)
            obs = obs["obs"] if isinstance(obs, dict) else obs
        succeeded |= base.ep_succeeded.bool()
    return base, robot, traj, succeeded, dict(bolt_pos=bolt_pos, bolt_quat=bolt_quat, nut_frame_to_centre=NUT_FRAME_TO_CENTRE)


def run_lift(task):
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.commands.object_pose.resampling_time_range = (100.0, 100.0)  # one goal per episode
    env_cfg.commands.object_pose.debug_vis = False
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    env = RslRlVecEnvWrapper(gym.make(task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    base = env.unwrapped
    robot, nut = base.scene["robot"], base.scene["object"]
    obs, _ = env.reset()
    origins = base.scene.env_origins
    goal_w = robot.data.root_pos_w + base.command_manager.get_command("object_pose")[:, :3]
    rec = slice(0, min(args.record_envs, base.num_envs))
    traj = {"joint_pos": [], "nut_pos": [], "nut_quat": []}
    running = torch.ones(base.num_envs, dtype=torch.bool, device=base.device)
    final_dist = torch.full((base.num_envs,), float("inf"), device=base.device)
    final_z = torch.zeros(base.num_envs, device=base.device)
    for _ in range(int(base.max_episode_length)):
        final_dist[running] = torch.norm(nut.data.root_pos_w - goal_w, dim=-1)[running]
        final_z[running] = (nut.data.root_pos_w[:, 2] - origins[:, 2])[running]
        traj["joint_pos"].append(robot.data.joint_pos[rec].cpu().numpy())
        traj["nut_pos"].append((nut.data.root_pos_w - origins)[rec].cpu().numpy())
        traj["nut_quat"].append(nut.data.root_quat_w[rec].cpu().numpy())
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        running &= ~dones.bool()
    succeeded = (final_dist < 0.05) & (final_z > 0.05)
    print(f"[gripper] median final goal distance {1e3 * final_dist.median().item():.0f} mm")
    return base, robot, traj, succeeded, dict(goal_pos=(goal_w - origins)[rec].cpu().numpy(), nut_frame_to_centre=0.0)


def main():
    task = TASKS[args.task]
    base, robot, traj, succeeded, extra = (run_thread if args.task == "thread" else run_lift)(task)
    n = base.num_envs
    print(f"[gripper] {args.task}: {args.checkpoint}")
    print(f"[gripper] success {succeeded.sum().item()}/{n} = {succeeded.float().mean().item():.1%}")
    rec = slice(0, min(args.record_envs, n))
    np.savez_compressed(
        args.output, task=args.task, joint_names=np.array(robot.joint_names), dt=base.step_dt,
        success=succeeded[rec].cpu().numpy(), **extra, **{k: np.stack(v, axis=1) for k, v in traj.items()},
    )
    print(f"[gripper] saved {args.output}")
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
