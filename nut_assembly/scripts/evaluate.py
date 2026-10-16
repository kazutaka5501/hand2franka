"""Measure how often a trained policy grasps and lifts the ball, and record trajectories for rendering.

Every episode starts with the ball resting on the table at a random place, under full gravity and with the
observation noise of the last curriculum stage. One goal position is held for the whole episode.

    python scripts/evaluate.py --headless --checkpoint logs/rsl_rl/franka_sharpa_ball_lift/<run>/model_<n>.pt

Reported per episode (4 s):
    lifted   the ball ends at least ``--lift_height`` above where it rested, held by thumb + another finger
    at goal  the ball ends within ``--goal_tolerance`` of the commanded position
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Franka-Sharpa-Ball-Lift-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point (v2 variants).")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--lift_height", type=float, default=0.10)
parser.add_argument("--goal_tolerance", type=float, default=0.05)
parser.add_argument("--record_envs", type=int, default=8, help="Number of envs whose trajectories are saved.")
parser.add_argument("--output", type=str, default=None, help="Trajectory file, defaults to <checkpoint>_eval.npz.")
parser.add_argument("--difficulty", type=int, default=None, help="Curriculum level to evaluate at, default the last one.")
parser.add_argument("--floating_start", action="store_true", help="Spawn the ball in the air as in training.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--keep_resampling", action="store_true", help="Keep the task's goal changes within an episode.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import franka_sharpa.tasks  # noqa: F401
from franka_sharpa import nut_geometry
from franka_sharpa.tasks.ball_lift import mdp
from franka_sharpa.tasks.nut_lift import mdp as nut_mdp


class PinnedDifficulty:
    """Stands in for Dexsuite's DifficultyScheduler and reports a constant difficulty fraction."""

    def __init__(self, fraction: float):
        # a tensor like the original's, the curriculum's interpolation calls .item() on what it derives from it
        self.difficulty_frac = torch.tensor(fraction)

    def __call__(self, env, env_ids, **kwargs):
        return self.difficulty_frac

    def reset(self, env_ids=None):
        pass


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    # ball at rest on the table instead of spawned up to 0.4 m above it, one goal per episode
    if not args.floating_start:
        env_cfg.events.reset_object.params["pose_range"]["z"] = [0.0, 0.0]
    if not args.keep_resampling:
        env_cfg.commands.object_pose.resampling_time_range = (100.0, 100.0)
    env_cfg.commands.object_pose.debug_vis = False
    if getattr(env_cfg.observations, "perception", None) is not None:
        env_cfg.observations.perception.object_point_cloud.params["visualize"] = False

    agent_cfg = load_cfg_from_registry(args.task, args.agent)
    env = gym.make(args.task, cfg=env_cfg)
    base = env.unwrapped
    # pin the curriculum, otherwise failed episodes would lower gravity and noise during the evaluation
    adr = base.curriculum_manager.cfg.adr
    level = adr.params["max_difficulty"] if args.difficulty is None else args.difficulty
    adr.func = PinnedDifficulty(level / adr.params["max_difficulty"])
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=base.device)

    robot, ball, table = base.scene["robot"], base.scene["object"], base.scene["table"]
    is_nut = "Nut" in args.task
    # height of the object's centre above the table when resting: the ball radius, or the nut lying flat
    radius = nut_geometry.REST_HEIGHT_FLAT * nut_mdp.nut_scale(base) if is_nut else mdp.ball_radius(base)

    # the first reset only switches gravity on, the second starts the measured episode under it
    env.reset()
    obs, _ = env.reset()
    origins = base.scene.env_origins
    rest_height = table.data.root_pos_w[:, 2] + 0.5 * env_cfg.scene.table.spawn.size[2] + radius
    goal_w = robot.data.root_pos_w + base.command_manager.get_command("object_pose")[:, :3]

    num_steps = base.max_episode_length
    rec = slice(0, min(args.record_envs, base.num_envs))
    traj = {"joint_pos": [], "joint_target": [], "ball_pos": [], "ball_quat": [], "finger_force": [],
            "goal_pos_t": [], "goal_quat_t": []}
    running = torch.ones(base.num_envs, dtype=torch.bool, device=base.device)
    height = torch.zeros(base.num_envs, device=base.device)
    goal_dist = torch.full((base.num_envs,), float("inf"), device=base.device)
    grasped = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    max_height = torch.zeros(base.num_envs, device=base.device)
    ended_by = {name: 0 for name in base.termination_manager.active_terms}

    for _ in range(num_steps):
        # state before the step; an env that terminates is reset inside step(), so this is its last valid state
        height[running] = (ball.data.root_pos_w[:, 2] - rest_height)[running]
        max_height = torch.maximum(max_height, height)
        goal_dist[running] = torch.norm(ball.data.root_pos_w - goal_w, dim=1)[running]
        grasped[running] = mdp.good_grasp_contact(base, threshold=1.0)[running]
        traj["joint_pos"].append(robot.data.joint_pos[rec].cpu().numpy())
        traj["ball_pos"].append((ball.data.root_pos_w - origins)[rec].cpu().numpy())
        traj["ball_quat"].append(ball.data.root_quat_w[rec].cpu().numpy())
        traj["finger_force"].append(mdp.finger_contact_forces(base)[rec].cpu().numpy())
        command = base.command_manager.get_command("object_pose")
        traj["goal_pos_t"].append((robot.data.root_pos_w + command[:, :3] - origins)[rec].cpu().numpy())
        traj["goal_quat_t"].append(command[rec, 3:7].cpu().numpy())  # robot base is not rotated: world frame
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        # joint position targets the step commanded: the absolute-target actions of a student policy
        traj["joint_target"].append(robot.data.joint_pos_target[rec].cpu().numpy())
        ended = running & dones.bool()
        for name in ended_by:
            ended_by[name] += int((ended & base.termination_manager.get_term(name)).sum())
        running &= ~dones.bool()

    lifted = (height > args.lift_height) & grasped
    at_goal = goal_dist < args.goal_tolerance
    n = base.num_envs
    print(f"[evaluate] checkpoint: {args.checkpoint}")
    print(f"[evaluate] curriculum level {level}/{adr.params['max_difficulty']}, gravity {tuple(round(g, 2) for g in base.sim.physics_sim_view.get_gravity())}")
    if is_nut:
        print(f"[evaluate] episodes: {n}, nut size factor {nut_mdp.nut_scale(base).min().item():.2f}-{nut_mdp.nut_scale(base).max().item():.2f}")
    else:
        print(f"[evaluate] episodes: {n}, ball diameter {2e3 * radius.min().item():.0f}-{2e3 * radius.max().item():.0f} mm")
    print(f"[evaluate] lifted > {args.lift_height:.2f} m and held at the end: {lifted.sum().item()}/{n} = {lifted.float().mean().item():.1%}")
    print(f"[evaluate] within {args.goal_tolerance:.2f} m of the goal at the end: {at_goal.sum().item()}/{n} = {at_goal.float().mean().item():.1%}")
    print(f"[evaluate] ever lifted > {args.lift_height:.2f} m: {(max_height > args.lift_height).float().mean().item():.1%}")
    print(f"[evaluate] median final height above rest: {height.median().item():.3f} m, median goal distance: {goal_dist.median().item():.3f} m")
    print(f"[evaluate] episode endings: {ended_by}")
    small = radius < radius.median()
    print(f"[evaluate] lifted, smaller half of the {'nuts' if is_nut else 'balls'}: {lifted[small].float().mean().item():.1%}, larger half: {lifted[~small].float().mean().item():.1%}")

    output = args.output or args.checkpoint.replace(".pt", "_eval.npz")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    np.savez_compressed(
        output,
        joint_names=np.array(robot.joint_names),
        dt=base.step_dt,
        ball_radius=radius[rec].cpu().numpy(),
        object="nut" if is_nut else "ball",
        object_scale=(nut_mdp.nut_scale(base) if is_nut else radius)[rec].cpu().numpy(),
        goal_pos=(goal_w - origins)[rec].cpu().numpy(),
        table_pos=(table.data.root_pos_w - origins)[rec].cpu().numpy(),
        table_size=np.array(env_cfg.scene.table.spawn.size),
        lifted=lifted[rec].cpu().numpy(),
        at_goal=at_goal[rec].cpu().numpy(),
        **{key: np.stack(value, axis=1) for key, value in traj.items()},
    )
    print(f"[evaluate] saved trajectories of {rec.stop} envs to {output}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
