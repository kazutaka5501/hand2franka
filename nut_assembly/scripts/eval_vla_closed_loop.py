"""Closed-loop evaluation of a SmolVLA student in Isaac: the student alone drives the robot, from camera images.

Start the policy server first (``scripts/vla_server.py``), then

    python scripts/eval_vla_closed_loop.py --headless --num_envs 32 --output logs/videos/vla_distilled.npz

Same episodes and criteria as ``scripts/evaluate.py`` for the RL teacher: the nut rests on the table, full gravity,
one goal per episode, 4 s. The student's absolute joint targets are turned into the environment's relative actions
(target = current joint positions + scale x action). The student acts at 30 Hz, the environment steps at 60 Hz, so each
target is held for two steps. Trajectories of the first envs are saved for ``scripts/render_mujoco.py``.
"""

import argparse
import io

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Franka-Sharpa-Nut-Lift-Dense-Play-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=1, help="Rounds of num_envs episodes.")
parser.add_argument("--lift_height", type=float, default=0.10)
parser.add_argument("--goal_tolerance", type=float, default=0.05)
parser.add_argument("--record_envs", type=int, default=8)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--keep_resampling", action="store_true", help="Keep the task's goal changes (stage 1).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from multiprocessing.connection import Client

import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg

import franka_sharpa.tasks  # noqa: F401
from franka_sharpa import nut_geometry
from franka_sharpa.tasks.ball_lift import mdp
from franka_sharpa.tasks.nut_lift import mdp as nut_mdp

ADDRESS = ("localhost", 6011)
AUTHKEY = b"franka-sharpa-vla"
STEPS_PER_TARGET = 2


def pack(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def unpack(blob: bytes) -> dict:
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


class PinnedDifficulty:
    def __init__(self, fraction):
        self.difficulty_frac = torch.tensor(fraction)

    def __call__(self, env, env_ids, **kwargs):
        return self.difficulty_frac

    def reset(self, env_ids=None):
        pass


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.events.reset_object.params["pose_range"]["z"] = [0.0, 0.0]
    if not args.keep_resampling:
        env_cfg.commands.object_pose.resampling_time_range = (100.0, 100.0)
    env_cfg.commands.object_pose.debug_vis = False
    env = gym.make(args.task, cfg=env_cfg)
    base = env.unwrapped
    base.curriculum_manager.cfg.adr.func = PinnedDifficulty(1.0)
    robot, nut, table = base.scene["robot"], base.scene["object"], base.scene["table"]
    term = base.action_manager.get_term("action")
    scale = term._scale if torch.is_tensor(term._scale) else torch.full((base.num_envs, robot.num_joints), term._scale)

    conn = Client(ADDRESS, authkey=AUTHKEY)
    origins = base.scene.env_origins
    env.reset()
    env.reset()  # the first reset switches gravity on
    conn.send_bytes(pack(
        joint_names=np.array(robot.joint_names),
        nut_scale=nut_mdp.nut_scale(base).cpu().numpy(),
        table_pos=(table.data.root_pos_w - origins).cpu().numpy(),
        table_size=np.array(env_cfg.scene.table.spawn.size),
    ))
    conn.recv_bytes()

    rest = nut_mdp.table_top_z(base) + nut_geometry.REST_HEIGHT_FLAT * nut_mdp.nut_scale(base)
    all_lifted, all_goal = [], []
    rec = slice(0, min(args.record_envs, base.num_envs))
    traj = {"joint_pos": [], "ball_pos": [], "ball_quat": [], "goal_pos_t": [], "goal_quat_t": []}
    torch.inference_mode().__enter__()  # all stepping and resetting below, as in rsl_rl's play loop
    for round_ in range(args.episodes):
        if round_ > 0:
            env.reset()
        goal_w = robot.data.root_pos_w + base.command_manager.get_command("object_pose")[:, :3]
        running = torch.ones(base.num_envs, dtype=torch.bool, device=base.device)
        height = torch.zeros(base.num_envs, device=base.device)
        goal_dist = torch.full((base.num_envs,), float("inf"), device=base.device)
        grasped = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
        first = True
        for _ in range(int(base.max_episode_length) // STEPS_PER_TARGET):
            command = base.command_manager.get_command("object_pose")
            goal_w = robot.data.root_pos_w + command[:, :3]
            conn.send_bytes(pack(
                goal_quat=command[:, 3:7].cpu().numpy(),
                joint_pos=robot.data.joint_pos.cpu().numpy(),
                nut_pos=(nut.data.root_pos_w - origins).cpu().numpy(),
                nut_quat=nut.data.root_quat_w.cpu().numpy(),
                goal_pos=(goal_w - origins).cpu().numpy(),
                reset=np.array(int(first)),
            ))
            first = False
            target = torch.from_numpy(unpack(conn.recv_bytes())["target"]).to(base.device)
            for _ in range(STEPS_PER_TARGET):
                height[running] = (nut.data.root_pos_w[:, 2] - rest)[running]
                goal_dist[running] = torch.norm(nut.data.root_pos_w - goal_w, dim=1)[running]
                grasped[running] = mdp.good_grasp_contact(base, threshold=1.0)[running]
                if round_ == 0:
                    traj["joint_pos"].append(robot.data.joint_pos[rec].cpu().numpy())
                    traj["ball_pos"].append((nut.data.root_pos_w - origins)[rec].cpu().numpy())
                    traj["ball_quat"].append(nut.data.root_quat_w[rec].cpu().numpy())
                    traj["goal_pos_t"].append((goal_w - origins)[rec].cpu().numpy())
                    traj["goal_quat_t"].append(command[rec, 3:7].cpu().numpy())
                action = ((target - robot.data.joint_pos) / scale).clamp(-1.0, 1.0)
                _, _, terminated, truncated, _ = env.step(action)
                running &= ~(terminated | truncated)
        lifted = (height > args.lift_height) & grasped
        at_goal = goal_dist < args.goal_tolerance
        all_lifted.append(lifted.cpu())
        all_goal.append(at_goal.cpu())
        if round_ == 0:
            first_round = dict(goal_pos=(goal_w - origins)[rec].cpu().numpy(), lifted=lifted[rec].cpu().numpy(),
                               at_goal=at_goal[rec].cpu().numpy())
    conn.send_bytes(b"done")

    lifted, at_goal = torch.cat(all_lifted), torch.cat(all_goal)
    n = len(lifted)
    print(f"[vla] episodes: {n}")
    print(f"[vla] lifted > {args.lift_height:.2f} m and held at the end: {lifted.sum().item()}/{n} = {lifted.float().mean().item():.1%}")
    print(f"[vla] within {args.goal_tolerance:.2f} m of the goal at the end: {at_goal.sum().item()}/{n} = {at_goal.float().mean().item():.1%}")
    np.savez_compressed(
        args.output,
        joint_names=np.array(robot.joint_names),
        dt=base.step_dt,
        ball_radius=nut_mdp.nut_scale(base)[rec].cpu().numpy() * nut_geometry.REST_HEIGHT_FLAT,
        object="nut",
        object_scale=nut_mdp.nut_scale(base)[rec].cpu().numpy(),
        table_pos=(table.data.root_pos_w - origins)[rec].cpu().numpy(),
        table_size=np.array(env_cfg.scene.table.spawn.size),
        **first_round,
        **{key: np.stack(value, axis=1) for key, value in traj.items()},
    )
    print(f"[vla] saved {args.output}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
