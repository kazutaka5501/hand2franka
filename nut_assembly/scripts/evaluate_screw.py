"""Evaluate a stage-2 (screwing) policy: does it turn the nut down the thread, or only push it?

Every episode starts with the nut resting on the bolt tip. Per episode we measure, from the nut's own pose:
    turns    net clockwise rotation about the bolt axis (seen from above), counted continuously
    descent  how far the nut went down from where it settled on the tip [mm]
    pitch    descent / turns: a real screw motion gives the thread pitch (6 mm per turn)
and save trajectories for ``scripts/render_screw.py``.

    python scripts/evaluate_screw.py --headless --checkpoint logs/rsl_rl/franka_sharpa_nut_screw/<run>/model_<n>.pt
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Franka-Sharpa-Nut-Screw-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--record_envs", type=int, default=8)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--seed", type=int, default=11)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.math import yaw_quat
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import franka_sharpa.tasks  # noqa: F401
from franka_sharpa import thread_geometry
from franka_sharpa.tasks.nut_lift import screw


def yaw(q):
    y = yaw_quat(q)
    return 2 * torch.atan2(y[:, 3], y[:, 0])


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env = gym.make(args.task, cfg=env_cfg)
    base = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=base.device)
    robot, nut = base.scene["robot"], base.scene["object"]
    command = base.command_manager.get_term("object_pose")
    origins = base.scene.env_origins

    env.reset()
    obs, _ = env.reset()  # the first reset switches the curriculum's gravity on
    n = base.num_envs
    running = torch.ones(n, dtype=torch.bool, device=base.device)
    turns = torch.zeros(n, device=base.device)
    descent = torch.zeros(n, device=base.device)
    off_axis = torch.zeros(n, device=base.device)
    waypoints = torch.zeros(n, device=base.device)
    prev = yaw(nut.data.root_quat_w)
    rec = slice(0, min(args.record_envs, n))
    traj = {"joint_pos": [], "nut_pos": [], "nut_quat": []}
    for step in range(int(base.max_episode_length)):
        traj["joint_pos"].append(robot.data.joint_pos[rec].cpu().numpy())
        traj["nut_pos"].append((nut.data.root_pos_w - origins)[rec].cpu().numpy())
        traj["nut_quat"].append(nut.data.root_quat_w[rec].cpu().numpy())
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        now = yaw(nut.data.root_quat_w)
        dyaw = torch.remainder(now - prev + math.pi, 2 * math.pi) - math.pi
        prev = now
        live = running & ~dones.bool()
        turns[live] -= dyaw[live] / (2 * math.pi)  # clockwise from above counts positive
        descent[live] = command.descent()[live]
        off_axis[live] = torch.norm(nut.data.root_pos_w[:, :2] - screw.bolt_pos_w(base)[:, :2], dim=-1)[live]
        waypoints[live] = command.waypoint[live].float()
        running &= ~dones.bool()

    t, d = turns.cpu().numpy(), 1e3 * descent.cpu().numpy()
    on = off_axis.cpu().numpy() < 0.004
    print(f"[screw] checkpoint {args.checkpoint}, {n} episodes of {base.max_episode_length_s:.0f} s")
    print(f"[screw] nut still on the bolt at the end: {on.mean():.0%}")
    print(f"[screw] clockwise turns: median {np.median(t):.2f}, mean {t.mean():.2f}; >= 1 turn: {(t >= 1).mean():.0%}, >= 2: {(t >= 2).mean():.0%}")
    print(f"[screw] descent: median {np.median(d):.1f} mm; >= 6 mm (one pitch): {(d >= 6).mean():.0%}")
    sel = on & (t > 0.25)
    if sel.any():
        print(f"[screw] descent per turn where turned > 1/4 and still on the bolt: median {np.median(d[sel] / t[sel]):.1f} mm "
              f"(thread pitch {1e3 * thread_geometry.PITCH:.0f} mm)")
    print(f"[screw] waypoints reached (8 = two turns): median {np.median(waypoints.cpu().numpy()):.0f}")
    np.savez_compressed(
        args.output,
        joint_names=np.array(robot.joint_names),
        dt=base.step_dt,
        bolt_pos=np.array(screw.BOLT_POS),
        turns=t[rec],
        descent_mm=d[rec],
        **{k: np.stack(v, axis=1) for k, v in traj.items()},
    )
    print(f"[screw] saved {args.output}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
