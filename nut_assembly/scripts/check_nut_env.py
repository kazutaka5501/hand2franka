"""Sanity-check the nut environment: the nut spawns, falls and settles flat or on its side; rewards stay finite.

    python scripts/check_nut_env.py --headless --num_envs 64
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Franka-Sharpa-Nut-Lift-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils import parse_env_cfg

import franka_sharpa.tasks  # noqa: F401
from franka_sharpa import nut_geometry
from franka_sharpa.tasks.nut_lift import mdp


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    # full gravity, so the nut falls onto the table
    env_cfg.curriculum.adr.params["init_difficulty"] = env_cfg.curriculum.adr.params["max_difficulty"]
    env = gym.make(args.task, cfg=env_cfg)
    base = env.unwrapped
    nut = base.scene["object"]
    print("[check_nut] obs dims:", {k: v for k, v in base.observation_manager.group_obs_dim.items()})
    print("[check_nut] reward terms:", base.reward_manager.active_terms)
    print("[check_nut] nut mass:", nut.root_physx_view.get_masses().flatten()[:6].tolist())
    print("[check_nut] nut scale:", mdp.nut_scale(base)[:6].tolist())

    # gravity is switched on by the curriculum at the first reset; the second reset spawns under it
    env.reset()
    env.reset()
    zero = torch.zeros(base.num_envs, base.action_manager.total_action_dim, device=base.device)
    # robot holds still for 1.5 s while the nut lands
    for _ in range(75):
        env.step(zero)
    z = nut.data.root_pos_w[:, 2] - mdp.table_top_z(base)
    axis = quat_apply(nut.data.root_quat_w, torch.tensor([0.0, 0.0, 1.0], device=base.device).expand(base.num_envs, 3))
    vertical = axis[:, 2].abs() > 0.9
    side = axis[:, 2].abs() < 0.1
    speed = torch.norm(nut.data.root_lin_vel_w, dim=-1)
    on_table = (z > 0.0) & (z < 0.05)
    print(f"[check_nut] after 1.5 s: on table {on_table.sum().item()}/{base.num_envs}, flat {vertical.sum().item()}, "
          f"on side {side.sum().item()}, max speed {speed.max().item():.3f} m/s")
    print(f"[check_nut] centre height above table, flat: {z[vertical].mean().item():.4f} (expected "
          f"{nut_geometry.REST_HEIGHT_FLAT:.4f}), side: {z[side].mean().item() if side.any() else float('nan'):.4f} "
          f"(expected {nut_geometry.REST_HEIGHT_SIDE:.4f})")

    env.reset()
    sums = {}
    for _ in range(args.steps):
        _, rew, _, _, _ = env.step(torch.randn_like(zero))
        assert torch.isfinite(rew).all(), "non-finite reward"
        for name, value in zip(base.reward_manager.active_terms, base.reward_manager._step_reward.mean(0).tolist()):
            sums[name] = sums.get(name, 0.0) + value
    print("[check_nut] random policy, mean step reward per term:", {k: round(v / args.steps, 5) for k, v in sums.items()})
    env.close()


if __name__ == "__main__":
    main()
    app.close()
