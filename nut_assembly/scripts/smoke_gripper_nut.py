"""Smoke test for the gripper nut tasks: build the env, step it, print where the nut is.

    python -u scripts/smoke_gripper_nut.py --task lift --num_envs 16 --headless
    python -u scripts/smoke_gripper_nut.py --task thread --num_envs 16 --headless

lift:   zero actions for 3 s, then random actions; prints the nut's resting height and axis tilt (flat = 0 deg,
        on its side = 90 deg) and the goal range.
thread: prints the nut pose relative to the bolt tip and gripper after reset, holds still with zero actions, then
        runs a scripted screw (centre over the bolt, push down, turn clockwise at the maximum rate) and prints the
        descent, yaw and Factory's engaged/success flags. ``--yaw_range 360`` samples the bolt yaw over a full turn
        to see which thread phases the rotation budget covers.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["lift", "thread", "m16"], required=True, help="m16: Factory's own NutThread, for reference.")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--yaw_deg", type=float, default=None, help="thread: bolt yaw start [deg] (default: task cfg).")
parser.add_argument("--yaw_range", type=float, default=None, help="thread: bolt yaw range [deg] (default: task cfg).")
parser.add_argument("--held_noise", type=float, default=None, help="thread: nut-in-gripper position noise [m] (y, z).")
parser.add_argument("--free_steps", type=int, default=45, help="thread: steps turning above the bolt before descending.")
parser.add_argument("--down", type=float, default=0.5, help="thread: z action while screwing (pushes down with 2 N per unit).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import math
import time

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply
from isaaclab_tasks.utils import parse_env_cfg

import franka_sharpa.tasks.gripper_nut  # noqa: F401

torch.set_printoptions(precision=4, sci_mode=False, linewidth=200)


def lift():
    task = "Franka-Gripper-Nut-Lift-v0"
    cfg = parse_env_cfg(task, num_envs=args.num_envs)
    env = gym.make(task, cfg=cfg)
    u = env.unwrapped
    print("[SMOKE] obs", {k: v.shape for k, v in u.observation_manager.compute().items()}, "action", u.action_manager.total_action_dim)
    print("[SMOKE] goal ranges", cfg.commands.object_pose.ranges)
    env.reset()
    nut = u.scene["object"]
    zero = torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
    t0 = time.time()
    for i in range(150):
        env.step(zero)
        if i in (0, 25, 50, 149):
            pos = nut.data.root_pos_w - u.scene.env_origins
            axis = quat_apply(nut.data.root_quat_w, torch.tensor([0.0, 0.0, 1.0], device=u.device).expand(u.num_envs, 3))
            tilt = torch.rad2deg(torch.acos(axis[:, 2].abs().clamp(max=1.0)))
            print(f"[SMOKE] step {i:3d} nut z {pos[:, 2].cpu()}\n[SMOKE]          tilt deg {tilt.cpu()}\n"
                  f"[SMOKE]          lin vel {nut.data.root_lin_vel_w.norm(dim=1).max().item():.4f}")
    print(f"[SMOKE] 150 zero-action steps in {time.time() - t0:.1f} s")
    flat = (tilt < 10).sum().item()
    side = (tilt > 80).sum().item()
    print(f"[SMOKE] resting: flat {flat}, on side {side}, other {u.num_envs - flat - side}; "
          f"expected z flat {0.0165:.4f}, side {0.0255:.4f}")
    print("[SMOKE] goal (robot frame) sample", u.command_manager.get_command("object_pose")[:4, :3].cpu())
    rew = 0.0
    for i in range(100):
        _, r, term, trunc, _ = env.step(2 * torch.rand_like(zero) - 1)
        rew += r.mean().item()
    pos = nut.data.root_pos_w - u.scene.env_origins
    print(f"[SMOKE] 100 random steps ok, mean reward/step {rew / 100:.4f}, nut z {pos[:, 2].min().item():.4f}..{pos[:, 2].max().item():.4f}")
    env.close()


def thread():
    task = "Isaac-Factory-NutThread-Direct-v0" if args.task == "m16" else "Franka-Gripper-Nut-Thread-v0"
    cfg = parse_env_cfg(task, num_envs=args.num_envs)
    if args.yaw_deg is not None:
        cfg.task.fixed_asset_init_orn_deg = args.yaw_deg
    if args.yaw_range is not None:
        cfg.task.fixed_asset_init_orn_range_deg = args.yaw_range
    if args.held_noise is not None:
        cfg.task.held_asset_pos_noise = [0.0, args.held_noise, args.held_noise]
    env = gym.make(task, cfg=cfg)
    u = env.unwrapped
    tc = u.cfg_task
    print(f"[SMOKE] task {tc.name} pitch {tc.fixed_asset_cfg.thread_pitch} base_height {tc.fixed_asset_cfg.base_height} "
          f"height {tc.fixed_asset_cfg.height} kp {tc.keypoint_coef_baseline} {tc.keypoint_coef_coarse} {tc.keypoint_coef_fine} "
          f"scale {tc.keypoint_scale} bolt yaw {tc.fixed_asset_init_orn_deg}+{tc.fixed_asset_init_orn_range_deg}")
    t0 = time.time()
    obs, _ = env.reset()
    print(f"[SMOKE] reset {time.time() - t0:.1f} s; obs {obs['policy'].shape} state {obs['critic'].shape}")
    tip_z = tc.fixed_asset_cfg.base_height + tc.fixed_asset_cfg.height

    def report(tag):
        bolt_yaw = euler_xyz_from_quat(u.fixed_quat)[2]
        nut_roll, nut_pitch, nut_yaw = euler_xyz_from_quat(u.held_quat)
        # nut pose in the bolt frame (bolt is upright, so a yaw rotation)
        d = u.held_pos - u.fixed_pos
        c, s = torch.cos(-bolt_yaw), torch.sin(-bolt_yaw)
        d_b = torch.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1], d[:, 2]], 1)
        base_above_tip = d_b[:, 2] + tc.fixed_asset_cfg.base_height - tip_z  # nut bottom face above the tip
        ft_above_tip = u.fingertip_midpoint_pos[:, 2] - u.fixed_pos[:, 2] - tip_z
        tilt = torch.rad2deg(torch.acos((torch.cos(nut_roll) * torch.cos(nut_pitch)).clamp(-1, 1)))
        rel_yaw = torch.rad2deg(torch.remainder(nut_yaw - bolt_yaw, 2 * math.pi))
        ft_yaw = euler_xyz_from_quat(u.fingertip_midpoint_quat)[2]
        engaged = u._get_curr_successes(tc.engage_threshold, check_rot=False)
        success = u._get_curr_successes(tc.success_threshold, check_rot=True)
        print(f"[SMOKE] {tag}\n"
              f"  nut bottom above tip [mm] {(1e3 * base_above_tip).cpu()}\n"
              f"  fingertip above tip [mm]  {(1e3 * ft_above_tip).cpu()}\n"
              f"  nut xy off axis [mm]      {(1e3 * d_b[:, :2].norm(dim=1)).cpu()}\n"
              f"  nut tilt [deg]            {tilt.cpu()}\n"
              f"  nut yaw - bolt yaw [deg]  {rel_yaw.cpu()}\n"
              f"  bolt yaw [deg]            {torch.rad2deg(bolt_yaw).cpu()}\n"
              f"  fingertip yaw [deg]       {torch.rad2deg(ft_yaw).cpu()}\n"
              f"  joint7 [rad]              {u.joint_pos[:, 6].cpu()}\n"
              f"  finger joints [mm]        {(1e3 * u.joint_pos[:, 7]).cpu()} {(1e3 * u.joint_pos[:, 8]).cpu()}\n"
              f"  engaged {engaged.sum().item()}/{u.num_envs}  success {success.sum().item()}/{u.num_envs}")
        return base_above_tip, success

    report("after reset")
    zero = torch.zeros(u.num_envs, 6, device=u.device)
    zero[:, 5] = -1.0  # unidirectional rot: -1 = no rotation
    for _ in range(15):
        env.step(zero)
    report("after 15 zero-action steps (1 s)")

    steps = args.steps or 400

    def nut_depth_and_rel_yaw():
        bolt_yaw = euler_xyz_from_quat(u.fixed_quat)[2]
        depth = u.held_pos[:, 2] - u.fixed_pos[:, 2] + tc.fixed_asset_cfg.base_height - tip_z
        rel = torch.rad2deg(torch.remainder(euler_xyz_from_quat(u.held_quat)[2] - bolt_yaw, 2 * math.pi))
        return depth, rel

    nan = torch.full((u.num_envs,), float("nan"), device=u.device)
    seat_yaw, seat_depth, catch_yaw = nan.clone(), nan.clone(), nan.clone()
    catch_step = torch.full((u.num_envs,), -1, device=u.device)
    first_success = torch.full((u.num_envs,), -1, device=u.device)
    t0 = time.time()
    for i in range(steps):
        a = torch.zeros(u.num_envs, 6, device=u.device)
        to_axis = u.fixed_pos_obs_frame[:, :2] - u.fingertip_midpoint_pos[:, :2]
        a[:, 0:2] = (to_axis / 0.02).clamp(-1, 1)
        if i < args.free_steps:  # turn in free space: unloaded yaw rate
            a[:, 5] = 1.0
            if i == 0:
                yaw0 = euler_xyz_from_quat(u.fingertip_midpoint_quat)[2]
            elif i == args.free_steps - 1:
                dyaw = torch.remainder(yaw0 - euler_xyz_from_quat(u.fingertip_midpoint_quat)[2], 2 * math.pi)
                print(f"[SMOKE] free turning: {torch.rad2deg(dyaw).mean().item() / i:.2f} deg/step clockwise")
        else:
            centred = to_axis.norm(dim=1) < 0.002
            a[:, 2] = torch.where(centred, -args.down, 0.0)
            a[:, 5] = torch.where(centred, 1.0, -1.0)  # full clockwise rate once centred
        _, rew, _, _, _ = env.step(a)
        success = u._get_curr_successes(tc.success_threshold, check_rot=True)
        first_success[(first_success < 0) & success] = i
        # thread phase scan: relative yaw when the nut first sits on the tip, and when it first gets one
        # millimetre lower (caught the thread)
        depth, rel_yaw = nut_depth_and_rel_yaw()
        seated = (seat_yaw.isnan()) & (depth < -0.0045)
        seat_yaw[seated], seat_depth[seated] = rel_yaw[seated], depth[seated]
        caught = (catch_yaw.isnan()) & (depth < -0.006)
        catch_yaw[caught] = rel_yaw[caught]
        catch_step[caught] = i
        if (i + 1) % 100 == 0 or i + 1 == args.free_steps:
            report(f"scripted screw step {i + 1} (reward mean {rew.mean().item():.3f})")
    print(f"[SMOKE] {steps} steps in {time.time() - t0:.1f} s ({u.num_envs} envs)")
    print(f"[SMOKE] first success step per env {first_success.cpu()}")
    depth, rel_yaw = nut_depth_and_rel_yaw()
    order = torch.argsort(seat_yaw.nan_to_num(999.0))
    print("[SMOKE] phase scan (sorted by nut-bolt yaw at seating; yaw decreases while screwing):")
    for k in order.tolist():
        print(f"[SCAN] bolt_yaw {math.degrees(euler_xyz_from_quat(u.fixed_quat)[2][k].item()):7.1f} "
              f"seat_rel_yaw {seat_yaw[k].item():7.1f} seat_depth_mm {1e3 * seat_depth[k].item():6.2f} "
              f"catch_rel_yaw {catch_yaw[k].item():7.1f} catch_step {catch_step[k].item():4d} "
              f"final_depth_mm {1e3 * depth[k].item():6.2f} final_rel_yaw {rel_yaw[k].item():7.1f} "
              f"success_step {first_success[k].item():4d}")
    print(f"[SMOKE] factory extras {[k for k in u.extras]}")
    env.close()


if args.task == "lift":
    lift()
else:
    thread()
app.close()
