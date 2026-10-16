"""MDP terms for picking up the hex nut and bringing it to a commanded position.

The main reward follows SimToolReal (Lum et al. 2026, github.com/tylerlum/simtoolreal,
``isaacsimenvs/tasks/simtoolreal/utils/reward_utils.py``): rewards are paid for *progress*, i.e. for beating the
best value reached so far in the episode, in three stages

    1. fingertips approach the object            (before the lift)
    2. the object rises off the table, one-off bonus at 10 cm   (before the lift)
    3. the object approaches the goal            (after the lift)

plus a per-step bonus while the object is at the goal and an L1 joint velocity penalty. Paying for progress
instead of for proximity means the policy cannot farm reward by hovering near a good state.

Changes to SimToolReal, which drops the object onto the table and trains under full gravity from the start:
    * Dexsuite's curriculum starts in zero gravity with the object floating where it spawned, so "lifted"
      additionally requires a thumb-opposed grasp contact, and stages 2-3 and the goal bonus are only paid while
      that contact holds. Without it a nut knocked towards the goal in zero gravity earns the reward.
    * The lift is measured from the table surface, not from the spawn height (the spawn is in the air here).
    * One goal per episode (Dexsuite's ADR curriculum scores the final goal error), not a chain of goals.
    * The goal is a position: the nut is symmetric under 60 deg turns and flips, so the orientation goal is left
      for the threading stage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

from franka_sharpa import nut_geometry
from franka_sharpa.robot import FINGERTIP_BODIES
from franka_sharpa.tasks.ball_lift.mdp import *  # noqa: F401, F403
from franka_sharpa.tasks.ball_lift.mdp import good_grasp_contact

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

CONTACT_THRESHOLD = 1.0  # N, per fingertip, as in the ball task


def nut_scale(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Size factor of each env's nut from the spawn-time scale randomisation, shape ``(num_envs,)``."""
    if not hasattr(env, "_nut_scale"):
        nut: RigidObject = env.scene[object_cfg.name]
        prim_paths = [nut.cfg.prim_path.replace("env_.*", f"env_{i}") for i in range(env.num_envs)]
        scales = []
        for path in prim_paths:
            attr = env.scene.stage.GetPrimAtPath(path).GetAttribute("xformOp:scale")
            scales.append(attr.Get()[0] if attr and attr.Get() is not None else 1.0)
        env._nut_scale = torch.tensor(scales, device=env.device, dtype=torch.float32)
    return env._nut_scale


def table_top_z(env: ManagerBasedRLEnv) -> torch.Tensor:
    table: RigidObject = env.scene["table"]
    return table.data.root_pos_w[:, 2] + 0.5 * table.cfg.spawn.size[2]


def goal_pos_w(env: ManagerBasedRLEnv, command_name: str = "object_pose") -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    pos, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    return pos


##
# SimToolReal progress state, shared by the reward terms below
##


class _Progress:
    """Best-so-far trackers of SimToolReal, updated once per environment step whichever term asks first."""

    def __init__(self, env: ManagerBasedRLEnv):
        robot: Articulation = env.scene["robot"]
        self.fingertip_ids = robot.find_bodies(FINGERTIP_BODIES, preserve_order=True)[0]
        n = env.num_envs
        self.step = -1
        self.lifted = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.closest_fingertip = torch.zeros(n, len(self.fingertip_ids), device=env.device)
        self.closest_goal = torch.zeros(n, device=env.device)
        # per-step outputs
        self.fingertip_progress = torch.zeros(n, device=env.device)
        self.lift_height = torch.zeros(n, device=env.device)
        self.just_lifted = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.goal_progress = torch.zeros(n, device=env.device)
        self.goal_dist = torch.zeros(n, device=env.device)
        self.grasped = torch.zeros(n, dtype=torch.bool, device=env.device)

    def update(self, env: ManagerBasedRLEnv, lift_threshold: float):
        if self.step == env.common_step_counter:
            return
        self.step = env.common_step_counter
        robot: Articulation = env.scene["robot"]
        nut: RigidObject = env.scene["object"]
        nut_pos = nut.data.root_pos_w

        fingertip_dist = torch.norm(robot.data.body_pos_w[:, self.fingertip_ids] - nut_pos[:, None, :], dim=-1)
        goal_dist = torch.norm(nut_pos - goal_pos_w(env), dim=-1)
        self.grasped = good_grasp_contact(env, CONTACT_THRESHOLD)

        # first step of an episode (the step counter is already incremented when rewards are computed):
        # the trackers start at the current values, SimToolReal's "-1 sentinel"
        fresh = env.episode_length_buf <= 1
        self.lifted[fresh] = False
        self.closest_fingertip[fresh] = fingertip_dist[fresh]
        self.closest_goal[fresh] = goal_dist[fresh]

        # 1. fingertip approach, summed over the five fingertips, paid until the lift
        self.fingertip_progress = (self.closest_fingertip - fingertip_dist).clamp(min=0.0).sum(-1) * (~self.lifted)
        self.closest_fingertip = torch.minimum(self.closest_fingertip, fingertip_dist)

        # 2. lift: height of the nut's lowest possible point above the table, latched once past the threshold
        self.lift_height = nut_pos[:, 2] - table_top_z(env) - nut_geometry.REST_HEIGHT_FLAT * nut_scale(env)
        lifted_now = self.lifted | ((self.lift_height > lift_threshold) & self.grasped)
        self.just_lifted = lifted_now & ~self.lifted
        self.lifted = lifted_now

        # 3. goal approach, after the lift and while held
        self.goal_progress = (self.closest_goal - goal_dist).clamp(min=0.0) * self.lifted * self.grasped
        self.closest_goal = torch.minimum(self.closest_goal, goal_dist)
        self.goal_dist = goal_dist


def _progress(env: ManagerBasedRLEnv, lift_threshold: float = 0.10) -> _Progress:
    if not hasattr(env, "_nut_progress"):
        env._nut_progress = _Progress(env)
    env._nut_progress.update(env, lift_threshold)
    return env._nut_progress


def fingertip_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Metres by which the fingertips beat their closest approach to the nut this step (SimToolReal stage 1)."""
    return _progress(env).fingertip_progress


def lift_progress(env: ManagerBasedRLEnv, max_height: float = 0.15) -> torch.Tensor:
    """Height above the table while not yet lifted, held by the grasp (SimToolReal stage 2, per step)."""
    p = _progress(env)
    return p.lift_height.clamp(0.0, max_height) * (~p.lifted) * p.grasped


def lift_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1 on the step the held nut first rises past the lift threshold."""
    return _progress(env).just_lifted.float()


def goal_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Metres by which the held nut beats its closest approach to the goal this step (SimToolReal stage 3)."""
    return _progress(env).goal_progress


def goal_bonus(
    env: ManagerBasedRLEnv,
    pos_std: float,
    command_name: str = "object_pose",
    asset_cfg: SceneEntityCfg | None = None,
    align_asset_cfg: SceneEntityCfg | None = None,
    rot_std: float | None = None,
) -> torch.Tensor:
    """1 per step while the held nut is within ``pos_std / 2`` of the goal.

    The tolerance is the one Dexsuite's ADR curriculum uses to promote an env (it reads ``pos_std`` from the
    ``success`` reward term), so the bonus and the curriculum agree on what counts as reaching the goal. The
    unused arguments are there because the Dexsuite base config writes them into the ``success`` term.
    """
    p = _progress(env)
    return ((p.goal_dist < 0.5 * pos_std) & p.grasped).float()


def joint_vel_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """SimToolReal's "action penalty", which is in fact an L1 penalty on joint velocities."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_vel[:, asset_cfg.joint_ids].abs().sum(-1)


##
# Dense terms of the ball v1 design, for the comparison run
##


def hand_to_nut_surface(
    env: ManagerBasedRLEnv,
    std_coarse: float,
    std_fine: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The ball task's reach reward with the nut's inscribed radius in place of the ball radius."""
    robot: Articulation = env.scene[asset_cfg.name]
    nut: RigidObject = env.scene[object_cfg.name]
    centre_dist = torch.norm(robot.data.body_pos_w[:, asset_cfg.body_ids] - nut.data.root_pos_w[:, None, :], dim=-1)
    gap = (centre_dist.max(dim=-1).values - 0.5 * nut_geometry.ACROSS_FLATS * nut_scale(env, object_cfg)).clamp(min=0.0)
    return 0.5 * (1 - torch.tanh(gap / std_coarse)) + 0.5 * (1 - torch.tanh(gap / std_fine))


##
# Stage 1 of the Play2Perfect-style plan: 6D goals (position + orientation)
##


def goal_quat_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Commanded nut orientation (w, x, y, z) in the robot base frame."""
    return env.command_manager.get_command(command_name)[:, 3:7]


def orientation_command_error_tanh_grasped(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Dexsuite's orientation tracking reward with the Sharpa grasp check in place of its Allegro-only contact term."""
    from isaaclab.utils.math import quat_error_magnitude, quat_mul

    robot: Articulation = env.scene[asset_cfg.name]
    nut: RigidObject = env.scene[align_asset_cfg.name]
    des_quat_w = quat_mul(robot.data.root_quat_w, env.command_manager.get_command(command_name)[:, 3:7])
    distance = quat_error_magnitude(nut.data.root_quat_w, des_quat_w)
    return (1 - torch.tanh(distance / std)) * good_grasp_contact(env, CONTACT_THRESHOLD).float()


def _nut_symmetries(device) -> torch.Tensor:
    """The 12 rotations that map the hex nut onto itself: 6 turns of 60 deg about its axis, each with or without a flip."""
    from isaaclab.utils.math import quat_from_euler_xyz

    yaw = torch.arange(6, device=device, dtype=torch.float32) * (torch.pi / 3)
    zeros = torch.zeros_like(yaw)
    upright = quat_from_euler_xyz(zeros, zeros, yaw)
    flipped = quat_from_euler_xyz(torch.full_like(yaw, torch.pi), zeros, yaw)
    return torch.cat([upright, flipped])


def nut_rotation_error(nut_quat_w: torch.Tensor, goal_quat_w: torch.Tensor) -> torch.Tensor:
    """Smallest rotation [rad] between the nut and the goal over the nut's 12 symmetries.

    A nut upside down or turned by 60 deg looks and threads the same; without this the goal demands one particular
    face up, which takes a 180 deg in-hand flip half of the time. The first 6D play run ignored the orientation goal
    (median error about 170 deg) for that reason.
    """
    from isaaclab.utils.math import quat_error_magnitude, quat_mul

    sym = _nut_symmetries(nut_quat_w.device)
    n, k = nut_quat_w.shape[0], sym.shape[0]
    candidates = quat_mul(nut_quat_w.repeat_interleave(k, 0), sym.repeat(n, 1))
    errors = quat_error_magnitude(candidates, goal_quat_w.repeat_interleave(k, 0)).view(n, k)
    return errors.min(dim=1).values


def _goal_pose_w(env: ManagerBasedRLEnv, command_name: str, robot: Articulation):
    from isaaclab.utils.math import quat_mul

    command = env.command_manager.get_command(command_name)
    pos, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    return pos, quat_mul(robot.data.root_quat_w, command[:, 3:7])


def orientation_tracking_symmetric(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Orientation tracking with the symmetric nut error, gated by the Sharpa grasp check."""
    robot: Articulation = env.scene[asset_cfg.name]
    nut: RigidObject = env.scene[align_asset_cfg.name]
    _, goal_quat = _goal_pose_w(env, command_name, robot)
    error = nut_rotation_error(nut.data.root_quat_w, goal_quat)
    return (1 - torch.tanh(error / std)) * good_grasp_contact(env, CONTACT_THRESHOLD).float()


def success_symmetric(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    pos_std: float,
    rot_std: float | None = None,
) -> torch.Tensor:
    """Dexsuite's success reward (position kernel x orientation kernel), symmetric nut error, held by the grasp."""
    robot: Articulation = env.scene[asset_cfg.name]
    nut: RigidObject = env.scene[align_asset_cfg.name]
    goal_pos, goal_quat = _goal_pose_w(env, command_name, robot)
    pos = 1 - torch.tanh(torch.norm(nut.data.root_pos_w - goal_pos, dim=1) / pos_std)
    rot = 1 - torch.tanh(nut_rotation_error(nut.data.root_quat_w, goal_quat) / rot_std) if rot_std else pos
    return pos * rot * good_grasp_contact(env, CONTACT_THRESHOLD).float()
