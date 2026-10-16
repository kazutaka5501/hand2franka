"""Stage 2 of the assembly plan: screw the nut onto the bolt, after Play2Perfect's screwing task.

The nut starts resting on the tip of a bolt fixed to the table (threaded meshes with SDF collision, see
``franka_sharpa.thread_geometry``). The goal is a sequence of waypoints, each a quarter turn clockwise (seen from above,
right-hand thread) and a quarter pitch lower than the last (Play2Perfect: ``_one_leg_screw_insert_waypoints``, 90 deg
steps). The next waypoint is issued once the nut has gone down the thread far enough, so the command the policy sees
has the same layout as in stage 1 (position + orientation) and the stage-1 policy is the starting point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import quat_from_euler_xyz, subtract_frame_transforms, yaw_quat

from franka_sharpa import nut_geometry, thread_geometry
from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp.commands.pose_commands import ObjectUniformPoseCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# bolt standing on the table in front of the robot (env frame); the table top is at z = 0.255
BOLT_POS = (0.55, 0.0, 0.255)
NUM_WAYPOINTS = 8  # two turns
QUARTER = 0.25 * thread_geometry.PITCH
SETTLE_STEPS = 10
LEAD_IN_TURNS = 1.0  # rotation credited before the nut has to go down the thread  # steps after a reset before the nut's resting height on the tip is taken as the start


def bolt_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene.env_origins + torch.tensor(BOLT_POS, device=env.device)


class ScrewWaypointCommand(ObjectUniformPoseCommand):
    """Quarter-turn screwing waypoints in the robot base frame, ``(x, y, z, qw, qx, qy, qz)`` like Dexsuite's command."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n = self.num_envs
        self.waypoint = torch.zeros(n, dtype=torch.long, device=self.device)
        self.start_z = torch.full((n,), float("nan"), device=self.device)
        self.start_yaw = torch.zeros(n, device=self.device)
        self.advanced = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.turns = torch.zeros(n, device=self.device)  # clockwise turns since the nut settled on the tip
        self._prev_yaw = torch.zeros(n, device=self.device)
        self._env = env

    def descent(self) -> torch.Tensor:
        """How far the nut has gone down the thread since it settled on the tip [m], 0 before it settled."""
        z = self.object.data.root_pos_w[:, 2]
        return torch.nan_to_num(self.start_z - z, nan=0.0)

    def screw_turns(self) -> torch.Tensor:
        """Turns the nut has made down the thread: its descent in pitches, but no more than it actually turned.

        The first stage-2 policy pushed the nut 17 mm down the bolt with no rotation at all (median 0.02 turns): the
        Franka can force the nut through the SDF thread contact. Counting only the descent that the rotation explains
        removes that shortcut.
        """
        # up to one turn counts without descent: in the physics test the first ~0.7 turn only catches the thread
        # (no descent), and without this a policy that starts turning gets no signal at all
        return torch.minimum(self.descent() / thread_geometry.PITCH + LEAD_IN_TURNS, self.turns).clamp(min=0.0)

    def _resample_command(self, env_ids):
        self.waypoint[env_ids] = 0
        self.start_z[env_ids] = float("nan")
        self.turns[env_ids] = 0.0
        self._write(env_ids)

    def _update_command(self):
        nut = self.object.data
        yaw_q = yaw_quat(nut.root_quat_w)
        yaw_now = 2 * torch.atan2(yaw_q[:, 3], yaw_q[:, 0])
        dyaw = torch.remainder(yaw_now - self._prev_yaw + torch.pi, 2 * torch.pi) - torch.pi
        self._prev_yaw = yaw_now
        settled = ~torch.isnan(self.start_z)
        self.turns = torch.where(settled, self.turns - dyaw / (2 * torch.pi), self.turns)  # clockwise = negative yaw
        settling = self._env.episode_length_buf == SETTLE_STEPS
        if settling.any():
            self.start_z[settling] = nut.root_pos_w[settling, 2]
            self.start_yaw[settling] = yaw_now[settling]
            self.turns[settling] = 0.0
        # next waypoint once the nut is within 1/3 of a quarter turn of the current one's, in depth and in rotation
        reached = self.screw_turns() >= (self.waypoint + 1).float() * 0.25 - 0.25 / 3
        self.advanced = reached & (self.waypoint < NUM_WAYPOINTS) & ~torch.isnan(self.start_z)
        self.waypoint += self.advanced.long()
        self._write(torch.arange(self.num_envs, device=self.device))

    def _write(self, env_ids):
        k = self.waypoint[env_ids].float()
        bolt = bolt_pos_w(self._env)[env_ids]
        nominal_z = bolt[:, 2] + thread_geometry.BOLT_LENGTH + 0.5 * nut_geometry.THICKNESS - QUARTER
        z0 = torch.where(torch.isnan(self.start_z[env_ids]), nominal_z, self.start_z[env_ids])
        target_pos = torch.stack([bolt[:, 0], bolt[:, 1], z0 - (k + 1) * QUARTER], dim=-1)
        # clockwise seen from above: negative yaw
        yaw = self.start_yaw[env_ids] - (k + 1) * (torch.pi / 2)
        zeros = torch.zeros_like(yaw)
        target_quat = quat_from_euler_xyz(zeros, zeros, yaw)
        robot: Articulation = self.robot
        pos_b, quat_b = subtract_frame_transforms(
            robot.data.root_pos_w[env_ids], robot.data.root_quat_w[env_ids], target_pos, target_quat
        )
        self.pose_command_b[env_ids, :3] = pos_b
        self.pose_command_b[env_ids, 3:] = quat_b


def _command(env: ManagerBasedRLEnv) -> ScrewWaypointCommand:
    return env.command_manager.get_term("object_pose")


def screw_depth(env: ManagerBasedRLEnv, max_turns: float = 2.5, max_offset: float = 0.004) -> torch.Tensor:
    """Turns of thread engaged since the nut settled on the tip (descent / pitch), per step.

    Only while the nut is centred on the bolt axis: a nut knocked off the bolt also goes down, and the first stage-2
    run collected this reward that way before the off-bolt termination fired.
    """
    nut: RigidObject = env.scene["object"]
    on_axis = torch.norm(nut.data.root_pos_w[:, :2] - bolt_pos_w(env)[:, :2], dim=-1) < max_offset
    turns = _command(env).screw_turns().clamp(0.0, max_turns)
    return turns * on_axis.float()


def forced_descent(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Pitches the nut went down beyond what its rotation explains (pushed through the thread), per step."""
    command = _command(env)
    excess = command.descent() / thread_geometry.PITCH - command.turns.clamp(min=0.0)
    return (excess - 0.25).clamp(min=0.0)


def waypoint_reached(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1 on the step a screwing waypoint is reached."""
    return _command(env).advanced.float()


def nut_off_bolt(env: ManagerBasedRLEnv, max_offset: float = 0.03) -> torch.Tensor:
    """The nut left the bolt: its centre is more than ``max_offset`` from the bolt axis, or lower than the tip on the table."""
    nut: RigidObject = env.scene["object"]
    bolt = bolt_pos_w(env)
    off_axis = torch.norm(nut.data.root_pos_w[:, :2] - bolt[:, :2], dim=-1) > max_offset
    return off_axis


def reset_nut_on_bolt(env: ManagerBasedRLEnv, env_ids: torch.Tensor, xy_noise: float = 0.001):
    """Place the nut upright 2 mm above the bolt tip, centred within ``xy_noise``, turned by a random yaw."""
    nut: RigidObject = env.scene["object"]
    n = len(env_ids)
    pos = bolt_pos_w(env)[env_ids].clone()
    pos[:, :2] += (2 * torch.rand(n, 2, device=env.device) - 1) * xy_noise
    pos[:, 2] += thread_geometry.BOLT_LENGTH + 0.5 * nut_geometry.THICKNESS + 0.002
    yaw = (2 * torch.rand(n, device=env.device) - 1) * torch.pi
    zeros = torch.zeros_like(yaw)
    quat = quat_from_euler_xyz(zeros, zeros, yaw)
    nut.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    nut.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)


def nut_spin(env: ManagerBasedRLEnv, max_rate: float = 2.0, max_offset: float = 0.004) -> torch.Tensor:
    """Clockwise spin rate of the nut about the bolt axis [turns/s], while it sits on the bolt.

    Not gated by the grasp: however the hand makes the nut turn, turning is what screws it in. Counter-clockwise
    spin counts negative.
    """
    nut: RigidObject = env.scene["object"]
    on_axis = torch.norm(nut.data.root_pos_w[:, :2] - bolt_pos_w(env)[:, :2], dim=-1) < max_offset
    rate = -nut.data.root_ang_vel_w[:, 2] / (2 * torch.pi)  # clockwise seen from above
    return rate.clamp(-max_rate, max_rate) * on_axis.float()
