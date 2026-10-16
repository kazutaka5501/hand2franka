"""MDP terms for the Franka + Sharpa Wave ball task.

Everything comes from Isaac Lab's Dexsuite task except the contact terms, which are hard-coded there
for the four Allegro fingers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply_inverse

from isaaclab_tasks.manager_based.manipulation.dexsuite.mdp import *  # noqa: F401, F403

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# The fingertip is two bodies in the Sharpa model: the hard distal phalanx and the soft pad glued onto it.
FINGERTIP_PARTS = ("DP", "elastomer")


def contact_sensor_name(finger: str, part: str) -> str:
    return f"{finger}_{part}_object_s"


def fingertip_contact_force_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Force the object exerts on each fingertip (distal phalanx + pad) in world frame, shape ``(num_envs, 5, 3)``."""
    forces = []
    for finger in FINGERS:
        parts = [env.scene.sensors[contact_sensor_name(finger, part)] for part in FINGERTIP_PARTS]
        forces.append(sum(sensor.data.force_matrix_w.view(env.num_envs, 3) for sensor in parts))
    return torch.stack(forces, dim=1)


def fingertip_contact_force_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Fingertip/object contact forces in the robot base frame, shape ``(num_envs, 5, 3)``."""
    robot: Articulation = env.scene[asset_cfg.name]
    force_w = fingertip_contact_force_w(env)
    return quat_apply_inverse(robot.data.root_link_quat_w.unsqueeze(1).repeat(1, force_w.shape[1], 1), force_w)


def finger_contact_forces(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Magnitude of the fingertip/object contact force per finger, shape ``(num_envs, 5)``."""
    return torch.norm(fingertip_contact_force_w(env), dim=-1)


def good_grasp_contact(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """True when the thumb and at least one opposing finger press on the object."""
    in_contact = finger_contact_forces(env) > threshold
    return in_contact[:, 0] & in_contact[:, 1:].any(dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward tracking of the commanded object position, gated by a thumb-opposed grasp contact."""
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, command[:, :3])
    distance = torch.norm(object.data.root_pos_w - des_pos_w, dim=1)
    return (1 - torch.tanh(distance / std)) * good_grasp_contact(env, contact_threshold).float()


##
# Ball-specific terms (v1 design)
##


def ball_radius(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Radius of each env's ball, shape ``(num_envs,)``. The spawn-time scale randomisation fixes it for the run."""
    if not hasattr(env, "_ball_radius"):
        ball: RigidObject = env.scene[object_cfg.name]
        prim_paths = [ball.cfg.prim_path.replace("env_.*", f"env_{i}") for i in range(env.num_envs)]
        scales = [env.scene.stage.GetPrimAtPath(path).GetAttribute("xformOp:scale").Get()[0] for path in prim_paths]
        env._ball_radius = ball.cfg.spawn.radius * torch.tensor(scales, device=env.device)
    return env._ball_radius


def ball_radius_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    return ball_radius(env).unsqueeze(1)


def goal_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Commanded ball position in the robot base frame, without the unused orientation part of the command."""
    return env.command_manager.get_command(command_name)[:, :3]


def hand_to_ball_surface(
    env: ManagerBasedRLEnv,
    std_coarse: float,
    std_fine: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward wrapping the hand around the ball.

    Uses the largest gap between the ball surface and any of the given hand bodies (palm and fingertips), so the
    value does not depend on the ball size. The coarse kernel pulls the hand in from far away, the fine one keeps
    a gradient over the last centimetres where the coarse one is flat.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[object_cfg.name]
    centre_dist = torch.norm(asset.data.body_pos_w[:, asset_cfg.body_ids] - ball.data.root_pos_w[:, None, :], dim=-1)
    gap = (centre_dist.max(dim=-1).values - ball_radius(env, object_cfg)).clamp(min=0.0)
    return 0.5 * (1 - torch.tanh(gap / std_coarse)) + 0.5 * (1 - torch.tanh(gap / std_fine))


def grasped_success_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    pos_std: float,
    rot_std: float | None = None,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Dexsuite's success reward, paid only while the ball is held by the thumb and an opposing finger.

    Without the gate a ball knocked towards the goal in the zero-gravity curriculum stage earns the largest
    reward of the task without ever being grasped.
    """
    reward = success_reward(env, command_name, asset_cfg, align_asset_cfg, pos_std, rot_std)  # noqa: F405
    return reward * good_grasp_contact(env, contact_threshold).float()


##
# Tactile terms (v2 design, after RoTO: proprioception / tactile / ground truth kept in separate groups)
##


def pad_contact_force(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Magnitude of the net contact force on each fingertip pad, shape ``(num_envs, 5)``.

    Unlike the reward terms this is the net force from anything the pad touches (ball, table, the hand itself),
    which is what a tactile sensor measures: it cannot tell what it is touching.
    """
    forces = [env.scene.sensors[contact_sensor_name(finger, "elastomer")].data.net_forces_w for finger in FINGERS]
    return torch.norm(torch.cat(forces, dim=1), dim=-1)


def pad_contact_binary(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """One bit per fingertip pad: is it touching something. RoTO's "sparse binary contacts" on Sharpa's five pads."""
    return (pad_contact_force(env) > threshold).float()


class ball_first_look(ManagerTermBase):
    """Ball centre and radius as seen once at the start of the episode, held constant afterwards.

    Models a camera that locates the ball before the hand moves in and occludes it. The position error is drawn
    once per episode, so it acts as a constant bias of the estimate rather than as per-step jitter.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._first_look = torch.zeros(env.num_envs, 4, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        pos_noise: float = 0.0,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        # the observation is computed right after a reset with the step counter at zero: that is the one look
        fresh = env.episode_length_buf == 0
        if fresh.any():
            pos_b = object_pos_b(env, robot_cfg, object_cfg)[fresh]  # noqa: F405
            pos_b = pos_b + pos_noise * (2.0 * torch.rand_like(pos_b) - 1.0)
            self._first_look[fresh] = torch.cat([pos_b, ball_radius(env, object_cfg)[fresh].unsqueeze(1)], dim=1)
        return self._first_look
