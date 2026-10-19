"""MDP terms for the gripper nut tasks: Isaac Lab's lift terms plus the nut orientation observation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_mul, quat_inv, quat_unique

from isaaclab_tasks.manager_based.manipulation.lift.mdp import *  # noqa: F401, F403

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_quat_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Orientation (w, x, y, z, w >= 0) of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    return quat_unique(quat_mul(quat_inv(robot.data.root_quat_w), obj.data.root_quat_w))
