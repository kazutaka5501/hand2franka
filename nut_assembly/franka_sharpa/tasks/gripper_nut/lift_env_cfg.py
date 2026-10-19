"""Franka Panda with its parallel gripper picking up the hex nut and bringing it to a commanded position.

Isaac Lab's manager-based Franka cube lift (``manipulation/lift/config/franka/joint_pos_env_cfg.py``: joint-position
arm actions, binary gripper, same rewards, terminations, PPO settings) with these changes:

    * object: the smooth-bore nut ``assets/nut/nut.usda`` (12 convex pieces, 51 mm across flats, 33 mm thick,
      35 g) instead of the DexCube. The gripper opens to 80 mm, so it fits across the flats (51 mm), the corners
      (59 mm) and the thickness (33 mm).
    * start pose: as in ``nut_lift`` the nut is dropped with a random orientation, so it lands flat (axis
      vertical, centre 16.5 mm above the table) or on its side (centre 25.5 mm), in a random spot.
    * policy observation: the nut orientation in the robot frame is added (the cube task only sees the position);
      the grasp depends on whether the nut lies flat or on its side and on its yaw.
    * goal region: includes the space just above the bolt of the screwing stage, see below.

Frames. Here the robot base is at z = 0 and the table top (SeattleLabTable) at z = -0.003; in ``nut_lift`` (Dexsuite
layout) the robot base is at z = 0 and the table top at z = 0.255, with the bolt standing on the table at
``screw.BOLT_POS = (0.55, 0, 0.255)``. Keeping the bolt's place on the table, it stands at (0.55, 0, -0.003) here
and its tip is at z = 0.082. The cube task samples goals in x 0.4..0.6, y -0.25..0.25, z 0.25..0.5 (robot base frame),
which never comes near the tip; the z range is lowered to start 25 mm above the tip (nut centre at z = 0.107, its
bottom face 8.5 mm above the tip when the nut is held flat).
"""

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg

from franka_sharpa import nut_geometry, thread_geometry

from . import mdp

NUT_USD = Path(__file__).resolve().parents[3] / "assets/nut/nut.usda"

# measured: the nut settles at z = 0.0135 flat / 0.0225 on its side, 3 mm below the nominal 0.0165 / 0.0255
TABLE_TOP_Z = -0.003
# screw.BOLT_POS moved from the Dexsuite table (top at z = 0.255) onto this table
BOLT_POS = (0.55, 0.0, TABLE_TOP_Z)
BOLT_TIP_Z = BOLT_POS[2] + thread_geometry.BOLT_LENGTH
GOAL_RANGE_X = (0.4, 0.6)
GOAL_RANGE_Y = (-0.25, 0.25)
GOAL_RANGE_Z = (BOLT_TIP_Z + 0.025, 0.5)

# nut spawn: centre 5..15 cm above the table, then dropped (circumscribed radius 34 mm, so never inside the table)
NUT_SPAWN_POS = (0.5, 0.0, TABLE_TOP_Z + 0.05)
NUT_SPAWN_RANGE = {
    "x": (-0.1, 0.1),
    "y": (-0.25, 0.25),
    "z": (0.0, 0.1),
    "roll": (-math.pi, math.pi),
    "pitch": (-math.pi, math.pi),
    "yaw": (-math.pi, math.pi),
}


@configclass
class FrankaGripperNutLiftEnvCfg(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=NUT_SPAWN_POS, rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(NUT_USD),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                # thin walls: keep the contact shell tight (as in nut_lift)
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=nut_geometry.MASS),
            ),
        )
        self.events.reset_object_position.params["pose_range"] = dict(NUT_SPAWN_RANGE)

        self.observations.policy.object_orientation = ObsTerm(func=mdp.object_quat_in_robot_root_frame)

        ranges = self.commands.object_pose.ranges
        ranges.pos_x, ranges.pos_y, ranges.pos_z = GOAL_RANGE_X, GOAL_RANGE_Y, GOAL_RANGE_Z


@configclass
class FrankaGripperNutLiftEnvCfg_PLAY(FrankaGripperNutLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

