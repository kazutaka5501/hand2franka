"""Franka Panda arm + Sharpa Wave hand grasping and lifting a ball.

The task is Isaac Lab's Dexsuite Lift (reach, grasp, bring the object to a commanded position) with the
Kuka-Allegro robot replaced by Franka-Sharpa and the object set replaced by a single ball. Dexsuite places its
workspace at -x in front of the Kuka; the Franka faces +x, so the layout is mirrored here.
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.dexsuite import dexsuite_env_cfg as dexsuite

from franka_sharpa.robot import ACTION_SCALE, FINGERTIP_BODIES, FRANKA_SHARPA_CFG, PALM_BODY

from . import mdp

# Tennis-ball sized at scale 1.0; the pre-startup scale randomization turns it into 5-10 cm balls.
BALL_RADIUS = 0.035
WORKSPACE_X = 0.55


@configclass
class FrankaSharpaRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=ACTION_SCALE)


@configclass
class FrankaSharpaRewardsCfg(dexsuite.RewardsCfg):
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.2,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    # bool term: thumb and at least one opposing fingertip are pressing on the ball
    good_finger_contact = RewTerm(func=mdp.good_grasp_contact, weight=0.5, params={"threshold": 1.0})


@configclass
class FrankaSharpaBallMixinCfg:
    rewards: FrankaSharpaRewardsCfg = FrankaSharpaRewardsCfg()
    actions: FrankaSharpaRelJointPosActionCfg = FrankaSharpaRelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()

        # -- robot
        self.scene.robot = FRANKA_SHARPA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        for finger in mdp.FINGERS:
            for part in mdp.FINGERTIP_PARTS:
                setattr(
                    self.scene,
                    mdp.contact_sensor_name(finger, part),
                    ContactSensorCfg(
                        prim_path="{ENV_REGEX_NS}/Robot/" + f"right_{finger}_{part}",
                        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                    ),
                )
        # contact force on a fingertip stays well under 20 N
        self.observations.proprio.contact = ObsTerm(func=mdp.fingertip_contact_force_b, clip=(-20.0, 20.0))
        hand_bodies = [PALM_BODY, *FINGERTIP_BODIES]
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = hand_bodies
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=hand_bodies)
        # Dexsuite offsets every joint by +-0.5 rad on reset. On this arm that puts the hand inside the table in
        # about a third of the resets; the resulting kick-out trips the abnormal-velocity termination and each of
        # those two-step episodes counts as a failure for the curriculum. Keep +-0.5 for the hand only.
        self.events.reset_robot_joints.params["asset_cfg"] = SceneEntityCfg("robot", joint_names="right_.*")
        self.events.reset_robot_arm_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names="panda_joint[1-6]"),
                "position_range": [-0.2, 0.2],
                "velocity_range": [0.0, 0.0],
            },
        )
        self.events.reset_robot_wrist_joint.params["asset_cfg"] = SceneEntityCfg("robot", joint_names="panda_joint7")
        self.events.reset_robot_wrist_joint.params["position_range"] = [-1.5, 1.5]
        # Dexsuite scales joint friction by up to 5x, which is fine for the Allegro's 0.01 Nm but would let the
        # calibrated Sharpa friction (0.1 Nm on a 1.9 Nm motor) eat most of a unit action.
        self.events.joint_friction.params["friction_distribution_params"] = [0.5, 1.5]

        # -- ball
        self.scene.object.spawn = sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.75, 0.1)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                # stands in for rolling resistance, otherwise a nudged ball rolls off the table forever
                angular_damping=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        )

        # -- workspace in front of the Franka (+x)
        self.scene.object.init_state.pos = (WORKSPACE_X, 0.0, 0.35)
        self.scene.table.init_state.pos = (WORKSPACE_X, 0.0, 0.235)
        self.commands.object_pose.ranges.pos_x = (WORKSPACE_X - 0.15, WORKSPACE_X + 0.1)
        self.commands.object_pose.ranges.pos_y = (-0.2, 0.2)
        self.commands.object_pose.ranges.pos_z = (0.45, 0.75)
        self.terminations.object_out_of_bound.params["in_bound_range"]["x"] = (-0.5, 1.5)
        self.viewer.eye = (2.25, 0.0, 0.75)


@configclass
class FrankaSharpaBallLiftEnvCfg(FrankaSharpaBallMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class FrankaSharpaBallLiftEnvCfg_PLAY(FrankaSharpaBallMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass
