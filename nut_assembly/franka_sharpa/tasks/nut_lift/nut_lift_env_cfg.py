"""Franka + Sharpa Wave picking up the 3D-printed hex nut and bringing it to a commanded position.

Built on the ball task (``ball_lift``): same robot, table, Dexsuite curriculum (zero gravity ramping to full
gravity, observation noise ramping up), contact sensors and action space. What changes:

    * object: the measured nut (``franka_sharpa.nut_geometry``), +-5 % size and 0.6-1.6x mass randomisation
      instead of Dexsuite's 0.75-1.5x size, since the target is one known part
    * start pose: as in Dexsuite and SimToolReal the nut spawns with a random orientation up to 0.4 m above the
      table, so under gravity it lands flat (axis vertical) or on its side (axis horizontal) in a random spot
    * observations: nut position and orientation, goal position, last action (+ the unchanged proprio group)
    * rewards: two designs trained side by side
        - v0 ``NutLiftEnvCfg``: SimToolReal's progress rewards, see ``mdp``
        - dense ``NutLiftDenseEnvCfg``: the ball v1 rewards (tanh distance kernels, contact-gated success)
"""

from pathlib import Path

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.manipulation.dexsuite import dexsuite_env_cfg as dexsuite
from isaaclab_tasks.manager_based.manipulation.dexsuite.adr_curriculum import CurriculumCfg

from franka_sharpa import nut_geometry
from franka_sharpa.tasks.ball_lift.ball_lift_env_cfg import FrankaSharpaBallMixinCfg, FrankaSharpaRewardsCfg

from . import mdp

NUT_USD = Path(__file__).resolve().parents[3] / "assets/nut/nut.usda"


@configclass
class NutObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Task state: where the nut is and how it is turned, where it has to go, and the previous action."""

        object_pos_b = ObsTerm(func=mdp.object_pos_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        object_quat_b = ObsTerm(func=mdp.object_quat_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        goal_pos_b = ObsTerm(func=mdp.goal_pos_b, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()
    proprio: dexsuite.ObservationsCfg.ProprioObsCfg = dexsuite.ObservationsCfg.ProprioObsCfg()


@configclass
class NutCurriculumCfg(CurriculumCfg):
    """Dexsuite's curriculum; the object position noise goes on the nut position instead of the point cloud."""

    object_obs_unoise_min_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.object_pos_b.noise.n_min",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": -0.01, "difficulty_term_str": "adr"},
        },
    )
    object_obs_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.object_pos_b.noise.n_max",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.01, "difficulty_term_str": "adr"},
        },
    )


# SimToolReal's reward scales (reward_utils.py / SimToolReal.yaml) times 0.01: their values are sized for rl_games
# with value normalisation, rsl_rl's PPO here sees raw rewards
_SCALE = 0.01


@configclass
class NutSimToolRealRewardsCfg(dexsuite.RewardsCfg):
    fingertip_progress = RewTerm(func=mdp.fingertip_progress, weight=50.0 * _SCALE)
    lift_progress = RewTerm(func=mdp.lift_progress, weight=20.0 * _SCALE)
    lift_bonus = RewTerm(func=mdp.lift_bonus, weight=300.0 * _SCALE)
    goal_progress = RewTerm(func=mdp.goal_progress, weight=200.0 * _SCALE)
    # SimToolReal pays 1000 spread over the 10 steps needed for success; here per step at the goal
    success = RewTerm(func=mdp.goal_bonus, weight=100.0 * _SCALE, params={"pos_std": 0.08, "rot_std": 0.5})
    arm_joint_vel = RewTerm(
        func=mdp.joint_vel_l1, weight=-0.03 * _SCALE, params={"asset_cfg": SceneEntityCfg("robot", joint_names="panda_.*")}
    )
    hand_joint_vel = RewTerm(
        func=mdp.joint_vel_l1, weight=-0.003 * _SCALE, params={"asset_cfg": SceneEntityCfg("robot", joint_names="right_.*")}
    )


@configclass
class NutDenseRewardsCfg(FrankaSharpaRewardsCfg):
    """The ball v1 rewards with the nut's inscribed radius in the reach term."""

    fingers_to_object = RewTerm(func=mdp.hand_to_nut_surface, params={"std_coarse": 0.4, "std_fine": 0.05}, weight=1.0)
    success = RewTerm(
        func=mdp.grasped_success_reward,
        weight=10,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_std": 0.08,
            "rot_std": 0.5,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class NutMixinCfg(FrankaSharpaBallMixinCfg):
    observations: NutObservationsCfg = NutObservationsCfg()
    rewards: NutSimToolRealRewardsCfg = NutSimToolRealRewardsCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        adr_params = self.curriculum.adr.params
        self.curriculum = NutCurriculumCfg()
        self.curriculum.adr.params = adr_params

        # Dexsuite's dense terms are replaced by the SimToolReal ones; action_l2, action_rate_l2 and the
        # early-termination penalty stay. The base configs above wrote into these terms, so drop them only now.
        if isinstance(self.rewards, NutSimToolRealRewardsCfg):
            self.rewards.fingers_to_object = None
            self.rewards.position_tracking = None
            self.rewards.orientation_tracking = None

        self.scene.object.spawn = sim_utils.UsdFileCfg(
            usd_path=str(NUT_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            # thin walls: keep the contact shell tight
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=nut_geometry.MASS),
        )
        self.events.randomize_object_scale.params["scale_range"] = (0.95, 1.05)
        self.events.object_scale_mass.params["mass_distribution_params"] = [0.6, 1.6]


@configclass
class NutLiftEnvCfg(NutMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class NutLiftEnvCfg_PLAY(NutMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass


@configclass
class NutV1MixinCfg(NutMixinCfg):
    """SimToolReal's rewards alone: v0 also kept Dexsuite's action_l2 and action_rate_l2 penalties.

    In v0 those two penalties (about -0.8 per episode for a moving policy) outweighed the scaled-down progress
    rewards a random policy collects (about 0.001 per episode), and the policy learned to keep still. SimToolReal
    itself only penalises joint velocities.
    """

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.rewards.action_l2 = None
        self.rewards.action_rate_l2 = None


@configclass
class NutLiftV1EnvCfg(NutV1MixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class NutLiftV1EnvCfg_PLAY(NutV1MixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass


@configclass
class NutDenseMixinCfg(NutMixinCfg):
    rewards: NutDenseRewardsCfg = NutDenseRewardsCfg()


@configclass
class NutLiftDenseEnvCfg(NutDenseMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class NutLiftDenseEnvCfg_PLAY(NutDenseMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass


##
# Stage 1 ("play", after Play2Perfect): reach 6D goals, i.e. also turn the nut in the hand
##


@configclass
class NutPlayObservationsCfg(NutObservationsCfg):
    @configclass
    class PolicyCfg(NutObservationsCfg.PolicyCfg):
        # appended last, so the stage-0 policy's inputs keep their positions (see scripts/warmstart.py)
        goal_quat_b = ObsTerm(func=mdp.goal_quat_b, params={"command_name": "object_pose"})

    policy: PolicyCfg = PolicyCfg()


@configclass
class NutPlayRewardsCfg(NutDenseRewardsCfg):
    # both orientation terms measure the error up to the nut's symmetries (see mdp.nut_rotation_error)
    success = RewTerm(
        func=mdp.success_symmetric,
        weight=10,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_std": 0.08,
            "rot_std": 0.5,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )
    orientation_tracking = RewTerm(
        func=mdp.orientation_tracking_symmetric,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 1.5,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class NutPlayMixinCfg(NutMixinCfg):
    """Continues from the dense lift policy, which already grasps and carries the nut under full gravity.

    * goals: position and orientation (Dexsuite Reorient), any orientation including turns about the nut axis
    * a new goal every 2-3 s in an 8 s episode, so the policy has to keep turning the nut it holds, as in
      Play2Perfect's play stage (there: goals chained on success, +-90 deg apart)
    * curriculum pinned at its last level (full gravity and observation noise): the lift policy is past it, and
      Dexsuite's scheduler scores only the last goal of an episode
    """

    observations: NutPlayObservationsCfg = NutPlayObservationsCfg()
    rewards: NutPlayRewardsCfg = NutPlayRewardsCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.episode_length_s = 8.0
        self.commands.object_pose.resampling_time_range = (2.0, 3.0)
        # goals with the nut axis vertical and any turn about it: the set the assembly needs (carry it upright over
        # the bolt, then turn it). A first try with Dexsuite's full random orientations collapsed within 30
        # iterations (position tracking 0.9 -> 0.01, half the episodes ended by the abnormal-state termination).
        self.commands.object_pose.ranges.roll = (0.0, 0.0)
        self.commands.object_pose.ranges.pitch = (0.0, 0.0)
        self.commands.object_pose.ranges.yaw = (-3.14, 3.14)
        top = self.curriculum.adr.params["max_difficulty"]
        self.curriculum.adr.params["init_difficulty"] = top
        self.curriculum.adr.params["min_difficulty"] = top


@configclass
class NutPlayEnvCfg(NutPlayMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class NutPlayEnvCfg_PLAY(NutPlayMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY):
    pass


##
# Stage 2 ("perfect", after Play2Perfect): screw the nut onto the bolt, continuing from the stage-1 policy
##

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from . import screw

THREADED_NUT_USD = NUT_USD.with_name("nut_threaded.usda")
BOLT_USD = NUT_USD.with_name("bolt.usda")


@configclass
class NutScrewRewardsCfg(NutPlayRewardsCfg):
    # the waypoints are a few millimetres apart: a finer position kernel than in stage 1, and the plain (not
    # symmetric) orientation error, since a 60 deg turn of the nut is progress here, not an equivalent pose
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.02,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )
    orientation_tracking = RewTerm(
        func=mdp.orientation_command_error_tanh_grasped,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 1.0,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )
    # physical progress along the thread, not gated by the grasp: however the nut is turned, going down is screwing
    screw_depth = RewTerm(func=screw.screw_depth, weight=5.0)
    forced_descent = RewTerm(func=screw.forced_descent, weight=-1.0)
    # turning the nut is the point: a large reward on its clockwise spin, whether or not the hand touches it
    nut_spin = RewTerm(func=screw.nut_spin, weight=20.0)
    # knocking the nut off ends the episode, which also ended the forced-descent penalty: the second stage-2 run
    # learned to do exactly that (episodes reaching the time limit fell from 33 % to 4 % in 30 min). Rewards are
    # scaled by the step (0.02 s), so -1500 is -30 per knocked-off nut.
    nut_off_bolt = RewTerm(func=mdp.is_terminated_term, weight=-1500.0, params={"term_keys": "nut_off_bolt"})
    waypoint = RewTerm(func=screw.waypoint_reached, weight=10.0)
    fingers_to_object = RewTerm(func=mdp.hand_to_nut_surface, params={"std_coarse": 0.4, "std_fine": 0.05}, weight=1.0)


@configclass
class NutScrewMixinCfg(NutPlayMixinCfg):
    rewards: NutScrewRewardsCfg = NutScrewRewardsCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.scene.num_envs = 1024  # SDF thread contacts are expensive
        # fingers pressed into the threads get pushed out at up to max_depenetration_velocity; the robot's 1000 m/s
        # flung the hand and tripped the abnormal-joint-velocity termination in 46 % of the first run's episodes
        self.scene.robot.spawn.rigid_props.max_depenetration_velocity = 5.0
        # The thread holds the nut up to ~100 N of downward force and gives way at 200 N
        # (scripts/test_thread_push.py); the Franka's 87 Nm joints press far harder than that, and the stage-2
        # policies pushed the nut through the thread instead of turning it. Limit the arm so the hand cannot press
        # past the thread (its links carry no gravity here, so a small torque still moves it and lifts the nut).
        arm = self.scene.robot.actuators["arm"]
        self.scene.robot.actuators["arm"] = arm.replace(effort_limit_sim={"panda_joint[1-4]": 20.0, "panda_joint[5-7]": 6.0})
        self.episode_length_s = 10.0
        # bolt fixed on the table, threaded nut
        self.scene.bolt = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Bolt",
            spawn=sim_utils.UsdFileCfg(usd_path=str(BOLT_USD)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=screw.BOLT_POS),
        )
        self.scene.object.spawn = self.scene.object.spawn.replace(usd_path=str(THREADED_NUT_USD))
        self.events.randomize_object_scale = None  # the threads have to mate
        self.events.reset_table.params["pose_range"] = {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]}
        self.events.reset_object = EventTerm(func=screw.reset_nut_on_bolt, mode="reset")
        # waypoint command in place of Dexsuite's random goals
        self.commands.object_pose.class_type = screw.ScrewWaypointCommand
        self.commands.object_pose.resampling_time_range = (1.0e6, 1.0e6)
        self.terminations.nut_off_bolt = DoneTerm(func=screw.nut_off_bolt)


@configclass
class NutScrewEnvCfg(NutScrewMixinCfg, dexsuite.DexsuiteReorientEnvCfg):
    pass


@configclass
class NutScrewEnvCfg_PLAY(NutScrewMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.resampling_time_range = (1.0e6, 1.0e6)
