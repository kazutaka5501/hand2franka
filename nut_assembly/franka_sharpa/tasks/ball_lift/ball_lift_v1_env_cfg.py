"""Ball-specific observation and reward design on top of the Dexsuite-style baseline.

Differences to ``ball_lift_env_cfg`` (v0, Dexsuite's design made for an Allegro hand and 16 object shapes):

Observations
    * The object is a sphere, so it is described by its centre and radius (4 numbers) instead of a 64-point
      surface cloud plus an orientation quaternion (196 numbers, the quaternion carrying no information).
    * The goal is its position only; v0 also feeds the unused orientation half of the command.
    Per step 187 numbers instead of 383; both versions stack the last 5 steps.

Rewards
    * Reaching is measured to the ball surface with a coarse and a fine kernel, see ``mdp.hand_to_ball_surface``.
    * The success reward is gated by the grasp contact, see ``mdp.grasped_success_reward``.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.manipulation.dexsuite import dexsuite_env_cfg as dexsuite
from isaaclab_tasks.manager_based.manipulation.dexsuite.adr_curriculum import CurriculumCfg

from . import mdp
from .ball_lift_env_cfg import FrankaSharpaBallMixinCfg, FrankaSharpaRewardsCfg


@configclass
class BallObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Task state: where the ball is, how big it is, where it has to go, and the previous action."""

        ball_pos_b = ObsTerm(func=mdp.object_pos_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        ball_radius = ObsTerm(func=mdp.ball_radius_obs)
        goal_pos_b = ObsTerm(func=mdp.goal_pos_b, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()
    # joint positions and velocities, palm and fingertip states, fingertip contact forces: unchanged
    proprio: dexsuite.ObservationsCfg.ProprioObsCfg = dexsuite.ObservationsCfg.ProprioObsCfg()


@configclass
class BallRewardsCfg(FrankaSharpaRewardsCfg):
    fingers_to_object = RewTerm(func=mdp.hand_to_ball_surface, params={"std_coarse": 0.4, "std_fine": 0.05}, weight=1.0)

    success = RewTerm(
        func=mdp.grasped_success_reward,
        weight=10,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_std": 0.1,
            # read by the Dexsuite base config; the lift task then disables the orientation part
            "rot_std": 0.5,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class BallCurriculumCfg(CurriculumCfg):
    """Dexsuite's curriculum with the object noise terms pointed at the ball centre observation."""

    object_quat_unoise_min_adr = None
    object_quat_unoise_max_adr = None
    object_obs_unoise_min_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.ball_pos_b.noise.n_min",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": -0.01, "difficulty_term_str": "adr"},
        },
    )
    object_obs_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.ball_pos_b.noise.n_max",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.01, "difficulty_term_str": "adr"},
        },
    )


@configclass
class FrankaSharpaBallV1MixinCfg(FrankaSharpaBallMixinCfg):
    observations: BallObservationsCfg = BallObservationsCfg()
    rewards: BallRewardsCfg = BallRewardsCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        # swap the curriculum but keep what the base classes configured on the difficulty scheduler
        adr_params = self.curriculum.adr.params
        self.curriculum = BallCurriculumCfg()
        self.curriculum.adr.params = adr_params


@configclass
class FrankaSharpaBallLiftV1EnvCfg(FrankaSharpaBallV1MixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class FrankaSharpaBallLiftV1EnvCfg_PLAY(FrankaSharpaBallV1MixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass
