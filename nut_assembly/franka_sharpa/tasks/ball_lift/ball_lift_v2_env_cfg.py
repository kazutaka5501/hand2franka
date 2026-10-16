"""Tactile version of the ball task, organised like RoTO (https://github.com/elle-miller/roto).

The environment exposes its observations in separate groups and the agent config decides which of them the
policy sees, so the same environment serves all comparisons:

    proprio     joint positions and velocities, palm and fingertip states (forward kinematics)   136
    task        goal position, previous action                                                    32
    tactile     one contact bit per fingertip pad, the five places a Sharpa Wave has sensors        5
    gt          ball centre and radius, tracked every step                                          4
    first_look  ball centre and radius seen once at the start of the episode                       4

Agents (``agents/rsl_rl_ppo_cfg.py``), all with a critic that sees proprio + task + tactile + gt:

    full_touch     proprio + task + tactile + gt
    full_notouch   proprio + task + gt
    blind_touch    proprio + task + tactile + first_look
    blind_notouch  proprio + task + first_look

Rewards and curriculum are those of v1. Rewards still use the simulator's ball state and ball-specific contact
forces; that is training signal only and never reaches the policy.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.manipulation.dexsuite import dexsuite_env_cfg as dexsuite

from . import mdp
from .ball_lift_env_cfg import FrankaSharpaBallMixinCfg
from .ball_lift_v1_env_cfg import BallCurriculumCfg, BallRewardsCfg

# Sharpa's own Isaac Lab example uses 0.05 N as its contact threshold
PAD_CONTACT_THRESHOLD = 0.05
# error of the one-shot ball position estimate, same size as the last curriculum stage's tracking noise
FIRST_LOOK_POS_NOISE = 0.01
HISTORY_LENGTH = 5


@configclass
class TactileObservationsCfg:
    @configclass
    class TaskCfg(ObsGroup):
        goal_pos_b = ObsTerm(func=mdp.goal_pos_b, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True
            self.history_length = HISTORY_LENGTH

    @configclass
    class TactileCfg(ObsGroup):
        pad_contact = ObsTerm(func=mdp.pad_contact_binary, params={"threshold": PAD_CONTACT_THRESHOLD})

        def __post_init__(self):
            self.concatenate_terms = True
            self.history_length = HISTORY_LENGTH

    @configclass
    class GroundTruthCfg(ObsGroup):
        ball_pos_b = ObsTerm(func=mdp.object_pos_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        ball_radius = ObsTerm(func=mdp.ball_radius_obs)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = HISTORY_LENGTH

    @configclass
    class FirstLookCfg(ObsGroup):
        ball = ObsTerm(func=mdp.ball_first_look, params={"pos_noise": FIRST_LOOK_POS_NOISE})

        def __post_init__(self):
            self.concatenate_terms = True

    proprio: dexsuite.ObservationsCfg.ProprioObsCfg = dexsuite.ObservationsCfg.ProprioObsCfg()
    task: TaskCfg = TaskCfg()
    tactile: TactileCfg = TactileCfg()
    gt: GroundTruthCfg = GroundTruthCfg()
    first_look: FirstLookCfg = FirstLookCfg()


@configclass
class TactileCurriculumCfg(BallCurriculumCfg):
    """The ball position noise now lives in the ``gt`` group."""

    object_obs_unoise_min_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.gt.ball_pos_b.noise.n_min",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": -0.01, "difficulty_term_str": "adr"},
        },
    )
    object_obs_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.gt.ball_pos_b.noise.n_max",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.01, "difficulty_term_str": "adr"},
        },
    )


@configclass
class FrankaSharpaBallV2MixinCfg(FrankaSharpaBallMixinCfg):
    observations: TactileObservationsCfg = TactileObservationsCfg()
    rewards: BallRewardsCfg = BallRewardsCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        # the base config puts ball-filtered 3D contact forces into proprio; here touch is the tactile group only
        self.observations.proprio.contact = None
        adr_params = self.curriculum.adr.params
        self.curriculum = TactileCurriculumCfg()
        self.curriculum.adr.params = adr_params


@configclass
class FrankaSharpaBallLiftV2EnvCfg(FrankaSharpaBallV2MixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class FrankaSharpaBallLiftV2EnvCfg_PLAY(FrankaSharpaBallV2MixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY):
    pass
