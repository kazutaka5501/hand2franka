"""Isaac Lab Factory NutThread with the measured M38x6 nut and bolt instead of Factory's M16 pair.

Runs on Factory's own ``FactoryEnv`` (``isaaclab_tasks.direct.factory``) unchanged: task name stays "nut_thread",
so every name-based branch in ``factory_env.py`` / ``factory_utils.py`` is taken as for M16. Robot: Factory's
``franka_mimic.usd`` (Franka Panda with the standard parallel gripper, fingers coupled), operational-space torque
control, actions = 6D pose deltas, gripper always closed; the nut starts in the gripper above the bolt tip.

Assets (``scripts/build_factory_assets.py``) follow the M16 USDs: the bolt is a fixed-base articulation (fixed joint
to the world in the USD, not a kinematic body), the nut a single-body articulation, both with SDF thread collision.
The nut frame is offset like Factory's: the nut's bottom face is at +``BoltM38.base_height`` in the nut frame.

Where Factory's M16 numbers enter and what they became:

    asset numbers (factory_tasks_cfg.NutM16 / BoltM16)
        nut  diameter 0.024 -> 0.051 (across flats; only used for the open gripper width at reset,
             diameter / 2 * 1.25 = 31.9 mm per finger, under the 40 mm limit), height 0.01 -> 0.033 (not read by
             the env), mass 0.03 -> 0.035
        bolt diameter 0.024 -> 0.038, height 0.025 -> 0.060 (threaded length), base_height 0.01 -> 0.025 (head),
             thread_pitch 0.002 -> 0.006
    factory_utils.get_held_base_pos_local: nut base = nut frame + base_height (bolt head height) -> nut bottom face,
        holds by construction of nut_factory.usda
    factory_utils.get_target_held_base_pose: target base = bolt frame + base_height + height - 1.5 pitch
        -> 9 mm below the tip (M16: 3 mm), the nut fully on the thread with the tip about flush
    factory_env._get_curr_successes: success if base within success_threshold * pitch above the target
        (0.375 * 6 mm = 2.25 mm), engaged within engage_threshold * pitch (0.5 * 6 = 3 mm), and the base within
        2.5 mm of the bolt axis (hard-coded, not scaled: the 1 mm radial clearance keeps a threaded nut within it);
        plus fingertip yaw < ee_success_yaw (0 rad), unchanged
    keypoint rewards: all lengths are 3x the M16 ones in pitch units, so the squashing slopes a are divided by 3
        and keypoint_scale multiplied by 3; the reward as a function of (descent / pitch, tilt) is the M16 one
    factory_env.get_handheld_asset_relative_pose: fingertip at the nut base, i.e. the finger pads (17.6 mm) hold
        the lower half of the 33 mm nut (M16: the whole 13 mm nut)
    hand_init_pos (fingertip 15 mm above the tip, +-20 mm xy, +-10 mm z), held_asset_pos_noise (3 mm),
        held_asset_rot_init (-90 deg; both nuts have their flats at +-x), action bounds (+-5 cm box around the tip),
        thresholds and gains: unchanged
    fixed_asset_init_orn_deg / _range_deg (M16: bolt yaw 120..150 deg): sets the thread phase between the nut in the
        gripper and the bolt, and with it how much turning the task needs; changed, see FIXED_YAW_* below
    friction: Factory sets per-asset values that PhysX averages per contact pair. Nut 0.01 (Factory's value),
        robot 0.75 (Factory's), bolt 0.59 -> nut-bolt 0.30 (PLA on PLA, as in the thread tests), nut-finger 0.38
"""

from pathlib import Path

from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_env_cfg import FactoryTaskNutThreadCfg
from isaaclab_tasks.direct.factory.factory_tasks_cfg import BoltM16, FixedAssetCfg, HeldAssetCfg, NutThread

from franka_sharpa import nut_geometry, thread_geometry as tg

ASSET_DIR = Path(__file__).resolve().parents[3] / "assets/nut"

# lengths relative to Factory's M16 thread, used to rescale the length-based reward settings
_M16 = NutThread()  # configclass fields are only readable on instances
LENGTH_SCALE = tg.PITCH / BoltM16().thread_pitch  # 3.0

# Bolt yaw at reset, uniform in [FIXED_YAW_DEG, FIXED_YAW_DEG + FIXED_YAW_RANGE_DEG] (Factory M16: 120 + 30).
# Factory's controller turns the hand by at most ~0.35 deg per policy step (measured), ~155 deg in a 30 s episode, and
# success also needs the fingertip yaw to go from ~105 +- 15 deg to below 0. With M16's 120..150 deg the nut sat on
# the bolt tip and did not catch the thread within an episode (smoke test: bottom 4.75 -> 5.6 mm below the tip after
# ~110 deg of turning). scripts/thread_phase_model.py: the thread catches near nut yaw - bolt yaw = 5..10 deg and
# success depth follows ~44 deg later, so from reset the nut needs about (nut yaw - bolt yaw at reset) + 44 deg of
# turning. The nut yaw is the fingertip yaw - 90 deg (measured), so with bolt yaw = -46 deg the turn to success depth
# equals the fingertip's initial yaw, i.e. depth and yaw success arrive together; +-15 deg around it.
# From the geometric model only, not yet checked in Isaac (see scripts/smoke_gripper_nut.py --yaw_deg/--yaw_range).
FIXED_YAW_DEG = 360.0 - 46.0 - 15.0
FIXED_YAW_RANGE_DEG = 30.0


@configclass
class NutM38(HeldAssetCfg):
    usd_path = str(ASSET_DIR / "nut_factory.usda")
    diameter = nut_geometry.ACROSS_FLATS  # gripper width at reset
    height = nut_geometry.THICKNESS
    mass = nut_geometry.MASS
    friction = 0.01


@configclass
class BoltM38(FixedAssetCfg):
    usd_path = str(ASSET_DIR / "bolt_factory.usda")
    diameter = tg.MAJOR_DIAMETER
    height = tg.BOLT_SHANK_LENGTH  # threaded length above the head
    base_height = tg.BOLT_HEAD_HEIGHT
    thread_pitch = tg.PITCH
    friction = 0.59


def _scaled(coef):
    a, b = coef
    return [a / LENGTH_SCALE, b]


@configclass
class NutThreadM38(NutThread):
    name = "nut_thread"  # keeps Factory's nut_thread code paths
    fixed_asset_cfg = BoltM38()
    held_asset_cfg = NutM38()
    asset_size = 38.0
    duration_s = 30.0

    fixed_asset_init_orn_deg = FIXED_YAW_DEG
    fixed_asset_init_orn_range_deg = FIXED_YAW_RANGE_DEG

    keypoint_coef_baseline = _scaled(_M16.keypoint_coef_baseline)
    keypoint_coef_coarse = _scaled(_M16.keypoint_coef_coarse)
    keypoint_coef_fine = _scaled(_M16.keypoint_coef_fine)
    keypoint_scale = _M16.keypoint_scale * LENGTH_SCALE

    # Factory builds these spawn configs from the M16 cfgs at class definition: swap the USDs and masses
    fixed_asset = _M16.fixed_asset.replace(
        spawn=_M16.fixed_asset.spawn.replace(
            usd_path=fixed_asset_cfg.usd_path,
            mass_props=_M16.fixed_asset.spawn.mass_props.replace(mass=fixed_asset_cfg.mass),
        )
    )
    held_asset = _M16.held_asset.replace(
        spawn=_M16.held_asset.spawn.replace(
            usd_path=held_asset_cfg.usd_path,
            mass_props=_M16.held_asset.spawn.mass_props.replace(mass=held_asset_cfg.mass),
        )
    )


@configclass
class FactoryTaskNutThreadM38Cfg(FactoryTaskNutThreadCfg):
    task_name = "nut_thread"
    task = NutThreadM38()
    episode_length_s = 30.0
