"""Hand tracks -> Franka demonstrations in the simulated twin.

The pinch point (thumb tip / index tip midpoint) becomes the gripper's tool centre point and the
contact interval becomes "closed". Three things are changed to make a human motion executable by
a parallel gripper:
  * time is stretched (people do this in ~1.5 s) and the robot pauses to close / open;
  * the fingers take the cube by its left and right faces instead of the hand's diagonal pinch;
  * heights are raised where the noisy monocular depth would drive the tool through the
    table or sideways into the cube.

    python -m h2f.retarget            # replay all clips, keep the successful ones as a LeRobot dataset
"""
import argparse
import glob
import shutil
from pathlib import Path

import imageio
import numpy as np

from .camera import Calib
from .perception import INDEX_TIP, THUMB_TIP
from .sim import CAMERA_SIZES, CONTROL_HZ, HOME_TCP, PLATE_THICKNESS, TASK, PickPlaceEnv

VIDEO_FPS = 30
SLOWDOWN = 2.0  # robot time / human time
MAX_SPEED = 0.4  # tool speed limit [m/s]
SETTLE = 4  # control steps to let the arm arrive before the fingers move
DWELL = 10  # control steps spent closing / opening the gripper
CLEARANCE = 0.035  # tool height above the cube top while not yet above it
DATA_ROOT = Path("data/lerobot")


def demo_root(cameras):
    return DATA_ROOT / ("demos_" + "_".join(cameras))


def create_dataset(root, repo_id, cameras=("phone",)):
    """The real recordings only have the phone's point of view, so that is the default camera set."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    features = {
        "observation.state": {"dtype": "float32", "shape": (5,), "names": ["x", "y", "z", "yaw", "width"]},
        "action": {"dtype": "float32", "shape": (5,), "names": ["x", "y", "z", "yaw", "grip"]},
        **{f"observation.images.{c}": {"dtype": "video", "shape": (CAMERA_SIZES[c][1], CAMERA_SIZES[c][0], 3), "names": ["height", "width", "channels"]}
           for c in cameras},
    }
    shutil.rmtree(root, ignore_errors=True)
    return LeRobotDataset.create(repo_id, fps=CONTROL_HZ, features=features, root=root, robot_type="franka_panda",
                                 image_writer_threads=4)


def save_episode(dataset, steps, task=TASK):
    """steps: [(observation before the action, action)]"""
    cameras = [k for k in dataset.features if k.startswith("observation.images.")]
    for obs, action in steps:
        dataset.add_frame({"observation.state": obs["state"], "action": np.asarray(action, np.float32), "task": task,
                           **{k: obs[k.rsplit(".", 1)[1]] for k in cameras}})
    dataset.save_episode()


def gripper_yaw(cube_yaw):
    """Fingers on the cube's left and right faces as the phone sees them, following the cube's own yaw.

    The hand pinches the cube diagonally (thumb -> index at about 130 deg in the table plane), which a
    parallel gripper cannot do on a cube, so a pair of faces has to be chosen. Choosing it per clip
    from the measured angle flipped between pairs on a few degrees of noise and gave the policy
    several grasps for the same scene; left/right is used for every clip, and keeps both fingers in view."""
    return (cube_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4


def limit_speed(path, max_step=MAX_SPEED / CONTROL_HZ):
    """Insert intermediate targets wherever consecutive ones (columns 0-2) are further apart than max_step."""
    out = [path[:1]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b[:3] - a[:3]) / max_step)))
        out.append(np.linspace(a, b, n + 1)[1:])
    return np.concatenate(out)


def gripper_plan(track, calib, shift=(0.0, 0.0)):
    """-> (N, 5) action targets [x, y, z, yaw, grip] and, for each, the video frame it came from.

    `shift` moves the cube's start position; the hand path follows it up to the grasp and blends
    back to the recorded path on the way to the plate."""
    k = track["keypoints"]
    pinch = k[:, [THUMB_TIP, INDEX_TIP]].mean(1)
    t_grasp, t_release = int(track["t_grasp"]), int(track["t_release"])
    t_end = min(len(pinch) - 1, t_release + VIDEO_FPS // 2)  # follow the hand for 0.5 s after release
    shift = np.r_[shift, 0, 0]
    cube, place = track["cube_start"] + shift[:3], track["cube_end"]
    yaw = gripper_yaw(cube[2])

    def hand(t0, t1):
        """Pinch path between two video frames at the stretched control rate, as rows [x, y, z, frame]."""
        frames = np.arange(t0, t1 + 1e-6, VIDEO_FPS / (CONTROL_HZ * SLOWDOWN))
        return np.c_[[np.interp(frames, np.arange(len(pinch)), pinch[:, a]) for a in range(3)] + [frames]].T

    above = calib.cube + CLEARANCE
    grasp = np.r_[cube[:2], calib.cube / 2, t_grasp]
    release = np.r_[place[:2], calib.cube / 2 + PLATE_THICKNESS + 0.002, t_release]

    reach = hand(0, t_grasp) + shift  # keep the hand's horizontal approach but stay above the cube, then come straight down
    reach[:, 2] = np.maximum(reach[:, 2], above)
    reach = np.r_[[np.r_[HOME_TCP, 0]], reach, [np.r_[grasp[:2], above, t_grasp]], np.linspace([*grasp[:2], above, t_grasp], grasp, 7)[1:]]
    carry = hand(t_grasp, t_release)  # never below the grasp height, and hop over the plate rim
    carry += shift * np.linspace(1, 0, len(carry))[:, None]
    carry[:, 2] = np.maximum(carry[:, 2], calib.cube / 2 + 0.03 * np.sin(np.linspace(0, np.pi, len(carry))))
    carry = np.r_[[grasp], carry[1:-1], [release]]
    leave = hand(t_release, t_end)  # straight up, then follow the hand away
    leave[:, 2] = np.maximum(leave[:, 2], above + 0.03)
    leave = np.r_[[release], np.linspace(release, [*release[:2], above + 0.03, t_release], 7)[1:], leave]

    hold = lambda row, n: np.repeat(row[None], n, 0)
    parts = [(limit_speed(reach), 0), (hold(grasp, SETTLE), 0), (hold(grasp, DWELL), 1), (limit_speed(carry), 1),
             (hold(release, SETTLE), 1), (hold(release, DWELL), 0), (limit_speed(leave), 0), (hold(leave[-1], DWELL), 0)]
    path = np.concatenate([p for p, _ in parts])
    grip = np.concatenate([np.full(len(p), g) for p, g in parts])
    return np.c_[path[:, :3], np.full(len(path), yaw), grip], path[:, 3].astype(int)


def usual_finger_angle(tracks):
    """Thumb -> index direction at grasp, median over the clips. The gripper has it at yaw 0."""
    fingers = [t["keypoints"][int(t["t_grasp"]), INDEX_TIP] - t["keypoints"][int(t["t_grasp"]), THUMB_TIP] for t in tracks]
    return np.median([np.arctan2(f[1], f[0]) % np.pi for f in fingers])


def follow_plan(track, calib, usual_angle):
    """Follow the tracked hand. -> (N, 5) targets [x, y, z, yaw, grip] and the video frame of each.

    Tool centre point = pinch point. Yaw = how far the thumb -> index direction is turned from its usual
    value. The gripper is closed from the frame the cube is seen to leave until it is seen to arrive
    (the distance between the fingertips is too noisy to say). Added: a constant playback slow-down,
    a straight move from the robot's start pose to where the hand is first seen, and a floor at the
    grasp height, since the tracked hand sometimes dips below the table."""
    k = track["keypoints"]
    seen = np.flatnonzero(track["valid"])
    finger = k[:, INDEX_TIP] - k[:, THUMB_TIP]
    angle = np.unwrap(np.arctan2(finger[:, 1], finger[:, 0]), period=np.pi)  # a finger axis has no sign
    angle -= np.pi * np.round((angle[int(track["t_grasp"])] - usual_angle) / np.pi)
    pinch = k[:, [THUMB_TIP, INDEX_TIP]].mean(1)
    signals = np.c_[pinch[:, :2], np.maximum(pinch[:, 2], calib.cube / 2), angle - usual_angle]

    frames = np.arange(seen[0], seen[-1] + 1e-6, VIDEO_FPS / (CONTROL_HZ * SLOWDOWN))
    plan = np.stack([np.interp(frames, np.arange(len(k)), signals[:, a]) for a in range(4)], 1)
    plan = np.c_[plan, (frames >= int(track["t_grasp"])) & (frames < int(track["t_release"]))]
    approach = limit_speed(np.stack([np.r_[HOME_TCP, 0, 0], plan[0]]))[:-1]
    plan = np.r_[approach, plan, np.repeat(plan[-1:], DWELL, 0)]
    frames = np.r_[np.full(len(approach), frames[0]), frames, np.full(DWELL, frames[-1])]
    return plan, frames.astype(int)


def replay(env, track, plan, shift=(0.0, 0.0)):
    """Run a plan open loop until the task is done. -> per-step (observation, executed action), success."""
    obs = env.reset(track["cube_start"] + np.r_[shift, 0], track["plate"])
    steps = []
    for action in plan:
        prev = obs
        obs, _, success, _ = env.step(action)
        steps.append((prev, np.r_[env.target, action[4]].astype(np.float32)))
        if success:
            return steps, True
    return steps, False


def side_by_side(path, video, steps, frames):
    """Real clip (left, at the frame each robot step was derived from) next to the same view in simulation."""
    import av
    import cv2
    with av.open(video) as c:
        real = [cv2.resize(f.to_ndarray(format="rgb24"), (384, 216)) for f in c.decode(video=0)]
    with imageio.get_writer(path, fps=CONTROL_HZ, macro_block_size=1) as out:
        for (obs, _), f in zip(steps, frames):
            out.append_data(np.hstack([real[f], obs["phone"]]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=int, default=6, help="how many side-by-side replays to write")
    parser.add_argument("--no-dataset", action="store_true")
    parser.add_argument("--wrist", action="store_true", help="also record the simulated wrist camera")
    parser.add_argument("--follow", action="store_true", help="follow the tracked hand instead of the planned grasp")
    parser.add_argument("--shifted", type=int, default=0, help="extra copies of each clip with the cube start moved")
    parser.add_argument("--margin", type=float, default=0.08, help="how far beyond the human data the moved starts reach [m]")
    args = parser.parse_args()

    cameras = ("phone", "wrist") if args.wrist else ("phone",)
    calib, env = Calib.load(), PickPlaceEnv(cameras)
    Path("outputs/retarget_follow" if args.follow else "outputs/retarget").mkdir(parents=True, exist_ok=True)
    root = DATA_ROOT / "demos_follow" if args.follow else demo_root(cameras) if not args.shifted else DATA_ROOT / f"demos_shifted{args.shifted}"
    dataset = None if args.no_dataset else create_dataset(root, "local/hand2franka_demos", cameras)
    paths = sorted(glob.glob("data/tracks/*.npz"))
    tracks = [np.load(p) for p in paths]
    usual = usual_finger_angle(tracks)
    starts = np.array([t["cube_start"][:2] for t in tracks])
    rng, results = np.random.default_rng(0), []
    for i, (path, track) in enumerate(zip(paths, tracks)):
        # the clip as recorded, then copies whose cube start is drawn from the widened box
        shifts = [np.zeros(2)] + [rng.uniform(starts.min(0) - args.margin, starts.max(0) + args.margin) - track["cube_start"][:2]
                                  for _ in range(args.shifted)]
        for j, shift in enumerate(shifts):
            plan, frames = follow_plan(track, calib, usual) if args.follow else gripper_plan(track, calib, shift)
            steps, success = replay(env, track, plan, shift)
            results.append(success)
            if i < args.videos and j == 0:
                side_by_side(f"outputs/retarget{'_follow' if args.follow else ''}/{Path(path).stem}.mp4", path.replace("tracks", "raw").replace(".npz", ".MOV"), steps, frames)
            if dataset is not None and success:
                save_episode(dataset, steps)
        print(f"{Path(path).stem}: {sum(results[-len(shifts):])}/{len(shifts)} replays succeeded")
    if dataset is not None:
        dataset.finalize()
    print(f"open-loop replay success: {sum(results)}/{len(results)}")


if __name__ == "__main__":
    main()
