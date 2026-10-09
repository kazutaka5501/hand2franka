"""Cup-stacking clips -> Franka demonstrations in the simulated twin.

What each clip contributes is where the three cups stand and when each one is picked up and put
down. The hand itself is hidden most of the time in these clips (sleeve in front, fingers inside
the cup), so the motion between those observed events is a plain lift-carry-lower at the arm's own pace.
(Making the arm wait so that each pick happened when it did in the clip was tried and dropped: the
idle frames taught the policy to stay put.)

The gripper takes a cup across its rim from outside (73 mm, inside the 80 mm opening). Held on a
diameter the cup hangs under the fingers like a bucket and settles upright by itself; pinching one
side of the rim, as the hand does, let it swing round the pinch and miss the cup below.

    python -m e2f.cups_retarget
"""
import argparse
import glob
from pathlib import Path

import av
import cv2
import imageio
import numpy as np

from .cups_sim import HEIGHT, NEST, TASK, CupStackEnv
from .retarget import DATA_ROOT, DWELL, SETTLE, create_dataset, limit_speed, save_episode
from .sim import CAMERA_SIZES, CONTROL_HZ, HOME_TCP

GRIP = np.array([0.0, 0.0, HEIGHT - 0.012])  # tool point on a cup: on its axis just below the rim, fingers left and right of it
RELEASE = 0.010  # let go this far above the seated position: the cup centres itself as it drops in
SLOW_ZONE = 0.05  # the last stretch towards the point above a cup is taken at the slow speed [m]
DESCENT = 0.008  # travel per control step while a cup is being approached or lifted [m]
DATASET_ROOT = DATA_ROOT / "cups_demos"


def stack_plan(track):
    """-> (N, 5) targets [x, y, z, yaw, grip] and the video frame each one corresponds to."""
    events = [0, *track["events"]]
    hold = lambda p, n: np.repeat(np.asarray(p, float)[None], n, 0)
    parts, here = [], np.asarray(HOME_TCP, float)
    for i, name in enumerate(("green", "pink")):
        level = i + 1
        grasp = np.r_[track[name], 0] + GRIP
        release = np.r_[track["blue"], level * NEST + RELEASE] + GRIP
        over = HEIGHT + release[2] + 0.03  # the carried cup's base clears the stack
        # The arm trails a fast-moving target by a few centimetres and a cup enters another with 7 mm to
        # spare, so it slows down as it arrives above a cup and comes down slowly. It never stops there:
        # frames in which nothing changes teach the policy to stay put.
        def arrive(a, b):
            near = b + (a - b) * min(1.0, SLOW_ZONE / np.linalg.norm(a - b))
            return np.r_[limit_speed(np.stack([a, near])), limit_speed(np.stack([near, b]), DESCENT)[1:]]
        up = lambda p: np.r_[p[:2], over]
        reach = np.r_[limit_speed(np.stack([here, up(here)])), arrive(up(here), up(grasp))[1:], limit_speed(np.stack([up(grasp), grasp]), DESCENT)[1:]]
        carry = np.r_[limit_speed(np.stack([grasp, up(grasp)]), DESCENT), arrive(up(grasp), up(release))[1:],
                      limit_speed(np.stack([up(release), release]), DESCENT)[1:]]
        pick, place = events[2 * i + 1], events[2 * i + 2]
        parts += [(reach, 0, events[2 * i], pick), (hold(grasp, SETTLE), 0, pick, pick), (hold(grasp, DWELL), 1, pick, pick),
                  (carry, 1, pick, place), (hold(release, SETTLE), 1, place, place), (hold(release, DWELL), 0, place, place)]
        here = release
    leave = limit_speed(np.stack([here, [*here[:2], over], HOME_TCP]))
    parts += [(leave, 0, events[-1], int(track["frames"]) - 1), (hold(HOME_TCP, DWELL), 0, *[int(track["frames"]) - 1] * 2)]
    xyz = np.concatenate([p for p, *_ in parts])
    grip = np.concatenate([np.full(len(p), g) for p, g, *_ in parts])
    frames = np.concatenate([np.linspace(a, b, len(p)) for p, _, a, b in parts])
    return np.c_[xyz, np.zeros(len(xyz)), grip], frames.astype(int)


def scene(track):
    return {name: track[name] for name in ("pink", "green", "blue")}


def replay(env, track, plan):
    """Run a plan open loop. -> per-step (observation, action), success."""
    obs = env.reset(scene(track))
    steps, success = [], False
    for action in plan:
        steps.append((obs, action.astype(np.float32)))
        obs, _, success, _ = env.step(action)
    return steps, success


def side_by_side(path, video, steps, frames):
    with av.open(video) as c:
        real = [cv2.resize(f.to_ndarray(format="rgb24"), CAMERA_SIZES["phone"]) for f in c.decode(video=0)]
    with imageio.get_writer(path, fps=CONTROL_HZ, macro_block_size=1) as out:
        for (obs, _), f in zip(steps, frames):
            out.append_data(np.hstack([real[min(f, len(real) - 1)], obs["phone"]]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-dataset", action="store_true")
    args = parser.parse_args()
    env = CupStackEnv()
    Path("outputs/cups/retarget").mkdir(parents=True, exist_ok=True)
    dataset = None if args.no_dataset else create_dataset(DATASET_ROOT, "local/ego2franka_cups")
    results = []
    for path in sorted(glob.glob("data/cups/tracks/*.npz")):
        track = np.load(path)
        plan, frames = stack_plan(track)
        steps, success = replay(env, track, plan)
        results.append(success)
        print(f"{Path(path).stem}: {len(plan)} steps, stacked {env.stacked()}/2, {'success' if success else 'FAIL'}", flush=True)
        side_by_side(f"outputs/cups/retarget/{Path(path).stem}.mp4", f"data/cups/raw/{Path(path).stem}.mp4", steps, frames)
        if dataset is not None and success:
            save_episode(dataset, steps, TASK)
    if dataset is not None:
        dataset.finalize()
    print(f"open-loop replay success: {sum(results)}/{len(results)}")


if __name__ == "__main__":
    main()
