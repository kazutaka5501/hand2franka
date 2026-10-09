"""RL by trial, selection and retraining: reward-weighted regression with reward = success, discounted by time.

One round
  1. rolls the current policy out on fresh cube positions;
  2. scores every episode: failures are dropped, successes are compared with how long a success
     usually takes from that cube position (a linear baseline), and only the faster half is kept;
  3. `lerobot-train` continues the policy on the kept episodes.

Selecting on speed alone would favour cubes that happen to start near the plate; the baseline removes that.

    python -m e2f.improve CHECKPOINT --round 1
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .evaluate import load_policy, rollout, write_video
from .retarget import DATA_ROOT, create_dataset, save_episode
from .sim import PickPlaceEnv


def faster_than_expected(cubes, steps):
    """Indices of the episodes that finished sooner than a linear fit of duration on cube position predicts."""
    A = np.c_[np.ones(len(cubes)), cubes[:, :2]]
    expected = A @ np.linalg.lstsq(A, steps, rcond=None)[0]
    return np.flatnonzero(steps < expected), expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--margin", type=float, default=0.08, help="cube start box, widened beyond the human data [m]")
    parser.add_argument("--videos", type=int, default=6)
    args = parser.parse_args()

    policy, pre, post = load_policy(args.checkpoint)
    env = PickPlaceEnv(policy.cameras)
    out = Path("outputs/rl") / f"round{args.round}"
    out.mkdir(parents=True, exist_ok=True)
    dataset = create_dataset(DATA_ROOT / f"round{args.round}", f"local/ego2franka_round{args.round}", policy.cameras)
    cubes, steps, attempts = [], [], 0
    # training scenes use their own seeds, so they never coincide with the evaluation scenes (seed 0)
    for i, scene in enumerate(itertools.islice(env.scenes(1000 + args.round, args.margin), args.episodes)):
        r = rollout(env, policy, pre, post, scene, record=True)
        attempts += 1
        if r["success"]:
            save_episode(dataset, r["trace"])
            cubes.append(scene[0])
            steps.append(r["steps"])
        if i < args.videos:
            write_video(out / f"episode_{i:02d}_{'success' if r['success'] else 'fail'}.mp4", r["trace"])
        print(f"episode {i:03d}: {'success' if r['success'] else 'fail   '} steps={r['steps']}", flush=True)
    dataset.finalize()

    cubes, steps = np.array(cubes), np.array(steps)
    keep, expected = faster_than_expected(cubes, steps)
    (out / "selection.json").write_text(json.dumps(dict(
        checkpoint=args.checkpoint, attempts=attempts, successes=len(steps), kept=keep.tolist(),
        median_steps=float(np.median(steps)), median_steps_kept=float(np.median(steps[keep])),
        steps=steps.tolist(), expected=expected.round(1).tolist()), indent=2))
    print(f"{len(steps)}/{attempts} succeeded (median {np.median(steps):.0f} steps); "
          f"kept the {len(keep)} faster than expected (median {np.median(steps[keep]):.0f} steps)")


if __name__ == "__main__":
    main()
