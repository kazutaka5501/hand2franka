"""Stage-1 (6D goals) tracking metrics from recorded trajectories, for the RL teacher and SmolVLA students alike.

For every goal an episode shows, the error is measured over the last 0.5 s before the goal changes (or the episode
ends): position error, and the orientation error up to the nut's 12 symmetries. A goal counts as reached when the
median over that window is within 3 cm and 30 deg.

    python scripts/play_metrics.py data/teacher/nut_play_rollouts.npz logs/videos/vla_play_distilled.npz
"""

import argparse

import numpy as np
from scipy.spatial.transform import Rotation as R

SYMMETRIES = [R.from_euler("xyz", [flip * np.pi, 0.0, k * np.pi / 3]) for flip in (0, 1) for k in range(6)]


def goal_errors(d, env, window):
    nut = R.from_quat(d["ball_quat"][env][:, [1, 2, 3, 0]])
    goal = R.from_quat(d["goal_quat_t"][env][:, [1, 2, 3, 0]])
    rot = np.min([np.degrees(((nut * s) * goal.inv()).magnitude()) for s in SYMMETRIES], axis=0)
    pos = 1e3 * np.linalg.norm(d["ball_pos"][env] - d["goal_pos_t"][env], axis=1)
    ends = np.r_[np.where(np.any(np.diff(d["goal_quat_t"][env], axis=0) != 0, axis=1))[0], len(rot) - 1]
    return [(np.median(pos[max(e - window, 0):e + 1]), np.median(rot[max(e - window, 0):e + 1])) for e in ends]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    for path in args.files:
        d = np.load(path)
        window = int(round(0.5 / float(d["dt"])))
        errors = np.array([e for env in range(d["ball_pos"].shape[0]) for e in goal_errors(d, env, window)])
        reached = (errors[:, 0] < 30) & (errors[:, 1] < 30)
        print(f"{path}: {d['ball_pos'].shape[0]} episodes, {len(errors)} goals | median position error "
              f"{np.median(errors[:, 0]):.0f} mm, orientation error {np.median(errors[:, 1]):.0f} deg | "
              f"goals reached (3 cm, 30 deg): {reached.mean():.0%} | lifted at the end: {d['lifted'].mean():.0%}")


if __name__ == "__main__":
    main()
