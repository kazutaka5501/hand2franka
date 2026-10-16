"""Human-video-only demonstrations for SmolVLA: the retargeted clips (scripts/retarget_mano.py) as a LeRobot dataset with
the same features, camera, image size, frame rate and task text as the teacher dataset (build_distill_dataset.py).

    MUJOCO_GL=egl python scripts/build_human_dataset.py

    observation.state            29 joint positions [rad] of the retargeted robot at video frame t
    observation.images.camera1   256 x 256 MuJoCo render (robot, nut from the marker tracking, green goal marker where
                                 the nut ends up on the bolt; the bolt itself is not drawn, as in the teacher images)
    action                       joint positions at frame t + 1 (absolute targets)

Each clip is trimmed to [seg_start, seg_end) from retarget_mano.py: first appearance of the hand to the nut resting at
its final place on the bolt and the hand leaving it. Clips with too little measured hand data are skipped.
"""

import argparse
import glob
import importlib.util
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


render = load("render_mujoco", "render_mujoco.py")
distill = load("build_distill_dataset", "build_distill_dataset.py")
TASK, SIZE, camera = distill.TASK, distill.SIZE, distill.camera
FPS = 30
TABLE_POS = np.array([0.55, 0.0, 0.235])
TABLE_SIZE = np.array([0.8, 1.5, 0.04])
NUT_SCALE = 1.0  # the printed nut has the nominal size of franka_sharpa.nut_geometry

MIN_FRAMES = 60  # 2 s
MIN_MEASURED = 0.6  # fraction of segment frames with a measured (not interpolated) hand
MAX_GAP = 45  # longest stretch of joint-space interpolation inside the segment [frames]
MAX_BAD_IK = 0.10  # fraction of frames whose palm is > 30 mm or > 30 deg off the human one (IK / speed limit)


def check(d):
    s, e = int(d["seg_start"]), int(d["seg_end"])
    st = json.loads(str(d["stats"]))
    if e - s < MIN_FRAMES:
        return f"segment only {e - s} frames"
    if st["hand_measured"] < MIN_MEASURED:
        return f"hand measured in only {st['hand_measured']:.0%} of the segment"
    if st["longest_gap"] > MAX_GAP:
        return f"hand missing for {st['longest_gap']} frames in a row"
    pr, rr = np.nan_to_num(d["pos_res"][s:e]), np.nan_to_num(d["rot_res"][s:e])
    bad = np.mean((pr > 0.03) | (rr > np.radians(30)))
    if bad > MAX_BAD_IK:
        return f"robot palm off the human palm (>30 mm or >30 deg) in {bad:.0%} of the segment"
    if not np.isfinite(d["joint_pos"][s:e]).all():
        return "non-finite joints"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", type=Path, default=ROOT / "data/human")
    parser.add_argument("--out", type=Path, default=ROOT / "data/lerobot/nut_human_train")
    parser.add_argument("--repo_id", default="local/nut_human_train")
    parser.add_argument("--task", default=TASK, help="Instruction; match the dataset this one is mixed with.")
    parser.add_argument("--wrist", action="store_true", help="Add the palm camera as camera2 (as in the stage-1 data).")
    args = parser.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    names = [str(n) for n in np.load(ROOT / "data/teacher/nut_dense_rollouts.npz")["joint_names"]]
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "action": {"dtype": "float32", "shape": (len(names),), "names": names},
        "observation.images.camera1": {"dtype": "video", "shape": (SIZE, SIZE, 3), "names": ["height", "width", "channels"]},
    }
    if args.wrist:
        features["observation.images.camera2"] = dict(features["observation.images.camera1"])
    shutil.rmtree(args.out, ignore_errors=True)
    ds = LeRobotDataset.create(args.repo_id, fps=FPS, features=features, root=args.out, robot_type="franka_sharpa",
                               image_writer_threads=4)
    model = render.build_model(0.03, TABLE_POS, TABLE_SIZE, SIZE, SIZE, NUT_SCALE, wrist_camera=args.wrist)
    qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
    nut = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")]
    goal = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal")]
    state = mujoco.MjData(model)
    cam = camera()
    kept, skipped, frames = [], {}, 0
    with mujoco.Renderer(model, SIZE, SIZE) as renderer:
        for path in sorted(glob.glob(str(args.human / "IMG_*.npz"))):
            clip = Path(path).stem
            d = np.load(path)
            why = check(d)
            if why:
                skipped[clip] = why
                print(f"[dataset] skip {clip}: {why}")
                continue
            assert [str(n) for n in d["joint_names"]] == names
            s, e = int(d["seg_start"]), int(d["seg_end"])
            q = d["joint_pos"]
            for t in range(s, e - 1):  # the last frame has no next state to serve as its action
                state.qpos[qadr] = q[t]
                state.mocap_pos[nut] = d["nut_pos"][t]
                state.mocap_quat[nut] = d["nut_quat"][t]
                state.mocap_pos[goal] = d["goal_pos"]
                state.mocap_quat[goal] = d["nut_quat"][e - 1]
                mujoco.mj_forward(model, state)
                renderer.update_scene(state, cam)
                frame = {
                    "observation.state": q[t].astype(np.float32),
                    "action": q[t + 1].astype(np.float32),
                    "observation.images.camera1": renderer.render(),
                    "task": args.task,
                }
                if args.wrist:
                    renderer.update_scene(state, "wrist")
                    frame["observation.images.camera2"] = renderer.render()
                ds.add_frame(frame)
            ds.save_episode()
            kept.append(clip)
            frames += e - 1 - s
            print(f"[dataset] {clip}: {e - 1 - s} frames")
    ds.finalize() if hasattr(ds, "finalize") else None
    (args.human / "dataset_episodes.json").write_text(json.dumps({"episodes": kept, "skipped": skipped}, indent=1))
    print(f"[dataset] {args.out}: {len(kept)} episodes, {frames} frames at {FPS} Hz; skipped {len(skipped)}: {skipped}")


if __name__ == "__main__":
    main()
