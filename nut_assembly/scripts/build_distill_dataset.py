"""Turn teacher rollouts (RL policy, privileged state) into a LeRobot dataset for SmolVLA: images + joint state -> action.

The teacher sees the nut and goal positions as numbers; the student sees a camera image instead, in which the goal is
drawn as a green sphere (as in the rendered videos). Images come from MuJoCo, replaying the recorded Isaac states,
because Isaac Sim's renderer crashes on this machine (see README).

    python scripts/build_distill_dataset.py data/teacher/nut_dense_rollouts.npz

Successful episodes only (lifted and at the goal at the end). The last ``--held_out`` of them go into a separate
dataset for the offline evaluation. 60 Hz control is subsampled to 30 Hz.

    observation.state            29 joint positions [rad]
    observation.images.camera1   256 x 256 RGB, fixed camera across the table
    action                       29 joint position targets [rad] that the teacher commanded (absolute)
"""

import argparse
import importlib.util
import shutil
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("render_mujoco", ROOT / "scripts/render_mujoco.py")
render = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render)

TASK = "pick up the nut and hold it at the green marker"
SIZE = 256
SUBSAMPLE = 2


def camera():
    cam = mujoco.MjvCamera()
    # across the table from the robot, high enough to see the table and the whole goal volume
    cam.lookat[:] = [0.5, 0.0, 0.42]
    cam.distance, cam.azimuth, cam.elevation = 1.15, 180.0, -28.0
    return cam


def goal_at(data, env, step):
    """Goal position and orientation at a step: per step when the goals changed within the episode (stage 1)."""
    if "goal_pos_t" in data:
        return data["goal_pos_t"][env, step], data["goal_quat_t"][env, step]
    return data["goal_pos"][env], np.array([1.0, 0.0, 0.0, 0.0])


def write(episodes, data, root: Path, repo_id: str, task: str = TASK, wrist: bool = False):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    names = [str(n) for n in data["joint_names"]]
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "action": {"dtype": "float32", "shape": (len(names),), "names": names},
        "observation.images.camera1": {"dtype": "video", "shape": (SIZE, SIZE, 3), "names": ["height", "width", "channels"]},
    }
    if wrist:
        features["observation.images.camera2"] = dict(features["observation.images.camera1"])
    shutil.rmtree(root, ignore_errors=True)
    fps = int(round(1.0 / (float(data["dt"]) * SUBSAMPLE)))
    dataset = LeRobotDataset.create(repo_id, fps=fps, features=features, root=root, robot_type="franka_sharpa",
                                    image_writer_threads=4)
    cam = camera()
    for env in episodes:
        model = render.build_model(float(data["ball_radius"][env]), data["table_pos"][env], data["table_size"], SIZE, SIZE,
                                   float(data["object_scale"][env]), wrist_camera=wrist)
        qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
        nut = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")]
        goal = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal")]
        state = mujoco.MjData(model)
        with mujoco.Renderer(model, SIZE, SIZE) as renderer:
            for step in range(0, data["joint_pos"].shape[1], SUBSAMPLE):
                state.qpos[qadr] = data["joint_pos"][env, step]
                state.mocap_pos[nut] = data["ball_pos"][env, step]
                state.mocap_quat[nut] = data["ball_quat"][env, step]
                state.mocap_pos[goal], state.mocap_quat[goal] = goal_at(data, env, step)
                mujoco.mj_forward(model, state)
                renderer.update_scene(state, cam)
                image = renderer.render()
                extra = {}
                if wrist:
                    renderer.update_scene(state, "wrist")
                    extra["observation.images.camera2"] = renderer.render()
                # the action of a 30 Hz frame: the target the teacher commanded at the last of its two 60 Hz steps
                last = min(step + SUBSAMPLE - 1, data["joint_target"].shape[1] - 1)
                dataset.add_frame({
                    "observation.state": data["joint_pos"][env, step].astype(np.float32),
                    "action": data["joint_target"][env, last].astype(np.float32),
                    "observation.images.camera1": image,
                    "task": task,
                    **extra,
                })
        dataset.save_episode()
    dataset.finalize() if hasattr(dataset, "finalize") else None
    print(f"[dataset] {root}: {len(episodes)} episodes at {fps} Hz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--held_out", type=int, default=20)
    parser.add_argument("--out", type=Path, default=ROOT / "data/lerobot")
    parser.add_argument("--name", type=str, default="nut_distill")
    parser.add_argument("--task", type=str, default=TASK)
    parser.add_argument("--wrist", action="store_true", help="Add a palm camera as camera2.")
    parser.add_argument("--shard", type=int, default=0, help="Build only every num_shards-th training episode, from this one.")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--lifted_only", action="store_true",
                        help="Keep episodes that lifted and held the nut (goals change within the episode, so the "
                             "single-goal check does not apply).")
    args = parser.parse_args()
    data = np.load(args.rollouts)
    keep = data["lifted"] if args.lifted_only else data["lifted"] & data["at_goal"]
    ok = np.where(keep)[0]
    train, held = ok[: len(ok) - args.held_out], ok[len(ok) - args.held_out:]
    print(f"[dataset] {len(ok)}/{len(data['lifted'])} successful teacher episodes: {len(train)} train, {len(held)} held out")
    if args.num_shards > 1:
        # parallel builds: shard k writes <name>_train_k (merged afterwards), shard 0 also the held-out set
        shard = f"{args.name}_train_{args.shard}"
        write(train[args.shard::args.num_shards], data, args.out / shard, f"local/{shard}", args.task, args.wrist)
        if args.shard == 0:
            write(held, data, args.out / f"{args.name}_heldout", f"local/{args.name}_heldout", args.task, args.wrist)
        return
    write(train, data, args.out / f"{args.name}_train", f"local/{args.name}_train", args.task, args.wrist)
    write(held, data, args.out / f"{args.name}_heldout", f"local/{args.name}_heldout", args.task, args.wrist)


if __name__ == "__main__":
    main()
