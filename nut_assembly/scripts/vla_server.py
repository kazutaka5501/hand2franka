"""SmolVLA policy server for closed-loop evaluation in Isaac (``scripts/eval_vla_closed_loop.py`` is the client).

Isaac Lab runs in the Python 3.11 venv and lerobot in the hand2franka (lerobot, Python 3.12) environment, so the policy runs in its own process.
Per control step (30 Hz, as in the datasets) the client sends the simulator state; this process renders the camera
image with MuJoCo exactly as the training datasets were rendered, runs SmolVLA on all environments as one batch, and
returns absolute joint position targets. A predicted chunk is executed for ``--replan`` steps before predicting again.

    MUJOCO_GL=egl python scripts/vla_server.py outputs/train/nut_distill/checkpoints/last/pretrained_model
    MUJOCO_GL=egl python scripts/vla_server.py lerobot/smolvla_base --zero_shot
"""

import argparse
import importlib.util
import io
from multiprocessing.connection import Listener
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_distill_dataset", ROOT / "scripts/build_distill_dataset.py")
dataset_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dataset_builder)
render = dataset_builder.render
ADDRESS = ("localhost", 6011)
AUTHKEY = b"franka-sharpa-vla"


def unpack(blob: bytes) -> dict:
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def pack(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def load_policy(checkpoint: str, zero_shot: bool, train_dataset: Path):
    from lerobot.policies.factory import make_pre_post_processors

    if zero_shot:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies.factory import make_policy

        meta = LeRobotDatasetMetadata(f"local/{train_dataset.name}", root=train_dataset)
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.pretrained_path = checkpoint
        cfg.input_features = {}
        cfg.device = "cuda"
        policy = make_policy(cfg, ds_meta=meta).to("cuda").eval()
        pre, post = make_pre_post_processors(policy.config, dataset_stats=meta.stats)
    else:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        policy = SmolVLAPolicy.from_pretrained(checkpoint).to("cuda").eval()
        pre, post = make_pre_post_processors(policy.config, pretrained_path=checkpoint)
    return policy, pre, post


class Renderer:
    """One MuJoCo scene per environment (nut size and table position differ), rendered from the dataset camera."""

    def __init__(self, setup: dict, wrist: bool):
        names = [str(n) for n in setup["joint_names"]]
        self.wrist = wrist
        self.envs = []
        for i in range(len(setup["nut_scale"])):
            model = render.build_model(0.03, setup["table_pos"][i], setup["table_size"], dataset_builder.SIZE,
                                       dataset_builder.SIZE, float(setup["nut_scale"][i]), wrist_camera=wrist)
            qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
            nut = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")]
            goal = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal")]
            self.envs.append((model, mujoco.MjData(model), mujoco.Renderer(model, dataset_builder.SIZE, dataset_builder.SIZE),
                              qadr, nut, goal))
        self.camera = dataset_builder.camera()

    def images(self, state: dict) -> dict:
        out, wrist = [], []
        for i, (model, data, renderer, qadr, nut, goal) in enumerate(self.envs):
            data.qpos[qadr] = state["joint_pos"][i]
            data.mocap_pos[nut] = state["nut_pos"][i]
            data.mocap_quat[nut] = state["nut_quat"][i]
            data.mocap_pos[goal] = state["goal_pos"][i]
            data.mocap_quat[goal] = state["goal_quat"][i] if "goal_quat" in state else (1.0, 0.0, 0.0, 0.0)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, self.camera)
            out.append(renderer.render())
            if self.wrist:
                renderer.update_scene(data, "wrist")
                wrist.append(renderer.render())
        images = {"camera1": np.stack(out)}
        if self.wrist:
            images["camera2"] = np.stack(wrist)
        return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--zero_shot", action="store_true")
    parser.add_argument("--train_dataset", type=Path, default=ROOT / "data/lerobot/nut_distill_train")
    parser.add_argument("--replan", type=int, default=10, help="Actions executed from each predicted chunk.")
    parser.add_argument("--task", type=str, default=dataset_builder.TASK, help="Instruction, as in the training dataset.")
    parser.add_argument("--infer_batch", type=int, default=64, help="Environments per forward pass (8 fits an 8 GB GPU).")
    args = parser.parse_args()

    policy, pre, post = load_policy(args.checkpoint, args.zero_shot, args.train_dataset)
    print(f"[server] {args.checkpoint} loaded, listening on {ADDRESS}", flush=True)
    with Listener(ADDRESS, authkey=AUTHKEY) as listener, listener.accept() as conn:
        setup = unpack(conn.recv_bytes())
        wrist = "observation.images.camera2" in policy.config.input_features
        renderer = Renderer(setup, wrist)
        print(f"[server] cameras: {'overview + wrist' if wrist else 'overview'}", flush=True)
        chunk, step = None, 0
        conn.send_bytes(pack(ok=np.array(1)))
        while True:
            blob = conn.recv_bytes()
            if blob == b"done":
                break
            state = unpack(blob)
            if state.get("reset", np.array(0)).item():
                chunk, step = None, 0
            if chunk is None or step % args.replan == 0:
                images = renderer.images(state)
                batch = {
                    "observation.state": torch.from_numpy(state["joint_pos"].astype(np.float32)).to("cuda"),
                    "task": [args.task] * len(state["joint_pos"]),
                    **{f"observation.images.{cam}": torch.from_numpy(img).to("cuda").permute(0, 3, 1, 2).float() / 255
                       for cam, img in images.items()},
                }
                n, parts = len(state["joint_pos"]), []
                for i in range(0, n, args.infer_batch):
                    part = {k: (v[i:i + args.infer_batch] if k != "task" else v[i:i + args.infer_batch]) for k, v in batch.items()}
                    with torch.no_grad():
                        parts.append(post(policy.predict_action_chunk(pre(part))).float().cpu().numpy())
                chunk = np.concatenate(parts)
                offset = step
            conn.send_bytes(pack(target=chunk[:, step - offset]))
            step += 1
    print("[server] done", flush=True)


if __name__ == "__main__":
    main()
