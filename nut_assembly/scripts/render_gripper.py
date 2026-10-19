"""Render ``scripts/evaluate_gripper.py`` trajectories with MuJoCo: Franka with its gripper, the nut, and for the
threading task the bolt (kinematic replay, like ``render_mujoco.py``).

    MUJOCO_GL=egl uv run --no-project --python 3.11 --with mujoco --with imageio --with imageio-ffmpeg \
        --with pillow --with numpy --with scipy python scripts/render_gripper.py logs/videos/gripper_thread.npz
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from franka_sharpa import nut_geometry, thread_geometry as tg  # noqa: E402

_spec = importlib.util.spec_from_file_location("render_mujoco", ROOT / "scripts/render_mujoco.py")
render = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render)


def franka_hand_urdf() -> Path:
    """Isaac Sim's Franka description with absolute mesh paths, written next to the other robot files."""
    spec = importlib.util.find_spec("isaacsim")
    desc = (Path(spec.origin).parent / "exts/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description"
            if spec else Path(sys.argv[0]))
    out = ROOT / "assets/franka_gripper/franka_gripper.urdf"
    if not out.exists():
        text = (desc / "robots/panda_arm_hand.urdf").read_text().replace("package://franka_description", str(desc))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    return out


def build(data, size):
    render.ROBOT_URDF = ROBOT_URDF
    original = render._finish

    def finish(spec, w, h):
        v, f = nut_geometry.visual_mesh()
        spec.add_mesh(name="nutmesh", uservert=v.ravel().tolist(), userface=f.ravel().tolist())
        nut = spec.worldbody.add_body(name="nut", mocap=True)
        nut.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="nutmesh", rgba=render.NUT_RGBA)  # collidable: discardvisual drops the rest
        if "bolt_pos" in data:
            sv, sf = tg.bolt_shank_mesh()
            spec.add_mesh(name="shank", uservert=sv.ravel().tolist(), userface=sf.ravel().tolist())
            spec.add_mesh(name="head", uservert=tg.bolt_head_vertices().ravel().tolist())
            for m in ("shank", "head"):
                spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=m, pos=list(data["bolt_pos"][0]),
                                        rgba=[0.12, 0.25, 0.85, 1.0])
        return original(spec, w, h)

    render._finish = finish
    try:
        table_top = float(data["bolt_pos"][0][2]) if "bolt_pos" in data else -0.003
        return render.build_model(0.03, (0.55, 0.0, table_top - 0.02), (0.8, 1.5, 0.04), size, size, None)
    finally:
        render._finish = original


ROBOT_URDF = None


def main():
    global ROBOT_URDF
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--envs", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--size", type=int, default=560)
    args = parser.parse_args()
    ROBOT_URDF = franka_hand_urdf()
    data = dict(np.load(args.trajectory))
    model = build(data, args.size)
    names = [str(n) for n in data["joint_names"]]
    qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
    nut = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "nut")]
    goal = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal")]
    ball = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")]
    state = mujoco.MjData(model)
    state.mocap_pos[ball] = [0, 0, -5]  # render_mujoco's ball, unused here
    offset = float(data["nut_frame_to_centre"])
    overview, close = mujoco.MjvCamera(), mujoco.MjvCamera()
    overview.lookat[:] = [0.45, 0.0, 0.25]
    overview.distance, overview.azimuth, overview.elevation = 1.4, 135.0, -22.0
    close.distance, close.azimuth, close.elevation = 0.35, 150.0, -20.0
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    frames = []
    fps = int(round(1.0 / float(data["dt"])))
    with mujoco.Renderer(model, args.size, args.size) as renderer:
        for env in args.envs:
            label = f"{data['task']} env {env}: {'success' if data['success'][env] else 'failure'}"
            for t in range(data["joint_pos"].shape[1]):
                state.qpos[qadr] = data["joint_pos"][env, t]
                rot = R.from_quat(data["nut_quat"][env, t][[1, 2, 3, 0]])
                state.mocap_pos[nut] = data["nut_pos"][env, t] + rot.apply([0.0, 0.0, offset])
                state.mocap_quat[nut] = data["nut_quat"][env, t]
                state.mocap_pos[goal] = data["goal_pos"][env] if "goal_pos" in data else [0, 0, -1]
                mujoco.mj_forward(model, state)
                close.lookat[:] = state.mocap_pos[nut]
                renderer.update_scene(state, overview)
                a = renderer.render()
                renderer.update_scene(state, close)
                img = Image.fromarray(np.concatenate([a, renderer.render()], axis=1))
                d = ImageDraw.Draw(img)
                d.rectangle([0, 0, img.width, 30], fill=(255, 255, 255))
                d.text((img.width // 2, 5), label, fill=(0, 0, 0), font=font, anchor="ma")
                frames.append(np.asarray(img))
    out = args.trajectory.with_suffix(".webm")
    imageio.mimsave(out, frames, fps=fps, macro_block_size=1, codec="libvpx-vp9",
                    ffmpeg_params=["-b:v", "0", "-crf", "34", "-row-mt", "1", "-deadline", "realtime", "-cpu-used", "8"])
    Image.fromarray(frames[len(frames) // 6]).save(args.trajectory.with_suffix(".png"))
    print(f"[render] wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
