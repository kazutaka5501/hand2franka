"""Render ``scripts/evaluate_screw.py`` trajectories with MuJoCo: robot, threaded bolt, nut (kinematic replay).

    MUJOCO_GL=egl uv run --no-project --python 3.11 --with mujoco --with imageio --with imageio-ffmpeg \
        --with pillow --with numpy python scripts/render_screw.py logs/videos/screw_<n>.npz

Left: overview, right: close-up of the bolt; the caption gives the turns and descent measured in Isaac.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from franka_sharpa import thread_geometry as tg  # noqa: E402

_spec = importlib.util.spec_from_file_location("render_mujoco", ROOT / "scripts/render_mujoco.py")
render = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render)


def build(bolt_pos, width, height):
    original = render._finish

    def finish(spec, w, h):
        v, f = tg.bolt_shank_mesh()
        spec.add_mesh(name="shank", uservert=v.ravel().tolist(), userface=f.ravel().tolist())
        spec.add_mesh(name="head", uservert=tg.bolt_head_vertices().ravel().tolist())
        for mesh in ("shank", "head"):
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, pos=list(bolt_pos),
                                    rgba=[0.12, 0.25, 0.85, 1.0])
        return original(spec, w, h)

    render._finish = finish
    try:
        # table as in the task; nut drawn with the smooth-bore mesh (the thread is hidden inside anyway)
        return render.build_model(0.03, (0.55, 0.0, 0.235), (0.8, 1.5, 0.04), width, height, 1.0)
    finally:
        render._finish = original


def caption(frame, text):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, frame.shape[1], 36], fill=(255, 255, 255))
    draw.text((frame.shape[1] // 2, 8), text, fill=(0, 0, 0), font=font, anchor="ma")
    return np.asarray(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--envs", type=int, nargs="*", default=[0, 1, 2, 3])
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()
    data = np.load(args.trajectory)
    names = [str(n) for n in data["joint_names"]]
    model = build(data["bolt_pos"], args.size, args.size)
    qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
    nut = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")]
    goal = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal")]
    state = mujoco.MjData(model)
    state.mocap_pos[goal] = [0, 0, -1]  # no goal marker
    overview, close = mujoco.MjvCamera(), mujoco.MjvCamera()
    overview.lookat[:] = [0.45, 0.0, 0.42]
    overview.distance, overview.azimuth, overview.elevation = 1.3, 135.0, -22.0
    close.lookat[:] = np.asarray(data["bolt_pos"]) + [0, 0, 0.07]
    close.distance, close.azimuth, close.elevation = 0.32, 150.0, -25.0
    frames = []
    fps = int(round(1.0 / float(data["dt"])))
    with mujoco.Renderer(model, args.size, args.size) as renderer:
        for env in args.envs:
            text = f"env {env}: {data['turns'][env]:+.2f} turns clockwise, {data['descent_mm'][env]:.1f} mm down the thread"
            for step in range(data["joint_pos"].shape[1]):
                state.qpos[qadr] = data["joint_pos"][env, step]
                state.mocap_pos[nut] = data["nut_pos"][env, step]
                state.mocap_quat[nut] = data["nut_quat"][env, step]
                mujoco.mj_forward(model, state)
                renderer.update_scene(state, overview)
                a = renderer.render()
                renderer.update_scene(state, close)
                frames.append(caption(np.concatenate([a, renderer.render()], axis=1), text))
    out = args.trajectory.with_suffix(".mp4")
    imageio.mimsave(out, frames, fps=fps, macro_block_size=1, ffmpeg_params=["-movflags", "+faststart"])
    imageio.mimsave(out.with_suffix(".webm"), frames, fps=fps, macro_block_size=1, codec="libvpx-vp9",
                    ffmpeg_params=["-b:v", "0", "-crf", "34", "-row-mt", "1", "-deadline", "realtime", "-cpu-used", "8"])
    Image.fromarray(frames[len(frames) // 8]).save(out.with_suffix(".png"))
    print(f"[render] wrote {out} and .webm ({len(frames)} frames)")


if __name__ == "__main__":
    main()
