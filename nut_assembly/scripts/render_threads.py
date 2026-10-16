"""Render ``scripts/test_thread_physics.py`` output with MuJoCo (kinematic replay, as ``render_mujoco.py``):

    MUJOCO_GL=egl uv run --no-project --python 3.11 --with mujoco --with imageio --with imageio-ffmpeg \
        --with pillow python scripts/render_threads.py logs/thread_test.npz

Writes <input>.mp4: an overview of all cases with labels, followed by a close-up of each case.
"""

import argparse
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from franka_sharpa import thread_geometry as tg  # noqa: E402

SPACING = 0.15  # as in test_thread_physics.py
BOLT_RGBA = [0.12, 0.25, 0.85, 1.0]
NUT_RGBA = [0.08, 0.08, 0.08, 1.0]
LABELS = {
    "tighten": "clockwise",
    "loosen": "counter-clockwise",
    "rest": "no torque",
    "tilted": "tilted 10 deg + clockwise",
    "phase": "turned 47 deg + clockwise",
}


def build(num_cases: int, width: int, height: int):
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth, spec.visual.global_.offheight = width, height
    shank_v, shank_f = tg.bolt_shank_mesh()
    spec.add_mesh(name="shank", uservert=shank_v.ravel().tolist(), userface=shank_f.ravel().tolist())
    head_v = tg.bolt_head_vertices()
    spec.add_mesh(name="head", uservert=head_v.ravel().tolist())  # convex: MuJoCo builds the hull
    nut_v, nut_f = tg.nut_mesh()
    spec.add_mesh(name="nut", uservert=nut_v.ravel().tolist(), userface=nut_f.ravel().tolist())
    world = spec.worldbody
    world.add_light(pos=[0.3, -0.6, 1.2], dir=[0.0, 0.5, -1.0], diffuse=[0.8, 0.8, 0.8])
    world.add_light(pos=[0.3, 0.6, 1.0], dir=[0.0, -0.5, -1.0], diffuse=[0.3, 0.3, 0.3])
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], rgba=[0.85, 0.85, 0.87, 1])
    for i in range(num_cases):
        for mesh in ("shank", "head"):
            world.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, pos=[i * SPACING, 0, 0], rgba=BOLT_RGBA,
                           contype=0, conaffinity=0)
        nut = world.add_body(name=f"nut{i}", mocap=True)
        nut.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="nut", rgba=NUT_RGBA, contype=0, conaffinity=0)
    return spec.compile()


def label(frame: np.ndarray, texts: list[tuple[float, str]], size: int = 22) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, frame.shape[1], size + 20], fill=(255, 255, 255))
    for x, text in texts:
        draw.text((x * frame.shape[1], 10), text, fill=(0, 0, 0), font=font, anchor="ma")
    return np.asarray(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    data = np.load(args.trajectory)
    names = [str(n) for n in data["names"]]
    pos, quat, fps = data["nut_pos"], data["nut_quat"], int(data["fps"])
    n, steps = pos.shape[0], pos.shape[1]
    model = build(n, args.width, args.height)
    state = mujoco.MjData(model)
    mocap = [model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"nut{i}")] for i in range(n)]

    def pose(step):
        for i in range(n):
            state.mocap_pos[mocap[i]] = pos[i, step] + [i * SPACING, 0, 0]
            state.mocap_quat[mocap[i]] = quat[i, step]
        mujoco.mj_forward(model, state)

    frames = []
    overview = mujoco.MjvCamera()
    overview.lookat[:] = [0.5 * (n - 1) * SPACING, 0.0, 0.07]
    overview.distance, overview.azimuth, overview.elevation = 0.62, 90.0, -12.0
    with mujoco.Renderer(model, args.height, args.width) as renderer:
        # overview: the label positions follow the bolts left to right
        texts = [((i + 0.5) / n, LABELS.get(name, name)) for i, name in enumerate(names)]
        for step in range(steps):
            pose(step)
            renderer.update_scene(state, overview)
            frames.append(label(renderer.render(), [(t[0], t[1]) for t in texts], 18))
        # close-ups
        close = mujoco.MjvCamera()
        close.distance, close.azimuth, close.elevation = 0.28, 60.0, -20.0
        for i, name in enumerate(names):
            close.lookat[:] = [i * SPACING, 0.0, 0.07]
            for step in range(steps):
                pose(step)
                renderer.update_scene(state, close)
                height_mm = 1e3 * pos[i, step, 2]
                frames.append(label(renderer.render(), [(0.5, f"{LABELS.get(name, name)}   nut centre {height_mm:5.1f} mm")]))

    output = args.trajectory.with_suffix(".mp4")
    imageio.mimsave(output, frames, fps=fps, macro_block_size=1, ffmpeg_params=["-movflags", "+faststart"])
    # VS Code's built-in player cannot decode H.264, it plays VP9 WebM
    imageio.mimsave(output.with_suffix(".webm"), frames, fps=fps, macro_block_size=1, codec="libvpx-vp9",
                    ffmpeg_params=["-b:v", "0", "-crf", "34", "-row-mt", "1", "-deadline", "realtime", "-cpu-used", "8"])
    print(f"[render] wrote {output} ({len(frames)} frames, {len(frames) / fps:.0f} s)")
    Image.fromarray(frames[steps - 1]).save(args.trajectory.with_suffix(".png"))


if __name__ == "__main__":
    main()
