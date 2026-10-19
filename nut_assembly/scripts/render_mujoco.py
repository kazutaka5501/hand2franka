"""Render trajectories recorded by ``scripts/evaluate.py`` with MuJoCo's offscreen renderer.

Isaac Sim 5.1's RTX renderer crashes on RTX 50-series cards with the 595 driver branch, so the Isaac rollout is
replayed kinematically here instead: the recorded joint angles and ball poses are simply displayed, no physics is
re-simulated. Runs in its own throwaway environment, not in the Isaac Lab venv:

    MUJOCO_GL=egl uv run --no-project --python 3.11 --with mujoco --with imageio --with imageio-ffmpeg \
        --with pillow python scripts/render_mujoco.py logs/.../model_1000_eval.npz
"""

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import sys

import imageio
import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from franka_sharpa import nut_geometry  # noqa: E402
ROBOT_URDF = ROOT / "assets/franka_sharpa/franka_sharpa.urdf"

ARM_RGBA = [0.85, 0.85, 0.87, 1.0]
HAND_RGBA = [0.25, 0.26, 0.28, 1.0]
PAD_RGBA = [0.08, 0.08, 0.08, 1.0]
BALL_RGBA = [0.9, 0.75, 0.1, 1.0]
GOAL_RGBA = [0.1, 0.8, 0.2, 0.35]
# the printed nut is black, but so is the hand: drawn orange so it can be followed
TABLE_RGBA = [0.55, 0.42, 0.3, 1.0]
NUT_RGBA = [1.0, 0.45, 0.0, 1.0]


def display_urdf() -> str:
    """The robot URDF with one displayable mesh set per link, declared as collision geometry.

    MuJoCo's URDF importer keeps either the visual or the collision geometry and cannot read the Franka's
    COLLADA visuals, so every link gets its STL visuals if it has any and its collision meshes otherwise.
    """
    robot = ET.parse(ROBOT_URDF).getroot()
    for link in robot.findall("link"):
        visuals = link.findall("visual")
        collisions = link.findall("collision")
        stl_visuals = [v for v in visuals if v.find("geometry/mesh").get("filename").lower().endswith(".stl")]
        for element in visuals + collisions:
            link.remove(element)
        for element in stl_visuals or collisions:
            element.tag = "collision"
            link.append(element)
    extension = ET.SubElement(robot, "mujoco")
    ET.SubElement(extension, "compiler", discardvisual="true", fusestatic="false", strippath="false")
    return ET.tostring(robot, encoding="unicode")


def add_wrist_camera(spec):
    """A camera on the palm looking along the fingers, slightly out of the palm, to see the grasp up close.

    Palm frame (see scripts/build_robot.py): z towards the fingertips, x out of the palm. MuJoCo cameras look along
    their -z axis with +y up in the image; here the image's up is "out of the palm".
    """
    palm = next(b for b in spec.bodies if b.name == "palm")
    pos = np.array([0.07, 0.0, -0.05])
    look = np.array([0.03, 0.0, 0.10]) - pos  # towards the space in front of the fingertips
    z = -look / np.linalg.norm(look)
    y = np.array([1.0, 0.0, 0.0]) - z * z[0]
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    palm.add_camera(name="wrist", pos=pos.tolist(), xyaxes=[*x, *y], fovy=80.0)


def build_model(ball_radius: float, table_pos, table_size, width: int, height: int, nut_scale: float | None = None,
                wrist_camera: bool = False):
    # one file per process: parallel builds (cluster array jobs) would otherwise overwrite each other's file
    urdf_path = ROBOT_URDF.with_name(f"franka_sharpa_display_{os.getpid()}.urdf")
    urdf_path.write_text(display_urdf())
    spec = mujoco.MjSpec.from_file(str(urdf_path))
    if wrist_camera:
        add_wrist_camera(spec)
    for geom in spec.geoms:
        body = geom.parent.name
        geom.rgba = ARM_RGBA if body.startswith("panda") else PAD_RGBA if "elastomer" in body else HAND_RGBA

    world = spec.worldbody
    world.add_light(pos=[0.5, -0.5, 2.5], dir=[0.0, 0.2, -1.0], diffuse=[0.8, 0.8, 0.8])
    world.add_light(pos=[1.5, 1.0, 2.0], dir=[-0.5, -0.4, -1.0], diffuse=[0.4, 0.4, 0.4])
    # floor under the table (the gripper tasks put the table top at the robot base, i.e. below z = 0)
    floor_z = min(0.0, table_pos[2] - 0.5 * table_size[2] - 0.3)
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3.0, 3.0, 0.1], pos=[0, 0, floor_z], rgba=[0.75, 0.77, 0.8, 1.0])
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=table_pos, size=0.5 * np.asarray(table_size), rgba=TABLE_RGBA)
    # a pedestal so the table does not appear to float
    world.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[table_pos[0], table_pos[1], 0.5 * (table_pos[2] - 0.5 * table_size[2] + floor_z)],
        size=[0.3, 0.5, 0.5 * (table_pos[2] - 0.5 * table_size[2] - floor_z)],
        rgba=[0.35, 0.35, 0.38, 1.0],
    )
    ball = world.add_body(name="ball", mocap=True)
    goal = world.add_body(name="goal", mocap=True)
    if nut_scale is not None:
        vertices, faces = nut_geometry.visual_mesh()
        spec.add_mesh(name="nut", uservert=(nut_scale * vertices).ravel().tolist(), userface=faces.ravel().tolist())
        ball.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="nut", rgba=NUT_RGBA)
        # the goal as a see-through nut, so its commanded orientation shows
        goal.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="nut", rgba=[0.1, 0.8, 0.2, 0.4])
        return _finish(spec, width, height)
    ball.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[ball_radius, 0, 0], rgba=BALL_RGBA)
    # a stripe makes the ball's rotation visible
    ball.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[1.005 * ball_radius, 0.15 * ball_radius, 0], rgba=[0.8, 0.2, 0.1, 1.0])
    # left collidable on purpose: discardvisual drops non-colliding geoms, and nothing is simulated here anyway
    goal.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[ball_radius, 0, 0], rgba=GOAL_RGBA)
    return _finish(spec, width, height)


def _finish(spec, width: int, height: int):
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    return spec.compile()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--envs", type=int, nargs="*", default=None, help="Recorded env indices to render, default all.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no_closeup", dest="closeup", action="store_false", help="Overview camera only.")
    args = parser.parse_args()

    data = np.load(args.trajectory)
    joint_names = [str(name) for name in data["joint_names"]]
    envs = args.envs if args.envs is not None else list(range(data["joint_pos"].shape[0]))
    output = args.output or args.trajectory.with_suffix(".mp4")
    fps = int(round(1.0 / float(data["dt"])))

    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.4, 0.0, 0.42]
    camera.distance, camera.azimuth, camera.elevation = 1.6, 135.0, -20.0

    frames, sheet = [], []
    for env in envs:
        is_nut = "object" in data and str(data["object"]) == "nut"
        model = build_model(float(data["ball_radius"][env]), data["table_pos"][env], data["table_size"], args.width, args.height,
                            float(data["object_scale"][env]) if is_nut else None)
        qpos_adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in joint_names]
        ball_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")]
        goal_id = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal")]
        state = mujoco.MjData(model)
        num_steps = data["joint_pos"].shape[1]
        close = mujoco.MjvCamera()
        close.lookat[:] = data["ball_pos"][env, 0]
        close.distance, close.azimuth, close.elevation = 0.35, 160.0, -25.0
        with mujoco.Renderer(model, args.height, args.width) as renderer:
            for step in range(num_steps):
                state.qpos[qpos_adr] = data["joint_pos"][env, step]
                state.mocap_pos[ball_id] = data["ball_pos"][env, step]
                state.mocap_quat[ball_id] = data["ball_quat"][env, step]
                if "goal_pos_t" in data:
                    state.mocap_pos[goal_id] = data["goal_pos_t"][env, step]
                    state.mocap_quat[goal_id] = data["goal_quat_t"][env, step]
                else:
                    state.mocap_pos[goal_id] = data["goal_pos"][env]
                mujoco.mj_forward(model, state)
                renderer.update_scene(state, camera)
                frame = renderer.render()
                if args.closeup:
                    close.lookat[:] = 0.5 * (np.asarray(data["ball_pos"][env, step]) + close.lookat)  # smoothed follow
                    renderer.update_scene(state, close)
                    frame = np.concatenate([frame, renderer.render()], axis=1)
                frames.append(frame)
                if step in (0, num_steps // 3, 2 * num_steps // 3, num_steps - 1):
                    sheet.append((env, frame))
        outcome = "lifted" if data["lifted"][env] else "not lifted"
        print(f"[render] env {env}: {outcome}")

    imageio.mimsave(output, frames, fps=fps, macro_block_size=1, ffmpeg_params=["-movflags", "+faststart"])
    print(f"[render] wrote {output} ({len(frames)} frames at {fps} fps)")
    # VS Code's built-in player cannot decode H.264, it plays VP9 WebM
    imageio.mimsave(output.with_suffix(".webm"), frames, fps=fps, macro_block_size=1, codec="libvpx-vp9",
                    ffmpeg_params=["-b:v", "0", "-crf", "34", "-row-mt", "1", "-deadline", "realtime", "-cpu-used", "8"])
    print(f"[render] wrote {output.with_suffix('.webm')}")

    # contact sheet: one row per env, columns at start, 1/3, 2/3 and end of the episode
    rows = [np.concatenate([frame[::2, ::2] for e, frame in sheet if e == env], axis=1) for env in envs]
    sheet_path = output.with_suffix(".png")
    Image.fromarray(np.concatenate(rows, axis=0)).save(sheet_path)
    print(f"[render] wrote {sheet_path}")


if __name__ == "__main__":
    main()
