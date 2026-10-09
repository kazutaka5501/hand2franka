"""Cup stacking in the simulated twin: green into blue, then pink on top.

The cups are hollow truncated cones built from flat wall segments, so they really nest: one cup
sits inside another until the walls touch, `NEST` higher than the cup below.
"""
import glob

import mujoco
import numpy as np

from .sim import ArmEnv, to_base, to_table

CALIB_PATH = "data/cups/calib.json"
TASK = "stack the green cup into the blue cup, then the pink cup on top"
CUPS = {"pink": [0.95, 0.72, 0.76, 1], "green": [0.35, 0.90, 0.10, 1], "blue": [0.10, 0.52, 0.95, 1]}
BASE_RADIUS, RIM_RADIUS, HEIGHT = 0.0275, 0.0365, 0.090  # base and height measured; rim fitted to the outlines in the videos
WALL = 0.002
NEST = 0.024  # rise of a nested cup, measured by dropping one into another (20 mm for ideal cones; the flat segments add a little)
SEGMENTS = 24


def add_cup(spec, name, rgba):
    body = spec.worldbody.add_body(name=name)
    body.add_freejoint(name=name)
    # thin, light walls squeezed between two fingers need stiffer contacts than MuJoCo's default
    part = dict(rgba=rgba, mass=0.02 / (SEGMENTS + 1), friction=[1.0, 0.02, 0.001], condim=4, solref=[0.004, 1])
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[BASE_RADIUS, WALL / 2, 0], pos=[0, 0, WALL / 2], **part)
    tilt = np.arctan2(RIM_RADIUS - BASE_RADIUS, HEIGHT)
    size = [WALL / 2, RIM_RADIUS * np.tan(np.pi / SEGMENTS), np.hypot(HEIGHT, RIM_RADIUS - BASE_RADIUS) / 2]
    r = (BASE_RADIUS + RIM_RADIUS) / 2 - WALL / 2
    for k in range(SEGMENTS):
        phi = 2 * np.pi * k / SEGMENTS
        quat, around, outward = np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(around, [0, 0, 1], phi)
        mujoco.mju_axisAngle2Quat(outward, [0, 1, 0], tilt)
        mujoco.mju_mulQuat(quat, around, outward)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=[r * np.cos(phi), r * np.sin(phi), HEIGHT / 2], quat=quat, **part)


class CupStackEnv(ArmEnv):
    task = TASK
    calib_path = CALIB_PATH
    stages = ("nothing", "one cup stacked", "both cups stacked")

    @staticmethod
    def scenes(seed=0, margin=0.0):
        """Endless stream of ({cup: (x, y)},): each cup uniform over the box it occupied across the clips,
        optionally widened by `margin` metres on every side."""
        tracks = [np.load(p) for p in sorted(glob.glob("data/cups/tracks/*.npz"))]
        rng = np.random.default_rng(seed)
        while True:
            yield ({name: rng.uniform(np.min([t[name] for t in tracks], 0) - margin, np.max([t[name] for t in tracks], 0) + margin)
                    for name in CUPS},)

    def progress(self):
        return self.stacked()

    def __init__(self, cameras=("phone",), calib=None, max_steps=500):
        super().__init__(cameras, calib, max_steps)

    def add_objects(self, spec):
        for name, rgba in CUPS.items():
            add_cup(spec, name, rgba)
        # The stock gripper is a soft position servo, so its squeeze grows with the object's thickness:
        # plenty on a cube, about 0.1 N on a 2 mm cup wall. Stiffen it and cap the force instead.
        grip = spec.actuator("actuator8")
        grip.gainprm[0], grip.biasprm[1], grip.biasprm[2] = 3000 * 0.04 / 255, -3000, -20
        grip.forcerange = [-4, 4]  # more than this pushes the fingers through the thin wall
        spec.option.noslip_iterations = 4  # a cup pinched off-centre must not creep round in the fingers while it is carried

    def cup_pos(self, name):
        q = self.model.joint(name).qposadr[0]
        return to_table(self.data.qpos[q:q + 3])

    def nested(self, inner, outer):
        """Inside the other cup: centred on it and no higher than a loose fit leaves it."""
        d = self.cup_pos(inner) - self.cup_pos(outer)
        return bool(np.linalg.norm(d[:2]) < 0.01 and 0.5 * NEST < d[2] < NEST + 0.02)

    def stacked(self):
        """How many cups sit in the stack on the blue one (0, 1 or 2)."""
        green = self.nested("green", "blue")
        return int(green) + int(green and self.nested("pink", "green"))

    def success(self):
        return bool(self.stacked() == 2 and self.gripper_width > 0.065 and self.tcp_pose()[0][2] > HEIGHT + 2 * NEST + 0.03)

    def reset(self, cups):
        """cups = {name: (x, y)} in the table frame."""
        mujoco.mj_resetData(self.model, self.data)
        for name, xy in cups.items():
            q = self.model.joint(name).qposadr[0]
            self.data.qpos[q:q + 3] = to_base([xy[0], xy[1], 0.0005])
        return self.start()
