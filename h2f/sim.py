"""MuJoCo twin of the recorded table: Franka Panda, the ArUco cube, the paper plate
and a camera placed exactly where the phone was.

Actions are end-effector targets in the *table frame* of the videos (origin at the
reference marker), so a hand trajectory measured in a clip can be sent to the robot
as it is: [x, y, z, yaw, grip], grip 0 = open, 1 = closed.
"""
import atexit
import glob
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import cv2
import mujoco
import numpy as np

from .camera import ARUCO_DICT, CALIB_PATH, H, ID_CUBE_FRONT, ID_CUBE_TOP, ID_TABLE, Calib

PANDA_XML = "assets/franka/panda.xml"
TEX_DIR = Path("assets/generated")

# Robot base in the table frame: on the far side of the table, facing the camera.
BASE_POS = np.array([-0.37, 0.52, 0.0])
BASE_R = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])  # table -> base rotation (+90 deg about z)

TCP_OFFSET = 0.1034  # hand frame -> point between the finger pads
PLATE_THICKNESS = 0.004
CONTROL_HZ = 20
MAX_STEP = 0.04  # largest TCP target move per control step [m]
MAX_TURN = 0.1  # largest yaw target change per control step [rad]
WORKSPACE = np.array([[-0.75, -0.25, 0.005], [0.15, 0.25, 0.45]])  # table frame
HOME_TCP = np.array([-0.37, 0.10, 0.22])  # start pose, visible at the top of the phone view
IK_SEED = np.array([0, -0.2, 0, -2.2, 0, 2.0, 0.785])
CAMERA_SIZES = {"phone": (384, 216), "wrist": (240, 240), "overview": (640, 480)}  # width, height
TASK = "pick up the cube and place it on the plate"


def to_base(p):
    return (np.asarray(p) - BASE_POS) @ BASE_R.T


def to_table(p):
    return np.asarray(p) @ BASE_R + BASE_POS


def marker_png(marker_id, fill, name, px=256):
    """White square with an ArUco marker covering `fill` of its side (blank if marker_id is None)."""
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    m = int(px * fill)
    img = np.full((px, px), 255, np.uint8)
    o = (px - m) // 2
    if marker_id is not None:
        img[o:o + m, o:o + m] = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), marker_id, m)
    path = (TEX_DIR / f"{name}.png").resolve()
    cv2.imwrite(str(path), img)
    return str(path)


def add_material(spec, name, **texture):
    spec.add_texture(name=name, **texture)
    spec.add_material(name=name).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
    return name


def base_scene(calib):
    """Franka on the far side of the table, the reference sheet, and a camera where the phone was."""
    spec = mujoco.MjSpec.from_file(PANDA_XML)
    for body in spec.bodies:
        body.gravcomp = body.name != "world"  # the arm holds its pose without sagging under gravity
    spec.option.timestep = 0.002
    spec.option.impratio = 10
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    world = spec.worldbody
    Rz90 = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]  # table axes expressed in the base frame

    world.add_light(pos=[0.5, 0, 1.5], dir=[0, 0, -1], diffuse=[0.6, 0.6, 0.6])
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], rgba=[0.07, 0.07, 0.08, 1])

    # A4 sheet with the reference marker (visual only)
    sheet = add_material(spec, "sheet", type=mujoco.mjtTexture.mjTEXTURE_2D, file=marker_png(ID_TABLE, 1.0, "table_marker"))
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.105, 0.1485, 0.0003], pos=[*to_base([0, 0.01, 0])[:2], 0.0003],
                   quat=Rz90, rgba=[0.95, 0.95, 0.95, 1], contype=0, conaffinity=0)
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[calib.table_marker / 2] * 2 + [0.01],
                   pos=[*to_base([0, 0, 0])[:2], 0.0008], quat=Rz90, material=sheet, contype=0, conaffinity=0)

    hand = spec.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.004] * 3, rgba=[1, 0, 0, 0])
    hand.add_camera(name="wrist", pos=[0.1, 0, 0.02], quat=[0, 0.9537, 0, 0.3007], fovy=75)

    # Phone camera: OpenCV axes (x right, y down, z forward) -> MuJoCo camera axes (x right, y up, z back)
    R_cam = BASE_R @ calib.R.T @ np.diag([1.0, -1, -1])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R_cam.ravel())
    world.add_camera(name="phone", pos=to_base(calib.cam_pos), quat=quat, fovy=np.degrees(2 * np.arctan(H / 2 / calib.f)))
    world.add_camera(name="overview", pos=[1.5, -0.9, 0.9], mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY, targetbody="link0", fovy=45)
    return spec


class ArmEnv:
    """The arm, its end-effector controller and the cameras. Subclasses add the objects and the task."""
    task = None
    calib_path = None

    def __init__(self, cameras=("phone", "wrist"), calib=None, max_steps=300):
        self.calib = calib or Calib.load(self.calib_path)
        spec = base_scene(self.calib)
        self.add_objects(spec)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self._ik = mujoco.MjData(self.model)  # scratch copy for kinematics
        self.substeps = round(1 / CONTROL_HZ / self.model.opt.timestep)
        self.max_steps = max_steps
        self.tcp = self.model.site("tcp").id
        self.renderers = {c: mujoco.Renderer(self.model, height=CAMERA_SIZES[c][1], width=CAMERA_SIZES[c][0]) for c in cameras}
        for renderer in self.renderers.values():
            atexit.register(renderer.close)  # free the GL context before the interpreter tears it down
        self.home_q = self.solve_ik(HOME_TCP, 0.0, IK_SEED, iters=200)

    # ------------------------------------------------------------------ state
    def tcp_pose(self, data=None):
        data = data or self.data
        R = data.site_xmat[self.tcp].reshape(3, 3)
        return to_table(data.site_xpos[self.tcp]), np.arctan2(R[1, 0], R[0, 0])

    @property
    def gripper_width(self):
        return float(self.data.qpos[7] + self.data.qpos[8])

    @property
    def empty_grasp(self):
        """Fingers closed on nothing - what a real gripper reports through its width."""
        return self.data.ctrl[7] < 128 and self.gripper_width < 0.01

    def state(self):
        pos, yaw = self.tcp_pose()
        return np.r_[pos, yaw, self.gripper_width].astype(np.float32)

    def render(self, camera):
        r = self.renderers[camera]
        r.update_scene(self.data, camera=camera)
        return r.render()

    def observe(self):
        return {"state": self.state(), **{name: self.render(name) for name in self.renderers}}

    # ---------------------------------------------------------------- control
    def solve_ik(self, pos, yaw, q0, iters=20, damping=1e-2):
        """Damped least squares for a top-down grasp pose (table-frame position, yaw about vertical)."""
        m, d = self.model, self._ik
        target_R = np.array([[np.cos(yaw), np.sin(yaw), 0], [np.sin(yaw), -np.cos(yaw), 0], [0, 0, -1.0]])
        target_p = to_base(pos)
        d.qpos[:] = self.data.qpos
        d.qpos[:7] = q0
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.site_xmat[self.tcp].reshape(3, 3)
            err_R = 0.5 * sum(np.cross(R[:, i], target_R[:, i]) for i in range(3))
            err = np.r_[target_p - d.site_xpos[self.tcp], err_R]
            mujoco.mj_jacSite(m, d, jacp, jacr, self.tcp)
            J = np.r_[jacp, jacr][:, :7]
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
            dq += (np.eye(7) - np.linalg.pinv(J) @ J) @ (0.05 * (IK_SEED - d.qpos[:7]))  # stay near the seed posture
            d.qpos[:7] = np.clip(d.qpos[:7] + dq, m.jnt_range[:7, 0], m.jnt_range[:7, 1])
        return d.qpos[:7].copy()

    def start(self):
        """Arm at its start pose, objects wherever the subclass put them. -> first observation"""
        self.data.qpos[:7] = self.home_q
        self.data.qpos[7:9] = 0.04
        self.data.ctrl[:7], self.data.ctrl[7] = self.home_q, 255
        mujoco.mj_forward(self.model, self.data)
        self.target = np.r_[self.tcp_pose()[0], self.tcp_pose()[1]]
        self.t = 0
        return self.observe()

    def step(self, action, render=True):
        action = np.asarray(action, float)
        pos = np.clip(action[:3], *WORKSPACE)
        move = pos - self.target[:3]
        dist = np.linalg.norm(move)
        pos = self.target[:3] + move * min(1.0, MAX_STEP / max(dist, 1e-9))
        yaw = self.target[3] + np.clip((action[3] - self.target[3] + np.pi) % (2 * np.pi) - np.pi, -MAX_TURN, MAX_TURN)
        self.target = np.r_[pos, yaw]
        q_from = self.data.ctrl[:7].copy()
        q_to = self.solve_ik(pos, yaw, q_from)
        self.data.ctrl[7] = 255 * (1 - np.clip(action[4], 0, 1))
        for i in range(self.substeps):
            self.data.ctrl[:7] = q_from + (q_to - q_from) * (i + 1) / self.substeps
            mujoco.mj_step(self.model, self.data)
        self.t += 1
        success = self.success()
        obs = self.observe() if render else {"state": self.state()}
        return obs, float(success), success, self.t >= self.max_steps


class PickPlaceEnv(ArmEnv):
    task = TASK
    calib_path = CALIB_PATH
    stages = ("nothing", "lifted")

    @staticmethod
    def scenes(seed=0, margin=0.0):
        """Endless stream of (cube (x, y, yaw), plate (x, y, radius)). The cube is uniform over the box spanned
        by the human clips, optionally widened by `margin` metres on every side to probe beyond the
        demonstrations; the plate is where it was in a random clip."""
        tracks = [np.load(p) for p in sorted(glob.glob("data/tracks/*.npz"))]
        starts = np.array([t["cube_start"] for t in tracks])
        plates = np.array([t["plate"] for t in tracks])
        widen = np.array([margin, margin, 0])
        rng = np.random.default_rng(seed)
        while True:
            yield rng.uniform(starts.min(0) - widen, starts.max(0) + widen), plates[rng.integers(len(plates))]

    def progress(self):
        return int(self.cube_pos[2] > self.calib.cube / 2 + 0.01)

    def privileged(self):
        """What the simulator knows and a camera does not: where the cube is, relative to the tool and to the plate."""
        tcp, yaw = self.tcp_pose()
        cube = self.cube_pos
        return np.r_[tcp, yaw, self.gripper_width, cube - tcp, cube[:2] - self.plate[:2], self.t / self.max_steps].astype(np.float32)

    def add_objects(self, spec):
        world, c = spec.worldbody, self.calib
        plate = world.add_body(name="plate", mocap=True)
        plate.add_geom(name="plate", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.116, PLATE_THICKNESS / 2, 0],
                       pos=[0, 0, PLATE_THICKNESS / 2], rgba=[0.93, 0.92, 0.88, 1], friction=[1, 0.005, 0.0001])
        # Cube with the same markers as the real one: id 1 on top, then 2, 3, 4, 5 around the sides starting
        # at the face towards the camera and going to the right. Cube frame = table frame at yaw 0.
        # MuJoCo cube maps list +x, -x, +y, -y, +z, -z in the geom frame.
        faces = {"+z": (ID_CUBE_TOP, c.top_marker), "-y": (ID_CUBE_FRONT, c.front_marker), "-x": (5, c.front_marker),
                 "+x": (3, c.front_marker), "+y": (4, c.front_marker), "-z": (None, 0)}
        files = [marker_png(faces[k][0], faces[k][1] / c.cube, f"cube_{k}") for k in ("+x", "-x", "+y", "-y", "+z", "-z")]
        cube_mat = add_material(spec, "cube", type=mujoco.mjtTexture.mjTEXTURE_CUBE, cubefiles=files)
        cube = world.add_body(name="cube", pos=[0.5, 0, c.cube / 2])
        cube.add_freejoint(name="cube")
        cube.add_geom(name="cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[c.cube / 2] * 3, material=cube_mat,
                      mass=0.05, friction=[1.5, 0.03, 0.0005], condim=4)

    @property
    def cube_pos(self):
        q = self.model.joint("cube").qposadr[0]
        return to_table(self.data.qpos[q:q + 3])

    def success(self):
        """Cube resting on the plate and released."""
        d = np.linalg.norm(self.cube_pos[:2] - self.plate[:2])
        resting = abs(self.cube_pos[2] - (PLATE_THICKNESS + self.calib.cube / 2)) < 0.01
        return bool(d < 0.116 - self.calib.cube / 2 and resting and self.gripper_width > 0.065
                    and self.tcp_pose()[0][2] > self.calib.cube + 0.03)

    def reset(self, cube, plate):
        """cube = (x, y, yaw), plate = (x, y), both in the table frame."""
        mujoco.mj_resetData(self.model, self.data)
        self.plate = np.r_[plate[:2], 0.0]
        self.data.mocap_pos[self.model.body("plate").mocapid[0]] = to_base(self.plate)
        q = self.model.joint("cube").qposadr[0]
        self.data.qpos[q:q + 3] = to_base([cube[0], cube[1], self.calib.cube / 2])
        half = (cube[2] + np.pi / 2) / 2  # cube frame = table frame, which is +90 deg from the base frame
        self.data.qpos[q + 3:q + 7] = [np.cos(half), 0, 0, np.sin(half)]
        return self.start()
