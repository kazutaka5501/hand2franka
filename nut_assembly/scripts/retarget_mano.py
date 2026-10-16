"""Retarget tracked human hand motion (WiLoR / MANO keypoints) onto the simulated Franka Panda + Sharpa Wave hand.

    MUJOCO_GL=egl python scripts/retarget_mano.py            # all clips with hand tracks
    MUJOCO_GL=egl python scripts/retarget_mano.py IMG_8274   # some clips
    MUJOCO_GL=egl python scripts/retarget_mano.py --estimate_yaw

Pipeline per clip (30 fps, one robot state per video frame):

1. Hand in 3D. ``data/hands/<clip>.npz`` (``../h2f/hand_mano.py``) holds, per frame, 21 MANO keypoints in
   metres in a hand-centred frame (no translation) and their pixels. The frames were decoded with PyAV, which does NOT
   apply the phone's -90 deg rotation flag, so the pixels are in the raw 1920 x 1080 landscape image; they are turned
   into the displayed 1080 x 1920 portrait image (90 deg clockwise: u' = 1079 - v, v' = u) in which the camera was
   calibrated (``data/calib.json``, see track_nut.py). solvePnP of the metric keypoints against those pixels gives the
   hand in the camera frame and, with the board pose, in the board frame. The depth comes from the hand's metric size,
   so it is only as good as MANO's hand-size estimate (checked against the nut while it is held: thumb/index midpoint
   about 3 cm from the nut centre). Frames with a large reprojection error or a jump are dropped, gaps of up to
   ``MAX_GAP`` frames are interpolated, and each run of frames is Savitzky-Golay smoothed.
2. Board -> sim world: a fixed yaw and offset (``BOARD_YAW_DEG``, ``BOARD_ORIGIN``). The yaw turns the direction from
   which the person's forearm comes in (median over the grasp frames of all clips, ``--estimate_yaw``) into +x, the
   direction the robot reaches from its base. The offset puts the bolt in front of the robot and the nut start
   positions inside the camera view of the teacher dataset.
3. Palm frame. On both hands an orthonormal frame from keypoints: origin = centroid of the index/middle/ring/pinky
   knuckles (MCP), z = wrist -> middle knuckle, y = pinky knuckle -> index knuckle (made orthogonal), x = y cross z. On
   the robot the same frame comes from the Sharpa MCP joint bodies and right_hand_C_MC at the zero pose and is rigid
   in the ``palm`` link.
4. Fingers: the 22 Sharpa joints by bounded least squares (warm started frame to frame) on PIP/DIP/tip positions of
   each finger relative to its own knuckle, the human vectors scaled by the robot/human finger-length ratio, plus
   thumb tip -> finger tip vectors at true (unscaled) size when the human fingers are close together (grasp aperture
   kept), plus a small smoothness term.
5. Arm: damped least squares IK (MuJoCo Jacobians) of panda_joint1..7 so that the robot palm frame matches the human
   one, joint limits enforced, warm started, with a weak null-space pull towards the teacher's start posture.

Writes data/human/<clip>.npz: joint_pos (T, 29) in the teacher's joint order, nut_pos / nut_quat (wxyz) in the sim
world, masks, residuals, and the trimmed segment [seg_start, seg_end) used by build_human_dataset.py.
"""

import argparse
import glob
import importlib.util
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R, Slerp

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("render_mujoco", ROOT / "scripts/render_mujoco.py")
render = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render)

FPS = 30
TABLE_TOP = 0.255
TABLE_POS = np.array([0.55, 0.0, 0.235])
TABLE_SIZE = np.array([0.8, 1.5, 0.04])
# board -> sim world (see module docstring). Yaw from --estimate_yaw (median forearm direction at grasp), rounded.
BOARD_YAW_DEG = -60.0
BOARD_ORIGIN = np.array([0.58, -0.10, TABLE_TOP])

MAX_REPROJ_PX = 12.0
ARM_MAX_STEP = np.array([2.1, 2.1, 2.1, 2.1, 2.5, 2.5, 2.5]) / FPS  # rad per frame, Franka joint speed limits
HAND_MAX_STEP = 4.0 / FPS
MAX_GAP = 15  # frames of missing hand that are interpolated
SMOOTH_WIN = 15

# MANO / OpenPose keypoint order
WRIST = 0
FINGERS = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12], "ring": [13, 14, 15, 16],
           "pinky": [17, 18, 19, 20]}
ROBOT_POINTS = {f: [f"right_{f}_MCP_VL", f"right_{f}_MP", f"right_{f}_DP", f"right_{f}_fingertip"]
                for f in ("index", "middle", "ring", "pinky")}
ROBOT_POINTS["thumb"] = ["right_thumb_CMC_VL", "right_thumb_MCP_VL", "right_thumb_DP", "right_thumb_fingertip"]
POINT_WEIGHTS = np.array([0.6, 0.6, 1.0])  # PIP, DIP, tip (thumb: MCP, IP, tip)


def calib():
    c = json.loads((ROOT / "data/calib.json").read_text())
    f = c["f"]
    w, h = c["image_size"]
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    return K, R.from_rotvec(c["board_rvec"]).as_matrix(), np.asarray(c["board_tvec"])


def raw_to_portrait(px):
    """Pixels of the raw 1920 x 1080 landscape frame -> the displayed 1080 x 1920 portrait frame (90 deg clockwise)."""
    return np.stack([1079.0 - px[..., 1], px[..., 0]], -1)


def board_to_world():
    Rz = R.from_euler("z", BOARD_YAW_DEG, degrees=True).as_matrix()
    return Rz, BOARD_ORIGIN.copy()


def hand_frame(k):
    """Orthonormal hand frame from 21 keypoints (..., 21, 3) -> rotation (..., 3, 3) (columns x, y, z) and origin."""
    origin = k[..., [5, 9, 13, 17], :].mean(-2)
    z = k[..., 9, :] - k[..., 0, :]
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    y = k[..., 5, :] - k[..., 17, :]
    y -= (y * z).sum(-1, keepdims=True) * z
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    x = np.cross(y, z)
    return np.stack([x, y, z], -1), origin


# ----------------------------------------------------------------------------------------------- hand in 3D

def hand_in_board(hands, K, R_cb, t_cb):
    """-> keypoints in the board frame (T, 21, 3) with NaN where unusable, reprojection error (T,)"""
    J, px, valid = hands["joints"].astype(float), raw_to_portrait(hands["pixels"].astype(float)), hands["valid"]
    T = len(valid)
    out = np.full((T, 21, 3), np.nan)
    err = np.full(T, np.nan)
    prev = None
    for i in np.flatnonzero(valid):
        if not np.isfinite(J[i]).all():
            continue
        if prev is not None and i - prev[0] < 5:
            ok, r, t = True, prev[1].copy(), prev[2].copy()
            ok, r, t = cv2.solvePnP(J[i], px[i], K, None, r, t, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            ok, r, t = cv2.solvePnP(J[i], px[i], K, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        r, t = cv2.solvePnPRefineLM(J[i], px[i], K, None, r, t)
        proj = cv2.projectPoints(J[i], r, t, K, None)[0].reshape(-1, 2)
        err[i] = np.sqrt(np.mean(np.sum((proj - px[i]) ** 2, -1)))
        if err[i] > MAX_REPROJ_PX or t[2, 0] < 0.2:
            continue
        Pc = J[i] @ R.from_rotvec(r.ravel()).as_matrix().T + t.ravel()
        out[i] = (Pc - t_cb) @ R_cb  # R_cb^T (Pc - t_cb)
        prev = (i, r, t)
    # jumps: drop frames whose knuckle centre is far from the median of its neighbourhood (wrong hand, PnP flip)
    centre = out[:, [5, 9, 13, 17]].mean(1)
    for i in np.flatnonzero(np.isfinite(centre[:, 0])):
        nb = centre[max(0, i - 7): i + 8]
        nb = nb[np.isfinite(nb[:, 0])]
        if len(nb) >= 3 and np.linalg.norm(centre[i] - np.median(nb, 0)) > 0.06:
            out[i] = np.nan
    return out, err


def fill_and_smooth(k):
    """Interpolate gaps of up to MAX_GAP frames, Savitzky-Golay smooth every run. -> keypoints, filled mask"""
    T = len(k)
    ok = np.isfinite(k[:, 0, 0])
    k = k.copy()
    idx = np.flatnonzero(ok)
    filled = np.zeros(T, bool)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= MAX_GAP + 1:
            w = (np.arange(a + 1, b) - a) / (b - a)
            k[a + 1:b] = (1 - w)[:, None, None] * k[a] + w[:, None, None] * k[b]
            filled[a + 1:b] = True
    ok = np.isfinite(k[:, 0, 0])
    # runs
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s >= SMOOTH_WIN:
            k[s:e] = savgol_filter(k[s:e], SMOOTH_WIN, 2, axis=0)
    return k, filled


# ----------------------------------------------------------------------------------------------- robot model

class Robot:
    def __init__(self):
        self.model = render.build_model(0.03, TABLE_POS, TABLE_SIZE, 64, 64, 1.0)
        self.data = mujoco.MjData(self.model)
        teacher = np.load(ROOT / "data/teacher/nut_dense_rollouts.npz")
        self.names = [str(n) for n in teacher["joint_names"]]
        m = self.model
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.qadr = np.array([m.jnt_qposadr[jid(n)] for n in self.names])
        self.arm = np.array([m.jnt_qposadr[jid(f"panda_joint{i}")] for i in range(1, 8)])
        self.arm_dof = np.array([m.jnt_dofadr[jid(f"panda_joint{i}")] for i in range(1, 8)])
        self.hand_names = [n for n in self.names if n.startswith("right_")]
        self.hand = np.array([m.jnt_qposadr[jid(n)] for n in self.hand_names])
        rng = lambda adr: np.array([m.jnt_range[np.flatnonzero(m.jnt_qposadr == a)[0]] for a in adr])
        self.arm_lim, self.hand_lim = rng(self.arm), rng(self.hand)
        self.home = np.median(teacher["joint_pos"][:, 0], 0)  # teacher start posture, teacher joint order
        self.palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm")
        self.cmc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_hand_C_MC")
        self.pts = {f: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies] for f, bodies in ROBOT_POINTS.items()}
        # robot hand frame in the palm link, from the zero pose
        self.data.qpos[:] = 0
        mujoco.mj_kinematics(m, self.data)
        kp = np.zeros((21, 3))
        kp[0] = self.data.xpos[self.cmc]
        for f, ids in FINGERS.items():
            kp[ids] = self.data.xpos[self.pts[f]]
        Rh, oh = hand_frame(kp)
        Rp, op = self.data.xmat[self.palm].reshape(3, 3), self.data.xpos[self.palm]
        self.R_pk, self.p_pk = Rp.T @ Rh, Rp.T @ (oh - op)
        self.zero_kp = (kp - oh) @ Rh  # robot keypoints in its hand frame at the zero pose
        self.cmc_in_palm = Rp.T @ (kp[0] - op)

    def set_q(self, q29):
        self.data.qpos[self.qadr] = q29

    def hand_points(self, qhand):
        """Robot finger keypoints (21, 3) in the robot hand frame for the given hand joints (arm untouched)."""
        self.data.qpos[self.hand] = qhand
        mujoco.mj_kinematics(self.model, self.data)
        Rp, op = self.data.xmat[self.palm].reshape(3, 3), self.data.xpos[self.palm]
        Rh, oh = Rp @ self.R_pk, op + Rp @ self.p_pk
        kp = np.zeros((21, 3))
        for f, ids in FINGERS.items():
            kp[ids] = self.data.xpos[self.pts[f]]
        kp[0] = self.data.xpos[self.cmc]
        return (kp - oh) @ Rh

    def frame_pose(self):
        Rp, op = self.data.xmat[self.palm].reshape(3, 3), self.data.xpos[self.palm]
        return Rp @ self.R_pk, op + Rp @ self.p_pk


def finger_lengths(kp):
    """Per finger chain length MCP -> PIP -> DIP -> tip (thumb: CMC -> MCP -> IP -> tip), (..., 5)"""
    out = []
    for ids in FINGERS.values():
        seg = np.diff(kp[..., ids, :], axis=-2)
        out.append(np.linalg.norm(seg, axis=-1).sum(-1))
    return np.stack(out, -1)


def retarget_fingers(robot, human_local, scales, q0, q_prev):
    """human_local: (21, 3) human keypoints in the human hand frame. -> hand joints (22,), rms tip error [m]"""
    targets = {}
    for (f, ids), s in zip(FINGERS.items(), scales):
        targets[f] = (ids, s * (human_local[ids[1:]] - human_local[ids[0]]))
    thumb_tip = human_local[4]
    pinch = []
    for f in ("index", "middle", "ring", "pinky"):
        v = human_local[FINGERS[f][3]] - thumb_tip
        w = np.clip((0.09 - np.linalg.norm(v)) / 0.04, 0.0, 1.0)
        if w > 0:
            pinch.append((FINGERS[f][3], v, 1.5 * w))

    def residual(q):
        kp = robot.hand_points(q)
        res = []
        for f, (ids, tgt) in targets.items():
            res.append((POINT_WEIGHTS[:, None] * ((kp[ids[1:]] - kp[ids[0]]) - tgt)).ravel())
        for tip, v, w in pinch:
            res.append(w * ((kp[tip] - kp[4]) - v))
        res.append(0.01 * (q - q_prev))
        return np.concatenate(res)

    lo, hi = robot.hand_lim[:, 0], robot.hand_lim[:, 1]
    sol = least_squares(residual, np.clip(q0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), method="trf", diff_step=1e-3,
                        max_nfev=60, x_scale=1.0)
    kp = robot.hand_points(sol.x)
    tips = [FINGERS[f][3] for f in FINGERS]
    err = np.sqrt(np.mean(np.sum(((kp[tips] - kp[[FINGERS[f][0] for f in FINGERS]]) -
                                  np.array([targets[f][1][-1] for f in FINGERS])) ** 2, -1)))
    return sol.x, err


ROT_WEIGHT = 0.15  # m per rad in the IK objective
CONT_WEIGHT = 0.12  # m per rad of joint change from the previous frame (keeps joint speeds feasible)
# posture prior (weak): shoulder and forearm rotations near the teacher start posture, keeps the arm out of the
# shoulder singularity (joints 1 and 3 trading off when joint 2 is near zero)
POSTURE_WEIGHT = np.array([0.01, 0.005, 0.02, 0.0, 0.005, 0.0, 0.0])


def arm_ik(robot, R_t, p_t, q_arm, q_prev, cont=CONT_WEIGHT, max_nfev=50):
    """IK of the 7 arm joints as bounded least squares (MuJoCo Jacobians) so the robot hand frame reaches (R_t, p_t),
    with a continuity term to the previous frame and a weak posture prior. Hand joints must already be set."""
    m, d = robot.model, robot.data
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    lo, hi = robot.arm_lim[:, 0], robot.arm_lim[:, 1]
    home = robot.home[:7]

    def fk(q):
        d.qpos[robot.arm] = q
        mujoco.mj_kinematics(m, d)
        return robot.frame_pose()

    def residual(q):
        Rc, pc = fk(q)
        return np.r_[p_t - pc, ROT_WEIGHT * R.from_matrix(R_t @ Rc.T).as_rotvec(), cont * (q - q_prev),
                     POSTURE_WEIGHT * (q - home)]

    def jac(q):
        Rc, pc = fk(q)
        mujoco.mj_comPos(m, d)
        mujoco.mj_jac(m, d, jacp, jacr, pc, robot.palm)
        return np.r_[-jacp[:, robot.arm_dof], -ROT_WEIGHT * jacr[:, robot.arm_dof], cont * np.eye(7), np.diag(POSTURE_WEIGHT)]

    x0 = np.clip(q_arm, lo + 1e-4, hi - 1e-4)
    sol = least_squares(residual, x0, jac=jac, bounds=(lo, hi), method="trf", max_nfev=max_nfev, xtol=1e-6, ftol=1e-8)
    Rc, pc = fk(sol.x)
    return sol.x, np.linalg.norm(p_t - pc), np.linalg.norm(R.from_matrix(R_t @ Rc.T).as_rotvec())


def ik_cost(pos, rot, q, q_prev):
    return pos + 0.05 * rot + 0.01 * np.linalg.norm(q - q_prev)


def solve_arm(robot, R_t, p_t, q_prev, first):
    """Warm-started IK; when it is stuck (usually the wrist joint 7 at a limit during the screwing twist) restart from
    seeds with the wrist rotated and keep the best. -> q, position / orientation residual, whether it restarted"""
    if first:
        q, pe, re = arm_ik(robot, R_t, p_t, q_prev, q_prev, cont=0.0, max_nfev=300)
    else:
        q, pe, re = arm_ik(robot, R_t, p_t, q_prev, q_prev)
    if pe < 0.015 and re < 0.15:
        return q, pe, re, False
    best = (ik_cost(pe, re, q, q_prev), q, pe, re)
    seeds = []
    for j7 in (-2.2, -1.1, 0.0, 1.1, 2.2):
        for base in (q_prev, robot.home[:7]):
            s = base.copy()
            s[6] = j7
            seeds.append(s)
    for s in seeds:
        qs, ps, rs = arm_ik(robot, R_t, p_t, s, q_prev, cont=0.0, max_nfev=150)
        c = ik_cost(ps, rs, qs, q_prev)
        if c < best[0] - 0.005:
            best = (c, qs, ps, rs)
    return best[1], best[2], best[3], True


# ----------------------------------------------------------------------------------------------- nut

def nut_track(poses, T, Rw, ow):
    pos, quat, valid = poses["pos"][:T], poses["quat"][:T], poses["valid"][:T].astype(bool)
    n = len(valid)
    if n < T:
        pos = np.r_[pos, np.full((T - n, 3), np.nan)]
        quat = np.r_[quat, np.full((T - n, 4), np.nan)]
        valid = np.r_[valid, np.zeros(T - n, bool)]
    idx = np.flatnonzero(valid)
    t = np.arange(T)
    p = np.stack([np.interp(t, idx, pos[idx, a]) for a in range(3)], 1)
    rots = R.from_quat(quat[idx][:, [1, 2, 3, 0]])
    tc = np.clip(t, idx[0], idx[-1])
    rot = Slerp(idx, rots)(tc) if len(idx) > 1 else R.from_quat(np.repeat(rots.as_quat(), T, 0))
    Rw_ = R.from_matrix(Rw)
    pw = p @ Rw.T + ow
    qw = (Rw_ * rot).as_quat()[:, [3, 0, 1, 2]]
    return pw, qw, valid


# ----------------------------------------------------------------------------------------------- segment

def segment(hand_ok, centre_w, nut_w, nut_valid):
    """[start, end): from the first frame with the hand to the nut being at its final place on the bolt and the hand
    leaving it (or the last hand frame)."""
    seen = np.flatnonzero(hand_ok)
    start, last = seen[0], seen[-1] + 1
    vi = np.flatnonzero(nut_valid)
    final = np.median(nut_w[vi[-15:]], 0)
    near = np.linalg.norm(nut_w - final, axis=1) < 0.008
    # first valid nut frame after which every valid nut frame is near its final place
    bad = vi[~near[vi]]
    settled = (bad[-1] + 1) if len(bad) else vi[0]
    end = last
    for i in range(max(settled, start), last):
        if hand_ok[i] and np.linalg.norm(centre_w[i] - final) > 0.12:
            end = i + 1
            break
    return int(start), int(max(end, start + 1)), final, int(settled)


# ----------------------------------------------------------------------------------------------- main

def estimate_yaw(clips, K, R_cb, t_cb):
    """Median horizontal direction wrist -> knuckles while the hand is at the nut, in the board frame [deg]."""
    angles = []
    for c in clips:
        h, p = np.load(ROOT / f"data/hands/{c}.npz"), np.load(ROOT / f"data/poses/{c}.npz")
        k, _ = hand_in_board(h, K, R_cb, t_cb)
        T = min(len(k), len(p["valid"]))
        for i in range(T):
            if np.isfinite(k[i, 0, 0]) and p["valid"][i]:
                pinch = k[i, [4, 8]].mean(0)
                if np.linalg.norm(pinch - p["pos"][i]) < 0.05:
                    d = k[i, 9] - k[i, 0]
                    angles.append(np.arctan2(d[1], d[0]))
        print(f"[yaw] {c}: {len(angles)} grasp frames so far")
    a = np.array(angles)
    mean = np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))
    print(f"[yaw] forearm direction in the board frame: circular mean {mean:.1f} deg "
          f"(spread {np.degrees(np.sqrt(-2 * np.log(np.hypot(np.sin(a).mean(), np.cos(a).mean())))):.0f} deg); "
          f"sim yaw that maps it to +x: {-mean:.1f} deg")


def process(clip, robot, K, R_cb, t_cb, out_dir):
    hands = np.load(ROOT / f"data/hands/{clip}.npz")
    poses = np.load(ROOT / f"data/poses/{clip}.npz")
    T = len(hands["valid"])
    kb, reproj = hand_in_board(hands, K, R_cb, t_cb)
    measured = np.isfinite(kb[:, 0, 0])
    kb, filled = fill_and_smooth(kb)
    hand_ok = np.isfinite(kb[:, 0, 0])
    Rw, ow = board_to_world()
    kw = kb @ Rw.T + ow
    nut_w, nut_q, nut_valid = nut_track(poses, T, Rw, ow)
    if hand_ok.sum() < 30 or nut_valid.sum() < 10:
        print(f"[retarget] {clip}: too little data (hand {hand_ok.sum()}, nut {nut_valid.sum()} frames), skipped")
        return None
    with np.errstate(invalid="ignore"):
        Rh, oh = hand_frame(kw)
    centre = oh
    start, end, final, settled = segment(hand_ok, centre, nut_w, nut_valid)

    # finger scale: robot / human finger chain lengths (median over the clip)
    human_len = np.nanmedian(finger_lengths(kb[hand_ok]), 0)
    robot_len = finger_lengths(robot.zero_kp)
    scales = robot_len / human_len

    joint_pos = np.full((T, 29), np.nan)
    pos_res, rot_res, fin_res = np.full(T, np.nan), np.full(T, np.nan), np.full(T, np.nan)
    q_hand = robot.home[[robot.names.index(n) for n in robot.hand_names]].copy()
    q_arm = robot.home[:7].copy()
    first = True
    restarted = np.zeros(T, bool)
    robot.data.qpos[:] = 0
    for i in range(start, end):
        if not hand_ok[i]:
            continue
        local = (kw[i] - oh[i]) @ Rh[i]
        q_hand, fin_res[i] = retarget_fingers(robot, local, scales, q_hand, q_hand)
        robot.data.qpos[robot.hand] = q_hand
        q_arm, pos_res[i], rot_res[i], restarted[i] = solve_arm(robot, Rh[i], oh[i], q_arm, first)
        first = False
        q = np.zeros(29)
        q[:7] = q_arm
        q[[robot.names.index(n) for n in robot.hand_names]] = q_hand
        joint_pos[i] = q
    # long gaps inside the segment: joint-space interpolation (flagged by hand_valid = False)
    seg = np.arange(start, end)
    have = seg[np.isfinite(joint_pos[seg, 0])]
    for j in range(29):
        joint_pos[seg, j] = np.interp(seg, have, joint_pos[have, j])
    # joint speed limit (IK restarts and near-singular stretches can jump): the robot lags and catches up
    ik_joint_pos = joint_pos.copy()
    for t in seg[1:]:
        step = np.r_[ARM_MAX_STEP, np.full(22, HAND_MAX_STEP)]
        joint_pos[t] = joint_pos[t - 1] + np.clip(joint_pos[t] - joint_pos[t - 1], -step, step)
    for i in seg:  # residuals of what is actually commanded
        robot.data.qpos[robot.qadr] = joint_pos[i]
        mujoco.mj_kinematics(robot.model, robot.data)
        Rc, pc = robot.frame_pose()
        if hand_ok[i]:
            pos_res[i] = np.linalg.norm(oh[i] - pc)
            rot_res[i] = np.linalg.norm(R.from_matrix(Rh[i] @ Rc.T).as_rotvec())
    hand_valid = measured.copy()
    gaps = np.diff(np.r_[have, end - 1])
    # contact check while the nut is carried (moved from its start, not yet settled): distance of the thumb/index
    # tip midpoint to the nut centre, human (tracked) and robot (retargeted)
    vi = np.flatnonzero(nut_valid)
    carried = nut_valid & (np.linalg.norm(nut_w - nut_w[vi[0]], axis=1) > 0.02) & (np.arange(T) < settled)
    carried[:start], carried[end:] = False, False
    robot_pinch = np.full((T, 3), np.nan)
    tip_ids = [mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, f"right_{f}_fingertip") for f in ("thumb", "index")]
    for i in seg:
        robot.data.qpos[robot.qadr] = joint_pos[i]
        mujoco.mj_kinematics(robot.model, robot.data)
        robot_pinch[i] = robot.data.xpos[tip_ids].mean(0)
    human_nut = np.linalg.norm(kw[:, [4, 8]].mean(1) - nut_w, axis=1)
    robot_nut = np.linalg.norm(robot_pinch - nut_w, axis=1)
    sel = carried & hand_valid
    dq = np.abs(np.diff(ik_joint_pos[seg, :7], axis=0)).max(1)
    stats = dict(frames=end - start, hand_measured=float(measured[seg].mean()), hand_usable=float(hand_ok[seg].mean()),
                 longest_gap=int(max(gaps.max(), 1) - 1), pos_res_mm=1e3 * np.nanmedian(pos_res[seg]),
                 pos_res_p90_mm=1e3 * np.nanpercentile(pos_res[seg], 90), rot_res_deg=np.degrees(np.nanmedian(rot_res[seg])),
                 rot_res_p90_deg=np.degrees(np.nanpercentile(rot_res[seg], 90)), finger_res_mm=1e3 * np.nanmedian(fin_res[seg]),
                 reproj_px=float(np.nanmedian(reproj)), ik_restarts=int(restarted.sum()), arm_jumps=int((dq > 0.25).sum()),
                 max_arm_step_rad=float(dq.max()), carried_frames=int(sel.sum()),
                 human_pinch_nut_mm=1e3 * float(np.median(human_nut[sel])) if sel.any() else None,
                 robot_pinch_nut_mm=1e3 * float(np.median(robot_nut[sel])) if sel.any() else None)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{clip}.npz", joint_pos=joint_pos, joint_names=np.array(robot.names), hand_valid=hand_valid,
                        hand_usable=hand_ok, hand_filled=filled, keypoints_world=kw, nut_pos=nut_w, nut_quat=nut_q,
                        nut_valid=nut_valid, goal_pos=final, seg_start=start, seg_end=end, nut_settled=settled,
                        pos_res=pos_res, rot_res=rot_res, finger_res=fin_res, reproj_px=reproj, finger_scale=scales,
                        human_pinch_nut=human_nut, robot_pinch_nut=robot_nut, carried=carried, ik_restarted=restarted, ik_joint_pos=ik_joint_pos,
                        board_yaw_deg=BOARD_YAW_DEG, board_origin=BOARD_ORIGIN, stats=json.dumps(stats))
    print(f"[retarget] {clip}: segment {start}-{end} ({(end - start) / FPS:.1f} s), hand measured {stats['hand_measured']:.0%} "
          f"usable {stats['hand_usable']:.0%}, longest gap {stats['longest_gap']} fr, palm pos res {stats['pos_res_mm']:.1f} mm "
          f"(p90 {stats['pos_res_p90_mm']:.1f}), rot res {stats['rot_res_deg']:.1f} deg (p90 {stats['rot_res_p90_deg']:.1f}), "
          f"finger tip res {stats['finger_res_mm']:.1f} mm,  IK restarts {stats['ik_restarts']}, raw IK jumps >0.25 rad "
          f"{stats['arm_jumps']} (max {stats['max_arm_step_rad']:.2f}), reproj {stats['reproj_px']:.1f} px, carried {sel.sum()} fr: "
          f"pinch-nut human {stats['human_pinch_nut_mm'] or float('nan'):.0f} mm robot "
          f"{stats['robot_pinch_nut_mm'] or float('nan'):.0f} mm, finger scale "
          f"{np.round(scales, 2).tolist()}", flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="*")
    parser.add_argument("--estimate_yaw", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "data/human")
    args = parser.parse_args()
    clips = args.clips or sorted(Path(p).stem for p in glob.glob(str(ROOT / "data/hands/*.npz")))
    K, R_cb, t_cb = calib()
    if args.estimate_yaw:
        estimate_yaw(clips, K, R_cb, t_cb)
        return
    robot = Robot()
    allstats = {}
    for c in clips:
        s = process(c, robot, K, R_cb, t_cb, args.out)
        if s is not None:
            allstats[c] = s
    (args.out / "summary.json").write_text(json.dumps(allstats, indent=1))


if __name__ == "__main__":
    main()
