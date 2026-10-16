"""Self-calibrate the phone camera from the clips and track the nut's 6D pose in the table-board frame.

Input: ``data/markers/*.npz`` from ``scripts/detect_markers.py``. The phone did not move between clips (the board
markers stay within 3 px), so all clips share one camera.

1. Calibration (no checkerboard, as in hand2franka's ``h2f/calibrate.py``): one bundle adjustment over
       focal length (principal point at the image centre, no distortion: the phone corrects its lens),
       board pose in the camera,
       the in-plane offset of each nut marker from the centre of its face (hand-cut and hand-stuck labels),
       the nut pose in a few hundred frames that see two or more nut markers.
   The board alone cannot fix the focal length from a nearly overhead view (focal length and distance trade off);
   the nut markers on faces at other angles can.
2. Tracking: every frame with a nut marker gets a PnP pose. With one marker the planar pose is two-fold ambiguous;
   the solution closer to the previous frame's pose wins (or, at the start, the one with the nut above the table).

    python scripts/track_nut.py

Writes data/calib.json and data/poses/<clip>.npz (time, position [m], quaternion wxyz, valid, markers, error [px]).
"""

import glob
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = json.loads((ROOT / "assets/markers/layout.json").read_text())
TABLE = {int(k): np.array(v) for k, v in LAYOUT["table"].items()}
NUT = {int(k): np.array(v) for k, v in LAYOUT["nut"].items()}
F0 = 1788.0  # same phone as hand2franka, its self-calibrated focal length at 1080p
MAX_CALIB_FRAMES = 400
MAX_ERROR_PX = 4.0


def load(path):
    d = np.load(path)
    off = d["offsets"]
    frames = [(d["ids"][off[i]:off[i + 1]], d["corners"][off[i]:off[i + 1]]) for i in range(len(off) - 1)]
    return frames, tuple(d["image_size"]), float(d["fps"])


def nut_face_axes(marker_id):
    """Right and up unit vectors of a nut marker in the nut frame (see make_markers.py)."""
    c = NUT[marker_id]
    right = (c[1] - c[0]) / np.linalg.norm(c[1] - c[0])
    up = (c[0] - c[3]) / np.linalg.norm(c[0] - c[3])
    return right, up


def nut_points(marker_id, offsets):
    right, up = nut_face_axes(marker_id)
    du, dv = offsets[marker_id]
    return NUT[marker_id] + du * right + dv * up


def K(f, size):
    return np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])


def project(points, f, rvec, tvec, size):
    return cv2.projectPoints(np.asarray(points, float), np.asarray(rvec, float), np.asarray(tvec, float), K(f, size), None)[0].reshape(-1, 2)


def calibrate(clips, size):
    # board: median corner pixels per marker over all frames of all clips (the camera never moved)
    board_obs = {}
    for frames in clips.values():
        for ids, corners in frames:
            for m, c in zip(ids, corners):
                if m in TABLE:
                    board_obs.setdefault(m, []).append(c)
    board_ids = sorted(board_obs)
    P_board = np.concatenate([TABLE[m] for m in board_ids])
    q_board = np.concatenate([np.median(np.stack(board_obs[m]), axis=0) for m in board_ids])

    # nut frames with at least two markers, spread over all clips
    candidates = [(ids, corners) for frames in clips.values() for ids, corners in frames if sum(m in NUT for m in ids) >= 2]
    pick = np.linspace(0, len(candidates) - 1, min(MAX_CALIB_FRAMES, len(candidates))).astype(int)
    nut_frames = []
    zero = {m: (0.0, 0.0) for m in NUT}
    for i in pick:
        ids, corners = candidates[i]
        sel = [k for k, m in enumerate(ids) if m in NUT]
        nut_frames.append(([int(ids[k]) for k in sel], np.concatenate([corners[k] for k in sel])))

    _, rb, tb = cv2.solvePnP(P_board, q_board, K(F0, size), None)
    poses0 = []
    for ids, q in nut_frames:
        P = np.concatenate([nut_points(m, zero) for m in ids])
        _, r, t = cv2.solvePnP(P, q, K(F0, size), None, flags=cv2.SOLVEPNP_SQPNP)
        poses0.append(np.r_[r.ravel(), t.ravel()])
    nut_ids = sorted(NUT)
    n_fixed = 1 + 6 + 2 * len(nut_ids)

    def unpack(x):
        offsets = {m: (x[7 + 2 * j], x[8 + 2 * j]) for j, m in enumerate(nut_ids)}
        return x[0], x[1:4], x[4:7], offsets, x[n_fixed:].reshape(-1, 6)

    def residuals(x):
        f, rb_, tb_, offsets, poses = unpack(x)
        res = [3.0 * (project(P_board, f, rb_, tb_, size) - q_board).ravel()]
        for (ids, q), pose in zip(nut_frames, poses):
            P = np.concatenate([nut_points(m, offsets) for m in ids])
            res.append((project(P, f, pose[:3], pose[3:], size) - q).ravel())
        # keep the label offsets small: a few millimetres at most
        res.append(np.r_[[x[7 + j] for j in range(2 * len(nut_ids))]] / 0.002)
        return np.concatenate(res)

    x0 = np.r_[F0, rb.ravel(), tb.ravel(), np.zeros(2 * len(nut_ids)), np.ravel(poses0)]
    # sparse Jacobian: every residual depends on the shared block and on its own frame's pose
    n_res = len(residuals(x0))
    sparsity = lil_matrix((n_res, len(x0)), dtype=int)
    row = 2 * len(q_board)
    sparsity[:row, :7] = 1
    for k, (ids, q) in enumerate(nut_frames):
        rows = slice(row, row + 2 * len(q))
        sparsity[rows, 0] = 1
        sparsity[rows, 7:n_fixed] = 1
        sparsity[rows, n_fixed + 6 * k:n_fixed + 6 * k + 6] = 1
        row += 2 * len(q)
    sparsity[row:, 7:n_fixed] = 1
    sol = least_squares(residuals, x0, loss="soft_l1", f_scale=2.0, jac_sparsity=sparsity, x_scale="jac")
    f, rb, tb, offsets, poses = unpack(sol.x)

    r = residuals(sol.x)
    board_rms = np.sqrt(np.mean((r[: 2 * len(q_board)] / 3.0) ** 2))
    nut_rms = np.sqrt(np.mean(r[2 * len(q_board): -2 * len(nut_ids)] ** 2))
    cam_pos = -R.from_rotvec(rb).as_matrix().T @ tb
    calib = {
        "f": float(f),
        "image_size": [int(size[0]), int(size[1])],
        "board_rvec": rb.tolist(),
        "board_tvec": tb.tolist(),
        "nut_marker_offsets": {str(m): [float(a), float(b)] for m, (a, b) in offsets.items()},
        "board_rms_px": float(board_rms),
        "nut_rms_px": float(nut_rms),
        "frames": len(nut_frames),
    }
    print(f"[calib] f = {f:.0f} px (start {F0:.0f}), {len(nut_frames)} nut frames, rms board {board_rms:.2f} px, "
          f"nut {nut_rms:.2f} px")
    print(f"[calib] camera at {np.round(cam_pos, 3).tolist()} m in the board frame")
    print("[calib] nut label offsets [mm]:", {m: (round(1e3 * a, 1), round(1e3 * b, 1)) for m, (a, b) in offsets.items()})
    return calib


def track(frames, calib, size):
    f = calib["f"]
    offsets = {int(m): v for m, v in calib["nut_marker_offsets"].items()}
    R_cb = R.from_rotvec(calib["board_rvec"]).as_matrix()
    t_cb = np.asarray(calib["board_tvec"])
    Kmat = K(f, size)
    out = {k: [] for k in ("pos", "quat", "valid", "markers", "error")}
    prev = None
    for ids, corners in frames:
        sel = [k for k, m in enumerate(ids) if m in NUT]
        pose = None
        if sel:
            P = np.concatenate([nut_points(int(ids[k]), offsets) for k in sel])
            q = np.concatenate([corners[k] for k in sel]).astype(float)
            if len(sel) >= 2:
                ok, r, t = cv2.solvePnP(P, q, Kmat, None, flags=cv2.SOLVEPNP_SQPNP)
                candidates = [(r, t)] if ok else []
            else:
                n, rs, ts, _ = cv2.solvePnPGeneric(P, q, Kmat, None, flags=cv2.SOLVEPNP_IPPE)
                candidates = list(zip(rs, ts))
            best = None
            for r, t in candidates:
                r, t = cv2.solvePnPRefineLM(P, q, Kmat, None, r.copy(), t.copy())
                err = np.sqrt(np.mean((project(P, f, r, t, size) - q) ** 2))
                # nut pose in the board frame
                R_cn = R.from_rotvec(r.ravel()).as_matrix()
                Rb = R_cb.T @ R_cn
                pb = R_cb.T @ (t.ravel() - t_cb)
                if prev is not None:
                    cost = np.linalg.norm(pb - prev[0]) / 0.02 + (R.from_matrix(Rb) * R.from_matrix(prev[1]).inv()).magnitude()
                else:
                    cost = 0.0 if pb[2] > 0.0 else 10.0  # above the table
                cost += err / MAX_ERROR_PX
                if best is None or cost < best[0]:
                    best = (cost, pb, Rb, err)
            if best is not None and best[3] < MAX_ERROR_PX:
                pose = best
        if pose is None:
            out["pos"].append(np.full(3, np.nan))
            out["quat"].append(np.full(4, np.nan))
            out["valid"].append(False)
            out["markers"].append(len(sel))
            out["error"].append(np.nan)
            continue
        _, pb, Rb, err = pose
        q_xyzw = R.from_matrix(Rb).as_quat()
        out["pos"].append(pb)
        out["quat"].append(np.r_[q_xyzw[3], q_xyzw[:3]])
        out["valid"].append(True)
        out["markers"].append(len(sel))
        out["error"].append(err)
        prev = (pb, Rb)
    return {k: np.asarray(v) for k, v in out.items()}


def main():
    paths = sorted(glob.glob(str(ROOT / "data/markers/*.npz")))
    clips, size, fps = {}, None, None
    for p in paths:
        clips[Path(p).stem], size, fps = load(p)
    calib = calibrate(clips, size)
    (ROOT / "data/calib.json").write_text(json.dumps(calib, indent=1))
    out_dir = ROOT / "data/poses"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frames in clips.items():
        tr = track(frames, calib, size)
        np.savez_compressed(out_dir / f"{name}.npz", t=np.arange(len(frames)) / fps, **tr)
        v = tr["valid"]
        z = tr["pos"][v, 2] if v.any() else np.array([np.nan])
        print(f"[track] {name}: pose in {v.mean():5.1%} of {len(frames)} frames, median error "
              f"{np.nanmedian(tr['error']):.2f} px, nut height {1e3 * np.nanmin(z):5.1f}..{1e3 * np.nanmax(z):5.1f} mm")


if __name__ == "__main__":
    main()
