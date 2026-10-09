"""Self-calibrate the phone camera from the videos themselves (no checkerboard).

The camera never moves across the 50 clips, so the first frames of every clip
see the same table marker and the cube resting at 50 different places. One
bundle adjustment over all of them recovers the focal length, the camera pose
and the cube dimensions.

Metric scale comes from the A4 sheet the table marker is printed on.
The cube height above the table is not observable from one view (a bigger cube
further down the ray looks the same), so the ratio cube edge / front marker is
measured once in the image and fixed.

    python -m h2f.calibrate
"""
import glob

import av
import cv2
import numpy as np
from scipy.optimize import least_squares

from .camera import (H, ID_CUBE_FRONT, ID_CUBE_TOP, ID_TABLE, W, Calib, aruco_detector,
                     cube_marker_points, detect, square)

A4 = (0.210, 0.297)
EDGE_OVER_FRONT_MARKER = 1.18  # measured in the image: cube edge / front marker side


def first_frames(path, n=10):
    with av.open(path) as c:
        for i, frame in enumerate(c.decode(video=0)):
            if i >= n:
                return
            yield frame.to_ndarray(format="bgr24")


def table_marker_size(bgr, q):
    """Side of marker 13 from the A4 sheet outline, both expressed in the marker plane."""
    _, mask = cv2.threshold(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 150, 255, cv2.THRESH_BINARY)
    cv2.fillConvexPoly(mask, q.astype(np.int32), 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sheet = [c for c in contours if cv2.pointPolygonTest(c, tuple(map(float, q.mean(0))), False) > 0]
    if not sheet:
        return None
    quad = cv2.approxPolyDP(sheet[0], 0.02 * cv2.arcLength(sheet[0], True), True)
    if len(quad) != 4:
        return None
    to_marker = cv2.getPerspectiveTransform(q.astype(np.float32), np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]))
    p = cv2.perspectiveTransform(quad.astype(np.float32), to_marker)[:, 0]
    e = np.linalg.norm(p - np.roll(p, 1, 0), axis=1)
    short, long = sorted([e[[0, 2]].mean(), e[[1, 3]].mean()])
    return (A4[0] / short + A4[1] / long) / 2


def collect(videos):
    """Median corner positions over the first frames of each clip (cube still at rest)."""
    detector, table, cubes, sizes = aruco_detector(), [], [], []
    for path in videos:
        acc = {}
        for bgr in first_frames(path):
            found = detect(detector, bgr)
            for k, q in found.items():
                acc.setdefault(k, []).append(q)
            if ID_TABLE in found and (s := table_marker_size(bgr, found[ID_TABLE])):
                sizes.append(s)
        med = {k: np.median(v, 0) for k, v in acc.items() if len(v) >= 3}
        if ID_TABLE in med:
            table.append(med[ID_TABLE])
        if ID_CUBE_TOP in med and ID_CUBE_FRONT in med:
            cubes.append(np.r_[med[ID_CUBE_TOP], med[ID_CUBE_FRONT]])
    return np.median(table, 0), np.array(cubes), float(np.median(sizes))


def project(P, f, rvec, tvec):
    pc = P @ cv2.Rodrigues(rvec)[0].T + tvec
    return pc[..., :2] / pc[..., 2:] * f + [W / 2, H / 2]


def residuals(x, q_table, q_cubes, marker):
    f, rvec, tvec, top, front, off, pose = x[0], x[1:4], x[4:7], x[7], x[8], x[9:12], x[12:].reshape(-1, 3)
    C = cube_marker_points(EDGE_OVER_FRONT_MARKER * front, top, front, off[:2], off[2])
    c, s = np.cos(pose[:, 2:]), np.sin(pose[:, 2:])
    P = np.stack([C[:, 0] * c - C[:, 1] * s + pose[:, :1],
                  C[:, 0] * s + C[:, 1] * c + pose[:, 1:2],
                  np.broadcast_to(C[:, 2], c.shape[:1] + C.shape[:1])], -1)
    return np.concatenate([(project(P, f, rvec, tvec) - q_cubes).ravel(),
                           3 * (project(square(marker), f, rvec, tvec) - q_table).ravel()])


def calibrate(videos, f0=1600.0):
    q_table, q_cubes, marker = collect(videos)
    K = np.array([[f0, 0, W / 2], [0, f0, H / 2], [0, 0, 1.0]])
    _, rvec, tvec = cv2.solvePnP(square(marker), q_table, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    init = Calib(f0, rvec.ravel().tolist(), tvec.ravel().tolist(), marker, 0.055, 0.04, 0.046, [0, 0], 0)
    poses = []
    for q in q_cubes:  # initial cube pose: top face corners dropped onto the plane z = cube edge
        t = init.on_plane(q[:4], init.cube)
        poses.append([*t[:, :2].mean(0), np.arctan2(*(t[1] - t[0])[[1, 0]])])
    x0 = np.r_[f0, rvec.ravel(), tvec.ravel(), init.top_marker, init.front_marker, 0, 0, 0, np.ravel(poses)]
    sol = least_squares(residuals, x0, loss="soft_l1", args=(q_table, q_cubes, marker))
    x = sol.x
    rms = float(np.sqrt(np.mean(sol.fun ** 2)))
    return Calib(x[0], x[1:4].tolist(), x[4:7].tolist(), marker, EDGE_OVER_FRONT_MARKER * x[8],
                 x[7], x[8], x[9:11].tolist(), x[11]), rms, len(q_cubes)


if __name__ == "__main__":
    calib, rms, n = calibrate(sorted(glob.glob("data/raw/*.MOV")))
    calib.save()
    print(f"{n} clips, reprojection rms {rms:.2f} px")
    print(f"f = {calib.f:.0f} px, camera at {np.round(calib.cam_pos, 3)} m in the table frame")
    print(f"table marker {calib.table_marker * 1e3:.1f} mm, cube {calib.cube * 1e3:.1f} mm, "
          f"top/front markers {calib.top_marker * 1e3:.1f}/{calib.front_marker * 1e3:.1f} mm")
