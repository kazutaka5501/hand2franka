"""Phone video -> metric hand and cube tracks in the table frame.

Cube and plate rest on the table, so their pixels are simply intersected with
a horizontal plane. The hand comes as 21 joints in its own frame, from MANO
parameters regressed by WiLoR (`h2f.hand_mano`, run first) or, without them,
from MediaPipe. PnP registers the joints to the image, which gives a 3D hand
up to one global scale. That scale is fixed by contact: at grasp and release
the pinch point must sit at the cube centre.

    python -m h2f.perception            # all clips -> data/tracks/*.npz
"""
import glob
from pathlib import Path

import av
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from scipy.ndimage import gaussian_filter1d, median_filter

from .camera import H, ID_CUBE_FRONT, ID_CUBE_TOP, ID_HAND, ID_TABLE, W, Calib, aruco_detector, detect

HAND_MODEL = "assets/hand_landmarker.task"
THUMB_TIP, INDEX_TIP = 4, 8  # MediaPipe landmark indices
PLATE_RIM = 0.015  # height of the paper plate rim [m]
TRACK_DIR = Path("data/tracks")
HAND_DIR = Path("data/hands")  # MANO joints from h2f.hand_mano; MediaPipe is used for clips without them
REST_FRAMES = 5  # the cube is untouched in the first frames and settled in the last ones


def hand_tracker():
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3, min_tracking_confidence=0.3))


def register(joints, px, K, max_err=40.0):
    """PnP of 21 hand joints (metres, hand frame) onto their pixels -> (21, 3) camera-frame points, unknown scale."""
    tips = px[[THUMB_TIP, INDEX_TIP]]
    if not np.isfinite(px).all() or (tips < 0).any() or (tips[:, 0] > W).any() or (tips[:, 1] > H).any():
        return None, None
    ok, rvec, tvec = cv2.solvePnP(joints.astype(np.float64), px.astype(np.float64), K, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None, None
    pc = joints @ cv2.Rodrigues(rvec)[0].T + tvec.ravel()
    err = np.linalg.norm(pc[:, :2] / pc[:, 2:] * K[0, 0] + K[:2, 2] - px, axis=1).mean()
    return (pc, px) if err < max_err and (pc[:, 2] > 0).all() else (None, None)


def hand_in_camera(result, K):
    """MediaPipe result -> registered hand."""
    if not result.hand_landmarks:
        return None, None
    px = np.array([[p.x * W, p.y * H] for p in result.hand_landmarks[0]])
    return register(np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]]), px, K)


def cube_on_table(calib, markers):
    """(x, y, yaw) of the cube resting on the table, from its top marker or else its front marker."""
    if ID_CUBE_TOP in markers:
        p = calib.on_plane(markers[ID_CUBE_TOP], calib.cube)
        centre, edge = p.mean(0), p[1] - p[0]
    elif ID_CUBE_FRONT in markers:
        p = calib.on_plane(markers[ID_CUBE_FRONT][[3, 2]], (calib.cube - calib.front_marker) / 2)
        edge = p[1] - p[0]
        centre = p.mean(0) + calib.cube / 2 * np.array([-edge[1], edge[0], 0]) / np.linalg.norm(edge)
    else:
        return None
    return np.r_[centre[:2], np.arctan2(edge[1], edge[0])]


def rest_corners(frames):
    """Median pixel corners of every cube marker over a few frames in which the cube is at rest."""
    acc = {}
    for markers in frames:
        for k, q in markers.items():
            if k not in (ID_HAND, ID_TABLE):
                acc.setdefault(k, []).append(q)
    return {k: np.median(v, 0) for k, v in acc.items()}


def at_rest(markers, ref, tol=4.0):
    return any(k in ref and np.abs(markers[k] - ref[k]).max() < tol for k in markers)


def find_plate(calib, bgr, table_marker_px):
    """Circle fit of the white plate outline, in table coordinates -> (x, y, radius)."""
    _, mask = cv2.threshold(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 150, 255, cv2.THRESH_BINARY)
    mask[: H // 6] = 0  # bright clutter behind the table
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = None
    for c in contours:
        if cv2.contourArea(c) < 2e4 or cv2.pointPolygonTest(c, tuple(map(float, table_marker_px.mean(0))), False) > 0:
            continue
        p = calib.on_plane(cv2.convexHull(c)[:, 0], PLATE_RIM)[:, :2]
        A = np.c_[2 * p, np.ones(len(p))]
        (cx, cy, k), *_ = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)
        r = np.sqrt(k + cx ** 2 + cy ** 2)
        fit = np.abs(np.linalg.norm(p - [cx, cy], axis=1) - r).mean() / r
        if best is None or fit < best[0]:
            best = (fit, np.array([cx, cy, r]))
    return best[1]


def track_video(path, calib, mano=None):
    """Raw per-frame measurements for one clip. `mano`: output of h2f.hand_mano for this clip, used
    instead of MediaPipe when given."""
    detector, tracker = aruco_detector(), None if mano is not None else hand_tracker()
    hand, hand_px, cube_obs = [], [], []
    with av.open(path) as container:
        for i, frame in enumerate(container.decode(video=0)):
            rgb = frame.to_ndarray(format="rgb24")
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if mano is not None:
                pc, px = register(mano["joints"][i], mano["pixels"][i], calib.K) if mano["valid"][i] else (None, None)
            else:
                result = tracker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / 30))
                pc, px = hand_in_camera(result, calib.K)
            hand.append(pc)
            hand_px.append(px)
            cube_obs.append(detect(detector, bgr))
            if i == 0:
                first = bgr
    if tracker is not None:
        tracker.close()
    T = len(hand)
    start = [c for m in cube_obs[:REST_FRAMES] if (c := cube_on_table(calib, m)) is not None]
    end = [c for m in cube_obs[-3 * REST_FRAMES:] if (c := cube_on_table(calib, m)) is not None]
    table_px = next(m[ID_TABLE] for m in cube_obs if ID_TABLE in m)
    return dict(
        name=Path(path).stem, T=T,
        valid=np.array([h is not None for h in hand]),
        hand=np.array([h if h is not None else np.full((21, 3), np.nan) for h in hand]),
        hand_px=np.array([p if p is not None else np.full((21, 2), np.nan) for p in hand_px]),
        cube_start=np.median(start, 0), cube_end=np.median(end, 0),
        plate=find_plate(calib, first, table_px),
        marker0=np.array([m[ID_HAND] if ID_HAND in m else np.full((4, 2), np.nan) for m in cube_obs]),
        markers=cube_obs,
    )


def find_contacts(tr, calib):
    """Grasp / release frames.

    The cube markers bracket the transport (last frame seen at the start pose, first frame seen at
    the end pose); inside a short window around each, take the closest approach of the pinch point.
    """
    obs = tr["markers"]
    start_ref, end_ref = rest_corners(obs[:REST_FRAMES]), rest_corners(obs[-2 * REST_FRAMES:])
    left = max(t for t, m in enumerate(obs) if at_rest(m, start_ref))
    arrived = min(t for t, m in enumerate(obs) if t > left and at_rest(m, end_ref))
    pinch = np.nanmean(tr["hand_px"][:, [THUMB_TIP, INDEX_TIP]], axis=1)
    t = np.arange(tr["T"])

    def closest(cube, lo, hi):
        d = np.linalg.norm(pinch - calib.project(np.r_[cube[:2], calib.cube / 2]), axis=1)
        d[(t < lo) | (t > hi)] = np.nan
        return int(np.nanargmin(d)) if np.isfinite(d).any() else int(np.clip((lo + hi) // 2, 0, tr["T"] - 1))

    return closest(tr["cube_start"], left - 5, left + 10), closest(tr["cube_end"], arrived - 15, arrived)


def hand_scale(tracks, calib):
    """One scale for the MediaPipe hand such that pinch points hit the cube at grasp and release."""
    num = den = 0.0
    for tr in tracks:
        for t, cube in ((tr["t_grasp"], tr["cube_start"]), (tr["t_release"], tr["cube_end"])):
            pinch = tr["hand"][t, [THUMB_TIP, INDEX_TIP]].mean(0)  # camera frame, unit scale
            target = np.r_[cube[:2], calib.cube / 2] @ calib.R.T + calib.tvec  # cube centre, camera frame
            num += pinch @ target
            den += pinch @ pinch
    return num / den


def to_table(tr, calib, scale, smooth=1.5, smooth_depth=4.0):
    """Scaled, gap-filled, smoothed hand keypoints (T, 21, 3) in the table frame.

    Depth from the hand's apparent size is far noisier than its image position, so the per-frame
    depth of the hand is low-pass filtered much harder than the keypoints themselves.
    """
    t, ok = np.arange(tr["T"]), tr["valid"]
    depth = np.interp(t, t[ok], median_filter(tr["hand"][ok, :, 2].mean(1), 5, mode="nearest"))
    depth_smooth = gaussian_filter1d(depth, smooth_depth, mode="nearest")
    pts = np.empty((tr["T"], 21, 3))
    for j in range(21):
        for a in range(3):
            pts[:, j, a] = np.interp(t, t[ok], tr["hand"][ok, j, a])
    pts = gaussian_filter1d(pts * (depth_smooth / depth)[:, None, None], smooth, axis=0, mode="nearest")
    return calib.cam_to_table(pts * scale)


def anchor_to_contacts(pts, tr, calib):
    """Per-clip depth correction: slide the hand along the camera rays so the pinch point meets the
    cube at grasp and at release (interpolated in between), then remove the small lateral residual."""
    cam, t = calib.cam_pos, np.arange(tr["T"])
    frames = [tr["t_grasp"], tr["t_release"]]
    cubes = np.array([np.r_[tr["cube_start"][:2], calib.cube / 2], np.r_[tr["cube_end"][:2], calib.cube / 2]])
    pinch = pts[frames][:, [THUMB_TIP, INDEX_TIP]].mean(1)
    gain = np.linalg.norm(cubes - cam, axis=1) / np.linalg.norm(pinch - cam, axis=1)
    pts = cam + (pts - cam) * np.interp(t, frames, gain)[:, None, None]
    residual = cubes - pts[frames][:, [THUMB_TIP, INDEX_TIP]].mean(1)
    return pts + np.stack([np.interp(t, frames, residual[:, a]) for a in range(3)], 1)[:, None]


def main():
    calib = Calib.load()
    tracks = []
    for path in sorted(glob.glob("data/raw/*.MOV")):
        mano = HAND_DIR / f"{Path(path).stem}.npz"
        tr = track_video(path, calib, np.load(mano) if mano.exists() else None)
        tr["t_grasp"], tr["t_release"] = find_contacts(tr, calib)
        tracks.append(tr)
        print(f"{tr['name']}: {tr['T']} frames, hand {tr['valid'].mean():.0%}, grasp@{tr['t_grasp']} release@{tr['t_release']}, "
              f"cube {np.round(tr['cube_start'][:2], 3)} -> {np.round(tr['cube_end'][:2], 3)}, plate {np.round(tr['plate'], 3)}")
    scale = hand_scale(tracks, calib)
    print(f"hand scale {scale:.3f} from {2 * len(tracks)} contacts")
    TRACK_DIR.mkdir(parents=True, exist_ok=True)
    for tr in tracks:
        raw = to_table(tr, calib, scale)
        np.savez(TRACK_DIR / f"{tr['name']}.npz", keypoints=anchor_to_contacts(raw, tr, calib), keypoints_raw=raw,
                 valid=tr["valid"], cube_start=tr["cube_start"], cube_end=tr["cube_end"], plate=tr["plate"],
                 t_grasp=tr["t_grasp"], t_release=tr["t_release"], marker0=tr["marker0"])


if __name__ == "__main__":
    main()
