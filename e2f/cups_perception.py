"""Cup-stacking clips -> where the cups are, when each one is picked up and put down, and the hand track.

The cups carry no markers: they are found by colour, and a tapered-cup model is fitted to each
outline through the calibrated camera. The clips are 1024x576 (messenger-compressed); pixel
coordinates are scaled to the 1920x1080 frame the camera model uses.

    python -m e2f.cups_perception            # -> data/cups/tracks/*.npz
"""
import glob
from pathlib import Path

import av
import cv2
import mediapipe as mp
import numpy as np
from scipy.optimize import minimize

from .camera import H, ID_TABLE, W, Calib, aruco_detector, detect, square
from .cups_sim import BASE_RADIUS, CALIB_PATH, HEIGHT, RIM_RADIUS
from .perception import hand_in_camera, hand_tracker

TRACK_DIR = Path("data/cups/tracks")
HSV = {"pink": [((0, 25, 150), (12, 110, 255)), ((165, 25, 150), (180, 110, 255))],
       "green": [((35, 120, 120), (60, 255, 255))],
       "blue": [((95, 150, 120), (112, 255, 255))]}
MOVED = ("green", "pink")  # the order in which the cups go onto the blue one
CIRCLE = np.c_[np.cos(np.linspace(0, 2 * np.pi, 48)), np.sin(np.linspace(0, 2 * np.pi, 48)), np.zeros(48)]


def calibrate(videos):
    """Camera pose for these clips from the reference marker. Same phone, so the focal length of the
    self-calibration on the cube clips is reused (scaled to 1920x1080 it is the same number)."""
    cube, detector = Calib.load(), aruco_detector()
    corners = []
    for path in videos:
        with av.open(path) as container:
            frame = next(container.decode(video=0)).to_ndarray(format="bgr24")
        corners.append(detect(detector, cv2.resize(frame, (W, H)))[ID_TABLE])
    _, rvec, tvec = cv2.solvePnP(square(cube.table_marker), np.median(corners, 0), cube.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    calib = Calib(cube.f, rvec.ravel().tolist(), tvec.ravel().tolist(), cube.table_marker)
    calib.save(CALIB_PATH)
    return calib


def colour_mask(bgr, name):
    """Largest blob of the cup's colour, at the 1920x1080 scale."""
    hsv = cv2.cvtColor(cv2.resize(bgr, (W, H)), cv2.COLOR_BGR2HSV)
    mask = sum(cv2.inRange(hsv, np.array(lo), np.array(hi)) for lo, hi in HSV[name])
    mask[:H // 9] = 0  # pale clutter behind the table
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    return (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])) if n > 1 else np.zeros_like(mask, bool)


def outline(calib, x, y, z=0.0):
    """Image region covered by a cup standing at (x, y) with its base at height z."""
    points = np.r_[BASE_RADIUS * CIRCLE + [x, y, z], RIM_RADIUS * CIRCLE + [x, y, z + HEIGHT]]
    region = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(region, cv2.convexHull(calib.project(points).astype(np.int32)), 1)
    return region.astype(bool)


def overlap(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def locate(calib, mask):
    """Table position of a cup standing on the table, from its colour mask."""
    ys, xs = np.nonzero(mask)
    foot = calib.on_plane([[xs[ys > ys.max() - 6].mean(), ys.max()]], 0.0)[0]
    fit = minimize(lambda p: 1 - overlap(outline(calib, *p), mask), foot[:2] + [0, BASE_RADIUS], method="Nelder-Mead",
                   options=dict(xatol=5e-4, fatol=1e-3, initial_simplex=foot[:2] + [0, BASE_RADIUS] + np.array([[0, 0], [0.02, 0], [0, 0.02]])))
    return fit.x, 1 - fit.fun


def track_video(path, calib):
    tracker = hand_tracker()
    hand, masks = [], {name: [] for name in HSV}
    with av.open(path) as container:
        for i, frame in enumerate(container.decode(video=0)):
            bgr = frame.to_ndarray(format="bgr24")
            rgb = np.ascontiguousarray(cv2.resize(bgr, (W, H))[:, :, ::-1])
            result = tracker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(i * 1000 / 30))
            hand.append(hand_in_camera(result, calib.K))
            for name in HSV:
                masks[name].append(colour_mask(bgr, name))
    tracker.close()
    T = len(hand)
    cups = {name: locate(calib, masks[name][0])[0] for name in HSV}
    track = dict(name=Path(path).stem, T=T, cups=cups, valid=np.array([h[0] is not None for h in hand]),
                 hand=np.array([h[0] if h[0] is not None else np.full((21, 3), np.nan) for h in hand]),
                 hand_px=np.array([h[1] if h[1] is not None else np.full((21, 2), np.nan) for h in hand]))
    # A cup "is at" a place in a frame if its colour fills most of the outline it would have there.
    # The arm often hides a cup that has not moved yet, so the last sighting at the start can be far too
    # early; a pick can never precede the previous cup's arrival, and if it would, it is put mid-way.
    events, previous = [], 0
    for level, name in enumerate(MOVED, start=1):
        start = outline(calib, *cups[name])
        end = outline(calib, *cups["blue"], z=level * 0.012)  # real cups nest about 12 mm apart
        at_start = np.array([overlap(m, start) > 0.6 for m in masks[name]])
        at_end = np.array([(m & end).sum() > 0.35 * end.sum() and not s for m, s in zip(masks[name], at_start)])
        left = int(np.flatnonzero(at_start).max())
        arrived = int(np.flatnonzero(at_end & (np.arange(T) > max(left, previous))).min())
        pick = left if left > previous else (previous + arrived) // 2
        events += [pick, arrived]
        previous = arrived
    track["events"] = np.array(events)  # frames: pick green, place green, pick pink, place pink
    return track


if __name__ == "__main__":
    videos = sorted(glob.glob("data/cups/raw/*.mp4"))
    calib = calibrate(videos)
    TRACK_DIR.mkdir(parents=True, exist_ok=True)
    for path in videos:
        tr = track_video(path, calib)
        print(f"{tr['name']}: {tr['T']} frames, hand seen {tr['valid'].mean():.0%}, events {tr['events'].tolist()} | "
              + " | ".join(f"{n} at {np.round(tr['cups'][n], 3)}" for n in HSV), flush=True)
        np.savez(TRACK_DIR / f"{tr['name']}.npz", events=tr["events"], valid=tr["valid"], frames=tr["T"],
                 **{n: tr["cups"][n] for n in HSV})
