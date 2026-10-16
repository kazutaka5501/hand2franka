"""Detect the ArUco markers (table board ids 10-17, nut ids 0-5) in every frame of the demonstration clips.

    python scripts/detect_markers.py data/raw/*.MOV

Writes data/markers/<clip>.npz: per frame the detected ids and corner pixels (frames are as displayed, i.e.
after the phone's rotation flag is applied), plus a per-clip summary on stdout.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = json.loads((ROOT / "assets/markers/layout.json").read_text())
TABLE_IDS = {int(k) for k in LAYOUT["table"]}
NUT_IDS = {int(k) for k in LAYOUT["nut"]}


def detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)


def detect_clip(path: Path, out_dir: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    det = detector()
    ids_per_frame, corners_per_frame = [], []
    shape = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        shape = frame.shape[:2]
        corners, ids, _ = det.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if ids is None:
            ids_per_frame.append(np.zeros(0, int))
            corners_per_frame.append(np.zeros((0, 4, 2), np.float32))
            continue
        keep = [i for i, m in enumerate(ids.ravel()) if m in TABLE_IDS | NUT_IDS]
        ids_per_frame.append(ids.ravel()[keep].astype(int))
        corners_per_frame.append(np.stack([corners[i][0] for i in keep]) if keep else np.zeros((0, 4, 2), np.float32))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    counts = np.array([len(i) for i in ids_per_frame])
    offsets = np.concatenate([[0], np.cumsum(counts)])
    np.savez_compressed(
        out_dir / f"{path.stem}.npz",
        ids=np.concatenate(ids_per_frame) if counts.sum() else np.zeros(0, int),
        corners=np.concatenate(corners_per_frame) if counts.sum() else np.zeros((0, 4, 2)),
        offsets=offsets,
        image_size=np.array(shape[::-1]),  # width, height
        fps=fps,
    )
    n_table = np.array([sum(m in TABLE_IDS for m in i) for i in ids_per_frame])
    n_nut = np.array([sum(m in NUT_IDS for m in i) for i in ids_per_frame])
    return {
        "frames": len(ids_per_frame),
        "size": tuple(shape[::-1]),
        "table>=4": float((n_table >= 4).mean()),
        "nut>=1": float((n_nut >= 1).mean()),
        "nut>=2": float((n_nut >= 2).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "data/markers")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for clip in sorted(args.clips):
        s = detect_clip(clip, args.out)
        print(f"{clip.stem}: {s['frames']:4d} frames {s['size'][0]}x{s['size'][1]}, table >= 4 markers {s['table>=4']:5.1%}, "
              f"nut >= 1 marker {s['nut>=1']:5.1%}, >= 2 {s['nut>=2']:5.1%}", flush=True)


if __name__ == "__main__":
    main()
