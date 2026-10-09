"""Track the hand in every clip with WiLoR, which regresses MANO hand-model parameters per frame.

Runs in its own environment because WiLoR needs NumPy 1.x:

    .venv-hand/bin/python -m e2f.hand_mano "data/raw/*.MOV" data/hands
    .venv-hand/bin/python -m e2f.hand_mano "data/cups/raw/*.mp4" data/cups/hands

The MANO model itself may not be redistributed. Register at https://mano.is.tue.mpg.de, download
"Models & Code" and put `MANO_RIGHT.pkl` in assets/mano/. The WiLoR weights are fetched from the
Hugging Face hub on first use.
"""
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # the WiLoR checkpoints predate torch's safe loader

import glob
import sys
from pathlib import Path

import av
import numpy as np
import torch

MANO = Path("assets/mano/MANO_RIGHT.pkl")
WILOR_DIR = Path("assets/wilor")


def load_tracker():
    if not MANO.exists():
        raise SystemExit(f"{MANO} is missing: download it from https://mano.is.tue.mpg.de (see the module docstring)")
    link = WILOR_DIR / "pretrained_models" / MANO.name
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(MANO.resolve())
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    return WiLorHandPose3dEstimationPipeline(device=torch.device("cuda"), dtype=torch.float16, wilor_pretrained_dir=str(WILOR_DIR), verbose=False)


def track(tracker, path):
    """Per frame: 21 joints in metres relative to the hand, their pixels, and the MANO parameters
    (wrist rotation and 15 joint rotations as axis-angle, 10 shape coefficients)."""
    keys = {"joints": ("pred_keypoints_3d", (21, 3)), "pixels": ("pred_keypoints_2d", (21, 2)),
            "wrist": ("global_orient", (1, 3)), "pose": ("hand_pose", (15, 3)), "shape": ("betas", (10,))}
    out, valid = {k: [] for k in keys}, []
    with av.open(path) as container:
        for frame in container.decode(video=0):
            hands = tracker.predict(frame.to_ndarray(format="rgb24"))
            if hands:
                box = lambda h: (h["hand_bbox"][2] - h["hand_bbox"][0]) * (h["hand_bbox"][3] - h["hand_bbox"][1])
                preds = max(hands, key=box)["wilor_preds"]
            for k, (name, shape) in keys.items():
                out[k].append(preds[name][0] if hands else np.full(shape, np.nan))
            valid.append(bool(hands))
    return dict({k: np.array(v) for k, v in out.items()}, valid=np.array(valid))


if __name__ == "__main__":
    videos, out = sys.argv[1], Path(sys.argv[2])
    tracker = load_tracker()
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(glob.glob(videos)):
        result = track(tracker, path)
        np.savez(out / f"{Path(path).stem}.npz", **result)
        print(f"{Path(path).stem}: hand found in {result['valid'].mean():.0%} of {len(result['valid'])} frames", flush=True)
