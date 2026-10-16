"""Side-by-side preview of retargeted human clips: the phone video (with the tracked hand keypoints drawn), the
simulated robot rendered from the phone's calibrated viewpoint (same pose and field of view, so the two can be compared
directly), and the robot in the dataset camera (build_human_dataset.py).

    MUJOCO_GL=egl python scripts/preview_human_retarget.py IMG_8274 IMG_8279 IMG_8281
"""

import argparse
import importlib.util
from pathlib import Path

import cv2
import imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


render = load("render_mujoco", "render_mujoco.py")
retarget = load("retarget_mano", "retarget_mano.py")
dataset = load("build_human_dataset", "build_human_dataset.py")

PH, PW = 854, 480  # portrait panel (shown turned to landscape)
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13),
         (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def phone_camera_model():
    """Scene model with an extra camera at the calibrated phone pose (OpenCV camera -> MuJoCo camera: flip y and z)."""
    K, R_cb, t_cb = retarget.calib()
    Rw, ow = retarget.board_to_world()
    R_wc = Rw @ R_cb.T
    p_w = Rw @ (-R_cb.T @ t_cb) + ow
    R_mj = R_wc @ np.diag([1.0, -1.0, -1.0])
    q = R.from_matrix(R_mj).as_quat()
    fovy = float(np.degrees(2 * np.arctan(0.5 * 1920 / K[0, 0])))
    original = render._finish

    def finish(spec, width, height):
        spec.worldbody.add_camera(name="phone", pos=p_w.tolist(), quat=[q[3], q[0], q[1], q[2]], fovy=fovy)
        return original(spec, width, height)

    render._finish = finish
    try:
        return render.build_model(0.03, retarget.TABLE_POS, retarget.TABLE_SIZE, 1024, 1024, 1.0)
    finally:
        render._finish = original


def frames(path):
    cap = cv2.VideoCapture(str(path))  # OpenCV applies the rotation flag: portrait 1080 x 1920
    while True:
        ok, f = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="+")
    parser.add_argument("--out", type=Path, default=ROOT / "logs/videos/human_retarget_preview.mp4")
    parser.add_argument("--human", type=Path, default=ROOT / "data/human")
    args = parser.parse_args()
    model = phone_camera_model()
    state = mujoco.MjData(model)
    names = [str(n) for n in np.load(ROOT / "data/teacher/nut_dense_rollouts.npz")["joint_names"]]
    qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
    nut = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")]
    goal = model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal")]
    cam = dataset.camera()
    out = []
    with mujoco.Renderer(model, PH, PW) as phone_r, mujoco.Renderer(model, PH, PH) as ds_r:
        for clip in args.clips:
            d = np.load(args.human / f"{clip}.npz")
            hands = np.load(ROOT / f"data/hands/{clip}.npz")
            px = retarget.raw_to_portrait(hands["pixels"])
            s, e = int(d["seg_start"]), int(d["seg_end"])
            for i, img in enumerate(frames(ROOT / f"data/raw/{clip}.MOV")):
                if i < s or i >= e:
                    continue
                img = cv2.resize(img, (PW, PH), interpolation=cv2.INTER_AREA)
                if hands["valid"][i]:
                    p = px[i] * PW / 1080.0
                    col = (0, 255, 0) if d["hand_valid"][i] else (255, 160, 0)
                    for a, b in BONES:
                        cv2.line(img, tuple(np.int32(p[a])), tuple(np.int32(p[b])), col, 1, cv2.LINE_AA)
                state.qpos[qadr] = d["joint_pos"][i]
                state.mocap_pos[nut] = d["nut_pos"][i]
                state.mocap_quat[nut] = d["nut_quat"][i]
                state.mocap_pos[goal] = d["goal_pos"]
                mujoco.mj_forward(model, state)
                phone_r.update_scene(state, "phone")
                sim_phone = phone_r.render()
                ds_r.update_scene(state, cam)
                sim_ds = ds_r.render().copy()
                tag = f"{clip} f{i}" + ("" if d["hand_valid"][i] else " (interp)")
                img = np.ascontiguousarray(img)
                cv2.putText(sim_ds, f"res {1e3 * np.nan_to_num(d['pos_res'][i]):.0f}mm "
                            f"{np.degrees(np.nan_to_num(d['rot_res'][i])):.0f}deg", (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 0, 0), 1, cv2.LINE_AA)
                # the phone was held upright; turned a quarter counter-clockwise the board reads as printed and the
                # panels sit side by side in landscape
                img, sim_phone = np.rot90(img), np.rot90(sim_phone)
                sim_ds = cv2.resize(sim_ds, (img.shape[0], img.shape[0]), interpolation=cv2.INTER_AREA)
                img = np.ascontiguousarray(img)
                cv2.putText(img, tag, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                out.append(np.ascontiguousarray(np.hstack([img, sim_phone, sim_ds])))
            print(f"[preview] {clip}: {e - s} frames")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, out, fps=30, macro_block_size=1, ffmpeg_params=["-movflags", "+faststart"])
    imageio.mimsave(args.out.with_suffix(".webm"), out, fps=30, macro_block_size=1, codec="libvpx-vp9",
                    ffmpeg_params=["-b:v", "0", "-crf", "34", "-row-mt", "1", "-deadline", "realtime", "-cpu-used", "8"])
    sheet = [out[k] for k in np.linspace(0, len(out) - 1, 6).astype(int)]
    cv2.imwrite(str(args.out.with_suffix(".png")), cv2.cvtColor(np.vstack([np.hstack(sheet[:3]), np.hstack(sheet[3:])]),
                                                               cv2.COLOR_RGB2BGR))
    print(f"[preview] wrote {args.out}, .webm and .png ({len(out)} frames)")


if __name__ == "__main__":
    main()
