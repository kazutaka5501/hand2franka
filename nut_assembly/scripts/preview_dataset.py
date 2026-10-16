"""Write a video of what a LeRobot training set contains: every camera of a few episodes side by side, with the
instruction and the episode index.

    python scripts/preview_dataset.py <dataset dir> <output.webm> --episodes 0 1 2
"""

import argparse
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(f"local/{args.root.name}", root=args.root, video_backend="pyav")
    cams = [k for k in ds.meta.features if k.startswith("observation.images.")]
    episode = np.array(ds.hf_dataset["episode_index"])
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    frames = []
    for ep in args.episodes:
        for i in np.where(episode == ep)[0]:
            item = ds[int(i)]
            panels = [(item[c].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for c in cams]
            img = Image.fromarray(np.concatenate(panels, axis=1)).resize(
                (panels[0].shape[1] * len(panels) * args.scale, panels[0].shape[0] * args.scale))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, img.width, 24], fill=(255, 255, 255))
            names = " | ".join(c.rsplit(".", 1)[1] for c in cams)
            draw.text((6, 4), f"{args.root.name} ep {ep}  [{names}]  \"{item['task']}\"", fill=(0, 0, 0), font=font)
            frames.append(np.asarray(img))
    imageio.mimsave(args.output, frames, fps=ds.meta.fps, macro_block_size=1, codec="libvpx-vp9",
                    ffmpeg_params=["-b:v", "0", "-crf", "34", "-row-mt", "1", "-deadline", "realtime", "-cpu-used", "8"])
    print(f"wrote {args.output}: {len(frames)} frames, cameras {cams}")


if __name__ == "__main__":
    main()
