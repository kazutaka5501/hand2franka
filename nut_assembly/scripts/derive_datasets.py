"""Assemble the SmolVLA training sets for the comparison from the shard builds.

    python scripts/derive_datasets.py

    nut_play_distill_train   RL distillation (stage-1 teacher), overview + wrist camera   <- merged shards
    nut_play_distill_1cam    the same without the wrist camera (camera ablation)
    nut_mix_train            RL distillation + human video retargeted with MANO (nut_human_wrist)
"""

import glob
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/lerobot"


def load(name):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(f"local/{name}", root=DATA / name, video_backend="pyav")


def fresh(name):
    shutil.rmtree(DATA / name, ignore_errors=True)
    return DATA / name


def main():
    from lerobot.datasets.dataset_tools import merge_datasets, remove_feature

    shards = sorted(glob.glob(str(DATA / "nut_play_distill_train_*")), key=lambda p: int(p.rsplit("_", 1)[1]))
    play = merge_datasets([load(Path(p).name) for p in shards], "local/nut_play_distill_train",
                          fresh("nut_play_distill_train"))
    print(f"[derive] nut_play_distill_train: {play.meta.total_episodes} episodes, {play.meta.total_frames} frames")
    one = remove_feature(play, "observation.images.camera2", fresh("nut_play_distill_1cam"), "local/nut_play_distill_1cam")
    print(f"[derive] nut_play_distill_1cam: {one.meta.total_episodes} episodes, cameras {[k for k in one.meta.features if 'images' in k]}")
    mix = merge_datasets([play, load("nut_human_wrist")], "local/nut_mix_train", fresh("nut_mix_train"))
    print(f"[derive] nut_mix_train: {mix.meta.total_episodes} episodes, {mix.meta.total_frames} frames")


if __name__ == "__main__":
    main()
