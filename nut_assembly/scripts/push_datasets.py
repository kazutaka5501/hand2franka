"""Upload the nut-assembly training sets to the Hugging Face Hub as public LeRobot datasets."""
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(__file__).resolve().parents[1] / "data/lerobot"
UPLOADS = {
    "nut_play2_train": "Kaz55/nut_rl_distill_2cam",
    "nut_human_wrist": "Kaz55/nut_human_mano_2cam",
    "nut_mix2_train": "Kaz55/nut_mix_2cam",
}
for local, repo in UPLOADS.items():
    ds = LeRobotDataset(repo, root=ROOT / local, video_backend="pyav")
    ds.push_to_hub(private=False, tags=["franka", "sharpa", "nut", "smolvla", "isaac-lab"])
    print(f"pushed {local} -> https://huggingface.co/datasets/{repo} ({ds.meta.total_episodes} episodes)", flush=True)
