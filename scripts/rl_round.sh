#!/usr/bin/env bash
# One RL round: collect and select with the current policy, continue training on the kept episodes, evaluate.
#   scripts/rl_round.sh <round> <checkpoint>
set -euo pipefail
round=$1
checkpoint=$2

python -m h2f.improve "$checkpoint" --round "$round"
episodes=$(python -c "import json; print(json.load(open('outputs/rl/round$round/selection.json'))['kept'])")

lerobot-train --policy.path="$checkpoint" \
    --dataset.repo_id="local/hand2franka_round$round" --dataset.root="data/lerobot/round$round" \
    --dataset.episodes="$episodes" --dataset.video_backend=pyav \
    --rename_map='{"observation.images.phone": "observation.images.camera1"}' \
    --batch_size=64 --steps=4000 --save_freq=4000 --num_workers=12 \
    --output_dir="outputs/train/rl_round$round" --job_name="rl_round$round" \
    --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false

python -m h2f.evaluate "outputs/train/rl_round$round/checkpoints/004000/pretrained_model" --episodes 100 --margin 0.08
