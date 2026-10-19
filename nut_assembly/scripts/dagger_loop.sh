#!/usr/bin/env bash
# DAgger for the SmolVLA student on the stage-1 task (pick the nut up, bring it to a 6D goal).
#
# Each round: the current student drives 128 episodes in Isaac (the teacher's action is executed instead with
# probability beta), the stage-1 RL teacher labels every visited state, the rollouts are rendered into a LeRobot
# dataset (overview + palm camera), all data so far is merged with the original teacher demonstrations, the student
# is fine-tuned from its previous checkpoint, and evaluated alone (beta = 0) on 64 episodes.
#
#   scripts/dagger_loop.sh <first student checkpoint> [rounds]     # log: logs/dagger/summary.txt
set -uo pipefail
cd "$(dirname "$0")/.."
STUDENT=${1:?usage: dagger_loop.sh <student checkpoint> [rounds]}
ROUNDS=${2:-4}
BETAS=(0.5 0.3 0.15 0.0 0.0 0.0)
TEACHER=${TEACHER:-logs/rsl_rl/franka_sharpa_nut_play/<run>/model_4250.pt}  # stage-1 RL policy
TASK_TEXT="pick up the nut and hold it like the green nut"
LEROBOT=${LEROBOT:-$(dirname "$(command -v lerobot-train)")}  # bin dir of the lerobot environment (Python 3.12)
mkdir -p logs/dagger
export OMNI_KIT_ACCEPT_EULA=YES

serve() {  # start the policy server for a checkpoint and wait until it listens
  rm -f logs/dagger/server.log
  (MUJOCO_GL=egl nohup $LEROBOT/python scripts/vla_server.py "$1" --task "$TASK_TEXT" > logs/dagger/server.log 2>&1 < /dev/null &)
  until grep -aq "listening\|Traceback" logs/dagger/server.log 2>/dev/null; do sleep 3; done
}

evaluate() {  # student alone, 64 episodes
  serve "$1"
  .venv/bin/python -u scripts/eval_vla_closed_loop.py --headless --task Franka-Sharpa-Nut-Play-Play-v0 --keep_resampling \
    --num_envs 64 --episodes 1 --record_envs 64 --output logs/dagger/eval_$2.npz < /dev/null > logs/dagger/eval_$2.log 2>&1
  echo "$(date +%H:%M) $2: $(grep -a 'lifted > 0.10' logs/dagger/eval_$2.log | sed 's/.*end: //') lifted | $(.venv/bin/python scripts/play_metrics.py logs/dagger/eval_$2.npz | sed 's/.*| goals/goals/')" | tee -a logs/dagger/summary.txt
}

echo "$(date +%H:%M) start: student $STUDENT" | tee -a logs/dagger/summary.txt
for r in $(seq 1 "$ROUNDS"); do
  beta=${BETAS[$((r - 1))]}
  # 1. rollouts with teacher labels
  serve "$STUDENT"
  .venv/bin/python -u scripts/dagger_rollout.py --headless --task Franka-Sharpa-Nut-Play-Play-v0 --keep_resampling \
    --num_envs 128 --episodes 1 --record_envs 128 --teacher "$TEACHER" --beta "$beta" \
    --output data/teacher/dagger_r$r.npz < /dev/null > logs/dagger/rollout_r$r.log 2>&1
  echo "$(date +%H:%M) round $r (beta $beta) rollouts: $(grep -a 'lifted > 0.10' logs/dagger/rollout_r$r.log | sed 's/.*end: //') lifted" | tee -a logs/dagger/summary.txt
  .venv/bin/python - "$r" <<'EOF'
import sys
import numpy as np
r = sys.argv[1]
d = dict(np.load(f"data/teacher/dagger_r{r}.npz"))
n = d["joint_pos"].shape[0]
d["lifted"] = np.ones(n, bool)  # keep every rollout: the teacher's labels are what matters
d["at_goal"] = np.ones(n, bool)
np.savez_compressed(f"data/teacher/dagger_r{r}_all.npz", **d)
EOF
  # 2. render (8 processes)
  for k in 0 1 2 3 4 5 6 7; do
    MUJOCO_GL=egl $LEROBOT/python scripts/build_distill_dataset.py data/teacher/dagger_r${r}_all.npz --name dagger_r$r --wrist \
      --held_out 1 --num_shards 8 --shard $k --task "$TASK_TEXT" > logs/dagger/build_r${r}_$k.log 2>&1 < /dev/null &
  done
  wait
  # 3. aggregate everything so far with the original demonstrations
  $LEROBOT/python - "$r" <<'EOF' >> logs/dagger/summary.txt 2>/dev/null
import glob, shutil, sys
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets
r = int(sys.argv[1])
D = Path("data/lerobot")
roots = [D / "nut_play2_train"] + [Path(p) for q in range(1, r + 1) for p in sorted(glob.glob(str(D / f"dagger_r{q}_train_*")))]
parts = [LeRobotDataset(f"local/{p.name}", root=p, video_backend="pyav") for p in roots]
out = D / f"nut_dagger_agg_r{r}"
shutil.rmtree(out, ignore_errors=True)
m = merge_datasets([p for p in parts if p.meta.total_episodes > 0], f"local/nut_dagger_agg_r{r}", out)
print(f"round {r} aggregate: {m.meta.total_episodes} episodes, {m.meta.total_frames} frames")
EOF
  # 4. fine-tune the student from its last checkpoint
  rm -rf outputs/train/dagger_r$r
  $LEROBOT/lerobot-train --policy.path="$STUDENT" --dataset.repo_id=local/nut_dagger_agg_r$r \
    --dataset.root=data/lerobot/nut_dagger_agg_r$r --dataset.video_backend=pyav --batch_size=64 --steps=3000 \
    --save_freq=3000 --output_dir=outputs/train/dagger_r$r --policy.device=cuda --policy.push_to_hub=false \
    --wandb.enable=false > logs/dagger/train_r$r.log 2>&1 < /dev/null
  STUDENT=outputs/train/dagger_r$r/checkpoints/last/pretrained_model
  # 5. evaluate the new student alone
  evaluate "$STUDENT" r$r
done
echo "$(date +%H:%M) done" | tee -a logs/dagger/summary.txt
