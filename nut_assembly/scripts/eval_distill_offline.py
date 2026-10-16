"""Offline check of the distilled SmolVLA on teacher episodes it was not trained on.

At every 10th frame of each held-out episode the student predicts an action chunk from the image, the joint state
and the instruction; its first 10 actions (1/3 s) are compared with what the RL teacher commanded. Reference:
"hold still" (target = current joint positions), which is what a policy that learned nothing would score.

    python scripts/eval_distill_offline.py outputs/train/nut_distill/checkpoints/last/pretrained_model
    python scripts/eval_distill_offline.py lerobot/smolvla_base --zero_shot

``--zero_shot`` evaluates the pretrained smolvla_base without any fine-tuning, set up exactly as fine-tuning starts
from it: input/output features of this dataset (29 joints, one camera) and the training set's normalisation stats.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 10
STRIDE = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/lerobot/nut_distill_heldout")
    parser.add_argument("--zero_shot", action="store_true")
    parser.add_argument("--train_dataset", type=Path, default=ROOT / "data/lerobot/nut_distill_train")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    ds = LeRobotDataset("local/nut_distill_heldout", root=args.dataset, video_backend="pyav")
    if args.zero_shot:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies.factory import make_policy

        train_meta = LeRobotDatasetMetadata("local/nut_distill_train", root=args.train_dataset)
        cfg = PreTrainedConfig.from_pretrained(str(args.checkpoint))
        cfg.pretrained_path = str(args.checkpoint)
        cfg.input_features = {}
        cfg.device = "cuda"
        policy = make_policy(cfg, ds_meta=train_meta).to("cuda").eval()
        pre, post = make_pre_post_processors(policy.config, dataset_stats=train_meta.stats)
    else:
        policy = SmolVLAPolicy.from_pretrained(args.checkpoint).to("cuda").eval()
        pre, post = make_pre_post_processors(policy.config, pretrained_path=str(args.checkpoint))
    names = ds.features["action"]["names"]
    arm = np.array([n.startswith("panda") for n in names])

    ep_index = np.array(ds.hf_dataset["episode_index"])
    student_err, hold_err, phase_err = [], [], {}
    for ep in np.unique(ep_index):
        idx = np.where(ep_index == ep)[0]
        actions = np.stack([ds.hf_dataset[int(i)]["action"] for i in idx])
        for k in range(0, len(idx) - HORIZON, STRIDE):
            item = ds[int(idx[k])]
            batch = {key: (v.unsqueeze(0).to("cuda") if torch.is_tensor(v) else v) for key, v in item.items()
                     if key.startswith("observation.")}
            batch["task"] = [item["task"]]
            with torch.no_grad():
                chunk = post(policy.predict_action_chunk(pre(batch)))[0].float().cpu().numpy()[:HORIZON]
            truth = actions[k:k + HORIZON]
            state = item["observation.state"].numpy()
            student_err.append(np.abs(chunk - truth).mean(0))
            hold_err.append(np.abs(state[None] - truth).mean(0))
            phase = "reach+grasp (0-1.3 s)" if k < 40 else "lift+hold (1.3-4 s)"
            phase_err.setdefault(phase, []).append((np.abs(chunk - truth).mean(0), np.abs(state[None] - truth).mean(0)))

    s, h = np.array(student_err), np.array(hold_err)
    print(f"[offline] {len(np.unique(ep_index))} held-out episodes, {len(s)} chunks, first {HORIZON} actions each")
    print(f"[offline] mean |action error| [rad]   student   hold-still")
    print(f"[offline]   arm (7 joints)            {s[:, arm].mean():.4f}    {h[:, arm].mean():.4f}")
    print(f"[offline]   hand (22 joints)          {s[:, ~arm].mean():.4f}    {h[:, ~arm].mean():.4f}")
    for phase, pairs in phase_err.items():
        ps, ph = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        print(f"[offline]   {phase:24s}  {ps.mean():.4f}    {ph.mean():.4f}")


if __name__ == "__main__":
    main()
