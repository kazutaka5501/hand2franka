"""Roll out a SmolVLA checkpoint in the simulated twin.

Start poses are drawn from the range the objects occupied in the videos; the same seed gives the
same scenes, so checkpoints can be compared episode by episode.

    python -m e2f.evaluate outputs/train/bc/checkpoints/last/pretrained_model --episodes 50
    python -m e2f.evaluate outputs/train/cups_bc/checkpoints/last/pretrained_model --task cups
"""
import argparse
import itertools
import json
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch

from .cups_sim import CupStackEnv
from .sim import CONTROL_HZ, PickPlaceEnv

ENVS = {"cube": PickPlaceEnv, "cups": CupStackEnv}
REPLAN = 10  # actions executed from each 50-step chunk (0.5 s)


def load_policy(path, device="cuda", float32=False):
    """SmolVLA runs in bfloat16 by default. RL needs float32: in bf16 the same input gives slightly different
    outputs depending on what else is in the batch, and small weight updates round away."""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(path).to(device).eval()
    if float32:
        policy.float()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(path))
    # which simulated cameras this checkpoint was trained on
    rename = json.loads((Path(path) / "train_config.json").read_text())["rename_map"]
    policy.cameras = [k.rsplit(".", 1)[1] for k in rename]
    return policy, pre, post


def to_batch(observations, cameras, device, task):
    """A list of environment observations -> the batch layout the policy's preprocessor expects."""
    batch = {"observation.state": torch.from_numpy(np.stack([o["state"] for o in observations])).to(device),
             "task": [task] * len(observations)}
    for cam in cameras:
        images = torch.from_numpy(np.stack([o[cam] for o in observations])).to(device)
        batch[f"observation.images.{cam}"] = images.permute(0, 3, 1, 2).float() / 255
    return batch


@torch.no_grad()
def plan(policy, pre, post, obs, task, noise=None):
    """One forward pass -> (chunk_size, 5) end-effector targets."""
    chunk = policy.predict_action_chunk(pre(to_batch([obs], policy.cameras, policy.config.device, task)), noise=noise)
    return post(chunk)[0].float().cpu().numpy()


def rollout(env, policy, pre, post, scene, replan=REPLAN, record=False):
    """Closed-loop episode: predict a chunk, execute its first `replan` actions, look again."""
    obs = env.reset(*scene)
    trace, progress, success = [], 0, False
    for t in range(env.max_steps):
        if t % replan == 0:
            actions = plan(policy, pre, post, obs, env.task)
        action = actions[t % replan]
        if record:
            trace.append((obs, action))
        obs, _, success, _ = env.step(action, render=record or (t + 1) % replan == 0)
        progress = max(progress, env.progress())
        if success:
            break
    return dict(success=success, progress=progress, steps=t + 1, trace=trace)


def write_video(path, trace):
    frames = [np.hstack([obs["phone"]] + ([cv2.resize(obs["wrist"], (216, 216))] if "wrist" in obs else [])) for obs, _ in trace]
    imageio.mimwrite(path, frames, fps=CONTROL_HZ, macro_block_size=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--task", choices=list(ENVS), default="cube")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--videos", type=int, default=6)
    parser.add_argument("--replan", type=int, default=REPLAN, help="actions executed from each predicted chunk")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.0, help="widen the cube start box beyond the human data [m]")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run = Path(args.checkpoint).parents[2].name + "_" + Path(args.checkpoint).parent.name
    out = Path(args.out or Path("outputs/eval") / f"{run}_margin{round(args.margin * 100)}cm")
    out.mkdir(parents=True, exist_ok=True)
    policy, pre, post = load_policy(args.checkpoint)
    env = ENVS[args.task](policy.cameras)
    results = []
    for i, scene in enumerate(itertools.islice(env.scenes(args.seed, args.margin), args.episodes)):
        r = rollout(env, policy, pre, post, scene, args.replan, record=i < args.videos)
        if trace := r.pop("trace"):
            write_video(out / f"episode_{i:02d}_{'success' if r['success'] else 'fail'}.mp4", trace)
        results.append(r)
        print(f"episode {i:02d}: {'success' if r['success'] else 'fail   '} {env.stages[r['progress']]}, {r['steps']} steps", flush=True)
    summary = dict(checkpoint=args.checkpoint, episodes=len(results), replan=args.replan, seed=args.seed, margin=args.margin,
                   success_rate=float(np.mean([r["success"] for r in results])),
                   reached={stage: float(np.mean([r["progress"] >= k for r in results])) for k, stage in enumerate(env.stages) if k},
                   results=results)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    steps = [r["steps"] for r in results if r["success"]]
    print(f"success {summary['success_rate']:.0%}, " + ", ".join(f"{s} {v:.0%}" for s, v in summary["reached"].items())
          + f", median {np.median(steps) if steps else float('nan'):.0f} steps to success over {len(results)} episodes -> {out}")


if __name__ == "__main__":
    main()
