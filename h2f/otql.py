"""RL post-training of the SmolVLA action head with OTQL (Sochopoulos et al., arXiv 2607.06262).

Starting from a policy fine-tuned on ten demonstrations, each round
  1. rolls the current policy out a few times and adds the episodes to a buffer that already
     holds the demonstrations;
  2. fits Q(s, a) by TD learning on the buffer: reward -1 per step until the task is done, chunks
     of 10 actions as one decision, the next action drawn from the current policy;
  3. scores every buffered action with its advantage, Q(s, a) minus the mean Q of actions the
     current policy would take there;
  4. retrains the flow on the buffer, where advantages decide how much each action counts and an
     optimal-transport plan decides which noise sample is paired with which action
     (weighted conditional OT flow matching).

The critic sees what the policy sees: the frozen VLM's features of the phone image and the
proprioceptive state. Nothing privileged from the simulator is used.

    python -m h2f.otql outputs/train/bc_follow10/checkpoints/last/pretrained_model --task cube
"""
import argparse
import copy
import glob
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .evaluate import ENVS, load_policy, rollout, to_batch

CHUNK = 10  # actions executed per decision; Q is defined on these
GAMMA = 0.99
ACTION_DIM = 5


def demonstrations(task, env, episodes):
    """The demonstrations the base policy was trained on, replayed to get observations. -> [(steps, success)]
    `episodes`: which hand-following cube clips; None for the corrected replays of all cube clips."""
    if task == "cups":
        from .cups_retarget import replay, stack_plan
        tracks = [np.load(p) for p in sorted(glob.glob("data/cups/tracks/*.npz"))]
        return [replay(env, t, stack_plan(t)[0]) for t in tracks]
    from .retarget import follow_plan, gripper_plan, replay, usual_finger_angle
    tracks = [np.load(p) for p in sorted(glob.glob("data/tracks/*.npz"))]
    if episodes is None:
        return [replay(env, t, gripper_plan(t, env.calib)[0]) for t in tracks]
    usual = usual_finger_angle(tracks)
    done = [r for r in (replay(env, t, follow_plan(t, env.calib, usual)[0]) for t in tracks) if r[1]]
    return [done[i] for i in episodes]


class Buffer:
    """Every step of every episode, as the start of one decision."""

    def __init__(self):
        self.obs, self.actions, self.pad, self.ret, self.next, self.done = [], [], [], [], [], []
        self.features = self.samples = None

    def add(self, steps, success, horizon):
        acts = np.stack([a for _, a in steps])
        T, start = len(steps), len(self.obs)
        reward = np.full(T, -(1 - GAMMA))  # scaled so that Q stays in [-1, 0]
        if success:
            reward[-1] = 0.0
        for t in range(T):
            idx = np.minimum(np.arange(t, t + horizon), T - 1)
            self.obs.append(steps[t][0])
            self.actions.append(acts[idx])
            self.pad.append(np.arange(t, t + horizon) >= T)
            n = min(CHUNK, T - t)
            self.ret.append((GAMMA ** np.arange(n) * reward[t:t + n]).sum())
            self.next.append(start + min(t + CHUNK, T - 1))
            self.done.append(success and t + CHUNK >= T)

    def __len__(self):
        return len(self.obs)


class Learner:
    def __init__(self, policy, pre, post, task, args):
        self.policy, self.pre, self.post, self.task, self.args = policy, pre, post, task, args
        self.device = policy.config.device
        hidden = policy.model.vlm_with_expert.config.text_config.hidden_size
        self.critic = nn.Sequential(nn.Linear(hidden + ACTION_DIM + CHUNK * ACTION_DIM, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(),
                                    nn.Linear(512, 1)).to(self.device)
        self.target = copy.deepcopy(self.critic)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.policy_opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)

    def batch(self, observations, actions=None, pad=None):
        raw = to_batch(observations, self.policy.cameras, self.device, self.task)
        if actions is not None:
            raw["action"] = torch.as_tensor(np.stack(actions), dtype=torch.float32, device=self.device)
            raw["action_is_pad"] = torch.as_tensor(np.stack(pad), device=self.device)
        return self.pre(raw)

    @torch.no_grad()
    def describe(self, buffer, samples):
        """Cache, for every buffered step, the frozen VLM's view of it, its action in the policy's
        normalised units, and `samples` action chunks the current policy would take there."""
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
        p, m = self.policy, self.policy.model
        first = 0 if buffer.features is None else len(buffer.features)
        feats, acts = [], []
        for i in range(first, len(buffer), 128):
            sl = slice(i, i + 128)
            b = self.batch(buffer.obs[sl], buffer.actions[sl], buffer.pad[sl])
            images, masks = p.prepare_images(b)
            embs, pad, att = m.embed_prefix(images, masks, b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK], state=p.prepare_state(b))
            out, _ = m.vlm_with_expert.forward(attention_mask=make_att_2d_masks(pad, att), position_ids=torch.cumsum(pad, 1) - 1,
                                               past_key_values=None, inputs_embeds=[embs, None], use_cache=True)
            pooled = (out[0].float() * pad[..., None]).sum(1) / pad.sum(1, keepdim=True)
            feats.append(torch.cat([pooled, b[OBS_STATE].float()], 1))
            acts.append(b["action"].float())
        if feats:
            buffer.features = torch.cat(([buffer.features] if first else []) + feats)
            buffer.norm_actions = torch.cat(([buffer.norm_actions] if first else []) + acts)
        drawn = []
        for i in range(0, len(buffer), 128):
            b = self.batch(buffer.obs[i:i + 128])
            drawn.append(torch.stack([p.predict_action_chunk(dict(b))[:, :CHUNK].float() for _ in range(samples)], 1))
        buffer.samples = torch.cat(drawn)  # (N, samples, CHUNK, 5)

    def q(self, net, features, chunk):
        return net(torch.cat([features, chunk.flatten(1)], 1)).squeeze(1)

    def fit_critic(self, buffer, steps):
        ret = torch.as_tensor(np.array(buffer.ret), dtype=torch.float32, device=self.device)
        nxt = torch.as_tensor(buffer.next, device=self.device)
        done = torch.as_tensor(buffer.done, dtype=torch.float32, device=self.device)
        for _ in range(steps):
            i = torch.randint(len(buffer), (256,), device=self.device)
            with torch.no_grad():
                k = torch.randint(buffer.samples.shape[1], (256,), device=self.device)
                target = ret[i] + GAMMA ** CHUNK * (1 - done[i]) * self.q(self.target, buffer.features[nxt[i]], buffer.samples[nxt[i], k])
            loss = (self.q(self.critic, buffer.features[i], buffer.norm_actions[i, :CHUNK]) - target).pow(2).mean()
            self.critic_opt.zero_grad()
            loss.backward()
            self.critic_opt.step()
            for a, b in zip(self.target.parameters(), self.critic.parameters()):
                a.data.lerp_(b.data, 0.005)
        return loss.item()

    @torch.no_grad()
    def advantages(self, buffer):
        """How much better each buffered chunk was than what the policy usually does there.

        "q" is the paper's form, Q(s, a) - V(s). Here Q turned out to follow the state closely and the
        action hardly at all, so "td" asks the state values instead: the reward collected over the chunk
        plus the value of where it led, minus the value of where it started."""
        f = buffer.features
        v = torch.stack([self.q(self.critic, f, buffer.samples[:, k]) for k in range(buffer.samples.shape[1])]).mean(0)
        if self.args.advantage == "td":
            ret = torch.as_tensor(np.array(buffer.ret), dtype=torch.float32, device=self.device)
            done = torch.as_tensor(buffer.done, dtype=torch.float32, device=self.device)
            advantage = ret + GAMMA ** CHUNK * (1 - done) * v[torch.as_tensor(buffer.next, device=self.device)] - v
        else:
            advantage = self.q(self.critic, f, buffer.norm_actions[:, :CHUNK]) - v
        # The paper weights by exp(lambda * A) and does not give lambda. A's scale depends on how the reward is
        # scaled, so it is expressed in units of its own spread over the buffer before the temperature applies.
        return advantage / (advantage.std() + 1e-8)

    def fit_policy(self, buffer, advantage, steps):
        """Weighted conditional OT flow matching."""
        a = self.args
        cond = (buffer.features - buffer.features.mean(0)) / (buffer.features.std(0) + 1e-6) / buffer.features.shape[1] ** 0.5
        size = (a.batch, self.policy.config.chunk_size, self.policy.config.max_action_dim)
        for _ in range(steps):
            j = torch.randint(len(buffer), (a.batch,), device=self.device)
            w = torch.exp(a.temperature * advantage[j]).clamp(max=a.max_weight)
            mass = w / w.sum()
            source = j[torch.multinomial(mass, a.batch, replacement=True)]  # each noise sample gets a condition, good ones more often
            z = torch.randn(size, device=self.device)
            target = torch.zeros(size, device=self.device)
            target[:, :, :ACTION_DIM] = buffer.norm_actions[j]
            cost = torch.cdist(z.flatten(1), target.flatten(1)).pow(2) + a.alpha ** 2 * torch.cdist(cond[source], cond[j]).pow(2)
            pick = torch.multinomial(transport(cost, mass, a.epsilon), 1).squeeze(1)  # which action each noise sample flows to
            rows = j[pick].tolist()
            batch = self.batch([buffer.obs[r] for r in rows], [buffer.actions[r] for r in rows], [buffer.pad[r] for r in rows])
            loss, _ = self.policy.forward(batch, noise=z)
            self.policy_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.policy_opt.step()
        return loss.item()


def transport(cost, mass, epsilon, iterations=50):
    """Entropic OT plan between uniform rows and columns of the given mass (log-domain Sinkhorn)."""
    cost = cost / cost.mean()
    log_a, log_b = -torch.log(torch.tensor(float(len(cost)), device=cost.device)), torch.log(mass)
    f, g = torch.zeros(len(cost), device=cost.device), torch.zeros(len(mass), device=cost.device)
    for _ in range(iterations):
        f = epsilon * (log_a - torch.logsumexp((g[None] - cost) / epsilon, 1))
        g = epsilon * (log_b - torch.logsumexp((f[:, None] - cost) / epsilon, 0))
    return torch.exp((f[:, None] + g[None] - cost) / epsilon)


def success_rate(env, learner, scenes, steps=None):
    """Closed-loop success on fixed scenes, optionally with fewer flow integration steps."""
    usual = learner.policy.config.num_steps
    learner.policy.config.num_steps = steps or usual
    results = [rollout(env, learner.policy, learner.pre, learner.post, s) for s in scenes]
    learner.policy.config.num_steps = usual
    return float(np.mean([r["success"] for r in results])), float(np.mean([r["progress"] > 0 for r in results]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--task", choices=list(ENVS), default="cube")
    parser.add_argument("--demos", type=int, nargs="+", default=list(range(0, 30, 3)), help="which hand-following demonstrations (cube task)")
    parser.add_argument("--corrected", action="store_true", help="the base policy was trained on the corrected replays of all cube clips")
    parser.add_argument("--out", default="outputs/train/otql")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--rollouts", type=int, default=10, help="episodes collected per round")
    parser.add_argument("--critic-steps", type=int, default=3000)
    parser.add_argument("--policy-steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=1.5, help="lambda: weight = exp(lambda * advantage in standard deviations)")
    parser.add_argument("--advantage", choices=["q", "td"], default="q")
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--alpha", type=float, default=10.0, help="how strongly the OT plan keeps noise with its own observation")
    parser.add_argument("--epsilon", type=float, default=0.05, help="entropic regularisation, relative to the mean cost")
    parser.add_argument("--samples", type=int, default=4, help="policy actions per state for the value baseline")
    parser.add_argument("--eval-episodes", type=int, default=30)
    args = parser.parse_args()

    policy, pre, post = load_policy(args.checkpoint)
    env = ENVS[args.task](policy.cameras)
    learner = Learner(policy, pre, post, env.task, args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("a")
    eval_scenes = list(itertools.islice(env.scenes(0), args.eval_episodes))
    train_scenes = env.scenes(1)

    def report(**row):
        print(row, flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()

    buffer = Buffer()
    for steps, success in demonstrations(args.task, env, None if args.corrected else args.demos):
        buffer.add(steps, success, policy.config.chunk_size)
    base = success_rate(env, learner, eval_scenes)
    report(round=0, buffer=len(buffer), eval_success=base[0], eval_progress=base[1])

    for k in range(1, args.rounds + 1):
        episodes = [rollout(env, policy, pre, post, next(train_scenes), record=True) for _ in range(args.rollouts)]
        for e in episodes:
            buffer.add(e["trace"], e["success"], policy.config.chunk_size)
        learner.describe(buffer, args.samples)
        critic_loss = learner.fit_critic(buffer, args.critic_steps)
        advantage = learner.advantages(buffer)
        policy_loss = learner.fit_policy(buffer, advantage, args.policy_steps)
        report(round=k, rollouts_succeeded=int(sum(e["success"] for e in episodes)), buffer=len(buffer), critic_loss=critic_loss,
               policy_loss=policy_loss)
        if k % 2 == 0 or k == args.rounds:
            s = success_rate(env, learner, eval_scenes)
            report(round=k, eval_success=s[0], eval_progress=s[1])

    fast = success_rate(env, learner, eval_scenes, steps=3)
    report(round=args.rounds, flow_steps=3, eval_success=fast[0], eval_progress=fast[1])
    path = out / "pretrained_model"
    path.mkdir(exist_ok=True)
    for f in Path(args.checkpoint).iterdir():  # same layout as a lerobot checkpoint, so h2f.evaluate loads it
        if f.name != "model.safetensors":
            shutil.copy(f, path / f.name)
    policy.save_pretrained(path)


if __name__ == "__main__":
    main()
