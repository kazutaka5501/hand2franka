"""Residual RL: SmolVLA stays as it is, and a small network learns to nudge its actions.

Every control step the frozen policy proposes an end-effector target and the residual adds up to
1 cm to it (and a little to the gripper command). The residual and its critics are trained with
TD3 on the simulator's state - tool, cube and plate positions - which a camera does not give, so
this is SmolVLA plus a state-based correction, not a better vision policy.

Reward: 1 when the cube rests on the plate, plus a potential that grows as the tool nears the
cube and the cube nears the plate (it only helps credit assignment; it does not change which
behaviour is best).

    python -m h2f.residual outputs/train/bc_follow10/checkpoints/last/pretrained_model
"""
import argparse
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .evaluate import REPLAN, load_policy, to_batch
from .sim import PickPlaceEnv

GAMMA = 0.99
SCALE = np.array([0.01, 0.01, 0.01, 0.0, 0.3])  # largest correction to [x, y, z, yaw, grip]
RESTRAINT = 0.5  # cost of a full-size correction, relative to the typical Q-value: correct only where it pays


def mlp(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, n_out))


def potential(env):
    """Higher is closer to done: tool near the cube while it lies on the table, then cube near the plate."""
    tcp, cube = env.tcp_pose()[0], env.cube_pos
    return -np.linalg.norm(cube[:2] - env.plate[:2]) - 0.5 * np.linalg.norm(tcp - cube)


class Robots:
    """Several simulated arms with the frozen policy's proposals, stepped together."""

    def __init__(self, n, policy, pre, post, seed):
        self.policy, self.pre, self.post = policy, pre, post
        self.envs = [PickPlaceEnv(policy.cameras) for _ in range(n)]
        self.scenes = self.envs[0].scenes(seed)
        self.chunks, self.planned_at = [None] * n, [None] * n
        for i in range(n):
            self.reset(i)

    def reset(self, i, scene=None):
        env = self.envs[i]
        env.obs = env.reset(*(scene or next(self.scenes)))
        self.planned_at[i] = None

    @torch.no_grad()
    def proposals(self):
        """The frozen policy's action for the current step of every arm; arms at a chunk boundary get a new chunk."""
        due = [i for i, env in enumerate(self.envs) if env.t % REPLAN == 0 and self.planned_at[i] != env.t]
        if due:
            p = self.policy
            batch = self.pre(to_batch([self.envs[i].obs for i in due], p.cameras, p.config.device, self.envs[0].task))
            chunks = self.post(p.predict_action_chunk(batch)).float().cpu().numpy()
            for i, chunk in zip(due, chunks):
                self.chunks[i], self.planned_at[i] = chunk, self.envs[i].t
        return np.stack([self.chunks[i][env.t % REPLAN] for i, env in enumerate(self.envs)])

    def features(self, base):
        """Residual / critic input: simulator state and where the frozen policy wants to go from here."""
        return np.stack([np.r_[env.privileged(), b[:3] - env.tcp_pose()[0], b[4]] for env, b in zip(self.envs, base)]).astype(np.float32)

    def step(self, i, action):
        env = self.envs[i]
        before = potential(env)
        env.obs, _, success, timeout = env.step(action, render=(env.t + 1) % REPLAN == 0)
        reward = float(success) + GAMMA * potential(env) - before
        return reward, success, timeout


def evaluate(robot, actor, scenes, device):
    """Success on fixed scenes with the residual applied without exploration noise (or not at all if actor is None)."""
    wins = 0
    for scene in scenes:
        robot.reset(0, scene)
        while True:
            base = robot.proposals()[:1]
            delta = 0 if actor is None else actor(torch.as_tensor(robot.features(base), device=device)).detach().cpu().numpy()[0] * SCALE
            _, success, timeout = robot.step(0, base[0] + delta)
            if success or timeout:
                wins += success
                break
    return wins / len(scenes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--out", default="outputs/train/residual")
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--robots", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20_000, help="steps of the frozen policy alone, to start the critics")
    parser.add_argument("--noise", type=float, default=0.2, help="exploration noise, as a fraction of the largest correction")
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    args = parser.parse_args()

    device = "cuda"
    policy, pre, post = load_policy(args.checkpoint)
    robots = Robots(args.robots, policy, pre, post, seed=1)
    tester = Robots(1, policy, pre, post, seed=0)
    eval_scenes = list(itertools.islice(tester.envs[0].scenes(0), args.eval_episodes))
    n_obs = robots.features(robots.proposals()).shape[1]

    actor = nn.Sequential(mlp(n_obs, 5), nn.Tanh()).to(device)
    nn.init.zeros_(actor[0][-1].weight)  # starts as "no correction"
    nn.init.zeros_(actor[0][-1].bias)
    critics = nn.ModuleList([mlp(n_obs + 5, 1), mlp(n_obs + 5, 1)]).to(device)
    actor_target, critics_target = copy.deepcopy(actor), copy.deepcopy(critics)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = torch.optim.Adam(critics.parameters(), lr=3e-4)
    q = lambda nets, o, a: [net(torch.cat([o, a], 1)).squeeze(1) for net in nets]

    size = args.steps + args.robots
    buf = dict(o=np.zeros((size, n_obs), np.float32), a=np.zeros((size, 5), np.float32), r=np.zeros(size, np.float32),
               o2=np.zeros((size, n_obs), np.float32), d=np.zeros(size, np.float32))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("a")

    def report(**row):
        print(row, flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()

    report(step=0, eval_success=evaluate(tester, None, eval_scenes, device))
    n, recent = 0, []
    while n < args.steps:
        base = robots.proposals()
        obs = robots.features(base)
        with torch.no_grad():
            act = actor(torch.as_tensor(obs, device=device)).cpu().numpy() if n >= args.warmup else np.zeros((args.robots, 5), np.float32)
        act = np.clip(act + args.noise * np.random.randn(*act.shape), -1, 1).astype(np.float32)
        outcome = [robots.step(i, base[i] + act[i] * SCALE) for i in range(args.robots)]
        for i, (reward, success, timeout) in enumerate(outcome):
            if success or timeout:
                recent.append(success)
                robots.reset(i)
        nxt = robots.features(robots.proposals())  # after a reset this row is unused: the episode ended
        for i, (reward, success, timeout) in enumerate(outcome):
            buf["o"][n], buf["a"][n], buf["r"][n], buf["o2"][n], buf["d"][n] = obs[i], act[i], reward, nxt[i], float(success or timeout)
            n += 1

        if n >= 5000:
            for k in range(args.robots):
                idx = np.random.randint(0, n, 256)
                o, a, r, o2, d = (torch.as_tensor(buf[key][idx], device=device) for key in ("o", "a", "r", "o2", "d"))
                with torch.no_grad():
                    a2 = (actor_target(o2) + (0.2 * torch.randn_like(a)).clamp(-0.5, 0.5)).clamp(-1, 1)
                    target = r + GAMMA * (1 - d) * torch.min(*q(critics_target, o2, a2))
                critic_loss = sum((qi - target).pow(2).mean() for qi in q(critics, o, a))
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()
                if k % 2 == 0 and n >= args.warmup:
                    proposed = actor(o)
                    value = q(critics[:1], o, proposed)[0]
                    actor_loss = -value.mean() / value.abs().mean().detach() + RESTRAINT * proposed.pow(2).mean()
                    actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_opt.step()
                    for net, tgt in ((actor, actor_target), (critics, critics_target)):
                        for p, pt in zip(net.parameters(), tgt.parameters()):
                            pt.data.lerp_(p.data, 0.005)

        if n % args.eval_every < args.robots:
            report(step=n, episodes=len(recent), train_success=float(np.mean(recent[-100:])) if recent else None,
                   eval_success=evaluate(tester, actor if n >= args.warmup else None, eval_scenes, device))
            torch.save(actor.state_dict(), out / "residual.pt")


if __name__ == "__main__":
    main()
