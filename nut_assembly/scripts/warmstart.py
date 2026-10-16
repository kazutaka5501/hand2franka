"""Turn a trained policy into the starting point of a task with extra observation columns.

The new task appends observation terms at the end of the ``policy`` group (e.g. the goal orientation for the 6D
play stage), so the first layers of actor and critic get zero weights for the new inputs and the observation
normalisers start them at mean 0, variance 1. The network then computes exactly what it computed before, and
training is free to start using the new inputs. The optimizer state is reset (its moments have the old shapes).

    python scripts/warmstart.py <checkpoint.pt> <output.pt> --insert_at 195 --count 20
"""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--insert_at", type=int, required=True, help="Index of the first new input column.")
    parser.add_argument("--count", type=int, required=True, help="Number of new input columns.")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    i, n = args.insert_at, args.count
    for net in ("actor", "critic"):
        w = sd[f"{net}.0.weight"]
        sd[f"{net}.0.weight"] = torch.cat([w[:, :i], torch.zeros(w.shape[0], n), w[:, i:]], dim=1)
        for stat, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
            key = f"{net}_obs_normalizer.{stat}"
            v = sd[key]
            sd[key] = torch.cat([v[:, :i], torch.full((1, n), fill), v[:, i:]], dim=1)
    # fresh Adam state over the parameters in the order ActorCritic.parameters() yields them
    params = [torch.nn.Parameter(v.clone()) for k, v in sd.items() if "normalizer" not in k]
    ck["optimizer_state_dict"] = torch.optim.Adam(params, lr=args.lr).state_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, args.output)
    print(f"wrote {args.output}: first layer {tuple(sd['actor.0.weight'].shape)}, from iteration {ck['iter']}")


if __name__ == "__main__":
    main()
