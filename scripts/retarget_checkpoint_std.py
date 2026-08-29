"""Create a reproducible low-variance continuation checkpoint.

This changes only the Gaussian actor standard deviation and clears Adam's
moments. Actor means, critic weights, observation normalizers, iteration, and
environment curriculum state are preserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

STD_KEY = "distribution.std_param"


def retarget_checkpoint(source: Path, destination: Path, std: float) -> None:
    if std <= 0.0:
        raise ValueError("std must be positive")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    actor = checkpoint.get("actor_state_dict")
    if not isinstance(actor, dict) or STD_KEY not in actor:
        raise KeyError(f"checkpoint does not contain actor_state_dict[{STD_KEY!r}]")
    actor[STD_KEY] = torch.full_like(actor[STD_KEY], float(std))

    optimizer = checkpoint.get("optimizer_state_dict")
    if isinstance(optimizer, dict):
        for state in optimizer.get("state", {}).values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = torch.zeros_like(value)

    infos = checkpoint.setdefault("infos", {})
    infos["retargeted_actor_std"] = float(std)
    infos["retargeted_from"] = str(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--std", type=float, required=True)
    args = parser.parse_args()
    retarget_checkpoint(args.source, args.destination, args.std)
    print(f"Wrote {args.destination} with actor std {args.std:g}")


if __name__ == "__main__":
    main()
