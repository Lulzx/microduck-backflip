"""Train the backflip discovery policy with threaded native MuJoCo rollout."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab_microduck.native_backflip_env import NativeBackflipVecEnv, TASK_ID


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-name", default="native-rollout-discovery")
    args = parser.parse_args()

    env = NativeBackflipVecEnv(
        num_envs=args.num_envs, nthread=args.threads, seed=args.seed
    )
    agent_cfg = load_rl_cfg(TASK_ID)
    agent_cfg.save_interval = args.save_interval
    agent_cfg.logger = "tensorboard"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = (
        Path("logs/rsl_rl/microduck_backflip")
        / f"{timestamp}_{args.run_name}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    runner_cls = load_runner_cls(TASK_ID)
    if runner_cls is None:
        raise RuntimeError(f"No runner registered for {TASK_ID}")
    runner = runner_cls(env, asdict(agent_cfg), log_dir=str(log_dir), device="cpu")
    if args.checkpoint is not None:
        runner.load(str(args.checkpoint), strict=True, map_location="cpu")
    try:
        runner.learn(num_learning_iterations=args.iterations)
    finally:
        env.close()


if __name__ == "__main__":
    main()
