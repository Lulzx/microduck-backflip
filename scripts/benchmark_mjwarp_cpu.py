"""Benchmark MJWarp's CPU physics path with Warp's native thread pool."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


TASK_ID = "Mjlab-Backflip-Flat-MicroDuck"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--min-tasks", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state-output", type=Path)
    args = parser.parse_args()

    # The native pool is constructed lazily on the first CPU kernel launch.
    # Configure it before importing Warp or MJLab.
    os.environ["WARP_NUM_THREADS"] = str(args.threads)
    os.environ["WARP_CPU_MIN_TASKS"] = str(args.min_tasks)

    import torch

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends
    from mjlab_microduck.tasks.microduck_backflip_env_cfg import (
        configure_backflip_standing_eval,
    )

    configure_torch_backends()
    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    configure_backflip_standing_eval(env_cfg)
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")

    try:
        env.reset()
        for _ in range(args.warmup_steps):
            env.sim.step()

        start = time.perf_counter()
        for _ in range(args.steps):
            env.sim.step()
        elapsed = time.perf_counter() - start

        result = {
            "num_envs": args.num_envs,
            "threads": args.threads,
            "min_tasks": args.min_tasks,
            "measured_steps": args.steps,
            "elapsed_s": elapsed,
            "ms_per_physics_step": elapsed * 1000.0 / args.steps,
            "physics_steps_per_s": args.num_envs * args.steps / elapsed,
            "finite_qpos": bool(torch.isfinite(env.sim.data.qpos).all().item()),
            "finite_qvel": bool(torch.isfinite(env.sim.data.qvel).all().item()),
        }
        if args.state_output is not None:
            import numpy as np

            args.state_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.state_output,
                qpos=env.sim.data.qpos.numpy(),
                qvel=env.sim.data.qvel.numpy(),
                qacc=env.sim.data.qacc.numpy(),
                actuator_force=env.sim.data.actuator_force.numpy(),
            )
            result["state_output"] = str(args.state_output)
        print("MJWARP_CPU_BENCHMARK=" + json.dumps(result, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
