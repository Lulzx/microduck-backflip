"""Benchmark native MuJoCo threaded rollout on the actual MicroDuck model.

This measures raw physics throughput only.  It is the first gate for replacing
MJLab's single-core MJWarp CPU path with a specialized native backflip vector
environment; it does not claim end-to-end trainer speed.
"""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np
from mujoco import rollout

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks.microduck_backflip_env_cfg import (
    configure_backflip_standing_eval,
)


TASK_ID = "Mjlab-Backflip-Flat-MicroDuck"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 12])
    args = parser.parse_args()

    cfg = load_env_cfg(TASK_ID, play=True)
    cfg.scene.num_envs = 1
    configure_backflip_standing_eval(cfg)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    model = env.sim.mj_model
    source_data = env.sim.mj_data

    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    initial = np.empty(nstate, dtype=mujoco.MJTNUM_DTYPE)
    mujoco.mj_getState(
        model, source_data, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS
    )
    initial = np.repeat(initial[None, :], args.worlds, axis=0)
    control = np.zeros(
        (args.worlds, args.steps, model.nu), dtype=mujoco.MJTNUM_DTYPE
    )

    print(
        f"MicroDuck native rollout: worlds={args.worlds} steps={args.steps} "
        f"nq={model.nq} nv={model.nv} nu={model.nu}"
    )
    for nthread in args.threads:
        workers = [mujoco.MjData(model) for _ in range(nthread)]
        with rollout.Rollout(nthread=nthread) as runner:
            # Warm the persistent pool and model caches.
            runner.rollout(model, workers, initial[:nthread], control[:nthread, :1])
            start = time.perf_counter()
            runner.rollout(model, workers, initial, control)
            elapsed = time.perf_counter() - start
        physics_steps = args.worlds * args.steps
        print(
            f"threads={nthread:2d} elapsed={elapsed:.4f}s "
            f"physics_steps_per_s={physics_steps / elapsed:,.0f}"
        )

    env.close()


if __name__ == "__main__":
    main()
