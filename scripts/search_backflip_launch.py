"""Parallel scripted search for MicroDuck's achievable jump impulse.

This is a feasibility diagnostic, not a controller for hardware. It sweeps a
supported crouch followed by symmetric leg extension in the randomized
MuJoCo/BAM model and reports the best collision-free launch velocities. The
result tells the RL curriculum whether insufficient flight time is a reward
problem or an actuator-envelope problem.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab_microduck.tasks.microduck_backflip_env_cfg import (
    CROUCH_OVERRIDES,
    configure_backflip_standing_eval,
)


TASK_ID = "Mjlab-Backflip-Flat-MicroDuck"


def search(seed: int, output: Path | None) -> dict:
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    depths = (0.55, 0.70, 0.85, 1.00, 1.15)
    crouch_steps = (5, 9, 13, 17, 21)
    extension_factors = (0.0, 0.35, 0.70, 1.05, 1.40)
    candidates = list(itertools.product(depths, crouch_steps, extension_factors))

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = len(candidates)
    env_cfg.seed = seed
    env_cfg.auto_reset = True
    configure_backflip_standing_eval(env_cfg)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset(seed=seed)

    action_term = env.action_manager.get_term("joint_pos")
    action_dim = action_term.action_dim
    default = env.scene["robot"].data.default_joint_pos[:, action_term.target_ids]
    crouch_delta = torch.zeros(len(candidates), action_dim, device=device)
    for action_index, target in CROUCH_OVERRIDES.items():
        crouch_delta[:, action_index] = target - default[:, action_index]

    depth = torch.tensor([item[0] for item in candidates], device=device)
    duration = torch.tensor([item[1] for item in candidates], device=device)
    extension = torch.tensor([item[2] for item in candidates], device=device)

    launched = torch.zeros(len(candidates), dtype=torch.bool, device=device)
    launch_vz = torch.full((len(candidates),), float("nan"), device=device)
    peak_vz = torch.full((len(candidates),), -float("inf"), device=device)
    peak_pre_takeoff_vz = torch.full_like(peak_vz, -float("inf"))
    peak_height = torch.zeros_like(peak_vz)
    first_body_contact = torch.zeros_like(launched)

    settle_steps = 5
    total_steps = settle_steps + max(crouch_steps) + 35
    with torch.inference_mode():
        for step in range(total_steps):
            relative_step = step - settle_steps
            crouching = (relative_step >= 0) & (relative_step < duration)
            crouch_action = crouch_delta * depth.unsqueeze(1)
            extension_action = -crouch_delta * extension.unsqueeze(1)
            actions = torch.where(crouching.unsqueeze(1), crouch_action, extension_action)
            if relative_step < 0:
                actions.zero_()
            env.step(actions)

            robot = env.scene["robot"].data
            vz = robot.root_link_lin_vel_w[:, 2]
            peak_vz = torch.maximum(peak_vz, vz)
            peak_pre_takeoff_vz = torch.maximum(
                peak_pre_takeoff_vz, torch.where(~launched, vz, -float("inf"))
            )
            height = robot.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]
            peak_height = torch.maximum(peak_height, height)
            launched_now = env._backflip_airborne_latch
            first_launch = launched_now & ~launched
            launch_vz[first_launch] = vz[first_launch]
            launched |= launched_now
            feet = env.scene.sensors["feet_ground_contact"].data.found
            robot_contact = env.scene.sensors["robot_ground_contact"].data.found
            feet_now = (feet.view(len(candidates), -1) > 0).any(dim=-1)
            robot_now = (robot_contact.view(len(candidates), -1) > 0).any(dim=-1)
            first_body_contact |= launched & robot_now & ~feet_now

    rows = []
    for index, (candidate_depth, candidate_steps, candidate_extension) in enumerate(candidates):
        rows.append(
            {
                "depth": candidate_depth,
                "crouch_steps": candidate_steps,
                "crouch_s": candidate_steps * env.step_dt,
                "extension_factor": candidate_extension,
                "launched": bool(launched[index].item()),
                "launch_vz_m_s": (
                    float(launch_vz[index].item())
                    if torch.isfinite(launch_vz[index])
                    else None
                ),
                "peak_pre_takeoff_vz_m_s": float(peak_pre_takeoff_vz[index].item()),
                "peak_vz_m_s": float(peak_vz[index].item()),
                "peak_height_m": float(peak_height[index].item()),
                "body_contact_after_launch": bool(first_body_contact[index].item()),
            }
        )
    rows.sort(key=lambda row: row["peak_pre_takeoff_vz_m_s"], reverse=True)
    result = {
        "task": TASK_ID,
        "device": device,
        "seed": seed,
        "num_candidates": len(rows),
        "best": rows[:20],
        "all": rows,
    }
    print(json.dumps({**result, "all": "omitted from stdout"}, indent=2))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {output}")
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    search(args.seed, args.output)


if __name__ == "__main__":
    main()
