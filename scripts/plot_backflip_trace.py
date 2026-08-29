"""Plot the kinematic trace emitted by ``eval_backflip.py --trace-output``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trace = json.loads(args.trace.read_text())
    steps = trace["steps"]
    t = [row["time_s"] for row in steps]
    series = (
        ("root_height_m", "Root height (m)"),
        ("upward_velocity_m_s", "Vertical velocity (m/s)"),
        ("backward_pitch_rate_rad_s", "Backward pitch rate (rad/s)"),
        ("airborne_rotation_deg", "Airborne rotation (deg)"),
    )

    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    for ax, (key, label) in zip(axes, series, strict=True):
        ax.plot(t, [row[key] for row in steps], color="#2457d6", linewidth=1.8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.22)
        airborne = False
        start = 0.0
        for row in steps:
            active_flight = (
                row["airborne_latched"]
                and not row.get("flight_ended_latched", False)
                and not row["robot_contact"]
            )
            if active_flight and not airborne:
                start = row["time_s"]
                airborne = True
            if airborne and not active_flight:
                ax.axvspan(start, row["time_s"], color="#f59e0b", alpha=0.15)
                airborne = False
        if airborne:
            ax.axvspan(start, t[-1], color="#f59e0b", alpha=0.15)

    axes[-1].axhline(360.0, color="#b91c1c", linestyle="--", linewidth=1.0)
    axes[-1].set_xlabel("Time (s); orange = collision-free airborne interval")
    fig.suptitle(
        f"MicroDuck backflip trace — {Path(trace['checkpoint']).stem}, "
        f"{trace['start_mode']} start, seed {trace['seed']}"
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
