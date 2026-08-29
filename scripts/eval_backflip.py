"""Headless, standing-start evaluation for a trained backflip checkpoint.

This intentionally does not use aggregate reward as the success criterion.
It reports the physical gates separately: takeoff, airborne rotation, feet-first
landing, and stable upright completion.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab_microduck.tasks.microduck_backflip_env_cfg import (
    configure_backflip_standing_eval,
)


TASK_ID = "Mjlab-Backflip-Flat-MicroDuck"
STABLE_HOLD_S = 0.5


class _SelectedEnvVideoRecorder(VideoRecorder):
    """VideoRecorder variant that renders a chosen vectorized environment."""

    def __init__(self, *args, environment_index: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.environment_index = environment_index

    def _record_frame(self) -> None:
        if self._wrapped_env.render_mode != "rgb_array":
            return
        frame = self._wrapped_env.render()
        if frame is None:
            return
        if isinstance(frame, np.ndarray) and frame.ndim == 4:
            frame = frame[self.environment_index]
        self.current_video_frames.append(frame)


def _summary(x: torch.Tensor) -> dict[str, float]:
    x = x.float()
    return {
        "min": float(x.min().item()),
        "mean": float(x.mean().item()),
        "p10": float(torch.quantile(x, 0.10).item()),
        "p50": float(torch.quantile(x, 0.50).item()),
        "p90": float(torch.quantile(x, 0.90).item()),
        "max": float(x.max().item()),
    }


def _finite_time_summary(x: torch.Tensor) -> dict[str, float | None]:
    finite = x[torch.isfinite(x)]
    if finite.numel() == 0:
        return {"mean": None, "p10": None, "p50": None, "p90": None}
    return _summary(finite)


def _specialist_observation(
    obs: torch.Tensor, elapsed_steps: torch.Tensor, step_dt: float
) -> torch.Tensor:
    """Rebase the two backflip context slots to a specialist-local episode.

    Actor indices 55:61 are the reused body-command slots. Specialists were
    trained with their own phase starting at zero and without the launch
    spotter, so forwarding the parent episode's phase/assist values is an
    observation-distribution error even though physical state is shared.
    """
    specialist_obs = obs.clone()
    elapsed_s = elapsed_steps.to(specialist_obs.dtype) * float(step_dt)
    x = elapsed_s / 0.30
    specialist_obs[:, 55] = x.pow(3) / (1.0 + x.pow(3))
    specialist_obs[:, 56] = 0.0
    return specialist_obs


def evaluate(
    checkpoint: Path,
    num_envs: int,
    seed: int,
    output: Path | None,
    start_mode: str = "standing",
    video_dir: Path | None = None,
    render_env_index: int = 0,
    trace_output: Path | None = None,
    assist_scale: float = 0.0,
    assist_force_n: float | None = None,
    assist_torque_nm: float | None = None,
    assist_start_s: float | None = None,
    assist_end_s: float | None = None,
    task_id: str = TASK_ID,
    recovery_checkpoint: Path | None = None,
    recovery_profile: str | None = None,
    recovery_switch_mode: str = "landing",
    post_landing_checkpoint: Path | None = None,
    recovery_approach_angle_deg: float = 330.0,
    recovery_approach_tilt_deg: float = 40.0,
    recovery_approach_height_m: float = 0.36,
    snapshot_output: Path | None = None,
    snapshot_angle_deg: float = 260.0,
) -> dict:
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    if render_env_index < 0 or render_env_index >= num_envs:
        raise ValueError("render_env_index must identify one of the vectorized envs")

    # The acceptance battery is standing-start only.  Reverse-curriculum
    # mid-flight states are valuable training data but are not full backflips.
    if start_mode != "task":
        configure_backflip_standing_eval(env_cfg)
    if assist_scale < 0.0 or assist_scale > 1.0:
        raise ValueError("assist_scale must be in [0, 1]")
    if not 0.0 <= recovery_approach_angle_deg <= 360.0:
        raise ValueError("recovery_approach_angle_deg must be in [0, 360]")
    if not 0.0 <= recovery_approach_tilt_deg <= 180.0:
        raise ValueError("recovery_approach_tilt_deg must be in [0, 180]")
    if recovery_approach_height_m <= 0.0:
        raise ValueError("recovery_approach_height_m must be positive")
    if not 0.0 <= snapshot_angle_deg <= 360.0:
        raise ValueError("snapshot_angle_deg must be in [0, 360]")
    if start_mode != "task":
        env_cfg.events["set_backflip_state"].params["initial_assist_scale"] = (
            assist_scale
        )
    assist_cfg = env_cfg.events["backflip_assistive_wrench"].params
    if assist_force_n is not None:
        assist_cfg["upward_force_n"] = assist_force_n
    if assist_torque_nm is not None:
        assist_cfg["backward_pitch_torque_nm"] = assist_torque_nm
    if assist_start_s is not None:
        assist_cfg["start_time_s"] = assist_start_s
    if assist_end_s is not None:
        assist_cfg["end_time_s"] = assist_end_s
    if start_mode == "crouch":
        reset = env_cfg.events["set_backflip_state"].params
        reset["standing_prob"] = 0.0
        reset["crouch_prob"] = 1.0
    elif start_mode == "midflight":
        reset = env_cfg.events["set_backflip_state"].params
        reset["standing_prob"] = 0.0
        reset["midflight_prob"] = 1.0
    elif start_mode == "recovery":
        reset = env_cfg.events["set_backflip_state"].params
        reset["standing_prob"] = 0.0
        reset["recovery_prob"] = 1.0
    elif start_mode not in ("standing", "task"):
        raise ValueError(f"Unsupported start mode: {start_mode}")
    if recovery_profile is not None:
        if start_mode != "recovery":
            raise ValueError("recovery_profile requires --start-mode recovery")
        profiles = {
            "easy": {
                "recovery_z_range": (0.112, 0.118),
                "recovery_tilt_max": math.radians(5.0),
                "recovery_lin_vel_max": 0.08,
                "recovery_ang_vel_max": 0.75,
                "recovery_vertical_velocity_range": (-0.40, -0.05),
                "joint_noise_std": 0.01,
            },
            "medium": {
                "recovery_z_range": (0.108, 0.12),
                "recovery_tilt_max": math.radians(15.0),
                "recovery_lin_vel_max": 0.30,
                "recovery_ang_vel_max": 3.0,
                "recovery_vertical_velocity_range": (-1.25, -0.10),
                "joint_noise_std": 0.03,
            },
            "hard": {
                "recovery_z_range": (0.105, 0.125),
                "recovery_tilt_max": math.radians(30.0),
                "recovery_lin_vel_max": 0.75,
                "recovery_ang_vel_max": 12.0,
                "recovery_vertical_velocity_range": (-2.50, -0.20),
                "joint_noise_std": 0.08,
            },
            "extreme": {
                "recovery_z_range": (0.100, 0.13),
                "recovery_tilt_max": math.radians(45.0),
                "recovery_lin_vel_max": 1.20,
                "recovery_ang_vel_max": 20.0,
                "recovery_vertical_velocity_range": (-3.25, -0.25),
                "joint_noise_std": 0.10,
            },
        }
        env_cfg.events["set_backflip_state"].params.update(profiles[recovery_profile])

    if video_dir is not None:
        env_cfg.viewer.width = 1280
        env_cfg.viewer.height = 720
        env_cfg.viewer.distance = 0.55
        env_cfg.viewer.azimuth = 90.0
        env_cfg.viewer.elevation = -8.0
    base_env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=device,
        render_mode="rgb_array" if video_dir is not None else None,
    )
    rollout_env = base_env
    if video_dir is not None:
        rollout_env = _SelectedEnvVideoRecorder(
            base_env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=base_env.max_episode_length,
            name_prefix=f"backflip-{checkpoint.stem}-{start_mode}",
            disable_logger=True,
            environment_index=render_env_index,
        )
    env = RslRlVecEnvWrapper(rollout_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id)
    if runner_cls is None:
        raise RuntimeError(f"No runner registered for {task_id}")
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)
    recovery_policy = None
    if recovery_checkpoint is not None:
        recovery_runner = runner_cls(env, asdict(agent_cfg), device=device)
        recovery_runner.load(
            str(recovery_checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        recovery_policy = recovery_runner.get_inference_policy(device=device)
    post_landing_policy = None
    if post_landing_checkpoint is not None:
        if recovery_checkpoint is None:
            raise ValueError(
                "post_landing_checkpoint requires an approach/recovery checkpoint"
            )
        post_landing_runner = runner_cls(env, asdict(agent_cfg), device=device)
        post_landing_runner.load(
            str(post_landing_checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        post_landing_policy = post_landing_runner.get_inference_policy(device=device)

    obs = env.get_observations()
    launched = torch.zeros(num_envs, dtype=torch.bool, device=device)
    rotated_300 = torch.zeros_like(launched)
    rotated_340 = torch.zeros_like(launched)
    rotated_360 = torch.zeros_like(launched)
    landed = torch.zeros_like(launched)
    stable = torch.zeros_like(launched)
    stable_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    max_stable_steps = torch.zeros_like(stable_steps)
    posture_steps = torch.zeros_like(stable_steps)
    max_posture_steps = torch.zeros_like(stable_steps)
    ever_landing_feet = torch.zeros_like(launched)
    ever_landing_upright = torch.zeros_like(launched)
    ever_landing_height = torch.zeros_like(launched)
    ever_landing_low_angular_speed = torch.zeros_like(launched)
    body_contact = torch.zeros_like(launched)
    nonfinite_state = torch.zeros_like(launched)
    recovery_active = torch.zeros_like(launched)
    post_landing_active = torch.zeros_like(launched)
    recovery_elapsed_steps = torch.zeros(
        num_envs, dtype=torch.long, device=device
    )
    post_landing_elapsed_steps = torch.zeros_like(recovery_elapsed_steps)
    peak_z = torch.zeros(num_envs, device=device)
    peak_rotation = torch.zeros(num_envs, device=device)
    peak_ang_vel = torch.zeros(num_envs, device=device)
    peak_backward_pitch_rate = torch.zeros(num_envs, device=device)
    peak_body_roll_rate = torch.zeros(num_envs, device=device)
    peak_axial_yaw_rate = torch.zeros(num_envs, device=device)
    peak_out_of_plane_rate = torch.zeros(num_envs, device=device)
    peak_lateral_axis_tilt = torch.zeros(num_envs, device=device)
    peak_upward_velocity = torch.zeros(num_envs, device=device)
    peak_pre_takeoff_upward_velocity = torch.zeros(num_envs, device=device)
    peak_pre_takeoff_backward_pitch_rate = torch.zeros(num_envs, device=device)
    peak_pre_takeoff_coupled_quality = torch.zeros(num_envs, device=device)
    backward_pitch_rate_at_takeoff = torch.full_like(
        peak_backward_pitch_rate, float("nan")
    )
    upward_velocity_at_takeoff = torch.full_like(
        peak_backward_pitch_rate, float("nan")
    )
    takeoff_time = torch.full((num_envs,), float("nan"), device=device)
    body_contact_time = torch.full_like(takeoff_time, float("nan"))
    first_ground_contact_time = torch.full_like(takeoff_time, float("nan"))
    rotation_at_first_ground_contact = torch.full_like(takeoff_time, float("nan"))
    first_ground_contact_was_feet = torch.zeros_like(launched)
    rotation_300_time = torch.full_like(takeoff_time, float("nan"))
    landing_time = torch.full_like(takeoff_time, float("nan"))
    stable_time = torch.full_like(takeoff_time, float("nan"))
    trace_steps: list[dict] = []
    trace_idx = render_env_index
    snapshot_taken = torch.zeros_like(launched)
    snapshots: list[dict] = []

    with torch.inference_mode():
        for step in range(base_env.max_episode_length):
            actions = policy(obs)
            if recovery_policy is not None and recovery_active.any():
                recovery_obs = _specialist_observation(
                    obs, recovery_elapsed_steps, base_env.step_dt
                )
                recovery_actions = recovery_policy(recovery_obs)
                actions = torch.where(
                    recovery_active.unsqueeze(1), recovery_actions, actions
                )
            if post_landing_policy is not None and post_landing_active.any():
                post_landing_obs = _specialist_observation(
                    obs, post_landing_elapsed_steps, base_env.step_dt
                )
                post_landing_actions = post_landing_policy(post_landing_obs)
                actions = torch.where(
                    post_landing_active.unsqueeze(1), post_landing_actions, actions
                )
            obs, _, _, _ = env.step(actions)
            # Count only steps on which a specialist actually acted. Newly
            # activated worlds therefore see local phase zero on their first
            # specialist action at the next control tick.
            recovery_elapsed_steps += recovery_active.long()
            post_landing_elapsed_steps += post_landing_active.long()
            now = (step + 1) * base_env.step_dt
            launched_now = base_env._backflip_airborne_latch
            landed_now = base_env._backflip_landed_latch
            if snapshot_output is not None:
                snapshot_now = (
                    launched_now
                    & ~base_env._backflip_flight_ended_latch
                    & ~snapshot_taken
                    & (
                        base_env._backflip_max
                        >= math.radians(snapshot_angle_deg)
                    )
                )
                snapshot_ids = torch.nonzero(snapshot_now, as_tuple=False).flatten()
                if snapshot_ids.numel() > 0:
                    qpos = base_env.sim.data.qpos[snapshot_ids].detach().clone()
                    qvel = base_env.sim.data.qvel[snapshot_ids].detach().clone()
                    origins = base_env.scene.terrain.env_origins[snapshot_ids]
                    qpos[:, :3] -= origins
                    for row, env_idx in enumerate(snapshot_ids.tolist()):
                        snapshots.append(
                            {
                                "source_environment_index": env_idx,
                                "time_s": float(
                                    base_env.episode_length_buf[env_idx].item()
                                    * base_env.step_dt
                                ),
                                "rotation_rad": float(
                                    base_env._backflip_max[env_idx].item()
                                ),
                                "qpos_local": [
                                    float(x) for x in qpos[row].tolist()
                                ],
                                "qvel": [float(x) for x in qvel[row].tolist()],
                                "previous_action": [
                                    float(x) for x in actions[env_idx].tolist()
                                ],
                            }
                        )
                    snapshot_taken |= snapshot_now
            # The launch actor owns the complete ballistic maneuver. Switch
            # only after the state machine validates a rotated, upright,
            # feet-first contact clear of the launch surface.
            recovery_active |= landed_now
            post_landing_active |= landed_now
            if recovery_policy is not None and recovery_switch_mode == "approach":
                robot_state = base_env.scene["robot"].data
                quat_now = robot_state.root_link_quat_w
                upright_now = 1.0 - 2.0 * (
                    quat_now[:, 1].square() + quat_now[:, 2].square()
                )
                relative_z = (
                    robot_state.root_link_pos_w[:, 2]
                    - base_env.scene.terrain.env_origins[:, 2]
                )
                approach_now = (
                    launched_now
                    & ~base_env._backflip_flight_ended_latch
                    & (
                        base_env._backflip_max
                        >= math.radians(recovery_approach_angle_deg)
                    )
                    & (
                        upright_now
                        >= math.cos(math.radians(recovery_approach_tilt_deg))
                    )
                    & (relative_z <= recovery_approach_height_m)
                    & (robot_state.root_link_lin_vel_w[:, 2] < 0.0)
                )
                recovery_active |= approach_now
                # The assisted launch actor is deliberately restricted to a
                # small residual around nominal PD. A zero-assist touchdown
                # specialist was trained with full joint-target authority, so
                # selecting it while leaving that clamp active is not a real
                # policy hand-off. The spotter pulse has already ended by this
                # late-flight gate; clear eligibility only for switched worlds
                # so the next specialist action uses its training authority.
                base_env._backflip_assist_eligible &= ~approach_now
            first_takeoff = launched_now & ~launched
            first_300 = (base_env._backflip_max >= math.radians(300.0)) & ~rotated_300
            first_landing = landed_now & ~landed
            takeoff_time[first_takeoff] = now
            pitch_rate_back = -base_env.scene["robot"].data.root_link_ang_vel_b[:, 1]
            upward_velocity = base_env.scene["robot"].data.root_link_lin_vel_w[:, 2]
            pre_takeoff = ~launched
            peak_pre_takeoff_upward_velocity = torch.maximum(
                peak_pre_takeoff_upward_velocity,
                torch.where(
                    pre_takeoff, torch.clamp(upward_velocity, min=0.0), 0.0
                ),
            )
            peak_pre_takeoff_backward_pitch_rate = torch.maximum(
                peak_pre_takeoff_backward_pitch_rate,
                torch.where(
                    pre_takeoff, torch.clamp(pitch_rate_back, min=0.0), 0.0
                ),
            )
            coupled_quality = torch.minimum(
                torch.clamp(upward_velocity / 1.2, min=0.0, max=1.0),
                torch.clamp(pitch_rate_back / 10.0, min=0.0, max=1.0),
            )
            peak_pre_takeoff_coupled_quality = torch.maximum(
                peak_pre_takeoff_coupled_quality,
                torch.where(pre_takeoff, coupled_quality, 0.0),
            )
            backward_pitch_rate_at_takeoff[first_takeoff] = pitch_rate_back[first_takeoff]
            upward_velocity_at_takeoff[first_takeoff] = upward_velocity[first_takeoff]
            rotation_300_time[first_300] = now
            landing_time[first_landing] = now
            launched |= launched_now
            landed |= landed_now
            peak_z = torch.maximum(peak_z, base_env._backflip_max_z)
            peak_rotation = torch.maximum(peak_rotation, base_env._backflip_max)
            rotated_300 |= base_env._backflip_max >= math.radians(300.0)
            rotated_340 |= base_env._backflip_max >= math.radians(340.0)
            rotated_360 |= base_env._backflip_max >= 2 * math.pi
            robot = base_env.scene["robot"].data
            quat = robot.root_link_quat_w
            upright = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
            lateral_axis_z = 2.0 * (
                quat[:, 2] * quat[:, 3] + quat[:, 0] * quat[:, 1]
            )
            height = robot.root_link_pos_w[:, 2] - base_env.scene.terrain.env_origins[:, 2]
            feet_found = base_env.scene.sensors["feet_ground_contact"].data.found
            feet_now = (feet_found.view(num_envs, -1) > 0).any(dim=-1)
            robot_found = base_env.scene.sensors["robot_ground_contact"].data.found
            robot_contact_now = (robot_found.view(num_envs, -1) > 0).any(dim=-1)
            first_ground_contact = (
                launched_now
                & robot_contact_now
                & ~torch.isfinite(first_ground_contact_time)
            )
            first_ground_contact_time[first_ground_contact] = now
            rotation_at_first_ground_contact[first_ground_contact] = (
                base_env._backflip_max[first_ground_contact]
            )
            first_ground_contact_was_feet[first_ground_contact] = feet_now[
                first_ground_contact
            ]
            body_contact_now = launched_now & robot_contact_now & ~feet_now
            first_body_contact = body_contact_now & ~body_contact
            body_contact_time[first_body_contact] = now
            body_contact |= body_contact_now
            landing_eligible = base_env._backflip_landed_latch & (
                base_env._backflip_max >= math.radians(340.0)
            )
            upright_ok = upright > math.cos(math.radians(20.0))
            height_ok = height >= 0.095
            angular_speed = robot.root_link_ang_vel_b.norm(dim=-1)
            low_angular_speed = angular_speed < 2.0
            ever_landing_feet |= landing_eligible & feet_now
            ever_landing_upright |= landing_eligible & upright_ok
            ever_landing_height |= landing_eligible & height_ok
            ever_landing_low_angular_speed |= landing_eligible & low_angular_speed
            posture_now = landing_eligible & feet_now & upright_ok & height_ok
            posture_steps = torch.where(
                posture_now, posture_steps + 1, torch.zeros_like(posture_steps)
            )
            max_posture_steps = torch.maximum(max_posture_steps, posture_steps)
            stable_now = posture_now & low_angular_speed
            stable_steps = torch.where(
                stable_now, stable_steps + 1, torch.zeros_like(stable_steps)
            )
            max_stable_steps = torch.maximum(max_stable_steps, stable_steps)
            required_stable_steps = math.ceil(STABLE_HOLD_S / base_env.step_dt)
            held_stable = stable_steps >= required_stable_steps
            first_stable = held_stable & ~stable
            stable_time[first_stable] = now - (
                required_stable_steps - 1
            ) * base_env.step_dt
            stable |= held_stable
            peak_ang_vel = torch.maximum(
                peak_ang_vel, robot.root_link_ang_vel_b.norm(dim=-1)
            )
            peak_backward_pitch_rate = torch.maximum(
                peak_backward_pitch_rate, pitch_rate_back
            )
            body_roll_rate = robot.root_link_ang_vel_b[:, 0].abs()
            axial_yaw_rate = robot.root_link_ang_vel_b[:, 2].abs()
            out_of_plane_rate = torch.sqrt(
                body_roll_rate.square() + axial_yaw_rate.square()
            )
            active_flight = (
                base_env._backflip_airborne_latch
                & ~base_env._backflip_flight_ended_latch
            )
            peak_body_roll_rate = torch.maximum(
                peak_body_roll_rate, torch.where(active_flight, body_roll_rate, 0.0)
            )
            peak_axial_yaw_rate = torch.maximum(
                peak_axial_yaw_rate, torch.where(active_flight, axial_yaw_rate, 0.0)
            )
            peak_out_of_plane_rate = torch.maximum(
                peak_out_of_plane_rate,
                torch.where(active_flight, out_of_plane_rate, 0.0),
            )
            peak_lateral_axis_tilt = torch.maximum(
                peak_lateral_axis_tilt,
                torch.where(
                    active_flight,
                    torch.asin(torch.clamp(lateral_axis_z.abs(), 0.0, 1.0)),
                    0.0,
                ),
            )
            peak_upward_velocity = torch.maximum(peak_upward_velocity, upward_velocity)
            sim_data = base_env.sim.data
            nonfinite_state |= ~(
                torch.isfinite(sim_data.qpos).all(dim=-1)
                & torch.isfinite(sim_data.qvel).all(dim=-1)
            )
            if trace_output is not None:
                # Keep the raw 14-D action for the selected deterministic world
                # so launch/contact mechanics can be inspected without changing
                # the actor observation contract.
                trace_steps.append(
                    {
                        "time_s": now,
                        "root_height_m": float(height[trace_idx].item()),
                        "upward_velocity_m_s": float(upward_velocity[trace_idx].item()),
                        "backward_pitch_rate_rad_s": float(pitch_rate_back[trace_idx].item()),
                        "body_roll_rate_rad_s": float(
                            robot.root_link_ang_vel_b[trace_idx, 0].item()
                        ),
                        "axial_yaw_rate_rad_s": float(
                            robot.root_link_ang_vel_b[trace_idx, 2].item()
                        ),
                        "lateral_axis_tilt_deg": float(
                            torch.rad2deg(
                                torch.asin(
                                    torch.clamp(
                                        lateral_axis_z[trace_idx].abs(), 0.0, 1.0
                                    )
                                )
                            ).item()
                        ),
                        "airborne_rotation_deg": float(
                            torch.rad2deg(base_env._backflip_accum[trace_idx]).item()
                        ),
                        "upright_tilt_deg": float(
                            torch.rad2deg(
                                torch.acos(
                                    torch.clamp(upright[trace_idx], -1.0, 1.0)
                                )
                            ).item()
                        ),
                        "angular_speed_rad_s": float(angular_speed[trace_idx].item()),
                        "feet_contact": bool(feet_now[trace_idx].item()),
                        "robot_contact": bool(robot_contact_now[trace_idx].item()),
                        "landing_eligible": bool(landing_eligible[trace_idx].item()),
                        "upright_ok": bool(upright_ok[trace_idx].item()),
                        "height_ok": bool(height_ok[trace_idx].item()),
                        "low_angular_speed": bool(low_angular_speed[trace_idx].item()),
                        "posture_ok": bool(posture_now[trace_idx].item()),
                        "stable_now": bool(stable_now[trace_idx].item()),
                        "stable_steps": int(stable_steps[trace_idx].item()),
                        "airborne_latched": bool(launched_now[trace_idx].item()),
                        "flight_ended_latched": bool(
                            base_env._backflip_flight_ended_latch[trace_idx].item()
                        ),
                        "landed_latched": bool(landed_now[trace_idx].item()),
                        "recovery_policy_active": bool(
                            recovery_active[trace_idx].item()
                        ),
                        "post_landing_policy_active": bool(
                            post_landing_active[trace_idx].item()
                        ),
                        "action": [float(x) for x in actions[trace_idx].tolist()],
                    }
                )

    top_indices = torch.topk(
        peak_rotation, k=min(5, num_envs), largest=True
    ).indices.tolist()
    top_rotation_trials = []
    for index in top_indices:
        flight_s = first_ground_contact_time[index] - takeoff_time[index]
        top_rotation_trials.append(
            {
                "environment_index": index,
                "rotation_deg": float(torch.rad2deg(peak_rotation[index]).item()),
                "peak_height_m": float(peak_z[index].item()),
                "takeoff_upward_velocity_m_s": (
                    float(upward_velocity_at_takeoff[index].item())
                    if torch.isfinite(upward_velocity_at_takeoff[index])
                    else None
                ),
                "takeoff_backward_pitch_rate_rad_s": (
                    float(backward_pitch_rate_at_takeoff[index].item())
                    if torch.isfinite(backward_pitch_rate_at_takeoff[index])
                    else None
                ),
                "peak_pre_takeoff_upward_velocity_m_s": float(
                    peak_pre_takeoff_upward_velocity[index].item()
                ),
                "peak_lateral_axis_tilt_deg": float(
                    torch.rad2deg(peak_lateral_axis_tilt[index]).item()
                ),
                "uninterrupted_flight_s": (
                    float(flight_s.item()) if torch.isfinite(flight_s) else None
                ),
                "first_ground_contact_was_feet": bool(
                    first_ground_contact_was_feet[index].item()
                ),
            }
        )

    top_stability_indices = torch.topk(
        max_stable_steps, k=min(5, num_envs), largest=True
    ).indices.tolist()
    top_stability_trials = [
        {
            "environment_index": index,
            "strict_stable_hold_s": float(
                (max_stable_steps[index].float() * base_env.step_dt).item()
            ),
            "posture_hold_s": float(
                (max_posture_steps[index].float() * base_env.step_dt).item()
            ),
            "rotation_deg": float(torch.rad2deg(peak_rotation[index]).item()),
            "first_ground_contact_was_feet": bool(
                first_ground_contact_was_feet[index].item()
            ),
        }
        for index in top_stability_indices
    ]

    result = {
        "task": task_id,
        "checkpoint": str(checkpoint.resolve()),
        "recovery_checkpoint": (
            str(recovery_checkpoint.resolve())
            if recovery_checkpoint is not None
            else None
        ),
        "post_landing_checkpoint": (
            str(post_landing_checkpoint.resolve())
            if post_landing_checkpoint is not None
            else None
        ),
        "device": device,
        "seed": seed,
        "episodes": num_envs,
        "start_mode": start_mode,
        "recovery_profile": recovery_profile,
        "recovery_switch_mode": recovery_switch_mode,
        "recovery_approach_gate": {
            "angle_deg": recovery_approach_angle_deg,
            "tilt_deg": recovery_approach_tilt_deg,
            "height_m": recovery_approach_height_m,
            "descending": True,
        },
        "standing_start_only": start_mode == "standing",
        "assist_scale": assist_scale,
        "assist_wrench": {
            "upward_force_n": assist_cfg["upward_force_n"],
            "backward_pitch_torque_nm": assist_cfg["backward_pitch_torque_nm"],
            "start_time_s": assist_cfg["start_time_s"],
            "end_time_s": assist_cfg["end_time_s"],
        },
        "stable_hold_s": STABLE_HOLD_S,
        "landed_environment_indices": torch.nonzero(landed).flatten().tolist(),
        "stable_environment_indices": torch.nonzero(stable).flatten().tolist(),
        "rates": {
            "takeoff": float(launched.float().mean().item()),
            "airborne_rotation_ge_300deg": float(rotated_300.float().mean().item()),
            "airborne_rotation_ge_340deg": float(rotated_340.float().mean().item()),
            "airborne_rotation_ge_360deg": float(rotated_360.float().mean().item()),
            "feet_first_landing": float(landed.float().mean().item()),
            "stable_landing": float(stable.float().mean().item()),
            "body_only_ground_contact": float(body_contact.float().mean().item()),
            "first_ground_contact_was_feet": float(
                first_ground_contact_was_feet.float().mean().item()
            ),
            "nonfinite_state": float(nonfinite_state.float().mean().item()),
            "recovery_policy_activated": float(
                recovery_active.float().mean().item()
            ),
            "post_landing_policy_activated": float(
                post_landing_active.float().mean().item()
            ),
            "ever_landing_feet": float(ever_landing_feet.float().mean().item()),
            "ever_landing_upright": float(
                ever_landing_upright.float().mean().item()
            ),
            "ever_landing_height": float(ever_landing_height.float().mean().item()),
            "ever_landing_low_angular_speed": float(
                ever_landing_low_angular_speed.float().mean().item()
            ),
        },
        "longest_posture_hold_s": _summary(max_posture_steps.float() * base_env.step_dt),
        "longest_strict_stable_hold_s": _summary(
            max_stable_steps.float() * base_env.step_dt
        ),
        "top_rotation_trials": top_rotation_trials,
        "top_stability_trials": top_stability_trials,
        "peak_height_m": _summary(peak_z),
        "peak_airborne_rotation_deg": _summary(torch.rad2deg(peak_rotation)),
        "peak_angular_speed_rad_s": _summary(peak_ang_vel),
        "peak_backward_pitch_rate_rad_s": _summary(peak_backward_pitch_rate),
        "peak_body_roll_rate_rad_s": _summary(peak_body_roll_rate),
        "peak_axial_yaw_rate_rad_s": _summary(peak_axial_yaw_rate),
        "peak_out_of_plane_rate_rad_s": _summary(peak_out_of_plane_rate),
        "peak_lateral_axis_tilt_deg": _summary(
            torch.rad2deg(peak_lateral_axis_tilt)
        ),
        "backward_pitch_rate_at_takeoff_rad_s": _finite_time_summary(
            backward_pitch_rate_at_takeoff
        ),
        "upward_velocity_at_takeoff_m_s": _finite_time_summary(
            upward_velocity_at_takeoff
        ),
        "peak_upward_velocity_m_s": _summary(peak_upward_velocity),
        "peak_pre_takeoff_upward_velocity_m_s": _summary(
            peak_pre_takeoff_upward_velocity
        ),
        "peak_pre_takeoff_backward_pitch_rate_rad_s": _summary(
            peak_pre_takeoff_backward_pitch_rate
        ),
        "peak_pre_takeoff_coupled_quality": _summary(
            peak_pre_takeoff_coupled_quality
        ),
        "rotation_at_first_ground_contact_deg": _finite_time_summary(
            torch.rad2deg(rotation_at_first_ground_contact)
        ),
        "timing_s": {
            "takeoff": _finite_time_summary(takeoff_time),
            "first_ground_contact": _finite_time_summary(first_ground_contact_time),
            "uninterrupted_flight": _finite_time_summary(
                first_ground_contact_time - takeoff_time
            ),
            "body_only_ground_contact": _finite_time_summary(body_contact_time),
            "airborne_until_body_contact": _finite_time_summary(
                body_contact_time - takeoff_time
            ),
            "rotation_300deg": _finite_time_summary(rotation_300_time),
            "feet_first_landing": _finite_time_summary(landing_time),
            "stable_landing": _finite_time_summary(stable_time),
        },
    }
    print(json.dumps(result, indent=2))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {output}")
    if trace_output is not None:
        trace_output.parent.mkdir(parents=True, exist_ok=True)
        trace = {
            "checkpoint": str(checkpoint.resolve()),
            "seed": seed,
            "start_mode": start_mode,
            "environment_index": trace_idx,
            "step_dt_s": base_env.step_dt,
            "steps": trace_steps,
        }
        trace_output.write_text(json.dumps(trace, indent=2) + "\n")
        print(f"Wrote {trace_output}")
    if snapshot_output is not None:
        snapshot_output.parent.mkdir(parents=True, exist_ok=True)
        snapshot_payload = {
            "schema_version": 1,
            "checkpoint": str(checkpoint.resolve()),
            "task": task_id,
            "seed": seed,
            "assist_scale": assist_scale,
            "episodes": num_envs,
            "capture_angle_deg": snapshot_angle_deg,
            "step_dt_s": base_env.step_dt,
            "qpos_coordinates": "root_xyz_relative_to_terrain_origin",
            "snapshots": snapshots,
        }
        snapshot_output.write_text(json.dumps(snapshot_payload, indent=2) + "\n")
        print(f"Wrote {snapshot_output} ({len(snapshots)} snapshots)")
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--task-id", default=TASK_ID)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--start-mode",
        choices=("standing", "crouch", "midflight", "recovery", "task"),
        default="standing",
    )
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument(
        "--render-env-index",
        type=int,
        default=0,
        help="Vectorized environment index to record (default: 0)",
    )
    parser.add_argument(
        "--assist-scale",
        type=float,
        default=0.0,
        help="Diagnostic only: virtual-spotter scale (strict acceptance is 0)",
    )
    parser.add_argument("--assist-force-n", type=float)
    parser.add_argument("--assist-torque-nm", type=float)
    parser.add_argument("--assist-start-s", type=float)
    parser.add_argument("--assist-end-s", type=float)
    parser.add_argument(
        "--recovery-checkpoint",
        type=Path,
        help="Optional second actor activated after a valid feet-first landing latch",
    )
    parser.add_argument(
        "--recovery-profile",
        choices=("easy", "medium", "hard", "extreme"),
        help="Explicit touchdown disturbance profile for recovery-only evaluation",
    )
    parser.add_argument(
        "--recovery-switch-mode",
        choices=("landing", "approach"),
        default="landing",
        help="Activate the specialist at validated contact or during final approach",
    )
    parser.add_argument(
        "--post-landing-checkpoint",
        type=Path,
        help="Optional third actor activated after the validated landing latch",
    )
    parser.add_argument(
        "--recovery-approach-angle-deg",
        type=float,
        default=330.0,
        help="Minimum completed rotation before an approach hand-off",
    )
    parser.add_argument(
        "--recovery-approach-tilt-deg",
        type=float,
        default=40.0,
        help="Maximum upright tilt before an approach hand-off",
    )
    parser.add_argument(
        "--recovery-approach-height-m",
        type=float,
        default=0.36,
        help="Maximum root height before an approach hand-off",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        help="Write per-step kinematics and policy actions for environment zero",
    )
    parser.add_argument(
        "--snapshot-output",
        type=Path,
        help="Capture every world's first full simulator state past an angle",
    )
    parser.add_argument(
        "--snapshot-angle-deg",
        type=float,
        default=260.0,
        help="Airborne rotation threshold for reference-state capture",
    )
    args = parser.parse_args()
    evaluate(
        args.checkpoint,
        args.num_envs,
        args.seed,
        args.output,
        args.start_mode,
        args.video_dir,
        args.render_env_index,
        args.trace_output,
        args.assist_scale,
        args.assist_force_n,
        args.assist_torque_nm,
        args.assist_start_s,
        args.assist_end_s,
        args.task_id,
        args.recovery_checkpoint,
        args.recovery_profile,
        args.recovery_switch_mode,
        args.post_landing_checkpoint,
        args.recovery_approach_angle_deg,
        args.recovery_approach_tilt_deg,
        args.recovery_approach_height_m,
        args.snapshot_output,
        args.snapshot_angle_deg,
    )


if __name__ == "__main__":
    main()
