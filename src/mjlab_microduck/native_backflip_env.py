"""Threaded native-MuJoCo discovery environment for the MicroDuck backflip.

MJLab currently hard-wires its simulation data model to MJWarp.  On Apple CPU,
MJWarp executes effectively on one core.  This task-specific VecEnv preserves
the backflip actor/critic layouts and physical success state machine while
using MuJoCo's persistent C++ rollout pool across all CPU cores.

This is a *discovery* backend.  It uses BAM's documented MuJoCo-equivalent
linear servo approximation and nominal dynamics.  Checkpoints must pass the
full MJLab domain-randomized strict evaluator before being accepted.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from bam.model import load_model
from mujoco import rollout
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.robot.microduck_constants import actuators
from mjlab_microduck.tasks.microduck_backflip_env_cfg import (
    CROUCH_OVERRIDES,
    STAND_Z,
    TUCK_OVERRIDES,
    configure_backflip_standing_eval,
)


TASK_ID = "Mjlab-Backflip-Flat-MicroDuck"


def _quat_from_euler(roll_angle, pitch_angle, yaw_angle):
    cr, sr = np.cos(roll_angle / 2), np.sin(roll_angle / 2)
    cp, sp = np.cos(pitch_angle / 2), np.sin(pitch_angle / 2)
    cy, sy = np.cos(yaw_angle / 2), np.sin(yaw_angle / 2)
    return np.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        axis=-1,
    )


class NativeBackflipVecEnv(VecEnv):
    """RSL-RL VecEnv backed by ``mujoco.rollout`` with persistent threads."""

    num_actions = 14
    step_dt = 0.02
    physics_dt = 0.005
    decimation = 4
    max_episode_length = 200

    def __init__(
        self,
        num_envs: int = 2048,
        nthread: int = 12,
        seed: int = 42,
        standing_prob: float = 0.75,
        crouch_prob: float = 0.20,
        midflight_prob: float = 0.05,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.rng = np.random.default_rng(seed)
        self.spawn_probs = np.asarray(
            (standing_prob, crouch_prob, midflight_prob), dtype=np.float64
        )
        self.spawn_probs /= self.spawn_probs.sum()
        self.cfg = SimpleNamespace(
            backend="native_mujoco_rollout",
            num_envs=num_envs,
            nthread=nthread,
            seed=seed,
            nominal_dynamics=True,
        )
        # ``MjlabOnPolicyRunner`` persists the environment curriculum clock
        # through ``env.unwrapped.common_step_counter``.  Keep the same small
        # interface so native discovery runs can resume ordinary MJLab
        # checkpoints without modifying rsl-rl or discarding optimizer state.
        self.unwrapped = self
        self.common_step_counter = 0

        # Build the exact compiled MicroDuck model once through MJLab so asset,
        # collision, actuator conversion, and sensor definitions remain shared.
        env_cfg = load_env_cfg(TASK_ID, play=True)
        env_cfg.scene.num_envs = 1
        configure_backflip_standing_eval(env_cfg)
        template = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
        self.model = template.sim.mj_model
        template_data = template.sim.mj_data
        robot = template.scene["robot"]
        self.default_joint_pos = (
            robot.data.default_joint_pos[0].detach().cpu().numpy().astype(np.float64)
        )

        self.nq, self.nv = self.model.nq, self.model.nv
        self.nstate = mujoco.mj_stateSize(
            self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS
        )
        initial = np.empty(self.nstate, dtype=mujoco.MJTNUM_DTYPE)
        mujoco.mj_getState(
            self.model,
            template_data,
            initial,
            mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )
        self._base_state = initial
        self._configure_native_bam_servo()

        self._sensor = {}
        for sensor_id in range(self.model.nsensor):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id
            )
            self._sensor[name] = (
                int(self.model.sensor_adr[sensor_id]),
                int(self.model.sensor_dim[sensor_id]),
            )

        self._workers = [mujoco.MjData(self.model) for _ in range(nthread)]
        self._rollout = rollout.Rollout(nthread=nthread)
        self._state = np.repeat(initial[None, :], num_envs, axis=0)
        self._control = np.zeros(
            (num_envs, self.decimation, self.model.nu), dtype=mujoco.MJTNUM_DTYPE
        )
        self._state_out = np.empty(
            (num_envs, self.decimation, self.nstate), dtype=mujoco.MJTNUM_DTYPE
        )
        self._sense_out = np.empty(
            (num_envs, self.decimation, self.model.nsensordata),
            dtype=mujoco.MJTNUM_DTYPE,
        )
        template.close()

        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.last_action = torch.zeros((num_envs, self.num_actions))
        self.foot_air_time = torch.zeros((num_envs, 2))
        self._sense = torch.zeros((num_envs, self.model.nsensordata))
        self._allocate_backflip_state()
        self.reset()

    def _configure_native_bam_servo(self) -> None:
        """Install BAM's documented native-MuJoCo linear approximation."""
        bam = load_model(actuators._resolved_json_path)
        bam.actuator.kp = 200.0
        vin = 0.5 * (6.5 + 8.2)
        kt, resistance = bam.kt.value, bam.R.value
        kp = bam.actuator.error_gain * bam.actuator.kp * vin * kt / resistance
        electrical_damping = kt * kt / resistance
        force_limit = 8.2 * kt / resistance

        self.model.actuator_dyntype[:] = mujoco.mjtDyn.mjDYN_NONE
        self.model.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
        self.model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
        self.model.actuator_gainprm[:] = 0.0
        self.model.actuator_biasprm[:] = 0.0
        self.model.actuator_gainprm[:, 0] = kp
        self.model.actuator_biasprm[:, 1] = -kp
        self.model.actuator_biasprm[:, 2] = -electrical_damping
        self.model.actuator_forcelimited[:] = 1
        self.model.actuator_forcerange[:] = (-force_limit, force_limit)
        self.model.actuator_ctrllimited[:] = 0
        dof_ids = self.model.jnt_dofadr[self.model.actuator_trnid[:, 0]]
        self.model.dof_frictionloss[dof_ids] = bam.friction_base.value
        self.model.dof_damping[dof_ids] = bam.friction_viscous.value

    def _allocate_backflip_state(self) -> None:
        n = self.num_envs
        z = lambda: torch.zeros(n)
        b = lambda: torch.zeros(n, dtype=torch.bool)
        self.accum = z()
        self.max_rotation = z()
        self.paid_rotation = z()
        self.max_z, self.paid_z = z(), z()
        self.max_vz, self.paid_vz = z(), z()
        self.max_preload, self.paid_preload = z(), z()
        self.max_launch_quality, self.paid_launch_quality = z(), z()
        self.max_push_quality, self.paid_push_quality = z(), z()
        self.max_feasible_push, self.paid_feasible_push = z(), z()
        self.had_support, self.airborne, self.flight_ended, self.landed = (
            b(), b(), b(), b()
        )

    @property
    def _qpos_np(self):
        return self._state[:, 1 : 1 + self.nq]

    @property
    def _qvel_np(self):
        start = 1 + self.nq
        return self._state[:, start : start + self.nv]

    def _sensor_values(self, suffix: str) -> torch.Tensor:
        name = next(name for name in self._sensor if name.endswith(suffix))
        address, dim = self._sensor[name]
        return self._sense[:, address : address + dim]

    def _reset_ids(self, ids: torch.Tensor) -> None:
        if ids.numel() == 0:
            return
        idx = ids.cpu().numpy()
        count = len(idx)
        modes = self.rng.choice(3, size=count, p=self.spawn_probs)
        is_standing, is_crouch, is_mid = modes == 0, modes == 1, modes == 2

        self._state[idx] = self._base_state
        qpos = self._qpos_np[idx]
        qvel = self._qvel_np[idx]
        qvel[:] = 0.0
        yaw = self.rng.uniform(-math.pi, math.pi, count)
        pitch = self.rng.uniform(-math.radians(4), math.radians(4), count)
        pitch[is_crouch] = self.rng.uniform(0.0, math.radians(10), is_crouch.sum())
        angle = self.rng.uniform(math.radians(160), math.radians(330), count)
        pitch[is_mid] = -angle[is_mid]
        roll_angle = self.rng.uniform(-math.radians(4), math.radians(4), count)
        qpos[:, 3:7] = _quat_from_euler(roll_angle, pitch, yaw)
        z = self.rng.uniform(0.11, 0.12, count)
        z[is_crouch] = self.rng.uniform(0.06, 0.085, is_crouch.sum())
        z[is_mid] = self.rng.uniform(0.16, 0.28, is_mid.sum())
        qpos[:, 2] = z
        qpos[:, 7:21] = self.default_joint_pos

        for mode_mask, overrides, factor_range in (
            (is_crouch, CROUCH_OVERRIDES, (0.55, 1.0)),
            (is_mid, TUCK_OVERRIDES, (0.50, 1.0)),
        ):
            if not mode_mask.any():
                continue
            factor = self.rng.uniform(*factor_range, mode_mask.sum())
            for joint_index, target in overrides.items():
                home = self.default_joint_pos[joint_index]
                qpos[mode_mask, 7 + joint_index] = home + factor * (target - home)
            qpos[mode_mask, 7:21] += self.rng.normal(
                0.0, 0.05, (mode_mask.sum(), 14)
            )

        omega = self.rng.uniform(10.0, 18.0, count)
        qvel[is_mid, 4] = -omega[is_mid]
        remaining = np.maximum(2 * math.pi - angle, 0.0)
        margin = self.rng.uniform(0.03, 0.10, count)
        flight_time = np.maximum(remaining / omega + margin, 0.05)
        vz = (STAND_Z - z + 0.5 * 9.81 * flight_time**2) / flight_time
        qvel[is_mid, 2] = vz[is_mid]
        # ``idx`` is advanced indexing, so qpos/qvel above are local copies.
        self._qpos_np[idx] = qpos
        self._qvel_np[idx] = qvel

        self.episode_length_buf[ids] = 0
        self.last_action[ids] = 0.0
        self.foot_air_time[ids] = 0.0
        self._sense[ids] = 0.0
        for tensor in (
            self.accum,
            self.max_rotation,
            self.paid_rotation,
            self.max_vz,
            self.paid_vz,
            self.max_preload,
            self.paid_preload,
            self.max_launch_quality,
            self.paid_launch_quality,
            self.max_push_quality,
            self.paid_push_quality,
            self.max_feasible_push,
            self.paid_feasible_push,
        ):
            tensor[ids] = 0.0
        z_t = torch.from_numpy(z.astype(np.float32))
        self.max_z[ids] = z_t
        self.paid_z[ids] = z_t
        mid_ids = ids[torch.from_numpy(is_mid)]
        shaped_ids = ids[torch.from_numpy(is_mid | is_crouch)]
        progress = torch.from_numpy(angle[is_mid].astype(np.float32))
        self.accum[mid_ids] = progress
        self.max_rotation[mid_ids] = progress
        self.paid_rotation[mid_ids] = progress
        initial_vz = torch.from_numpy(np.maximum(vz[is_mid], 0).astype(np.float32))
        self.max_vz[mid_ids] = initial_vz
        self.paid_vz[mid_ids] = initial_vz
        for maximum, paid in (
            (self.max_launch_quality, self.paid_launch_quality),
            (self.max_push_quality, self.paid_push_quality),
            (self.max_feasible_push, self.paid_feasible_push),
        ):
            maximum[mid_ids], paid[mid_ids] = 1.0, 1.0
        self.max_preload[shaped_ids], self.paid_preload[shaped_ids] = 1.0, 1.0
        self.had_support[ids] = True
        self.airborne[ids] = torch.from_numpy(is_mid)
        self.flight_ended[ids] = False
        self.landed[ids] = False

    def reset(self):
        self._reset_ids(torch.arange(self.num_envs))
        return self.get_observations(), {}

    def _kinematics(self):
        qpos = torch.from_numpy(self._qpos_np.astype(np.float32, copy=False))
        qvel = torch.from_numpy(self._qvel_np.astype(np.float32, copy=False))
        quat = qpos[:, 3:7]
        ang_vel = self._sensor_values("imu_ang_vel")
        lin_vel_b = self._sensor_values("imu_lin_vel")
        return qpos, qvel, quat, ang_vel, lin_vel_b

    @staticmethod
    def _projected_gravity(quat):
        w, x, y, z = quat.unbind(-1)
        return torch.stack(
            (
                -2 * (x * z - w * y),
                -2 * (y * z + w * x),
                -(1 - 2 * (x.square() + y.square())),
            ),
            dim=-1,
        )

    def _contacts(self):
        left = self._sensor_values("left_foot_collision_found")[:, 0] > 0
        right = self._sensor_values("right_foot_collision_found")[:, 0] > 0
        trunk = self._sensor_values("robot_ground_contact_trunk_base_found")[:, 0] > 0
        feet = left | right
        return left, right, feet, trunk | feet, trunk

    def _update_state_and_reward(self):
        qpos, qvel, quat, ang_vel, lin_vel_b = self._kinematics()
        left, right, feet, robot_contact, trunk_contact = self._contacts()
        height, vz = qpos[:, 2], qvel[:, 2]
        pitch_rate = -ang_vel[:, 1]
        lateral_axis_z = 2 * (quat[:, 2] * quat[:, 3] + quat[:, 0] * quat[:, 1])
        upright = 1 - 2 * (quat[:, 1].square() + quat[:, 2].square())

        self.had_support |= feet
        newly_airborne = self.had_support & ~robot_contact & (height >= 0.135)
        self.airborne |= newly_airborne
        recontact = self.airborne & robot_contact & ~self.flight_ended
        active_flight = self.airborne & ~robot_contact & ~self.flight_ended
        flat = torch.clamp((0.866 - lateral_axis_z.abs()) / (0.866 - 0.5), 0, 1)
        flat = flat.square() * (3 - 2 * flat)
        self.accum += pitch_rate * self.step_dt * active_flight.float() * flat
        self.max_rotation = torch.maximum(self.max_rotation, self.accum)
        self.max_z = torch.where(
            self.flight_ended, self.max_z, torch.maximum(self.max_z, height)
        )
        landed_now = (
            recontact
            & feet
            & (self.max_rotation >= math.radians(320))
            & (upright >= math.cos(math.radians(35)))
            & (height >= 0.085)
        )
        self.landed |= landed_now
        self.flight_ended |= recontact

        reward = torch.zeros(self.num_envs)
        logs = {}

        def add(name, weighted):
            nonlocal reward
            reward += weighted
            logs[f"Episode_Reward/{name}"] = weighted.mean()

        # Height potential.
        frontier = torch.clamp(self.max_z, max=0.30)
        delta = torch.clamp(frontier - torch.clamp(self.paid_z, max=0.30), min=0)
        self.paid_z = torch.maximum(self.paid_z, frontier)
        add("backflip_takeoff", 60.0 * delta / (0.30 - STAND_Z))

        # Post-takeoff vertical velocity potential.
        active = self.airborne & ~self.flight_ended
        self.max_vz = torch.maximum(
            self.max_vz, torch.clamp(vz, 0, 1.5) * active.float()
        )
        delta = torch.clamp(self.max_vz - self.paid_vz, min=0)
        self.paid_vz = torch.maximum(self.paid_vz, self.max_vz)
        add("backflip_launch_velocity", 40.0 * delta / 1.5)

        # Standing-to-crouch preload potential.
        joints = qpos[:, 7:21]
        preload_terms = []
        for joint_index, target in CROUCH_OVERRIDES.items():
            home = self.default_joint_pos[joint_index]
            denominator = target - home
            preload_terms.append(
                torch.clamp((joints[:, joint_index] - home) / denominator, 0, 1)
            )
        preload = torch.stack(preload_terms, dim=-1).mean(-1) * (~self.airborne)
        self.max_preload = torch.maximum(self.max_preload, preload)
        delta = torch.clamp(self.max_preload - self.paid_preload, min=0)
        self.paid_preload = torch.maximum(self.paid_preload, self.max_preload)
        add("backflip_preload", 5.0 * delta)

        launch_quality = torch.minimum(
            torch.clamp(vz / 1.2, 0, 1), torch.clamp(pitch_rate / 10.0, 0, 1)
        ) * active.float()
        self.max_launch_quality = torch.maximum(
            self.max_launch_quality, launch_quality
        )
        delta = torch.clamp(
            self.max_launch_quality - self.paid_launch_quality, min=0
        )
        self.paid_launch_quality = torch.maximum(
            self.paid_launch_quality, self.max_launch_quality
        )
        add("backflip_launch_quality", 40.0 * delta)

        supported = feet & ~self.airborne & (self.max_preload >= 0.55)
        push_quality = torch.minimum(
            torch.clamp(vz / 1.2, 0, 1), torch.clamp(pitch_rate / 10.0, 0, 1)
        ) * supported.float()
        self.max_push_quality = torch.maximum(self.max_push_quality, push_quality)
        delta = torch.clamp(self.max_push_quality - self.paid_push_quality, min=0)
        self.paid_push_quality = torch.maximum(
            self.paid_push_quality, self.max_push_quality
        )
        add("backflip_supported_push", 60.0 * delta)

        feasible = (
            torch.clamp((vz - 0.6) / 0.9, 0, 1)
            * torch.clamp((pitch_rate - 4.0) / 8.0, 0, 1)
            * supported.float()
        )
        self.max_feasible_push = torch.maximum(self.max_feasible_push, feasible)
        delta = torch.clamp(
            self.max_feasible_push - self.paid_feasible_push, min=0
        )
        self.paid_feasible_push = torch.maximum(
            self.paid_feasible_push, self.max_feasible_push
        )
        add("backflip_feasible_push", 120.0 * delta)

        frontier = torch.clamp(self.max_rotation, max=2 * math.pi)
        delta = torch.clamp(
            frontier - torch.clamp(self.paid_rotation, max=2 * math.pi), min=0
        )
        delta = torch.clamp(delta, max=18.0 * self.step_dt)
        self.paid_rotation = torch.maximum(self.paid_rotation, frontier)
        add("backflip_rotation", 20.0 * delta / (2 * math.pi))

        # Compact early-flight pose, faded before landing extension.
        tuck_error = []
        for joint_index, target in TUCK_OVERRIDES.items():
            tuck_error.append((joints[:, joint_index] - target).square())
        tuck_score = torch.exp(-torch.stack(tuck_error, -1).mean(-1) / 0.35**2)
        enter = torch.clamp(
            (self.max_rotation - math.radians(20)) / math.radians(30), 0, 1
        )
        enter = enter.square() * (3 - 2 * enter)
        leave = torch.clamp(
            (self.max_rotation - math.radians(250)) / math.radians(50), 0, 1
        )
        leave = 1 - leave.square() * (3 - 2 * leave)
        add(
            "backflip_flight_tuck",
            8.0 * tuck_score * enter * leave * active.float() * self.step_dt,
        )

        add("backflip_landing", 8.0 * self.landed.float() * self.step_dt)
        add("backflip_success", 2.0 * self.landed.float() * self.step_dt)
        add(
            "backflip_body_contact",
            -2.0 * (self.airborne & trunk_contact & ~feet).float() * self.step_dt,
        )
        add(
            "backflip_wrong_direction",
            -0.02 * torch.clamp(-pitch_rate, min=0).square() * self.step_dt,
        )
        add("backflip_flatness", -0.02 * lateral_axis_z.square() * self.step_dt)
        add(
            "backflip_lateral_velocity",
            -0.05 * lin_vel_b[:, 1].square() * self.step_dt,
        )
        add("body_ang_vel", -0.001 * ang_vel.square().sum(-1) * self.step_dt)
        return reward, logs

    def _compute_observations(self) -> TensorDict:
        qpos, qvel, quat, ang_vel, lin_vel_b = self._kinematics()
        gravity = self._projected_gravity(quat)
        joints = qpos[:, 7:21] - torch.from_numpy(
            self.default_joint_pos.astype(np.float32)
        )
        joint_vel = qvel[:, 6:20]
        zeros3 = torch.zeros((self.num_envs, 3))
        actor = torch.cat(
            (
                ang_vel,
                gravity,
                joints,
                joint_vel,
                self.last_action,
                zeros3,
                torch.zeros((self.num_envs, 4)),
                torch.zeros((self.num_envs, 6)),
            ),
            dim=-1,
        )
        left, right, _, _, _ = self._contacts()
        contacts = torch.stack((left, right), dim=-1).float()
        self.foot_air_time = torch.where(
            contacts.bool(), torch.zeros_like(self.foot_air_time), self.foot_air_time
        )
        left_force = self._sensor_values("left_foot_collision_force")
        right_force = self._sensor_values("right_foot_collision_force")
        critic = torch.cat(
            (
                lin_vel_b,
                ang_vel,
                gravity,
                joints,
                joint_vel,
                self.last_action,
                zeros3,
                self.foot_air_time,
                contacts,
                left_force,
                right_force,
                torch.zeros((self.num_envs, 4)),
                torch.zeros((self.num_envs, 6)),
            ),
            dim=-1,
        )
        assert actor.shape[-1] == 61 and critic.shape[-1] == 74
        return TensorDict({"actor": actor, "critic": critic}, batch_size=[self.num_envs])

    def get_observations(self) -> TensorDict:
        return self._compute_observations()

    def step(self, actions: torch.Tensor):
        actions = torch.clamp(actions.detach().cpu(), -100.0, 100.0)
        targets = actions.numpy() + self.default_joint_pos[None, :]
        self._control[:] = targets[:, None, :]
        self._rollout.rollout(
            self.model,
            self._workers,
            self._state,
            self._control,
            state=self._state_out,
            sensordata=self._sense_out,
        )
        np.copyto(self._state, self._state_out[:, -1])
        self._sense = torch.from_numpy(
            self._sense_out[:, -1].astype(np.float32, copy=False)
        )
        self.last_action = actions
        self.foot_air_time += self.step_dt
        reward, logs = self._update_state_and_reward()
        self.common_step_counter += 1
        self.episode_length_buf += 1
        timeouts = self.episode_length_buf >= self.max_episode_length
        nonfinite = ~torch.from_numpy(np.isfinite(self._state).all(axis=-1))
        dones = timeouts | nonfinite
        if dones.any():
            self._reset_ids(dones.nonzero(as_tuple=False).flatten())
        observations = self._compute_observations()
        extras = {"time_outs": timeouts, "log": logs}
        return observations, reward, dones.long(), extras

    def close(self) -> None:
        self._rollout.close()
