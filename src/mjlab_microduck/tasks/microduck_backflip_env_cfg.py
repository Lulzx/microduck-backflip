"""MicroDuck standing backflip task.

The task is deliberately separate from ``roulade``: a roulade only credits
supported forward rotation, while a backflip only credits collision-free
airborne backward rotation after a real takeoff.  Reverse-curriculum spawns
teach the landing half before the full jump is discovered.
"""

import math
from copy import deepcopy
from pathlib import Path

from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.backflip_pedestal_terrain import (
    BackflipMatTerrainCfg,
    BackflipPedestalTerrainCfg,
    LANDING_MAT_HEIGHT,
    PEDESTAL_HEIGHT,
    PEDESTAL_WIDTH,
)
from mjlab_microduck.tasks.backflip_actions import (
    BackflipResidualJointPositionActionCfg,
)
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    MicroduckRouladeRlCfg,
    make_microduck_roulade_env_cfg,
)


STAND_Z = 0.115
EPISODE_LENGTH_S = 4.0
SOFT_MAT_HEIGHT = 0.22
REFERENCE_STATE_PATH = (
    Path(__file__).parent / "data" / "pedestal_model225_angle260_seed42.json"
)
EARLY_REFERENCE_STATE_PATH = (
    Path(__file__).parent / "data" / "pedestal_model225_angle180_seed42.json"
)
MAT_LANDING_REFERENCE_STATE_PATH = (
    Path(__file__).parent / "data" / "mat_model460_landing_seed42.json"
)
LAUNCH650_MAT510_LANDING_STATE_PATH = (
    Path(__file__).parent
    / "data"
    / "mat_launch650_mat510_landing_t1p50_angle180_seed45_warp1.json"
)
LAUNCH650_MAT510_ANGLE260_STATE_PATH = (
    Path(__file__).parent
    / "data"
    / "mat_launch650_mat510_angle260_t1p50_seeds40-47_warp1.json"
)
CURRENT_FLOOR_REFERENCE_STATE_PATH = (
    Path(__file__).parent / "data" / "pedestal_floor_model600_angle180_seed45.json"
)
CURRENT_FLOOR_LANDING_STATE_PATH = (
    Path(__file__).parent / "data" / "pedestal_floor_model600_landing_seed45.json"
)
LAUNCH650_FLOOR_REFERENCE_STATE_PATH = (
    Path(__file__).parent
    / "data"
    / "pedestal_floor_launch650_nominal_angle160_seed45.json"
)

_LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]

# Compact flight pose.  It is only used for reverse-curriculum initialization,
# never as an imitated trajectory or a rewarded waypoint.
TUCK_OVERRIDES = {
    2: -1.15,
    3: 1.25,
    4: 1.05,
    5: -0.55,
    6: 0.55,
    11: 1.15,
    12: -1.25,
    13: -1.05,
}

# Supported deep-crouch anchor already validated by the velstand recovery task.
# Prelaunch resets interpolate HOME toward this pose so PPO can first learn the
# explosive extension, then connect standing -> crouch -> launch.
CROUCH_OVERRIDES = {
    2: -1.15,
    3: 1.25,
    4: 1.05,
    11: 1.15,
    12: -1.25,
    13: -1.05,
}


def make_microduck_backflip_env_cfg(play: bool = False):
    """Create the airborne backflip environment from the sim2real trick base."""
    cfg = make_microduck_roulade_env_cfg(play=play)
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ZEST-style residual teacher: constrain stochastic policy deviations around
    # the mechanically validated nominal-PD trajectory at full spotting, then
    # restore full authority continuously as the external wrench is withdrawn.
    old_action = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = BackflipResidualJointPositionActionCfg(
        entity_name=old_action.entity_name,
        clip=old_action.clip,
        actuator_names=old_action.actuator_names,
        scale=old_action.scale,
        offset=old_action.offset,
        preserve_order=old_action.preserve_order,
        use_default_offset=old_action.use_default_offset,
        min_assisted_authority=0.05,
    )

    # Remove every supported-roll objective.  Keeping any of these would make
    # the two maneuver definitions compete and reward head/ground contact.
    for name in tuple(cfg.rewards):
        if name.startswith("roulade_"):
            del cfg.rewards[name]

    cfg.rewards["backflip_takeoff"] = RewardTermCfg(
        func=microduck_mdp.backflip_takeoff_progress,
        weight=60.0,
        params={"target_height": 0.30, "start_height": STAND_Z},
    )
    cfg.rewards["backflip_launch_velocity"] = RewardTermCfg(
        func=microduck_mdp.backflip_launch_velocity_progress,
        weight=40.0,
        params={"target_velocity": 1.5},
    )
    cfg.rewards["backflip_preload"] = RewardTermCfg(
        func=microduck_mdp.backflip_preload_progress,
        weight=30.0,
        params={"joint_targets": CROUCH_OVERRIDES},
    )
    cfg.rewards["backflip_launch_quality"] = RewardTermCfg(
        func=microduck_mdp.backflip_launch_quality_progress,
        weight=40.0,
        params={"target_velocity": 1.2, "target_pitch_rate": 10.0},
    )
    cfg.rewards["backflip_supported_push"] = RewardTermCfg(
        func=microduck_mdp.backflip_supported_push_quality_progress,
        weight=60.0,
        params={
            "target_velocity": 1.2,
            "target_pitch_rate": 10.0,
            "min_preload": 0.55,
        },
    )
    cfg.rewards["backflip_feasible_push"] = RewardTermCfg(
        func=microduck_mdp.backflip_feasible_push_progress,
        weight=120.0,
        params={
            "target_velocity": 1.5,
            "min_velocity": 0.6,
            "target_pitch_rate": 12.0,
            "min_pitch_rate": 4.0,
            "min_preload": 0.55,
        },
    )
    cfg.rewards["backflip_rotation"] = RewardTermCfg(
        func=microduck_mdp.backflip_rotation_progress,
        weight=80.0,
        params={"target_angle": 2 * math.pi, "max_paid_rate": 18.0},
    )
    # EFGCL-style virtual spotting: a small upward + backward-pitch impulse
    # lets the critic observe successful trajectories early. It is annealed
    # from strict landing success, not from wall-clock training steps.
    cfg.events["backflip_assistive_wrench"] = EventTermCfg(
        func=microduck_mdp.apply_backflip_assistive_wrench,
        mode="step",
        params={
            "start_time_s": 0.30,
            "end_time_s": 0.40,
            "upward_force_n": 16.0,
            "backward_pitch_torque_nm": 1.40,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["backflip_flight_tuck"] = RewardTermCfg(
        func=microduck_mdp.backflip_flight_tuck,
        weight=8.0,
        params={
            "joint_targets": TUCK_OVERRIDES,
            "pose_std": 0.35,
            "gate_lo": math.radians(20.0),
            "gate_hi": math.radians(250.0),
            "fade_out": math.radians(300.0),
        },
    )
    cfg.rewards["backflip_prepare_landing"] = RewardTermCfg(
        func=microduck_mdp.backflip_prepare_landing,
        weight=20.0,
        params={
            "gate_lo": math.radians(250.0),
            "gate_hi": math.radians(340.0),
        },
    )
    cfg.rewards["backflip_landing_approach"] = RewardTermCfg(
        func=microduck_mdp.backflip_landing_approach,
        weight=50.0,
        params={
            "target_height": STAND_Z,
            "joint_indices": _LEG_JOINTS,
            "gate_lo": math.radians(270.0),
            "gate_hi": math.radians(340.0),
        },
    )
    cfg.rewards["backflip_landing"] = RewardTermCfg(
        func=microduck_mdp.backflip_landing_composite,
        # Keep the narrow all-at-once score as a near-goal bonus, but expose
        # its factors separately below so a poor pose cannot erase the entire
        # post-impact learning signal.
        weight=100.0,
        params={
            "target_height": STAND_Z,
            "height_std": 0.025,
            "upright_std": 0.30,
            "pose_std": 0.40,
            "joint_indices": _LEG_JOINTS,
            "target_overrides": None,
        },
    )
    cfg.rewards["backflip_landing_upright"] = RewardTermCfg(
        func=microduck_mdp.backflip_landing_upright,
        weight=100.0,
    )
    cfg.rewards["backflip_landing_height"] = RewardTermCfg(
        func=microduck_mdp.backflip_landing_height,
        weight=75.0,
        params={"target_height": STAND_Z, "minimum_height": 0.06},
    )
    cfg.rewards["backflip_landing_stillness"] = RewardTermCfg(
        func=microduck_mdp.backflip_landing_stillness,
        weight=75.0,
        params={"angular_speed_std": 4.0},
    )
    cfg.rewards["backflip_landing_foot_support"] = RewardTermCfg(
        func=microduck_mdp.backflip_landing_foot_support,
        weight=50.0,
    )
    cfg.rewards["backflip_stability_progress"] = RewardTermCfg(
        func=microduck_mdp.backflip_stability_progress,
        weight=200.0,
    )
    cfg.rewards["backflip_success"] = RewardTermCfg(
        func=microduck_mdp.backflip_success,
        weight=200.0,
    )
    cfg.rewards["backflip_body_contact"] = RewardTermCfg(
        func=microduck_mdp.backflip_body_contact_cost,
        weight=0.0,
    )
    cfg.rewards["backflip_assisted_action"] = RewardTermCfg(
        func=microduck_mdp.backflip_assisted_action_cost,
        weight=-2.0,
    )
    cfg.rewards["backflip_late_pitch_rate"] = RewardTermCfg(
        func=microduck_mdp.backflip_late_pitch_rate_cost,
        weight=-10.0,
        params={
            "gate_lo": math.radians(300.0),
            "gate_hi": math.radians(360.0),
            "rate_scale": 15.0,
        },
    )
    cfg.rewards["backflip_wrong_direction"] = RewardTermCfg(
        func=microduck_mdp.backflip_wrong_direction_cost,
        weight=-0.02,
    )
    # A small corkscrew is allowed.  The inherited roulade task needed a
    # strongly sagittal, mirror-symmetric motion to avoid rolling over a
    # shoulder, but an airborne flip can use left/right asymmetry to generate
    # launch torque.  The rotation accumulator still gives full pitch credit
    # only while the lateral axis is within 30 degrees of horizontal and the
    # landing gate still requires an upright feet-first finish.
    cfg.rewards["backflip_sagittal"] = RewardTermCfg(
        func=microduck_mdp.roulade_sagittal_penalty,
        weight=0.0,
    )
    cfg.rewards["backflip_flatness"] = RewardTermCfg(
        func=microduck_mdp.roulade_flatness_penalty,
        weight=-0.02,
    )
    cfg.rewards["backflip_lateral_velocity"] = RewardTermCfg(
        func=microduck_mdp.roulade_lateral_velocity_penalty,
        weight=-0.05,
    )

    # Large angular velocity is intrinsic to the maneuver.  Smoothness taxes
    # are introduced only after discovery by the inherited curricula.
    cfg.rewards["body_ang_vel"].weight = -0.001
    cfg.rewards["angular_momentum"].weight = 0.0
    cfg.rewards["action_rate_l2"].weight = 0.0
    cfg.rewards["joint_torque_rate_l2"].weight = 0.0
    cfg.rewards["gentle_landing"].weight = 0.0
    cfg.rewards["arrival_damping"].weight = 0.0

    del cfg.events["set_roulade_state"]
    cfg.events["set_backflip_state"] = EventTermCfg(
        func=microduck_mdp.reset_backflip_state,
        mode="reset",
        params={
            # Full spotting already exposes the launch. Allocate substantial
            # early capacity to the now-measured bottleneck: feet-first contact
            # followed by a continuous stable hold.
            "standing_prob": 0.45,
            "crouch_prob": 0.10,
            "midflight_prob": 0.25,
            "recovery_prob": 0.20,
            "standing_z_range": (0.11, 0.12),
            "standing_tilt_max": math.radians(4.0),
            "crouch_z_range": (0.06, 0.085),
            "crouch_tilt_max": math.radians(10.0),
            "crouch_overrides": CROUCH_OVERRIDES,
            "crouch_factor_range": (0.55, 1.0),
            "midflight_angle_range": (math.radians(160.0), math.radians(330.0)),
            "midflight_z_range": (0.16, 0.28),
            "midflight_omega_range": (10.0, 18.0),
            "midflight_vz_range": (0.2, 1.5),
            "midflight_ballistic_landing": True,
            "midflight_landing_z": STAND_Z,
            "midflight_time_margin_range": (0.03, 0.10),
            "recovery_z_range": (0.105, 0.12),
            "recovery_tilt_max": math.radians(15.0),
            "recovery_lin_vel_max": 0.20,
            "recovery_ang_vel_max": 1.50,
            "tuck_overrides": TUCK_OVERRIDES,
            "tuck_factor_range": (0.50, 1.0),
            "joint_noise_std": 0.05,
            "initial_assist_scale": 1.0,
            "assist_decay_step": 0.05,
            "assist_success_threshold": 0.60,
            "assist_evaluation_window": 256,
        },
    )

    # Reuse two pre-existing zero-padding slots so warm-start checkpoints keep
    # exactly the same network shape: [bounded phase, assist scale, 0, 0, 0, 0].
    for group in ("actor", "critic"):
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.backflip_assist_observation,
            params={"dim": 6, "timing_scale_s": 0.30},
        )

    # Consolidate landing/recovery first, then shift capacity toward replacing
    # the spotter from ordinary standing starts within this 600-iteration run.
    del cfg.curriculum["roulade_spawn_mix"]
    cfg.curriculum["backflip_spawn_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_backflip_state",
            "param_stages": [
                {
                    "step": 0,
                    "params": {
                        "standing_prob": 0.45,
                        "crouch_prob": 0.10,
                        "midflight_prob": 0.25,
                        "recovery_prob": 0.20,
                    },
                },
                {
                    "step": 200 * 24,
                    "params": {
                        "standing_prob": 0.60,
                        "crouch_prob": 0.10,
                        "midflight_prob": 0.20,
                        "recovery_prob": 0.10,
                    },
                },
                {
                    "step": 400 * 24,
                    "params": {
                        "standing_prob": 0.75,
                        "crouch_prob": 0.10,
                        "midflight_prob": 0.10,
                        "recovery_prob": 0.05,
                    },
                },
            ],
        },
    )
    cfg.curriculum["backflip_assist_scale"] = CurriculumTermCfg(
        func=microduck_mdp.backflip_assist_scale_metric,
    )
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 2500 * 24, "weight": -0.05},
                {"step": 3500 * 24, "weight": -0.15},
            ],
        },
    )
    cfg.curriculum["preload_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "backflip_preload",
            "weight_stages": [
                {"step": 0, "weight": 30.0},
                {"step": 200 * 24, "weight": 15.0},
                {"step": 350 * 24, "weight": 5.0},
            ],
        },
    )
    cfg.curriculum["body_contact_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "backflip_body_contact",
            "weight_stages": [
                {"step": 0, "weight": -2.0},
                {"step": 1500 * 24, "weight": -5.0},
                {"step": 3000 * 24, "weight": -8.0},
            ],
        },
    )
    cfg.curriculum["arrival_damping_weight"].params["weight_stages"] = [
        {"step": 0, "weight": 0.0},
        {"step": 3000 * 24, "weight": -0.02},
        {"step": 4000 * 24, "weight": -0.05},
    ]
    cfg.curriculum["torque_rate_weight"].params["weight_stages"] = [
        {"step": 0, "weight": 0.0},
        {"step": 3000 * 24, "weight": -2.5e-4},
        {"step": 4000 * 24, "weight": -5e-4},
    ]
    cfg.curriculum["gentle_landing_weight"].params["weight_stages"] = [
        {"step": 0, "weight": 0.0},
        {"step": 2500 * 24, "weight": 0.001},
        {"step": 3500 * 24, "weight": 0.003},
    ]

    return cfg


def configure_backflip_standing_eval(cfg) -> None:
    """Force all resets to standing without the training curriculum undoing it."""
    reset_cfg = cfg.events["set_backflip_state"]
    if "standing_probability" in reset_cfg.params:
        # Three-way distillation reset: standing versus captured flight and
        # landing states.  Do not inject the incompatible ``*_prob`` names
        # accepted only by ``reset_backflip_state``.
        reset_cfg.params["standing_probability"] = 1.0
        reset_cfg.params["landing_probability"] = 0.0
    else:
        reset_cfg.params["standing_prob"] = 1.0
        reset_cfg.params["crouch_prob"] = 0.0
        reset_cfg.params["midflight_prob"] = 0.0
        reset_cfg.params["recovery_prob"] = 0.0
    reset_cfg.params["initial_assist_scale"] = 0.0
    cfg.curriculum.pop("backflip_spawn_mix", None)
    cfg.curriculum.pop("backflip_reference_spawn_mix", None)


def make_microduck_backflip_recovery_env_cfg(play: bool = False):
    """Phase-final specialist used before returning to the complete maneuver.

    This implements a reference-state-initialization curriculum: every world
    begins immediately after a valid feet-first revolution, first near the
    nominal standing state and then with progressively wider touchdown errors.
    Network dimensions and strict success semantics are identical to the full
    task, so its actor can warm-start the subsequent integrated training stage.
    """
    cfg = make_microduck_backflip_env_cfg(play=play)
    reset = cfg.events["set_backflip_state"].params
    reset.update(
        {
            "standing_prob": 0.0,
            "crouch_prob": 0.0,
            "midflight_prob": 0.0,
            "recovery_prob": 1.0,
            "recovery_z_range": (0.112, 0.118),
            "recovery_tilt_max": math.radians(5.0),
            "recovery_lin_vel_max": 0.08,
            "recovery_ang_vel_max": 0.75,
            "recovery_vertical_velocity_range": (-0.40, -0.05),
            "joint_noise_std": 0.01,
            "initial_assist_scale": 0.0,
        }
    )
    cfg.curriculum.pop("backflip_spawn_mix", None)
    cfg.curriculum["backflip_recovery_difficulty"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_backflip_state",
            "param_stages": [
                {
                    "step": 0,
                    "params": {
                        "recovery_z_range": (0.112, 0.118),
                        "recovery_tilt_max": math.radians(5.0),
                        "recovery_lin_vel_max": 0.08,
                        "recovery_ang_vel_max": 0.75,
                        "recovery_vertical_velocity_range": (-0.40, -0.05),
                        "joint_noise_std": 0.01,
                    },
                },
                {
                    "step": 100 * 24,
                    "params": {
                        "recovery_z_range": (0.108, 0.12),
                        "recovery_tilt_max": math.radians(15.0),
                        "recovery_lin_vel_max": 0.30,
                        "recovery_ang_vel_max": 3.0,
                        "recovery_vertical_velocity_range": (-1.25, -0.10),
                        "joint_noise_std": 0.03,
                    },
                },
                {
                    "step": 200 * 24,
                    "params": {
                        "recovery_z_range": (0.105, 0.125),
                        "recovery_tilt_max": math.radians(30.0),
                        "recovery_lin_vel_max": 0.75,
                        "recovery_ang_vel_max": 12.0,
                        "recovery_vertical_velocity_range": (-2.50, -0.20),
                        "joint_noise_std": 0.08,
                    },
                },
                {
                    "step": 350 * 24,
                    "params": {
                        "recovery_z_range": (0.100, 0.13),
                        "recovery_tilt_max": math.radians(45.0),
                        "recovery_lin_vel_max": 1.20,
                        "recovery_ang_vel_max": 20.0,
                        "recovery_vertical_velocity_range": (-3.25, -0.25),
                        "joint_noise_std": 0.10,
                    },
                },
            ],
        },
    )
    cfg.terminations["backflip_body_only_contact"] = TerminationTermCfg(
        func=microduck_mdp.backflip_body_only_contact,
        time_out=False,
    )
    return cfg


def make_microduck_backflip_touchdown_env_cfg(play: bool = False):
    """Late-flight specialist that learns braking through post-impact recovery.

    Unlike the post-contact recovery task, these worlds start while still
    airborne near the end of the revolution. The actor must choose the leg
    configuration before impact, arrest the remaining pitch, touch feet first,
    and satisfy the unchanged strict hold in one continuous rollout.
    """
    cfg = make_microduck_backflip_env_cfg(play=play)
    reset = cfg.events["set_backflip_state"].params
    reset.update(
        {
            "standing_prob": 0.0,
            "crouch_prob": 0.0,
            "midflight_prob": 1.0,
            "recovery_prob": 0.0,
            "midflight_angle_range": (
                math.radians(330.0),
                math.radians(350.0),
            ),
            "midflight_z_range": (0.22, 0.30),
            "midflight_omega_range": (12.0, 18.0),
            "midflight_ballistic_landing": True,
            "midflight_landing_z": STAND_Z,
            "midflight_time_margin_range": (0.02, 0.05),
            "tuck_factor_range": (0.35, 0.85),
            "joint_noise_std": 0.02,
            "initial_assist_scale": 0.0,
        }
    )
    cfg.curriculum.pop("backflip_spawn_mix", None)
    cfg.curriculum.pop("body_contact_weight", None)
    cfg.rewards["backflip_late_pitch_rate"].weight = -120.0
    cfg.rewards["backflip_late_pitch_rate"].params.update(
        {
            "gate_lo": math.radians(300.0),
            "gate_hi": math.radians(350.0),
            "rate_scale": 15.0,
        }
    )
    cfg.rewards["backflip_prepare_landing"].weight = 100.0
    cfg.rewards["backflip_landing_approach"].weight = 100.0
    cfg.rewards["backflip_body_contact"].weight = -20.0
    cfg.rewards["backflip_rotation_overshoot"] = RewardTermCfg(
        func=microduck_mdp.backflip_rotation_overshoot_cost,
        weight=-100.0,
        params={
            "target_angle": 2 * math.pi,
            "angle_scale": math.radians(45.0),
        },
    )
    cfg.curriculum["backflip_touchdown_difficulty"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_backflip_state",
            "param_stages": [
                {
                    "step": 0,
                    "params": {
                        "midflight_angle_range": (
                            math.radians(330.0),
                            math.radians(350.0),
                        ),
                        "midflight_z_range": (0.22, 0.30),
                        "midflight_omega_range": (12.0, 18.0),
                        "midflight_time_margin_range": (0.02, 0.05),
                        "tuck_factor_range": (0.35, 0.85),
                        "joint_noise_std": 0.02,
                    },
                },
                {
                    "step": 100 * 24,
                    "params": {
                        "midflight_angle_range": (
                            math.radians(310.0),
                            math.radians(350.0),
                        ),
                        "midflight_z_range": (0.22, 0.34),
                        "midflight_omega_range": (12.0, 20.0),
                        "midflight_time_margin_range": (0.02, 0.07),
                        "tuck_factor_range": (0.25, 1.0),
                        "joint_noise_std": 0.04,
                    },
                },
                {
                    "step": 250 * 24,
                    "params": {
                        "midflight_angle_range": (
                            math.radians(280.0),
                            math.radians(355.0),
                        ),
                        "midflight_z_range": (0.20, 0.38),
                        "midflight_omega_range": (10.0, 22.0),
                        "midflight_time_margin_range": (0.01, 0.10),
                        "tuck_factor_range": (0.15, 1.0),
                        "joint_noise_std": 0.07,
                    },
                },
            ],
        },
    )
    cfg.terminations["backflip_body_only_contact"] = TerminationTermCfg(
        func=microduck_mdp.backflip_postflight_body_only_contact,
        time_out=False,
    )
    return cfg


def make_microduck_backflip_pedestal_env_cfg(play: bool = False):
    """Launch from a 25 cm cube and require landing on the lower floor.

    The elevated start adds about 0.20 s of ballistic fall time without
    changing actuator limits. It is a curriculum task, not a redefinition of
    the flat-ground acceptance gate.
    """
    cfg = make_microduck_backflip_env_cfg(play=play)
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=42,
            size=(2.0, 2.0),
            curriculum=False,
            num_rows=1,
            num_cols=1,
            color_scheme="none",
            sub_terrains={"backflip_pedestal": BackflipPedestalTerrainCfg()},
        ),
        max_init_terrain_level=0,
    )

    reset = cfg.events["set_backflip_state"].params
    reset.update(
        {
            "standing_prob": 0.35,
            "crouch_prob": 0.10,
            "midflight_prob": 0.35,
            "recovery_prob": 0.20,
            "standing_z_range": (
                PEDESTAL_HEIGHT + 0.11,
                PEDESTAL_HEIGHT + 0.12,
            ),
            "crouch_z_range": (
                PEDESTAL_HEIGHT + 0.06,
                PEDESTAL_HEIGHT + 0.085,
            ),
            "standing_edge_offset": 0.035,
            "floor_reset_distance": PEDESTAL_WIDTH / 2.0 + 0.25,
            "landing_min_horizontal_distance": PEDESTAL_WIDTH / 2.0 + 0.04,
        }
    )
    cfg.curriculum["backflip_spawn_mix"].params["param_stages"] = [
        {
            "step": 0,
            "params": {
                "standing_prob": 0.35,
                "crouch_prob": 0.10,
                "midflight_prob": 0.35,
                "recovery_prob": 0.20,
            },
        },
        {
            "step": 200 * 24,
            "params": {
                "standing_prob": 0.55,
                "crouch_prob": 0.10,
                "midflight_prob": 0.25,
                "recovery_prob": 0.10,
            },
        },
        {
            "step": 400 * 24,
            "params": {
                "standing_prob": 0.75,
                "crouch_prob": 0.10,
                "midflight_prob": 0.10,
                "recovery_prob": 0.05,
            },
        },
    ]
    cfg.events["backflip_assistive_wrench"].params["backward_force_n"] = 5.0
    cfg.rewards["backflip_takeoff"].params.update(
        {
            "start_height": PEDESTAL_HEIGHT + STAND_Z,
            "target_height": PEDESTAL_HEIGHT + 0.30,
        }
    )
    return cfg


def make_microduck_backflip_pedestal_braking_env_cfg(play: bool = False):
    """Fine-tune launch and touchdown together on the real cube trajectory.

    The assisted teacher remains protected through the mechanically difficult
    launch and tuck. After 280 degrees the same actor receives full authority,
    so landing gradients can change leg extension before impact rather than
    trying to recover from an already committed high-spin contact. Most worlds
    start on the cube; a shrinking late-flight slice keeps the landing basin
    populated while the integrated skill adapts.
    """
    cfg = make_microduck_backflip_pedestal_env_cfg(play=play)
    action = cfg.actions["joint_pos"]
    action.full_authority_after_angle_rad = math.radians(280.0)

    reset = cfg.events["set_backflip_state"].params
    reset.update(
        {
            "standing_prob": 0.75,
            "crouch_prob": 0.0,
            "midflight_prob": 0.25,
            "recovery_prob": 0.0,
            "midflight_angle_range": (
                math.radians(280.0),
                math.radians(350.0),
            ),
            "midflight_z_range": (0.20, 0.38),
            "midflight_omega_range": (10.0, 22.0),
            "midflight_ballistic_landing": True,
            "midflight_landing_z": STAND_Z,
            "midflight_time_margin_range": (0.01, 0.10),
            "tuck_factor_range": (0.15, 1.0),
        }
    )
    cfg.curriculum["backflip_spawn_mix"].params["param_stages"] = [
        {
            "step": 0,
            "params": {
                "standing_prob": 0.75,
                "crouch_prob": 0.0,
                "midflight_prob": 0.25,
                "recovery_prob": 0.0,
            },
        },
        {
            "step": 100 * 24,
            "params": {
                "standing_prob": 0.90,
                "crouch_prob": 0.0,
                "midflight_prob": 0.10,
                "recovery_prob": 0.0,
            },
        },
        {
            "step": 200 * 24,
            "params": {
                "standing_prob": 1.0,
                "crouch_prob": 0.0,
                "midflight_prob": 0.0,
                "recovery_prob": 0.0,
            },
        },
    ]

    cfg.curriculum.pop("body_contact_weight", None)
    cfg.rewards["backflip_late_pitch_rate"].weight = -120.0
    cfg.rewards["backflip_late_pitch_rate"].params.update(
        {
            "gate_lo": math.radians(280.0),
            "gate_hi": math.radians(350.0),
            "rate_scale": 15.0,
        }
    )
    cfg.rewards["backflip_prepare_landing"].weight = 100.0
    cfg.rewards["backflip_landing_approach"].weight = 100.0
    cfg.rewards["backflip_body_contact"].weight = -20.0
    cfg.rewards["backflip_rotation_overshoot"] = RewardTermCfg(
        func=microduck_mdp.backflip_rotation_overshoot_cost,
        weight=-100.0,
        params={
            "target_angle": 2 * math.pi,
            "angle_scale": math.radians(45.0),
        },
    )
    cfg.terminations["backflip_body_only_contact"] = TerminationTermCfg(
        func=microduck_mdp.backflip_postflight_body_only_contact,
        time_out=False,
    )
    return cfg


def make_microduck_backflip_reference_env_cfg(
    play: bool = False,
    reference_state_path: Path = REFERENCE_STATE_PATH,
):
    """Iterative-reference continuation from states the cube actor produced."""
    cfg = make_microduck_backflip_pedestal_braking_env_cfg(play=play)
    cfg.events["set_backflip_state"] = EventTermCfg(
        func=microduck_mdp.reset_backflip_reference_state,
        mode="reset",
        params={
            "reference_state_path": str(reference_state_path),
            "landing_min_horizontal_distance": PEDESTAL_WIDTH / 2.0 + 0.04,
        },
    )
    cfg.curriculum.pop("backflip_spawn_mix", None)
    cfg.rewards["backflip_landing_stillness"].weight = 300.0
    cfg.rewards["backflip_landing_stillness"].params["angular_speed_std"] = 8.0
    cfg.rewards["backflip_post_landing_angular_speed"] = RewardTermCfg(
        func=microduck_mdp.backflip_post_landing_angular_speed_cost,
        weight=-200.0,
        params={"speed_scale": 20.0},
    )
    cfg.rewards["backflip_stability_progress"].weight = 400.0
    cfg.rewards["backflip_success"].weight = 400.0
    return cfg


def make_microduck_backflip_early_reference_env_cfg(play: bool = False):
    """Continue the real cube launch from its apex instead of final approach.

    The 260-degree reference slice left about 0.15 seconds to dissipate impact
    energy.  The 180-degree slice is near the ballistic apex and roughly
    doubles that control window.  A timing cost asks the actor to retain only
    the pitch rate required to reach upright at predicted foot-height contact;
    this preserves the revolution objective while teaching earlier untucking.
    """
    cfg = make_microduck_backflip_reference_env_cfg(
        play=play,
        reference_state_path=EARLY_REFERENCE_STATE_PATH,
    )
    cfg.rewards["backflip_flight_tuck"].params.update(
        {
            "gate_hi": math.radians(210.0),
            "fade_out": math.radians(250.0),
        }
    )
    cfg.rewards["backflip_prepare_landing"].params.update(
        {
            "gate_lo": math.radians(190.0),
            "gate_hi": math.radians(280.0),
        }
    )
    cfg.rewards["backflip_landing_approach"].params.update(
        {
            "gate_lo": math.radians(220.0),
            "gate_hi": math.radians(310.0),
        }
    )
    cfg.rewards["backflip_late_pitch_rate"].weight = -20.0
    cfg.rewards["backflip_rotation_timing"] = RewardTermCfg(
        func=microduck_mdp.backflip_rotation_timing_cost,
        weight=-100.0,
        params={
            "target_angle": 2 * math.pi,
            "target_height": STAND_Z,
            "gate_lo": math.radians(170.0),
            "gate_hi": math.radians(220.0),
            "rate_error_scale": 8.0,
        },
    )
    return cfg


def make_microduck_backflip_mat_reference_env_cfg(play: bool = False):
    """Apex continuation onto a raised compliant landing mat."""
    cfg = make_microduck_backflip_early_reference_env_cfg(play=play)
    # Before the reset event loads an airborne reference, MuJoCo briefly sees
    # the default low standing pose intersecting the raised mat.  Reserve the
    # same contact capacity used by the repository's rough-terrain tasks.
    cfg.sim.nconmax = 200
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=42,
            size=(2.0, 2.0),
            curriculum=False,
            num_rows=1,
            num_cols=1,
            color_scheme="none",
            sub_terrains={"backflip_mat": BackflipMatTerrainCfg()},
        ),
        max_init_terrain_level=0,
    )
    target_height = LANDING_MAT_HEIGHT + STAND_Z
    cfg.rewards["backflip_landing_approach"].params["target_height"] = target_height
    cfg.rewards["backflip_landing"].params["target_height"] = target_height
    cfg.rewards["backflip_landing_height"].params.update(
        {
            "target_height": target_height,
            "minimum_height": LANDING_MAT_HEIGHT + 0.06,
        }
    )
    cfg.rewards["backflip_rotation_timing"].params["target_height"] = target_height
    # Feet touch while the root is still above its nominal standing height.
    # Aim modestly past one revolution in the ballistic predictor so contact
    # occurs after the strict 360-degree airborne gate rather than at 340-ish.
    cfg.rewards["backflip_rotation_timing"].params["target_angle"] = math.radians(
        385.0
    )
    return cfg


def make_microduck_backflip_mat_landing_reference_env_cfg(
    play: bool = False,
    reference_state_path: Path = MAT_LANDING_REFERENCE_STATE_PATH,
):
    """Post-impact RSI from exact full-revolution model-460 mat contacts."""
    cfg = make_microduck_backflip_mat_reference_env_cfg(play=play)
    cfg.events["set_backflip_state"] = EventTermCfg(
        func=microduck_mdp.reset_backflip_landing_reference_state,
        mode="reset",
        params={
            "reference_state_path": str(reference_state_path),
            "landing_min_horizontal_distance": PEDESTAL_WIDTH / 2.0 + 0.04,
        },
    )
    # These episodes begin after the flight frontier has been frozen. Retain
    # only post-impact objectives so the critic spends its capacity on the
    # 0.5-second balance problem rather than constant zero flight terms.
    for name in (
        "backflip_takeoff",
        "backflip_launch_velocity",
        "backflip_preload",
        "backflip_launch_quality",
        "backflip_supported_push",
        "backflip_feasible_push",
        "backflip_rotation",
        "backflip_flight_tuck",
        "backflip_prepare_landing",
        "backflip_landing_approach",
        "backflip_late_pitch_rate",
        "backflip_rotation_timing",
        "backflip_rotation_overshoot",
    ):
        cfg.rewards[name].weight = 0.0
    return cfg


def make_microduck_backflip_launch650_mat_landing_env_cfg(
    play: bool = False,
):
    """Recover exact contacts from the best cube-to-mat composition."""
    return make_microduck_backflip_mat_landing_reference_env_cfg(
        play=play,
        reference_state_path=LAUNCH650_MAT510_LANDING_STATE_PATH,
    )


def make_microduck_backflip_launch650_mat_approach_env_cfg(
    play: bool = False,
):
    """Continue deterministic live trajectories from 260 degrees to contact."""
    cfg = make_microduck_backflip_mat_reference_env_cfg(play=play)
    cfg.events["set_backflip_state"].params["reference_state_path"] = str(
        LAUNCH650_MAT510_ANGLE260_STATE_PATH
    )
    return cfg


def make_microduck_backflip_mat_mixed_reference_env_cfg(
    play: bool = False,
    flight_reference_state_path: Path = EARLY_REFERENCE_STATE_PATH,
):
    """Unified apex-to-landing curriculum with exact touchdown rehearsal."""
    cfg = make_microduck_backflip_mat_reference_env_cfg(play=play)
    cfg.events["set_backflip_state"] = EventTermCfg(
        func=microduck_mdp.reset_backflip_mixed_reference_state,
        mode="reset",
        params={
            "flight_reference_state_path": str(flight_reference_state_path),
            "landing_reference_state_path": str(MAT_LANDING_REFERENCE_STATE_PATH),
            "landing_probability": 0.5,
            "landing_min_horizontal_distance": PEDESTAL_WIDTH / 2.0 + 0.04,
        },
    )
    # Give the sparse exact-hold event enough leverage to survive PPO updates
    # from the much longer flight slice while preserving all approach terms.
    cfg.rewards["backflip_stability_progress"].weight = 600.0
    cfg.rewards["backflip_success"].weight = 800.0
    return cfg


def make_microduck_backflip_mat_current_mixed_env_cfg(play: bool = False):
    """Rehearse apex states emitted by the latest standing-cube actor."""
    return make_microduck_backflip_mat_mixed_reference_env_cfg(
        play=play,
        flight_reference_state_path=CURRENT_FLOOR_REFERENCE_STATE_PATH,
    )


def make_microduck_backflip_mat_current_reference_env_cfg(play: bool = False):
    """Pure current-actor apex continuation for uncontaminated evaluation."""
    cfg = make_microduck_backflip_mat_reference_env_cfg(play=play)
    cfg.events["set_backflip_state"] = EventTermCfg(
        func=microduck_mdp.reset_backflip_reference_state,
        mode="reset",
        params={
            "reference_state_path": str(CURRENT_FLOOR_REFERENCE_STATE_PATH),
            "landing_min_horizontal_distance": PEDESTAL_WIDTH / 2.0 + 0.04,
        },
    )
    return cfg


def make_microduck_backflip_current_floor_reference_env_cfg(play: bool = False):
    """Pure model-600 apex continuation to the cube's lower floor."""
    cfg = make_microduck_backflip_early_reference_env_cfg(play=play)
    cfg.events["set_backflip_state"].params["reference_state_path"] = str(
        CURRENT_FLOOR_REFERENCE_STATE_PATH
    )
    return cfg


def make_microduck_backflip_launch650_floor_reference_env_cfg(
    play: bool = False,
):
    """Continue from exact 160-degree states emitted by launch model 650.

    The hierarchical evaluator gives the landing actor full authority at the
    handoff. ``reset_backflip_reference_state`` likewise disables the launch
    wrench, so training and inference see the same state/action distribution.
    """
    cfg = make_microduck_backflip_early_reference_env_cfg(play=play)
    cfg.events["set_backflip_state"].params["reference_state_path"] = str(
        LAUNCH650_FLOOR_REFERENCE_STATE_PATH
    )
    return cfg


def make_microduck_backflip_current_floor_mixed_env_cfg(play: bool = False):
    """Match both ends of model 600's real cube-to-lower-floor trajectory."""
    cfg = make_microduck_backflip_current_floor_reference_env_cfg(play=play)
    cfg.events["set_backflip_state"] = EventTermCfg(
        func=microduck_mdp.reset_backflip_mixed_reference_state,
        mode="reset",
        params={
            "flight_reference_state_path": str(CURRENT_FLOOR_REFERENCE_STATE_PATH),
            "landing_reference_state_path": str(CURRENT_FLOOR_LANDING_STATE_PATH),
            "landing_probability": 0.5,
            "landing_min_horizontal_distance": PEDESTAL_WIDTH / 2.0 + 0.04,
        },
    )
    cfg.rewards["backflip_stability_progress"].weight = 600.0
    cfg.rewards["backflip_success"].weight = 800.0
    return cfg


def make_microduck_backflip_current_floor_distillation_env_cfg(
    play: bool = False,
):
    """Distill cube launch and the matched lower-floor continuation into one actor."""
    cfg = make_microduck_backflip_current_floor_mixed_env_cfg(play=play)
    cfg.events["set_backflip_state"] = EventTermCfg(
        func=microduck_mdp.reset_backflip_distillation_state,
        mode="reset",
        params={
            "flight_reference_state_path": str(CURRENT_FLOOR_REFERENCE_STATE_PATH),
            "landing_reference_state_path": str(CURRENT_FLOOR_LANDING_STATE_PATH),
            "standing_probability": 0.20,
            "landing_probability": 0.40,
            "standing_z_range": (
                PEDESTAL_HEIGHT + 0.11,
                PEDESTAL_HEIGHT + 0.12,
            ),
            "standing_edge_offset": 0.035,
            "landing_min_horizontal_distance": PEDESTAL_WIDTH / 2.0 + 0.04,
            "initial_assist_scale": 1.0,
        },
    )
    cfg.curriculum["backflip_reference_spawn_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_backflip_state",
            "param_stages": [
                {
                    # This task warm-starts model 690, and RSL restores its
                    # global iteration counter instead of rebasing to zero.
                    "step": 690 * 24,
                    "params": {
                        "standing_probability": 0.20,
                        "landing_probability": 0.40,
                    },
                },
                {
                    "step": 730 * 24,
                    "params": {
                        "standing_probability": 0.35,
                        "landing_probability": 0.325,
                    },
                },
                {
                    "step": 760 * 24,
                    "params": {
                        "standing_probability": 0.50,
                        "landing_probability": 0.25,
                    },
                },
            ],
        },
    )
    return cfg


def make_microduck_backflip_cube_launch_env_cfg(play: bool = False):
    """Learn a reproducible cube takeoff before composing the landing policy.

    The general curriculum clamps policy targets to five percent while the
    virtual spotter is at full strength. That deadlocks launch discovery:
    landing success is needed to anneal assistance, while annealing is needed
    to expose useful action authority. This specialist keeps the time-limited
    wrench, grants full joint-target authority, and recycles worlds at 180 deg.
    """
    cfg = make_microduck_backflip_pedestal_env_cfg(play=play)
    # First solve the mechanically nominal maneuver. The general sim-to-real
    # task spans battery sag, COM, inertia, friction, encoders, and armature;
    # that makes most worlds infeasible before a launch motion exists. Robust
    # continuation will widen each range only after nominal apex success.
    actuator = cfg.scene.entities["robot"].articulation.actuators[0]
    actuator.vin_range = (8.2, 8.2)
    actuator.vin_drop_gain_range = (0.0, 0.0)
    actuator.delay_min_lag = 3
    actuator.delay_max_lag = 3
    actuator.delay_per_env_phase = False
    twist_ranges = cfg.commands["twist"].ranges
    twist_ranges.lin_vel_x = (0.0, 0.0)
    twist_ranges.lin_vel_y = (0.0, 0.0)
    twist_ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands["twist"].rel_forward_envs = 0.0
    cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
    cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)
    cfg.events["base_com"].params["ranges"] = {
        0: (0.0, 0.0),
        1: (0.0, 0.0),
        2: (0.0, 0.0),
    }
    cfg.events["randomize_mass_inertia"].params["alpha_range"] = (0.0, 0.0)
    cfg.events["randomize_com"].params["ranges"] = (0.0, 0.0)
    cfg.events["randomize_head_com"].params["ranges"] = (0.0, 0.0)
    cfg.events["randomize_armature"].params["ranges"] = (1.0, 1.0)
    cfg.events["randomize_joint_friction"].params["scale_range"] = (1.0, 1.0)
    cfg.curriculum.pop("com_range", None)
    cfg.curriculum.pop("head_com_range", None)
    reset = cfg.events["set_backflip_state"].params
    reset.update(
        {
            "standing_prob": 1.0,
            "crouch_prob": 0.0,
            "midflight_prob": 0.0,
            "recovery_prob": 0.0,
            "initial_assist_scale": 1.0,
            "standing_tilt_max": 0.0,
            "joint_noise_std": 0.0,
            "standing_z_range": (
                PEDESTAL_HEIGHT + 0.115,
                PEDESTAL_HEIGHT + 0.115,
            ),
        }
    )
    cfg.curriculum.pop("backflip_spawn_mix", None)
    # First takeover rung: enough authority to shape the push without letting
    # an untrained full-scale action destroy the nominal assisted trajectory.
    cfg.actions["joint_pos"].min_assisted_authority = 0.20
    cfg.actions["joint_pos"].full_authority_after_angle_rad = None
    # Training recycles successes immediately. Play/evaluation must retain the
    # post-apex trajectory so the external evaluator can measure composition.
    if not play:
        cfg.terminations["backflip_launch_apex"] = TerminationTermCfg(
            func=microduck_mdp.backflip_rotation_reached,
            time_out=False,
            params={"target_angle": math.pi},
        )
    cfg.rewards["backflip_rotation"].weight = 240.0
    cfg.rewards["backflip_rotation"].params["target_angle"] = math.pi
    cfg.rewards["backflip_flight_tuck"].weight = 24.0
    for name in (
        "backflip_prepare_landing",
        "backflip_landing_approach",
        "backflip_landing",
        "backflip_landing_upright",
        "backflip_landing_height",
        "backflip_landing_stillness",
        "backflip_landing_foot_support",
        "backflip_stability_progress",
        "backflip_success",
        "backflip_late_pitch_rate",
        "backflip_rotation_overshoot",
        "backflip_post_landing_angular_speed",
        "backflip_rotation_timing",
    ):
        if name in cfg.rewards:
            cfg.rewards[name].weight = 0.0
    return cfg


def make_microduck_backflip_mat_distillation_env_cfg(play: bool = False):
    """Distill the verified apex landing back into complete cube starts."""
    cfg = make_microduck_backflip_mat_mixed_reference_env_cfg(play=play)
    cfg.events["set_backflip_state"] = EventTermCfg(
        func=microduck_mdp.reset_backflip_distillation_state,
        mode="reset",
        params={
            "flight_reference_state_path": str(EARLY_REFERENCE_STATE_PATH),
            "landing_reference_state_path": str(MAT_LANDING_REFERENCE_STATE_PATH),
            "standing_probability": 0.20,
            "landing_probability": 0.40,
            "standing_z_range": (
                PEDESTAL_HEIGHT + 0.11,
                PEDESTAL_HEIGHT + 0.12,
            ),
            "standing_edge_offset": 0.035,
            "landing_min_horizontal_distance": PEDESTAL_WIDTH / 2.0 + 0.04,
            "initial_assist_scale": 1.0,
        },
    )
    cfg.curriculum["backflip_reference_spawn_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_backflip_state",
            "param_stages": [
                {
                    # This task warm-starts model 510; curriculum counters are
                    # restored from the checkpoint rather than reset to zero.
                    "step": 510 * 24,
                    "params": {
                        "standing_probability": 0.20,
                        "landing_probability": 0.40,
                    },
                },
                {
                    "step": 560 * 24,
                    "params": {
                        "standing_probability": 0.35,
                        "landing_probability": 0.325,
                    },
                },
                {
                    "step": 610 * 24,
                    "params": {
                        "standing_probability": 0.50,
                        "landing_probability": 0.25,
                    },
                },
            ],
        },
    )
    return cfg


def make_microduck_backflip_soft_mat_distillation_env_cfg(
    play: bool = False,
):
    """Standing-cube curriculum with a higher, softer gymnastics mat."""
    cfg = make_microduck_backflip_mat_distillation_env_cfg(play=play)
    mat = cfg.scene.terrain.terrain_generator.sub_terrains["backflip_mat"]
    mat.landing_mat_height = SOFT_MAT_HEIGHT
    mat.mat_contact_time_s = 0.12
    target_height = SOFT_MAT_HEIGHT + STAND_Z
    cfg.rewards["backflip_landing_approach"].params["target_height"] = target_height
    cfg.rewards["backflip_landing"].params["target_height"] = target_height
    cfg.rewards["backflip_landing_height"].params.update(
        {
            "target_height": target_height,
            "minimum_height": SOFT_MAT_HEIGHT + 0.06,
        }
    )
    cfg.rewards["backflip_rotation_timing"].params["target_height"] = target_height
    return cfg


MicroduckBackflipRlCfg = deepcopy(MicroduckRouladeRlCfg)
MicroduckBackflipRlCfg.experiment_name = "microduck_backflip"
MicroduckBackflipRlCfg.run_name = "microduck_backflip"
MicroduckBackflipRlCfg.max_iterations = 5_000
# The roulade runner mirrors every sample and penalizes asymmetric actor
# outputs.  That is counterproductive here: asymmetric leg/arm timing is now a
# valid way to create takeoff torque and a mild axial correction in flight.
MicroduckBackflipRlCfg.algorithm.symmetry_cfg = None

MicroduckBackflipRecoveryRlCfg = deepcopy(MicroduckBackflipRlCfg)
MicroduckBackflipRecoveryRlCfg.experiment_name = "microduck_backflip_recovery"
MicroduckBackflipRecoveryRlCfg.run_name = "microduck_backflip_recovery"

MicroduckBackflipTouchdownRlCfg = deepcopy(MicroduckBackflipRlCfg)
MicroduckBackflipTouchdownRlCfg.experiment_name = "microduck_backflip_touchdown"
MicroduckBackflipTouchdownRlCfg.run_name = "microduck_backflip_touchdown"

MicroduckBackflipPedestalRlCfg = deepcopy(MicroduckBackflipRlCfg)
MicroduckBackflipPedestalRlCfg.experiment_name = "microduck_backflip_pedestal"
MicroduckBackflipPedestalRlCfg.run_name = "microduck_backflip_pedestal"

MicroduckBackflipPedestalBrakingRlCfg = deepcopy(MicroduckBackflipRlCfg)
MicroduckBackflipPedestalBrakingRlCfg.experiment_name = (
    "microduck_backflip_pedestal_braking"
)
MicroduckBackflipPedestalBrakingRlCfg.run_name = (
    "microduck_backflip_pedestal_braking"
)

MicroduckBackflipReferenceRlCfg = deepcopy(MicroduckBackflipRlCfg)
MicroduckBackflipReferenceRlCfg.experiment_name = "microduck_backflip_reference"
MicroduckBackflipReferenceRlCfg.run_name = "microduck_backflip_reference"
MicroduckBackflipReferenceRlCfg.algorithm.learning_rate = 3.0e-4
MicroduckBackflipReferenceRlCfg.algorithm.entropy_coef = 0.002

MicroduckBackflipEarlyReferenceRlCfg = deepcopy(MicroduckBackflipReferenceRlCfg)
MicroduckBackflipEarlyReferenceRlCfg.experiment_name = (
    "microduck_backflip_reference_early"
)
MicroduckBackflipEarlyReferenceRlCfg.run_name = "microduck_backflip_reference_early"

MicroduckBackflipMatReferenceRlCfg = deepcopy(MicroduckBackflipEarlyReferenceRlCfg)
MicroduckBackflipMatReferenceRlCfg.experiment_name = (
    "microduck_backflip_reference_mat"
)
MicroduckBackflipMatReferenceRlCfg.run_name = "microduck_backflip_reference_mat"

MicroduckBackflipMatLandingReferenceRlCfg = deepcopy(
    MicroduckBackflipMatReferenceRlCfg
)
MicroduckBackflipMatLandingReferenceRlCfg.experiment_name = (
    "microduck_backflip_reference_mat_landing"
)
MicroduckBackflipMatLandingReferenceRlCfg.run_name = (
    "microduck_backflip_reference_mat_landing"
)

MicroduckBackflipLaunch650MatLandingRlCfg = deepcopy(
    MicroduckBackflipMatLandingReferenceRlCfg
)
MicroduckBackflipLaunch650MatLandingRlCfg.experiment_name = (
    "microduck_backflip_launch650_mat_landing"
)
MicroduckBackflipLaunch650MatLandingRlCfg.run_name = (
    "microduck_backflip_launch650_mat_landing"
)
MicroduckBackflipLaunch650MatLandingRlCfg.algorithm.learning_rate = 1.0e-4
MicroduckBackflipLaunch650MatLandingRlCfg.algorithm.entropy_coef = 0.001

MicroduckBackflipLaunch650MatApproachRlCfg = deepcopy(
    MicroduckBackflipMatReferenceRlCfg
)
MicroduckBackflipLaunch650MatApproachRlCfg.experiment_name = (
    "microduck_backflip_launch650_mat_approach"
)
MicroduckBackflipLaunch650MatApproachRlCfg.run_name = (
    "microduck_backflip_launch650_mat_approach"
)
MicroduckBackflipLaunch650MatApproachRlCfg.algorithm.learning_rate = 1.0e-4
MicroduckBackflipLaunch650MatApproachRlCfg.algorithm.entropy_coef = 0.001

MicroduckBackflipMatMixedReferenceRlCfg = deepcopy(
    MicroduckBackflipMatReferenceRlCfg
)
MicroduckBackflipMatMixedReferenceRlCfg.experiment_name = (
    "microduck_backflip_reference_mat_mixed"
)
MicroduckBackflipMatMixedReferenceRlCfg.run_name = (
    "microduck_backflip_reference_mat_mixed"
)

MicroduckBackflipMatDistillationRlCfg = deepcopy(
    MicroduckBackflipMatMixedReferenceRlCfg
)
MicroduckBackflipMatDistillationRlCfg.experiment_name = (
    "microduck_backflip_reference_mat_distillation"
)
MicroduckBackflipMatDistillationRlCfg.run_name = (
    "microduck_backflip_reference_mat_distillation"
)

MicroduckBackflipMatCurrentMixedRlCfg = deepcopy(
    MicroduckBackflipMatMixedReferenceRlCfg
)
MicroduckBackflipMatCurrentMixedRlCfg.experiment_name = (
    "microduck_backflip_reference_mat_current"
)
MicroduckBackflipMatCurrentMixedRlCfg.run_name = (
    "microduck_backflip_reference_mat_current"
)

MicroduckBackflipCurrentFloorRlCfg = deepcopy(
    MicroduckBackflipMatCurrentMixedRlCfg
)
MicroduckBackflipCurrentFloorRlCfg.experiment_name = (
    "microduck_backflip_reference_current_floor"
)
MicroduckBackflipCurrentFloorRlCfg.run_name = (
    "microduck_backflip_reference_current_floor"
)

MicroduckBackflipLaunch650FloorRlCfg = deepcopy(
    MicroduckBackflipCurrentFloorRlCfg
)
MicroduckBackflipLaunch650FloorRlCfg.experiment_name = (
    "microduck_backflip_reference_launch650_floor"
)
MicroduckBackflipLaunch650FloorRlCfg.run_name = (
    "microduck_backflip_reference_launch650_floor"
)
MicroduckBackflipLaunch650FloorRlCfg.algorithm.learning_rate = 1.0e-4
MicroduckBackflipLaunch650FloorRlCfg.algorithm.entropy_coef = 0.001

MicroduckBackflipCurrentFloorDistillationRlCfg = deepcopy(
    MicroduckBackflipCurrentFloorRlCfg
)
MicroduckBackflipCurrentFloorDistillationRlCfg.experiment_name = (
    "microduck_backflip_reference_current_floor_distillation"
)
MicroduckBackflipCurrentFloorDistillationRlCfg.run_name = (
    "microduck_backflip_reference_current_floor_distillation"
)

MicroduckBackflipCubeLaunchRlCfg = deepcopy(MicroduckBackflipPedestalRlCfg)
MicroduckBackflipCubeLaunchRlCfg.experiment_name = "microduck_backflip_cube_launch"
MicroduckBackflipCubeLaunchRlCfg.run_name = "microduck_backflip_cube_launch"
MicroduckBackflipCubeLaunchRlCfg.algorithm.learning_rate = 1.0e-4
MicroduckBackflipCubeLaunchRlCfg.algorithm.entropy_coef = 0.001
