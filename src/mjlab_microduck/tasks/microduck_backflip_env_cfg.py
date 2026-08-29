"""MicroDuck standing backflip task.

The task is deliberately separate from ``roulade``: a roulade only credits
supported forward rotation, while a backflip only credits collision-free
airborne backward rotation after a real takeoff.  Reverse-curriculum spawns
teach the landing half before the full jump is discovered.
"""

import math
from copy import deepcopy

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
    BackflipPedestalTerrainCfg,
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
    reset_cfg.params["standing_prob"] = 1.0
    reset_cfg.params["crouch_prob"] = 0.0
    reset_cfg.params["midflight_prob"] = 0.0
    reset_cfg.params["recovery_prob"] = 0.0
    reset_cfg.params["initial_assist_scale"] = 0.0
    cfg.curriculum.pop("backflip_spawn_mix", None)


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

MicroduckBackflipPedestalRlCfg = deepcopy(MicroduckBackflipRlCfg)
MicroduckBackflipPedestalRlCfg.experiment_name = "microduck_backflip_pedestal"
MicroduckBackflipPedestalRlCfg.run_name = "microduck_backflip_pedestal"
