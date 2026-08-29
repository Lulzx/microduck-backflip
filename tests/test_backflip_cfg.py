"""Configuration invariants for the airborne MicroDuck backflip task."""

import math

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.backflip_actions import (
    BackflipResidualJointPositionActionCfg,
)
from mjlab_microduck.tasks.microduck_backflip_env_cfg import (
    MicroduckBackflipRlCfg,
    make_microduck_backflip_env_cfg,
    make_microduck_backflip_recovery_env_cfg,
    make_microduck_backflip_touchdown_env_cfg,
    make_microduck_backflip_pedestal_env_cfg,
)
from mjlab_microduck.tasks.backflip_pedestal_terrain import (
    PEDESTAL_HEIGHT,
    PEDESTAL_WIDTH,
)
from mjlab_microduck.tasks.microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
)


def test_backflip_has_airborne_task_rewards_and_no_roulade_rewards():
    cfg = make_microduck_backflip_env_cfg()
    assert not any(name.startswith("roulade_") for name in cfg.rewards)
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
        "backflip_landing",
        "backflip_landing_upright",
        "backflip_landing_height",
        "backflip_landing_stillness",
        "backflip_landing_foot_support",
        "backflip_stability_progress",
        "backflip_success",
        "backflip_body_contact",
        "backflip_assisted_action",
        "backflip_late_pitch_rate",
    ):
        assert name in cfg.rewards


def test_reward_signs_do_not_pay_crashes_or_wrong_way_rotation():
    cfg = make_microduck_backflip_env_cfg()
    assert cfg.rewards["backflip_rotation"].weight > 0.0
    assert cfg.rewards["backflip_landing"].weight > 0.0
    assert cfg.rewards["backflip_stability_progress"].weight > 0.0
    assert cfg.rewards["backflip_body_contact"].weight == 0.0
    assert cfg.rewards["backflip_wrong_direction"].weight < 0.0
    assert cfg.rewards["backflip_assisted_action"].weight < 0.0
    assert cfg.rewards["backflip_late_pitch_rate"].weight < 0.0
    body_contact_stages = cfg.curriculum["body_contact_weight"].params["weight_stages"]
    assert body_contact_stages[0]["weight"] < 0.0
    assert body_contact_stages[-1]["weight"] < 0.0
    # trunk_vertical_accel_penalty is self-negating; its curriculum weights
    # must therefore be non-negative.
    assert all(
        stage["weight"] >= 0.0
        for stage in cfg.curriculum["gentle_landing_weight"].params["weight_stages"]
    )


def test_reset_and_curriculum_use_midflight_reverse_curriculum():
    cfg = make_microduck_backflip_env_cfg()
    assert "set_roulade_state" not in cfg.events
    reset = cfg.events["set_backflip_state"]
    assert reset.func is microduck_mdp.reset_backflip_state
    assert reset.params["crouch_prob"] > 0.0
    assert reset.params["crouch_overrides"]
    lo, hi = reset.params["midflight_angle_range"]
    assert lo < math.pi < hi
    stages = cfg.curriculum["backflip_spawn_mix"].params["param_stages"]
    standing = [s["params"]["standing_prob"] for s in stages]
    midflight = [s["params"]["midflight_prob"] for s in stages]
    recovery = [s["params"]["recovery_prob"] for s in stages]
    # Spotting supplies early launch exploration, so recovery states are
    # front-loaded and standing starts take over as autonomy grows.
    assert standing == sorted(standing)
    assert midflight == sorted(midflight, reverse=True)
    assert standing[0] < standing[-1]
    assert midflight[0] > midflight[-1]
    assert recovery == sorted(recovery, reverse=True)
    assert recovery[0] > recovery[-1] > 0.0
    assert all(s["params"]["crouch_prob"] > 0.0 for s in stages)


def test_strict_eval_disables_every_reverse_curriculum_start():
    from mjlab_microduck.tasks.microduck_backflip_env_cfg import (
        configure_backflip_standing_eval,
    )

    cfg = make_microduck_backflip_env_cfg(play=True)
    configure_backflip_standing_eval(cfg)
    reset = cfg.events["set_backflip_state"].params
    assert reset["standing_prob"] == 1.0
    assert reset["crouch_prob"] == 0.0
    assert reset["midflight_prob"] == 0.0
    assert reset["recovery_prob"] == 0.0


def test_recovery_specialist_uses_easy_to_full_reference_state_curriculum():
    cfg = make_microduck_backflip_recovery_env_cfg()
    reset = cfg.events["set_backflip_state"].params
    assert reset["standing_prob"] == 0.0
    assert reset["crouch_prob"] == 0.0
    assert reset["midflight_prob"] == 0.0
    assert reset["recovery_prob"] == 1.0
    assert reset["initial_assist_scale"] == 0.0
    assert "backflip_spawn_mix" not in cfg.curriculum
    assert "backflip_body_only_contact" in cfg.terminations

    stages = cfg.curriculum["backflip_recovery_difficulty"].params["param_stages"]
    assert stages[0]["params"]["recovery_tilt_max"] < stages[-1]["params"][
        "recovery_tilt_max"
    ]
    assert stages[0]["params"]["recovery_ang_vel_max"] < stages[-1]["params"][
        "recovery_ang_vel_max"
    ]
    assert stages[-1]["params"]["recovery_ang_vel_max"] == 20.0
    assert stages[0]["params"]["recovery_vertical_velocity_range"][0] < 0.0
    assert (
        stages[-1]["params"]["recovery_vertical_velocity_range"][0]
        < stages[0]["params"]["recovery_vertical_velocity_range"][0]
    )


def test_pedestal_curriculum_starts_on_cube_and_requires_lower_floor_clearance():
    cfg = make_microduck_backflip_pedestal_env_cfg()
    reset = cfg.events["set_backflip_state"].params
    assert cfg.scene.terrain.terrain_type == "generator"
    assert reset["standing_prob"] > 0.0
    assert reset["midflight_prob"] > 0.0
    assert reset["recovery_prob"] > 0.0
    assert reset["standing_z_range"][0] >= PEDESTAL_HEIGHT + 0.11
    assert reset["crouch_z_range"][0] >= PEDESTAL_HEIGHT + 0.06
    assert reset["standing_edge_offset"] > 0.0
    assert reset["floor_reset_distance"] > PEDESTAL_WIDTH / 2.0
    assert reset["landing_min_horizontal_distance"] > PEDESTAL_WIDTH / 2.0
    assert cfg.rewards["backflip_takeoff"].params["start_height"] > PEDESTAL_HEIGHT
    assert "backflip_spawn_mix" in cfg.curriculum
    assert cfg.events["backflip_assistive_wrench"].params["backward_force_n"] > 0.0


def test_touchdown_specialist_starts_airborne_and_expands_approach_distribution():
    cfg = make_microduck_backflip_touchdown_env_cfg()
    reset = cfg.events["set_backflip_state"].params
    assert reset["standing_prob"] == 0.0
    assert reset["midflight_prob"] == 1.0
    assert reset["recovery_prob"] == 0.0
    assert reset["midflight_ballistic_landing"] is True
    assert reset["initial_assist_scale"] == 0.0
    assert "backflip_body_only_contact" in cfg.terminations
    assert (
        cfg.terminations["backflip_body_only_contact"].func
        is microduck_mdp.backflip_postflight_body_only_contact
    )
    assert cfg.rewards["backflip_late_pitch_rate"].weight <= -100.0
    assert cfg.rewards["backflip_rotation_overshoot"].weight < 0.0
    assert "body_contact_weight" not in cfg.curriculum
    stages = cfg.curriculum["backflip_touchdown_difficulty"].params["param_stages"]
    assert stages[0]["params"]["midflight_angle_range"][0] > stages[-1][
        "params"
    ]["midflight_angle_range"][0]
    assert stages[-1]["params"]["midflight_omega_range"][1] >= 20.0
    assert stages[-1]["params"]["midflight_z_range"][1] >= 0.35


def test_rotation_credit_is_a_full_revolution_and_is_rate_limited():
    cfg = make_microduck_backflip_env_cfg()
    params = cfg.rewards["backflip_rotation"].params
    assert params["target_angle"] == 2 * math.pi
    assert 10.0 <= params["max_paid_rate"] <= 20.0


def test_discovery_targets_enough_launch_speed_for_ballistic_flight():
    cfg = make_microduck_backflip_env_cfg()
    launch = cfg.rewards["backflip_launch_velocity"]
    assert launch.weight > 0.0
    assert launch.params["target_velocity"] >= 1.0
    reset = cfg.events["set_backflip_state"].params
    assert reset["midflight_omega_range"][1] >= 15.0
    assert reset["midflight_ballistic_landing"] is True
    assert reset["midflight_time_margin_range"][0] > 0.0
    quality = cfg.rewards["backflip_launch_quality"]
    assert quality.weight > 0.0
    assert quality.params["target_pitch_rate"] >= 8.0
    push = cfg.rewards["backflip_supported_push"]
    assert push.weight > quality.weight
    assert push.params["target_velocity"] >= 1.0
    assert push.params["target_pitch_rate"] >= 8.0
    assert push.params["min_preload"] >= 0.5
    feasible = cfg.rewards["backflip_feasible_push"]
    assert feasible.weight > push.weight
    assert feasible.params["min_velocity"] < feasible.params["target_velocity"]
    preload = cfg.rewards["backflip_preload"]
    assert preload.weight > 0.0
    assert preload.params["joint_targets"] == reset["crouch_overrides"]
    preload_stages = cfg.curriculum["preload_weight"].params["weight_stages"]
    assert preload_stages[0]["weight"] > preload_stages[-1]["weight"] > 0.0
    tuck = cfg.rewards["backflip_flight_tuck"]
    assert tuck.weight > 0.0
    assert tuck.params["joint_targets"] == reset["tuck_overrides"]
    assert tuck.params["gate_hi"] < tuck.params["fade_out"] < 2 * math.pi


def test_motion_blockers_are_off_during_discovery():
    cfg = make_microduck_backflip_env_cfg()
    assert cfg.rewards["angular_momentum"].weight == 0.0
    assert cfg.rewards["action_rate_l2"].weight == 0.0
    first_action_stage = cfg.curriculum["action_rate_weight"].params["weight_stages"][0]
    assert first_action_stage == {"step": 0, "weight": 0.0}


def test_assisted_policy_is_a_bounded_residual_with_full_strict_authority():
    cfg = make_microduck_backflip_env_cfg()
    action = cfg.actions["joint_pos"]
    assert isinstance(action, BackflipResidualJointPositionActionCfg)
    assert 0.0 < action.min_assisted_authority < 1.0
    assert action.scale == 1.0


def test_actor_and_critic_observation_layout_matches_existing_trick_policy():
    flip = make_microduck_backflip_env_cfg()
    roll = make_microduck_roulade_env_cfg()
    for group in ("actor", "critic"):
        assert list(flip.observations[group].terms) == list(
            roll.observations[group].terms
        )
    assert flip.observations["actor"].terms["head_command"].params["dim"] == 4
    assert flip.observations["actor"].terms["body_command"].params["dim"] == 6
    assert (
        flip.observations["actor"].terms["body_command"].func
        is microduck_mdp.backflip_assist_observation
    )


def test_assistive_wrench_is_success_gated_and_eval_disables_it():
    cfg = make_microduck_backflip_env_cfg()
    assist = cfg.events["backflip_assistive_wrench"]
    assert assist.mode == "step"
    assert assist.params["upward_force_n"] > 0.0
    assert assist.params["backward_pitch_torque_nm"] > 0.0
    reset = cfg.events["set_backflip_state"].params
    assert reset["initial_assist_scale"] == 1.0
    assert reset["assist_success_threshold"] == 0.60
    assert 0.0 < reset["assist_decay_step"] < 1.0

    from mjlab_microduck.tasks.microduck_backflip_env_cfg import (
        configure_backflip_standing_eval,
    )

    configure_backflip_standing_eval(cfg)
    assert cfg.events["set_backflip_state"].params["initial_assist_scale"] == 0.0


def test_runner_has_a_distinct_experiment_and_normalized_actor():
    assert MicroduckBackflipRlCfg.experiment_name == "microduck_backflip"
    assert MicroduckBackflipRlCfg.actor.obs_normalization is True


def test_backflip_allows_asymmetric_launch_and_mild_corkscrew():
    cfg = make_microduck_backflip_env_cfg()
    assert MicroduckBackflipRlCfg.algorithm.symmetry_cfg is None
    assert cfg.rewards["backflip_sagittal"].weight == 0.0
    assert -0.1 < cfg.rewards["backflip_flatness"].weight < 0.0
    assert -0.1 < cfg.rewards["backflip_lateral_velocity"].weight < 0.0


def test_evaluator_removes_the_spawn_curriculum_before_forcing_standing():
    # The live CurriculumManager would otherwise restore the 50/50 training
    # mix before the wrapper's reset and silently invalidate the battery.
    from mjlab_microduck.tasks.microduck_backflip_env_cfg import (
        configure_backflip_standing_eval,
    )

    cfg = make_microduck_backflip_env_cfg(play=True)
    configure_backflip_standing_eval(cfg)
    reset = cfg.events["set_backflip_state"]
    assert reset.params["standing_prob"] == 1.0
    assert reset.params["crouch_prob"] == 0.0
    assert reset.params["midflight_prob"] == 0.0
    assert "backflip_spawn_mix" not in cfg.curriculum
