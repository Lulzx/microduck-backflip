"""State-machine regression tests for what physically counts as a backflip."""

from types import SimpleNamespace

import torch

from mjlab_microduck.tasks import mdp


class _Scene:
    def __init__(self, asset):
        self.asset = asset
        self.sensors = {
            "feet_ground_contact": SimpleNamespace(
                data=SimpleNamespace(found=torch.ones((1, 1), dtype=torch.int32))
            ),
            "robot_ground_contact": SimpleNamespace(
                data=SimpleNamespace(found=torch.ones((1, 1), dtype=torch.int32))
            ),
        }
        self.terrain = SimpleNamespace(env_origins=torch.zeros((1, 3)))

    def __getitem__(self, name):
        assert name == "robot"
        return self.asset


def _fake_env():
    data = SimpleNamespace(
        root_link_pos_w=torch.tensor([[0.0, 0.0, 0.115]]),
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        root_link_ang_vel_b=torch.zeros((1, 3)),
    )
    asset = SimpleNamespace(data=data)
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        common_step_counter=0,
        episode_length_buf=torch.zeros(1, dtype=torch.long),
    )
    env.scene = _Scene(asset)
    return env, asset


def test_assist_observation_is_bounded_and_exposes_scale():
    env, _ = _fake_env()
    env._backflip_assist_scale = 0.65
    env.episode_length_buf[:] = 15
    obs = mdp.backflip_assist_observation(env, timing_scale_s=0.30)
    assert obs.shape == (1, 6)
    assert torch.isclose(obs[0, 0], torch.tensor(0.5))
    assert torch.isclose(obs[0, 1], torch.tensor(0.65))
    assert torch.equal(obs[0, 2:], torch.zeros(4))


def test_assist_curriculum_decays_only_after_success_threshold():
    env, _ = _fake_env()
    mdp._backflip_state(env)
    ids = torch.tensor([0])
    # Initial reset configures the scale but has no completed episode to count.
    mdp._update_backflip_assist_curriculum(env, ids, 1.0, 0.1, 0.6, 1)
    assert env._backflip_assist_scale == 1.0

    env._backflip_assist_initialized[:] = True
    env._backflip_assist_eligible[:] = True
    env._backflip_landed_latch[:] = False
    mdp._update_backflip_assist_curriculum(env, ids, 1.0, 0.1, 0.6, 1)
    assert env._backflip_assist_scale == 1.0

    env._backflip_landed_latch[:] = True
    env._backflip_stable_latch[:] = True
    mdp._update_backflip_assist_curriculum(env, ids, 1.0, 0.1, 0.6, 1)
    assert abs(env._backflip_assist_scale - 0.9) < 1e-9


def _contacts(env, feet: bool, robot: bool):
    env.scene.sensors["feet_ground_contact"].data.found.fill_(int(feet))
    env.scene.sensors["robot_ground_contact"].data.found.fill_(int(robot))


def test_grounded_backward_roll_gets_no_backflip_rotation_credit():
    env, asset = _fake_env()
    _contacts(env, feet=True, robot=True)
    asset.data.root_link_ang_vel_b[0, 1] = -8.0
    for step in range(1, 6):
        env.common_step_counter = step
        mdp._update_backflip_state(env, asset)
    assert env._backflip_airborne_latch.item() is False
    assert env._backflip_max.item() == 0.0


def test_supported_takeoff_then_airborne_backward_pitch_accumulates():
    env, asset = _fake_env()
    mdp._update_backflip_state(env, asset)  # establish supported start
    _contacts(env, feet=False, robot=False)
    asset.data.root_link_pos_w[0, 2] = 0.16
    asset.data.root_link_ang_vel_b[0, 1] = -5.0
    env.common_step_counter = 1
    mdp._update_backflip_state(env, asset)
    assert env._backflip_airborne_latch.item() is True
    assert torch.isclose(env._backflip_max, torch.tensor([0.1])).all()


def test_forward_pitch_after_takeoff_does_not_advance_frontier():
    env, asset = _fake_env()
    mdp._update_backflip_state(env, asset)
    _contacts(env, feet=False, robot=False)
    asset.data.root_link_pos_w[0, 2] = 0.16
    asset.data.root_link_ang_vel_b[0, 1] = 5.0
    env.common_step_counter = 1
    mdp._update_backflip_state(env, asset)
    assert env._backflip_accum.item() < 0.0
    assert env._backflip_max.item() == 0.0


def test_feet_first_upright_contact_after_320deg_latches_landing():
    env, asset = _fake_env()
    mdp._backflip_state(env)
    env._backflip_had_support[:] = True
    env._backflip_airborne_latch[:] = True
    env._backflip_max[:] = torch.deg2rad(torch.tensor([325.0]))
    _contacts(env, feet=True, robot=True)
    env.common_step_counter = 1
    mdp._update_backflip_state(env, asset)
    assert env._backflip_landed_latch.item() is True


def test_success_requires_continuous_half_second_stable_hold():
    env, asset = _fake_env()
    mdp._backflip_state(env)
    env._backflip_had_support[:] = True
    env._backflip_airborne_latch[:] = True
    env._backflip_max[:] = torch.deg2rad(torch.tensor([350.0]))
    env._backflip_landed_latch[:] = True
    _contacts(env, feet=True, robot=True)

    for step in range(1, 25):
        env.common_step_counter = step
        mdp._update_backflip_state(env, asset)
    assert mdp.backflip_success(env).item() == 0.0

    env.common_step_counter = 25
    mdp._update_backflip_state(env, asset)
    assert mdp.backflip_success(env).item() == 1.0

    # Once proven, success remains latched even if the robot later moves.
    asset.data.root_link_ang_vel_b[0, 1] = 4.0
    env.common_step_counter = 26
    mdp._update_backflip_state(env, asset)
    assert mdp.backflip_success(env).item() == 1.0


def test_landing_phase_components_remain_dense_when_one_factor_is_bad():
    env, asset = _fake_env()
    mdp._backflip_state(env)
    env._backflip_landed_latch[:] = True

    # A low body height no longer erases the independent upright, stillness,
    # and foot-support objectives.
    asset.data.root_link_pos_w[0, 2] = 0.06
    assert mdp.backflip_landing_height(env, target_height=0.115).item() == 0.0
    assert mdp.backflip_landing_upright(env).item() == 1.0
    assert mdp.backflip_landing_stillness(env).item() == 1.0
    assert mdp.backflip_landing_foot_support(env).item() == 1.0


def test_stability_progress_only_pays_for_a_new_hold_frontier():
    env, asset = _fake_env()
    mdp._backflip_state(env)
    env._backflip_had_support[:] = True
    env._backflip_airborne_latch[:] = True
    env._backflip_landed_latch[:] = True
    env._backflip_max[:] = torch.deg2rad(torch.tensor([350.0]))
    _contacts(env, feet=True, robot=True)

    first_streak = []
    for step in range(1, 6):
        env.common_step_counter = step
        first_streak.append(mdp.backflip_stability_progress(env).item())
    assert all(value > 0.0 for value in first_streak)

    # Interrupt the hold, then replay the same five-step prefix. Previously
    # earned progress cannot be farmed by repeated near-falls.
    asset.data.root_link_ang_vel_b[0, 1] = 4.0
    env.common_step_counter = 6
    assert mdp.backflip_stability_progress(env).item() == 0.0
    asset.data.root_link_ang_vel_b.zero_()
    for step in range(7, 12):
        env.common_step_counter = step
        assert mdp.backflip_stability_progress(env).item() == 0.0
    env.common_step_counter = 12
    assert mdp.backflip_stability_progress(env).item() > 0.0


def test_body_only_contact_cannot_latch_landing():
    env, asset = _fake_env()
    mdp._backflip_state(env)
    env._backflip_had_support[:] = True
    env._backflip_airborne_latch[:] = True
    env._backflip_max[:] = torch.deg2rad(torch.tensor([360.0]))
    _contacts(env, feet=False, robot=True)
    env.common_step_counter = 1
    mdp._update_backflip_state(env, asset)
    assert env._backflip_landed_latch.item() is False


def test_rotation_frontier_freezes_after_first_ground_recontact():
    env, asset = _fake_env()
    mdp._backflip_state(env)
    env._backflip_had_support[:] = True
    env._backflip_airborne_latch[:] = True
    asset.data.root_link_ang_vel_b[0, 1] = -8.0

    _contacts(env, feet=False, robot=False)
    env.common_step_counter = 1
    mdp._update_backflip_state(env, asset)
    first_flight = env._backflip_max.clone()

    _contacts(env, feet=True, robot=True)
    env.common_step_counter = 2
    mdp._update_backflip_state(env, asset)
    assert env._backflip_flight_ended_latch.item() is True

    _contacts(env, feet=False, robot=False)
    env.common_step_counter = 3
    mdp._update_backflip_state(env, asset)
    assert torch.equal(env._backflip_max, first_flight)


def test_launch_velocity_progress_is_one_shot_and_stops_after_impact():
    env, asset = _fake_env()
    asset.data.root_link_lin_vel_w = torch.tensor([[0.0, 0.0, 0.75]])
    mdp._backflip_state(env)
    env._backflip_had_support[:] = True
    env._backflip_airborne_latch[:] = True
    _contacts(env, feet=False, robot=False)
    first = mdp.backflip_launch_velocity_progress(env, target_velocity=1.5)
    env.common_step_counter = 1
    second = mdp.backflip_launch_velocity_progress(env, target_velocity=1.5)
    assert first.item() > 0.0
    assert second.item() == 0.0

    env._backflip_airborne_latch[:] = True
    _contacts(env, feet=False, robot=True)
    asset.data.root_link_lin_vel_w[0, 2] = 1.5
    env.common_step_counter = 2
    after_impact = mdp.backflip_launch_velocity_progress(env, target_velocity=1.5)
    assert after_impact.item() == 0.0


def test_full_spotter_assistance_suppresses_free_launch_credit():
    env, asset = _fake_env()
    asset.data.root_link_lin_vel_w = torch.tensor([[0.0, 0.0, 0.75]])
    mdp._backflip_state(env)
    env._backflip_assist_scale = 1.0
    env._backflip_had_support[:] = True
    env._backflip_airborne_latch[:] = True
    _contacts(env, feet=False, robot=False)
    assisted = mdp.backflip_launch_velocity_progress(env, target_velocity=1.5)
    assert assisted.item() == 0.0

    # A fresh autonomous episode receives the same physical-progress credit.
    env2, asset2 = _fake_env()
    asset2.data.root_link_lin_vel_w = torch.tensor([[0.0, 0.0, 0.75]])
    mdp._backflip_state(env2)
    env2._backflip_assist_scale = 0.0
    env2._backflip_had_support[:] = True
    env2._backflip_airborne_latch[:] = True
    _contacts(env2, feet=False, robot=False)
    autonomous = mdp.backflip_launch_velocity_progress(env2, target_velocity=1.5)
    assert autonomous.item() > 0.0


def test_assisted_action_cost_fades_to_zero_with_spotter():
    env, _ = _fake_env()
    env.action_manager = SimpleNamespace(action=torch.ones((1, 14)))
    env._backflip_assist_scale = 1.0
    env._backflip_assist_eligible = torch.tensor([True])
    assert mdp.backflip_assisted_action_cost(env).item() == 1.0
    env._backflip_assist_eligible[:] = False
    assert mdp.backflip_assisted_action_cost(env).item() == 0.0
    env._backflip_assist_eligible[:] = True
    env._backflip_flight_ended_latch = torch.tensor([True])
    assert mdp.backflip_assisted_action_cost(env).item() == 0.0
    env._backflip_flight_ended_latch[:] = False
    env._backflip_assist_scale = 0.0
    assert mdp.backflip_assisted_action_cost(env).item() == 0.0


def test_launch_quality_requires_simultaneous_jump_and_backward_pitch():
    env, asset = _fake_env()
    mdp._backflip_state(env)
    env._backflip_had_support[:] = True
    env._backflip_airborne_latch[:] = True
    _contacts(env, feet=False, robot=False)
    asset.data.root_link_lin_vel_w = torch.tensor([[0.0, 0.0, 1.2]])
    asset.data.root_link_ang_vel_b[0, 1] = 0.0
    no_spin = mdp.backflip_launch_quality_progress(env)
    assert no_spin.item() == 0.0

    asset.data.root_link_ang_vel_b[0, 1] = -5.0
    env.common_step_counter = 1
    coupled = mdp.backflip_launch_quality_progress(env)
    assert coupled.item() > 0.0


def test_supported_push_requires_crouch_support_and_coupled_rates():
    env, asset = _fake_env()
    asset.data.root_link_lin_vel_w = torch.tensor([[0.0, 0.0, 1.2]])
    asset.data.root_link_ang_vel_b[0, 1] = -10.0
    mdp._backflip_state(env)

    # A standing hop has not earned the crouch prerequisite.
    no_crouch = mdp.backflip_supported_push_quality_progress(env)
    assert no_crouch.item() == 0.0

    env._backflip_max_preload[:] = 0.75
    env.common_step_counter = 1
    supported = mdp.backflip_supported_push_quality_progress(env)
    assert supported.item() > 0.0

    # The potential is one-shot and disappears as soon as support is gone.
    env.common_step_counter = 2
    repeated = mdp.backflip_supported_push_quality_progress(env)
    assert repeated.item() == 0.0
    _contacts(env, feet=False, robot=False)
    env.common_step_counter = 3
    airborne = mdp.backflip_supported_push_quality_progress(env)
    assert airborne.item() == 0.0


def test_feasible_push_reserves_credit_for_high_energy_coupled_launch():
    env, asset = _fake_env()
    mdp._backflip_state(env)
    env._backflip_max_preload[:] = 0.75
    asset.data.root_link_lin_vel_w = torch.tensor([[0.0, 0.0, 0.59]])
    asset.data.root_link_ang_vel_b[0, 1] = -12.0
    below_threshold = mdp.backflip_feasible_push_progress(env)
    assert below_threshold.item() == 0.0

    asset.data.root_link_lin_vel_w[0, 2] = 1.5
    env.common_step_counter = 1
    feasible = mdp.backflip_feasible_push_progress(env)
    assert feasible.item() > 0.0

    # High vertical speed without simultaneous backward pitch is not useful.
    env2, asset2 = _fake_env()
    mdp._backflip_state(env2)
    env2._backflip_max_preload[:] = 0.75
    asset2.data.root_link_lin_vel_w = torch.tensor([[0.0, 0.0, 1.5]])
    asset2.data.root_link_ang_vel_b[0, 1] = 0.0
    straight_jump = mdp.backflip_feasible_push_progress(env2)
    assert straight_jump.item() == 0.0
