"""Tests for SSA physics primitives."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from src.core.config_loader import ExperimentConfig, apply_overrides, load_config
from src.core.experiment_runner import ExperimentRunner
from src.core.satellite_env import scope_observation
from src.orbital.isl import effective_data_rate_bps, is_isl_feasible, vector_range_km
from src.ssa.env import SSAEnvironment, SSA_MODES
from src.ssa.metrics import SSAMetricsCollector, geometric_utility_ceiling
from src.ssa.symbolic import RuleBasedSSA
from src.ssa.targets import (
    DetectionAccess,
    RSOTarget,
    apparent_magnitude,
    detection_draw,
    detection_probability,
    generate_family_catalog,
    optical_accesses,
    propagate_rso_position_km,
    sun_unit_eci,
    target_sunlit,
)


def test_committed_ssa_scenario_uses_microsat_platform_contract() -> None:
    scenario = yaml.safe_load(
        Path("configs/scenarios/ssa.yaml").read_text(encoding="utf-8")
    )

    assert scenario["orbit"] | {
        "constellation_geometry": scenario["constellation_geometry"]
    } == {
        "type": "SSO",
        "altitude_km": 775,
        "inclination_deg": 98.6,
        "eccentricity": 0.001,
        "raan_deg": 0.0,
        "arg_perigee_deg": 0.0,
        "orbital_period_s": 6012,
        "eclipse_fraction": 0.31,
        "propagator": "j2",
        "launch_lottery": True,
        "constellation_geometry": {
            "share_plane": True,
            "in_plane_spacing_deg": 2.0,
        },
    }
    assert scenario["power"]["solar_panels"] == {
        "config": "microsat body-mounted array",
        "generation_peak_w": 120.0,
    }
    assert scenario["power"]["battery"] == {
        "count": 1,
        "capacity_wh": 300.0,
        "initial_soc": 0.8,
        "min_soc": 0.2,
        "max_soc": 1.0,
        "charge_efficiency": 0.9,
    }
    assert scenario["power"]["consumption"] == {
        "charging": {"sun_w": 9.6, "eclipse_w": 9.2},
        "payload_observe": {"sun_w": 25.4, "eclipse_w": 25.0},
        "payload_detect": {"sun_w": 30.2, "eclipse_w": 29.8},
        "communication": {"sun_w": 61.0, "eclipse_w": 60.6},
        "safe": {"sun_w": 12.0, "eclipse_w": 12.0},
        "detumbling": {"sun_w": 5.0, "eclipse_w": 4.6},
    }
    assert scenario["communications"]["xband"] == {
        "downlink_rate_kbps": 50000,
        "uplink_rate_kbps": 2000,
    }
    assert "sband" not in scenario["communications"]
    assert scenario["storage"] == {
        "obc_capacity_mb": 4096,
        "jetson_capacity_mb": 249036.8,
        "observation_size_mb": 2016,
    }
    assert scenario["modes"]["available"] == SSA_MODES
    assert scenario["ssa"]["record_size_kb"] == 10.0
    assert {
        key: scenario["targets"][key]
        for key in (
            "count",
            "parent_altitude_km",
            "parent_inclination_deg",
            "raan_spread_deg",
            "sigma_dv_along_ms",
            "sigma_dv_normal_ms",
            "size_power_law_bounds_m",
        )
    } == {
        "count": 100,
        "parent_altitude_km": 805.0,
        "parent_inclination_deg": 98.6,
        "raan_spread_deg": 0.3,
        "sigma_dv_along_ms": 13.0,
        "sigma_dv_normal_ms": 26.0,
        "size_power_law_bounds_m": [0.01, 0.10],
    }
    for removed_key in (
        "seed",
        "altitude_range_km",
        "inclination_range_deg",
        "object_size_m",
    ):
        assert removed_key not in scenario["targets"]
    assert {
        key: scenario["targets"][key]
        for key in (
            "fov_half_angle_deg",
            "boresight_pitch_deg",
            "r_cap_km",
            "m_lim",
            "sigma_m",
            "albedo",
        )
    } == {
        "fov_half_angle_deg": 1.9,
        "boresight_pitch_deg": 12.0,
        "r_cap_km": 150.0,
        "m_lim": 15.0,
        "sigma_m": 0.5,
        "albedo": 0.13,
    }
    assert "max_range_km" not in scenario["targets"]

    env = SSAEnvironment({
        "scenario_config": "configs/scenarios/ssa.yaml",
        "constellation_size": 1,
        "max_steps": 1,
    })
    sub = env._subenvs["sat_0"]
    assert sub.battery_capacity_wh == 300.0
    assert sub.solar_generation_w == 120.0
    assert sub.downlink_rate_kbps == 50000
    assert env.record_size_bytes == 10.0 * 1024.0
    env.reset(seed=1)
    sat_orbit = env._sat_orbits["sat_0"]
    assert sat_orbit.semi_major_axis_km == pytest.approx(6371.0 + 775.0)
    assert sat_orbit.inclination_deg == pytest.approx(98.6)


def test_apparent_magnitude_matches_photometric_anchors() -> None:
    phase = math.radians(45.0)

    assert apparent_magnitude(0.10, 1000.0, phase) == pytest.approx(
        14.0, abs=0.15
    )
    assert apparent_magnitude(0.01, 160.0, phase) == pytest.approx(
        15.0, abs=0.15
    )
    assert math.isinf(apparent_magnitude(0.10, 1000.0, math.pi))


def test_sun_unit_eci_is_deterministic_and_normalized() -> None:
    epoch = datetime(2026, 6, 1, tzinfo=timezone.utc)

    first = sun_unit_eci(1234.0, epoch)
    second = sun_unit_eci(1234.0, epoch)

    assert first == second
    assert math.sqrt(sum(component * component for component in first)) == pytest.approx(
        1.0
    )


def test_detection_probability_is_half_at_limiting_magnitude() -> None:
    assert detection_probability(15.0, m_lim=15.0, sigma_m=0.5) == pytest.approx(
        0.5
    )


def test_cylindrical_shadow_gate_covers_day_night_and_clear_dark_side() -> None:
    sun = (1.0, 0.0, 0.0)

    assert target_sunlit((7000.0, 0.0, 0.0), sun)
    assert not target_sunlit((-7000.0, 0.0, 0.0), sun)
    assert target_sunlit((-7000.0, 6400.0, 0.0), sun)


def test_detection_draw_is_pure_bounded_and_tuple_sensitive() -> None:
    draw = detection_draw(17, "rso_4", "sat_2", 91)

    assert 0.0 <= draw < 1.0
    assert detection_draw(17, "rso_4", "sat_2", 91) == draw
    assert detection_draw(18, "rso_4", "sat_2", 91) != draw
    assert detection_draw(17, "rso_5", "sat_2", 91) != draw
    assert detection_draw(17, "rso_4", "sat_1", 91) != draw
    assert detection_draw(17, "rso_4", "sat_2", 92) != draw


def test_vector_range_helper_uses_3d_euclidean_distance() -> None:
    assert vector_range_km([0.0, 0.0, 0.0], [3.0, 4.0, 12.0]) == pytest.approx(13.0)


def test_isl_closes_in_range_and_fails_out_of_range() -> None:
    assert is_isl_feasible([0.0, 0.0, 0.0], [1000.0, 0.0, 0.0])
    assert effective_data_rate_bps(1000.0 * 1000.0) > 0.0
    assert not is_isl_feasible([0.0, 0.0, 0.0], [5000.0, 0.0, 0.0])


def test_isl_requires_both_endpoints_idle() -> None:
    assert not is_isl_feasible(
        [0.0, 0.0, 0.0],
        [1000.0, 0.0, 0.0],
        endpoint_a_idle=False,
        endpoint_b_idle=True,
    )


def test_optical_accesses_apply_pitched_fov_range_and_shadow_gates() -> None:
    observer = (7000.0, 0.0, 0.0)
    velocity = (0.0, 1.0, 0.0)
    pitch = math.radians(12.0)
    boresight = (math.sin(pitch), math.cos(pitch), 0.0)
    wide_angle = math.radians(3.0)
    wide = (
        math.cos(wide_angle) * boresight[0] - math.sin(wide_angle) * boresight[1],
        math.sin(wide_angle) * boresight[0] + math.cos(wide_angle) * boresight[1],
        0.0,
    )
    positions = {
        "inside": tuple(o + 50.0 * b for o, b in zip(observer, boresight)),
        "too_wide": tuple(o + 50.0 * b for o, b in zip(observer, wide)),
        "too_far": tuple(o + 151.0 * b for o, b in zip(observer, boresight)),
    }
    targets = [
        RSOTarget(
            object_id=object_id,
            semi_major_axis_km=7050.0,
            eccentricity=0.0,
            inclination_deg=0.0,
            raan_deg=0.0,
            arg_perigee_deg=0.0,
            true_anomaly_deg=0.0,
            size_m=0.1,
        )
        for object_id in positions
    ]

    accesses = optical_accesses(
        observer,
        velocity,
        targets,
        positions,
        (1.0, 0.0, 0.0),
        fov_half_angle_deg=1.9,
    )

    assert [access.object_id for access in accesses] == ["inside"]
    assert accesses[0].quality == pytest.approx(15.0 - accesses[0].m)
    assert accesses[0].p_detect == pytest.approx(
        detection_probability(accesses[0].m)
    )

    shadow_observer = (-7000.0, 0.0, 0.0)
    shadow_boresight = (-math.sin(pitch), math.cos(pitch), 0.0)
    shadow_position = tuple(
        origin + 50.0 * direction
        for origin, direction in zip(shadow_observer, shadow_boresight)
    )
    assert optical_accesses(
        shadow_observer,
        velocity,
        targets[:1],
        {"inside": shadow_position},
        (1.0, 0.0, 0.0),
        fov_half_angle_deg=1.9,
    ) == []


def test_geometric_utility_ceiling_counts_visible_targets_before_final_pass() -> None:
    timeline = [
        {"step": 9, "visible_target_ids": ["rso_0"]},
        {"step": 10, "visible_target_ids": ["rso_1"]},
    ]

    ceiling = geometric_utility_ceiling(
        [{"start_step": 10, "end_step": 10}],
        timeline,
        target_count=2,
    )

    assert ceiling == pytest.approx(0.5)


def test_geometric_utility_ceiling_counts_targets_seen_during_multistep_final_pass() -> None:
    # rso_1 can be observed at step 11 and delivered by a distinct
    # communication action on the final usable pass step, 12.  Cutting off at
    # the pass start would report 0.5 even though full delivery is feasible.
    ceiling = geometric_utility_ceiling(
        [{"start_step": 10, "end_step": 12}],
        [
            {"step": 9, "visible_target_ids": ["rso_0"]},
            {"step": 11, "visible_target_ids": ["rso_1"]},
        ],
        target_count=2,
    )

    assert ceiling == pytest.approx(1.0)


def test_geometric_utility_ceiling_handles_unsorted_geometry_and_overlapping_passes() -> None:
    ceiling = geometric_utility_ceiling(
        [
            {"satellite_id": "sat_1", "start_step": 5, "end_step": 7},
            {"satellite_id": "sat_0", "start_step": 4, "end_step": 6},
        ],
        [
            {"step": 6, "visible_target_ids": ["rso_1"]},
            {"step": 7, "visible_target_ids": ["too_late"]},
            {"step": 3, "visible_target_ids": ["rso_0"]},
        ],
        target_count=2,
    )

    assert ceiling == pytest.approx(1.0)


def test_fragmentation_family_catalog_is_seeded_clipped_and_size_sane() -> None:
    kwargs = {
        "raan_center_deg": 359.9,
        "parent_altitude_km": 805.0,
        "parent_inclination_deg": 98.6,
    }
    first = generate_family_catalog(1000, 7, **kwargs)
    second = generate_family_catalog(1000, 7, **kwargs)
    third = generate_family_catalog(1000, 8, **kwargs)
    parent_a_km = 6371.0 + 805.0

    assert first == second
    assert first != third
    assert all(
        abs(target.semi_major_axis_km - parent_a_km) <= 25.0 + 1e-12
        for target in first
    )
    assert all(
        abs(target.inclination_deg - 98.6) <= 0.2 + 1e-12
        for target in first
    )
    assert all(
        abs(((target.raan_deg - 359.9 + 180.0) % 360.0) - 180.0) <= 0.3 + 1e-12
        for target in first
    )
    assert all(0.0 <= target.eccentricity <= 0.001 for target in first)
    sizes = sorted(target.size_m for target in first)
    assert all(0.01 <= size <= 0.10 for size in sizes)
    assert sizes[len(sizes) // 2] < 0.03


def test_target_two_body_propagation_returns_finite_position() -> None:
    target = generate_family_catalog(1, 3, raan_center_deg=0.0)[0]
    position = propagate_rso_position_km(target, 120.0, prefer_orekit=False)

    assert len(position) == 3
    assert all(math.isfinite(value) for value in position)
    assert 6500.0 < vector_range_km([0.0, 0.0, 0.0], position) < 7600.0


def _ssa_env_config(*, n: int = 2, settling_s: float = 0.0) -> dict:
    fixed_positions = {
        "rso_0": [0.0, 0.0, 530.0],
        "rso_1": [0.0, 1.0, 530.0],
    }
    return {
        "constellation_size": n,
        "step_duration_s": 60,
        "max_steps": 20,
        "satellite_positions_km": {
            "sat_0": [0.0, 0.0, 500.0],
            "sat_1": [100.0, 0.0, 500.0],
        },
        "scenario_params": {
            "modes": {
                "transition_overhead": {
                    "settling_time_s": settling_s,
                    "attitude_maneuver_modes": ["payload_observe", "communication"],
                }
            },
            "payload": {"compression_time_factor": 1.0, "detection_time_s": 60.0},
        },
        "targets": {
            "fixed_positions_km": fixed_positions,
            "fov_half_angle_deg": 5.0,
            "boresight_pitch_deg": 90.0,
            "r_cap_km": 52.7,
            "m_lim": 100.0,
        },
        "ground_station": {"always_visible": True},
        "reward_config": {"local_weight": 1.0, "team_weight": 0.0, "collective_weight": 1.0},
    }


def _single_target_ssa_config(*, settling_s: float = 0.0) -> dict:
    cfg = _ssa_env_config(n=1, settling_s=settling_s)
    cfg["anomaly_prob"] = 0.0
    cfg["targets"]["fixed_positions_km"] = {"rso_0": [0.0, 0.0, 530.0]}
    return cfg


def _zero_base_reward_terms(cfg: dict) -> None:
    cfg["reward_config"].update({
        "reward_scale": 0.0,
        "local_weight": 1.0,
        "team_weight": 0.0,
        "collective_weight": 1.0,
        "mission_scale": 1.0,
    })


def _force_physical_contact(
    env: SSAEnvironment, sat_id: str = "sat_0", contact_seconds: float = 60.0
) -> None:
    """Pin the real sub-environment resolution path to a bounded contact."""
    sub = env._subenvs[sat_id]
    sub._is_ground_pass_active = lambda: contact_seconds > 0.0
    sub._contact_seconds = lambda: contact_seconds


def _seed_undelivered_record(env: SSAEnvironment, sat_id: str = "sat_0") -> dict:
    record = {
        "object_id": "rso_0",
        "satellite_id": sat_id,
        "step": 0,
        "quality": 1.0,
        "relay_hops": 0,
    }
    env._undelivered_records[sat_id]["rso_0"] = record
    return record


def test_detection_draw_keeps_crn_after_different_action_histories(
    monkeypatch,
) -> None:
    left = SSAEnvironment(_ssa_env_config())
    right = SSAEnvironment(_ssa_env_config())
    left.reset(seed=53)
    right.reset(seed=53)
    calls: list[tuple[int, str, str, int, float]] = []

    def spy(seed, object_id, sat_id, step):
        draw = detection_draw(seed, object_id, sat_id, step)
        calls.append((seed, object_id, sat_id, step, draw))
        return draw

    monkeypatch.setattr("src.ssa.env.detection_draw", spy)
    left.step({
        "sat_0": {"mode": "payload_observe"},
        "sat_1": {"mode": "charging"},
    })
    right.step({
        "sat_0": {"mode": "charging"},
        "sat_1": {"mode": "payload_observe"},
    })
    common_action = {
        "sat_0": {"mode": "payload_observe"},
        "sat_1": {"mode": "charging"},
    }
    left.step(common_action)
    right.step(common_action)

    paired = [
        draw
        for seed, object_id, sat_id, step, draw in calls
        if (seed, object_id, sat_id, step) == (53, "rso_0", "sat_0", 1)
    ]
    assert len(paired) == 2
    assert paired[0] == paired[1]


def test_productive_observe_uses_strict_probability_threshold(monkeypatch) -> None:
    env = SSAEnvironment(_single_target_ssa_config())
    env.reset(seed=9)
    access = DetectionAccess(
        object_id="rso_0",
        position_km=(0.0, 0.0, 530.0),
        range_km=30.0,
        angle_deg=0.0,
        m=15.2,
        p_detect=0.4,
        quality=-0.2,
    )
    monkeypatch.setattr(env, "_optical_accesses_at", lambda *args, **kwargs: [access])
    draws = iter((0.4, 0.399999))
    monkeypatch.setattr(
        "src.ssa.env.detection_draw",
        lambda *args, **kwargs: next(draws),
    )

    env.step({"sat_0": {"mode": "payload_observe"}})
    assert env.detection_matrix == [[0]]

    env.step({"sat_0": {"mode": "payload_observe"}})
    assert env.detection_matrix == [[1]]
    assert env.onboard_estimates["sat_0"]["rso_0"]["m"] == pytest.approx(15.2)


def test_observation_removes_oracle_and_cues_only_own_knowledge() -> None:
    cfg = _ssa_env_config()
    cfg["anomaly_prob"] = 0.0
    cfg["satellite_positions_km"]["sat_1"] = [0.0, 0.0, 500.0]
    env = SSAEnvironment(cfg)
    initial = env.reset(seed=5)

    for satellite in initial.constellation_state.satellites.values():
        assert "visible_rso_ids" not in satellite.metadata
        assert "visible_rso_count" not in satellite.metadata
        assert satellite.metadata["ssa_predicted_in_fov"] == []
        assert satellite.metadata["ssa_known_object_ages"] == {}
    assert all(task.get("type") != "observe_rso" for task in initial.tasks)

    env.onboard_estimates["sat_0"]["rso_0"] = {
        "object_id": "rso_0",
        "time_step": 0,
        "quality": 1.0,
    }
    env.onboard_estimates["sat_0"]["rso_0"]["last_refresh_step"] = 0
    assert {
        access.object_id
        for access in env._optical_accesses_at("sat_1", 0.0)
    } >= {"rso_0"}

    env.current_step = 3
    observation = env.get_observation()
    sat_0 = observation.constellation_state.satellites["sat_0"].metadata
    sat_1 = observation.constellation_state.satellites["sat_1"].metadata

    assert sat_0["ssa_predicted_in_fov"] == ["rso_0"]
    assert sat_0["ssa_known_object_ages"] == {"rso_0": 3}
    assert sat_1["ssa_predicted_in_fov"] == []
    assert sat_1["ssa_known_object_ages"] == {}


def test_support_cut_kept_set_is_paired_and_stable_across_resets() -> None:
    config = {
        "scenario_config": "configs/scenarios/ssa.yaml",
        "constellation_size": 2,
        "max_steps": 12,
        "targets": {
            "count": 30,
            "fov_half_angle_deg": 180.0,
            "r_cap_km": 20000.0,
        },
    }
    left = SSAEnvironment(config)
    right = SSAEnvironment(config)

    left.reset(seed=71)
    right.reset(seed=71)
    first_ids = list(left.target_ids)
    first_targets = list(left.targets)
    first_cut = left.ssa_support_cut_count

    assert first_ids
    assert right.target_ids == first_ids
    assert right.targets == first_targets
    assert right.ssa_support_cut_count == first_cut

    left.reset(seed=71)
    assert left.target_ids == first_ids
    assert left.targets == first_targets
    assert left.ssa_support_cut_count == first_cut
    assert left.catalog_count == 30


def test_visibility_timeline_is_computed_once_per_reset_and_reused(monkeypatch) -> None:
    env = SSAEnvironment(_ssa_env_config(n=1))
    original = env._compute_visibility_timeline
    calls = 0

    def counted_timeline():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(env, "_compute_visibility_timeline", counted_timeline)
    env.reset(seed=3)
    assert calls == 1

    env._compute_physical_utility_ceiling()
    assert calls == 1

    env.reset(seed=3)
    assert calls == 2


def test_ssa_support_cut_drops_never_visible_object_and_uses_kept_denominator() -> None:
    cfg = _ssa_env_config()
    cfg["targets"]["fixed_positions_km"] = {
        "rso_0": [0.0, 0.0, 530.0],
        "rso_1": [200.0, 0.0, 530.0],
    }
    env = SSAEnvironment(cfg)

    env.reset(seed=1)

    assert env.target_ids == ["rso_0"]
    assert env.target_count == 1
    assert env.ssa_catalog_size == 1
    assert env.ssa_support_cut_count == 1
    assert env.physical_utility_ceiling == pytest.approx(1.0)
    assert env.get_metrics()["ssa_catalog_size"] == 1.0
    assert env.get_metrics()["ssa_support_cut_count"] == 1.0

    result = env.step({
        "sat_0": {"mode": "payload_observe"},
        "sat_1": {"mode": "charging"},
    })
    assert result.info["ssa_onboard_coverage"] == pytest.approx(1.0)


def test_ssa_observe_updates_fixed_binary_detection_matrix() -> None:
    env = SSAEnvironment(_ssa_env_config())
    env.reset(seed=1)

    result = env.step({"sat_0": {"mode": "payload_observe"}, "sat_1": {"mode": "charging"}})

    assert env.detection_matrix == [[1, 1], [0, 0]]
    assert result.info["ssa_onboard_coverage"] == pytest.approx(1.0)
    assert result.info["ssa_delivered_coverage"] == 0.0
    row = result.observation.constellation_state.satellites["sat_0"].metadata["ssa_detection_row"]
    assert row == [1, 1]


@pytest.mark.parametrize(
    ("free_space_delta_mb", "accepted"),
    [
        (0.0, True),       # Exact fit.
        (-1e-6, False),    # One epsilon beyond capacity.
        (-9.0, False),     # Near-full buffer with almost no free space.
    ],
)
def test_ssa_observation_requires_atomic_base_admission(
    free_space_delta_mb: float, accepted: bool
) -> None:
    cfg = _single_target_ssa_config()
    _zero_base_reward_terms(cfg)
    env = SSAEnvironment(cfg)
    env.reset(seed=1)
    sub = env._subenvs["sat_0"]
    free_space_mb = sub.observation_size_mb + free_space_delta_mb
    sub.jetson_raw_mb = sub.jetson_capacity_mb - free_space_mb

    result = env.step({"sat_0": {"mode": "payload_observe"}})
    sat_info = result.info["per_satellite"]["sat_0"]

    assert sat_info["observation_accepted"] is accepted
    assert bool(sat_info["storage_overflow"]) is (not accepted)
    # Base terms are zero, so rejection cannot manufacture a positive reward;
    # only the single undelivered-coverage term remains.
    assert result.rewards["sat_0"] == -1.0
    if accepted:
        assert sub.uncompressed_observations == 1
        assert sub.total_observation_s == pytest.approx(60.0)
        assert env.detection_matrix == [[1]]
        assert set(env.onboard_estimates["sat_0"]) == {"rso_0"}
        assert set(env._undelivered_records["sat_0"]) == {"rso_0"}
        assert env.successful_observations == 1
        assert env.total_observation_records == 1
        assert result.info["ssa_onboard_coverage"] == 1.0
    else:
        assert sub.uncompressed_observations == 0
        assert sub.total_observation_s == 0.0
        assert env.detection_matrix == [[0]]
        assert env.onboard_estimates["sat_0"] == {}
        assert env._undelivered_records["sat_0"] == {}
        assert env.successful_observations == 0
        assert env.total_observation_records == 0
        assert result.info["ssa_last_step_detections"] == {"sat_0": []}
        assert result.info["ssa_onboard_coverage"] == 0.0
        assert result.info["ssa_delivered_coverage"] == 0.0


def test_ssa_executed_detection_uses_same_zero_based_timestep_as_ceiling() -> None:
    cfg = _ssa_env_config(n=1)
    cfg["targets"]["fixed_positions_km"] = {"rso_0": [0.0, 0.0, 530.0]}
    env = SSAEnvironment(cfg)

    # A moving target is visible only at the first action epoch. Before the
    # alignment fix the ceiling saw t=0 while step() evaluated detection at
    # t=60 s, so delivered utility could exceed the reported ceiling.
    env._target_positions_at = lambda epoch_s: {
        "rso_0": (
            (0.0, 0.0, 530.0)
            if epoch_s == 0.0
            else (1000.0, 0.0, 530.0)
        )
    }
    env._satellite_position = lambda sat_id, epoch_s=None: (0.0, 0.0, 500.0)
    env.reset(seed=1)

    result = env.step({"sat_0": {"mode": "payload_observe"}})

    assert env.physical_utility_ceiling == pytest.approx(1.0)
    assert env.detection_matrix == [[1]]
    assert result.info["ssa_onboard_coverage"] == pytest.approx(1.0)


def test_one_hot_binary_action_selects_payload_observe_mode() -> None:
    env = SSAEnvironment(_ssa_env_config())
    env.reset(seed=1)
    one_hot = [0] * len(SSA_MODES)
    one_hot[SSA_MODES.index("payload_observe")] = 1

    env.step({"sat_0": one_hot, "sat_1": {"mode": "charging"}})

    assert env.detection_matrix[0] == [1, 1]


def test_adcs_settling_blocks_observation_on_observe_entry() -> None:
    env = SSAEnvironment(_ssa_env_config(settling_s=135.0))
    env.reset(seed=1)

    result = env.step({"sat_0": {"mode": "payload_observe"}, "sat_1": {"mode": "charging"}})

    assert env.detection_matrix == [[0, 0], [0, 0]]
    assert result.info["per_satellite"]["sat_0"]["in_transition"] is True


def test_onboard_keeps_best_estimate_and_ground_archives_best_track() -> None:
    env = SSAEnvironment(_ssa_env_config())
    env.reset(seed=1)

    env.step({"sat_0": {"mode": "payload_observe"}, "sat_1": {"mode": "charging"}})
    env.step({"sat_0": {"mode": "payload_observe"}, "sat_1": {"mode": "charging"}})
    _force_physical_contact(env)
    downlink = env.step({"sat_0": {"mode": "communication"}, "sat_1": {"mode": "charging"}})

    assert set(env.onboard_estimates["sat_0"]) == {"rso_0", "rso_1"}
    assert len(env.onboard_estimates["sat_0"]) == 2
    # The undelivered buffer holds one best track per object (custody
    # semantics), so each object delivers exactly one record here.
    assert len(env.ground_archive["rso_0"]) == 1
    assert len(env.ground_archive["rso_1"]) == 1
    assert env._undelivered_records["sat_0"] == {}
    assert downlink.info["per_satellite"]["sat_0"]["resolved_mode"] == "communication"
    assert downlink.info["per_satellite"]["sat_0"]["contact_seconds"] == 60.0


def test_isl_merge_ors_matrix_and_keeps_higher_quality_estimate() -> None:
    env = SSAEnvironment(_ssa_env_config())
    env.reset(seed=1)
    env.step({"sat_0": {"mode": "payload_observe"}, "sat_1": {"mode": "charging"}})

    env.step({"sat_0": {"mode": "isl_share"}, "sat_1": {"mode": "charging"}})

    assert env.detection_matrix == [[1, 1], [1, 1]]
    assert set(env.onboard_estimates["sat_1"]) == {"rso_0", "rso_1"}
    assert env.get_metrics()["isl_connectivity"] > 0.0


def test_isl_power_is_in_observation_info_and_resource_efficiency() -> None:
    cfg = _ssa_env_config(n=1)
    cfg["anomaly_prob"] = 0.0
    cfg["scenario_params"]["power"] = {
        "solar_panels": {"generation_peak_w": 0.0},
        "battery": {"capacity_wh": 100.0, "initial_soc": 0.8},
        "consumption": {
            "charging": {"sun_w": 6.0, "eclipse_w": 6.0},
        },
    }
    cfg["isl"] = {"power_overhead_w": 5.0}
    env = SSAEnvironment(cfg)
    before_soc = env.reset(seed=1).constellation_state.satellites["sat_0"].resources[
        "battery_soc"
    ]

    result = env.step({"sat_0": {"mode": "isl_share"}})

    expected_base_wh = 6.0 * 60.0 / 3600.0
    expected_isl_wh = 5.0 * 60.0 / 3600.0
    expected_total_wh = expected_base_wh + expected_isl_wh
    observed_soc = result.observation.constellation_state.satellites["sat_0"].resources[
        "battery_soc"
    ]
    assert result.info["isl_energy_consumed_wh"] == pytest.approx(expected_isl_wh)
    assert result.info["gross_energy_consumed_wh"] == pytest.approx(expected_total_wh)
    assert result.info["solar_generation_wh"] == pytest.approx(0.0)
    assert result.info["net_battery_depletion_wh"] == pytest.approx(expected_total_wh)
    assert result.info["per_satellite"]["sat_0"]["battery_soc"] == pytest.approx(
        observed_soc
    )
    assert (before_soc - observed_soc) * 100.0 == pytest.approx(expected_total_wh)
    assert result.info["battery_soc_delta_sum"] * 100.0 == pytest.approx(
        expected_total_wh
    )

    collector = SSAMetricsCollector({
        "max_steps": 1,
        "step_duration_s": 60.0,
        "constellation_size": 1,
        "battery_capacity_wh": 100.0,
    })
    metric_info = {
        **result.info,
        "ssa_delivered_coverage": 1.0,
        "physical_utility_ceiling": 1.0,
    }
    collector.record_step(
        timestep=0,
        wall_clock_seconds=0.0,
        env_state=result.observation,
        actions={"sat_0": {"mode": "isl_share"}},
        rewards={"sat_0": 0.0},
        info=metric_info,
        decision_metrics={"inference_allowed": True},
    )
    episode = collector.finalise_episode(0)
    assert episode.aggregated["total_energy_consumed_wh"] == pytest.approx(
        expected_total_wh
    )
    assert episode.aggregated["net_battery_depletion_wh"] == pytest.approx(
        expected_total_wh
    )
    assert episode.aggregated["resource_efficiency"] == pytest.approx(
        1.0 / expected_total_wh
    )


def test_isl_energy_reports_gross_load_separately_from_net_soc_in_sunlight() -> None:
    cfg = _ssa_env_config(n=1)
    cfg["anomaly_prob"] = 0.0
    cfg["scenario_params"]["power"] = {
        "solar_panels": {"generation_peak_w": 10.0},
        "battery": {
            "capacity_wh": 100.0,
            "initial_soc": 0.8,
            "charge_efficiency": 0.9,
        },
        "consumption": {
            "charging": {"sun_w": 6.0, "eclipse_w": 6.0},
        },
    }
    cfg["isl"] = {"power_overhead_w": 5.0}
    env = SSAEnvironment(cfg)
    env.reset(seed=1)
    for sub in env._subenvs.values():
        sub._orbital_ctx.is_in_sunlight = lambda step: True

    result = env.step({"sat_0": {"mode": "isl_share"}})

    base_charge_wh = (10.0 - 6.0) * 60.0 / 3600.0 * 0.9
    base_load_wh = 6.0 * 60.0 / 3600.0
    solar_generation_wh = 10.0 * 60.0 / 3600.0
    isl_wh = 5.0 * 60.0 / 3600.0
    expected_net_drop_wh = isl_wh - base_charge_wh
    assert result.info["isl_energy_consumed_wh"] == pytest.approx(isl_wh)
    assert result.info["gross_energy_consumed_wh"] == pytest.approx(
        base_load_wh + isl_wh
    )
    assert result.info["solar_generation_wh"] == pytest.approx(solar_generation_wh)
    assert result.info["net_battery_depletion_wh"] == pytest.approx(
        expected_net_drop_wh
    )
    assert result.info["battery_soc_delta_sum"] * 100.0 == pytest.approx(
        expected_net_drop_wh
    )


def test_isl_power_debits_the_selected_satellites_battery_capacity() -> None:
    env = SSAEnvironment(_ssa_env_config(n=1))
    env.reset(seed=1)
    sub = env._subenvs["sat_0"]
    sub.battery_capacity_wh = 200.0
    env.battery_capacity_wh = 100.0  # Deliberately different prototype value.
    before_soc = sub.battery_soc

    consumed_wh = env._bill_isl_power("sat_0")

    expected_wh = env.isl_power_overhead_w * env.step_duration_s / 3600.0
    assert consumed_wh == pytest.approx(expected_wh)
    assert (before_soc - sub.battery_soc) * 200.0 == pytest.approx(expected_wh)


def test_delivered_utility_credits_only_downlinked_objects() -> None:
    env = SSAEnvironment(_ssa_env_config())
    env.reset(seed=1)

    observe = env.step({"sat_0": {"mode": "payload_observe"}, "sat_1": {"mode": "charging"}})
    _force_physical_contact(env)
    downlink = env.step({"sat_0": {"mode": "communication"}, "sat_1": {"mode": "charging"}})

    assert observe.info["ssa_delivered_coverage"] == 0.0
    assert downlink.info["ssa_delivered_coverage"] == pytest.approx(1.0)
    assert downlink.rewards["sat_0"] > observe.rewards["sat_0"]


def test_ssa_failed_communication_during_adcs_transition_keeps_records() -> None:
    cfg = _single_target_ssa_config(settling_s=135.0)
    _zero_base_reward_terms(cfg)
    env = SSAEnvironment(cfg)
    env.reset(seed=1)
    record = _seed_undelivered_record(env)
    _force_physical_contact(env)

    result = env.step({"sat_0": {"mode": "communication"}})

    sat_info = result.info["per_satellite"]["sat_0"]
    assert sat_info["requested_mode"] == "communication"
    assert sat_info["resolved_mode"] == "charging"
    assert sat_info["in_transition"] is True
    assert result.info["ssa_step_downlinked_records"] == 0.0
    assert env._undelivered_records["sat_0"] == {"rso_0": record}
    assert env.ground_archive["rso_0"] == []
    assert result.info["ssa_delivered_coverage"] == 0.0


def test_ssa_failed_communication_in_safe_mode_keeps_records() -> None:
    cfg = _single_target_ssa_config()
    _zero_base_reward_terms(cfg)
    env = SSAEnvironment(cfg)
    env.reset(seed=1)
    record = _seed_undelivered_record(env)
    _force_physical_contact(env)
    env._subenvs["sat_0"].battery_soc = 0.0

    result = env.step({"sat_0": {"mode": "communication"}})

    sat_info = result.info["per_satellite"]["sat_0"]
    assert sat_info["requested_mode"] == "communication"
    assert sat_info["resolved_mode"] == "safe"
    assert result.info["ssa_step_downlinked_records"] == 0.0
    assert env._undelivered_records["sat_0"] == {"rso_0": record}
    assert env.ground_archive["rso_0"] == []
    assert result.info["ssa_delivered_coverage"] == 0.0


def test_ssa_clean_resolved_communication_delivers_records() -> None:
    cfg = _single_target_ssa_config()
    _zero_base_reward_terms(cfg)
    env = SSAEnvironment(cfg)
    env.reset(seed=1)
    _seed_undelivered_record(env)
    _force_physical_contact(env)

    result = env.step({"sat_0": {"mode": "communication"}})

    sat_info = result.info["per_satellite"]["sat_0"]
    assert sat_info["resolved_mode"] == "communication"
    assert sat_info["in_transition"] is False
    assert sat_info["physical_ground_pass_active"] == 1.0
    assert sat_info["contact_seconds"] == 60.0
    assert result.info["ssa_step_downlinked_records"] == 1.0
    assert env._undelivered_records["sat_0"] == {}
    assert len(env.ground_archive["rso_0"]) == 1
    assert result.info["ssa_delivered_coverage"] == 1.0
    # Base terms are zero and full delivered coverage closes the sole SSA gap.
    assert result.rewards["sat_0"] == 0.0


def test_ssa_reward_zero_coverage_is_applied_once_through_real_step() -> None:
    cfg = _single_target_ssa_config()
    _zero_base_reward_terms(cfg)
    env = SSAEnvironment(cfg)
    env.reset(seed=1)

    result = env.step({"sat_0": {"mode": "charging"}})

    assert result.info["ssa_delivered_coverage"] == 0.0
    assert result.rewards["sat_0"] == -1.0


def test_ssa_local_team_blend_is_applied_once_through_real_step() -> None:
    cfg = _ssa_env_config(n=2)
    cfg["anomaly_prob"] = 0.0
    cfg["reward_config"].update({
        "reward_scale": 1.0,
        "standby_penalty": 2.0,
        "mission_observation_weight": 0.0,
        "mission_downlink_weight": 0.0,
        "local_weight": 0.25,
        "team_weight": 0.5,
        "team_reducer": "mean",
        "collective_weight": 1.0,
        "mission_scale": 1.0,
    })
    env = SSAEnvironment(cfg)
    env.reset(seed=1)

    result = env.step({
        "sat_0": {"mode": "charging"},
        "sat_1": {"mode": "charging"},
    })

    # Each raw sub-environment reward is -2.  One local/team blend gives
    # .25*(-2) + .5*mean(-2,-2) = -1.5, then one SSA gap gives -2.5.
    assert result.rewards == {"sat_0": -2.5, "sat_1": -2.5}


def test_ssa_metrics_adds_coverage_duplicate_connectivity_and_m10() -> None:
    collector = SSAMetricsCollector({
        "max_steps": 2,
        "step_duration_s": 60,
        "constellation_size": 2,
        "baseline_utility_n1": 0.5,
        "battery_capacity_wh": 100.0,
    })
    for step, delivered in enumerate((0.0, 1.0)):
        collector.record_step(
            timestep=step,
            wall_clock_seconds=0.01,
            env_state=None,
            actions={},
            rewards={"sat_0": 1.0},
            info={
                "battery_soc": 0.8,
                "prev_battery_soc": 0.81,
                "battery_soc_delta_sum": 0.02,
                "physical_utility_ceiling": 1.0,
                "ssa_onboard_coverage": 1.0,
                "ssa_delivered_coverage": delivered,
                "duplicate_observation_rate": 0.25,
                "mean_revisit_steps": 3.0,
                "isl_connectivity": 0.5,
                "ssa_delivered_objects": delivered * 2,
                "ssa_known_objects": 2,
            },
            decision_metrics={"inference_allowed": True, "has_rationale": True},
        )

    episode = collector.finalise_episode(0)

    assert episode.aggregated["utility"] == 1.0
    assert episode.aggregated["ssa_delivered_coverage"] == 1.0
    assert episode.aggregated["duplicate_observation_rate"] == 0.25
    assert episode.aggregated["eta_scale"] == pytest.approx(1.0)
    assert episode.aggregated["eta_scale_vs_1overN"] == pytest.approx(1.0)
    assert episode.aggregated["total_energy_consumed_wh"] == pytest.approx(4.0)
    assert episode.aggregated["resource_efficiency"] == pytest.approx(
        episode.aggregated["utility"] / episode.aggregated["total_energy_consumed_wh"]
    )
    assert episode.aggregated["mission_goal_utility"] == pytest.approx(1.0)
    assert episode.aggregated["physical_utility_ceiling"] == pytest.approx(1.0)
    assert episode.aggregated["utility_fraction_of_physical_ceiling"] == pytest.approx(1.0)


def test_ssa_metrics_exposes_utility_above_inconsistent_ceiling() -> None:
    collector = SSAMetricsCollector({"max_steps": 1, "step_duration_s": 60.0})
    collector.record_step(
        timestep=0,
        wall_clock_seconds=0.0,
        env_state=None,
        actions={},
        rewards={},
        info={
            "battery_soc_delta_sum": 0.0,
            "ssa_delivered_coverage": 1.0,
            "physical_utility_ceiling": 0.5,
        },
        decision_metrics={"inference_allowed": True},
    )

    episode = collector.finalise_episode(0)

    assert episode.aggregated["utility_fraction_of_physical_ceiling"] == pytest.approx(2.0)


def test_rule_based_ssa_drains_pipeline_before_observation_backpressure() -> None:
    representation = RuleBasedSSA({})
    base_state = {
        "battery_soc": 0.9,
        "health_status": "nominal",
        "ground_pass_active": False,
        "storage_used_fraction": 0.5,
        "obc_data_mb": 0.0,
        "jetson_compressed_mb": 0.0,
        "uncompressed_observations": 0,
        "undetected_observations": 0,
        "achievable_downlink_mb": 1.0,
        "undelivered_records": 0,
        "visible_new_rso_ids": [],
        "known_objects": [],
    }
    saturated = {
        **base_state,
        "jetson_compressed_mb": 10.0,
    }
    storage_pressure_outside_pass = {
        **base_state,
        "storage_used_fraction": 0.8,
        "obc_data_mb": 10.0,
    }
    relay_not_blocked_by_observation_backpressure = {
        **base_state,
        "obc_data_mb": 10.0,
        "undelivered_records": 1,
    }

    saturated_mode = representation._mode_for_satellite(
        "sat_0", saturated, set(), coordinated=False
    )[0]
    outside_pass_mode = representation._mode_for_satellite(
        "sat_0", storage_pressure_outside_pass, set(), coordinated=False
    )[0]
    relay_mode = representation._mode_for_satellite(
        "sat_0",
        relay_not_blocked_by_observation_backpressure,
        set(),
        coordinated=True,
    )[0]

    assert saturated_mode == "payload_send"
    assert outside_pass_mode == "charging"
    assert relay_mode == "isl_share"


def test_rule_based_ssa_backpressure_projects_next_product() -> None:
    representation = RuleBasedSSA({})
    state = {
        "battery_soc": 0.9,
        "health_status": "nominal",
        "ground_pass_active": False,
        "storage_used_fraction": 0.1,
        "obc_data_mb": 0.9,
        "jetson_compressed_mb": 0.0,
        "uncompressed_observations": 0,
        "undetected_observations": 0,
        "achievable_downlink_mb": 1.0,
        "undelivered_records": 0,
        "visible_new_rso_ids": ["rso_0"],
        "known_objects": [],
    }
    blocked = representation._mode_for_satellite(
        "sat_0", state, set(), coordinated=False
    )[0]
    state["achievable_downlink_mb"] = 0.9 + 9.41 / 5.11
    exact_fit = representation._mode_for_satellite(
        "sat_0", state, set(), coordinated=False
    )[0]

    assert blocked == "charging"
    assert exact_fit == "payload_observe"


def test_rule_based_ssa_safely_handles_removed_visibility_oracle() -> None:
    env = SSAEnvironment({
        **_ssa_env_config(n=2),
        "satellite_positions_km": {
            "sat_0": [0.0, 0.0, 500.0],
            "sat_1": [0.4, 0.0, 500.0],
        },
        "targets": {
            "fixed_positions_km": {"rso_0": [0.0, 0.0, 530.0]},
            "fov_half_angle_deg": 5.0,
            "r_cap_km": 52.7,
            "boresight_pitch_deg": 90.0,
            "m_lim": 100.0,
        },
    })
    observation = env.reset(seed=1)
    central = RuleBasedSSA({})

    central_action = central.select_action(type("Context", (), {"state": central.encode_observation(observation)}))

    assert central_action["sat_0"]["mode"] == "charging"
    assert central_action["sat_1"]["mode"] == "charging"

    step_result = env.step(central_action)
    local = RuleBasedSSA({"satellite_id": "sat_1"})
    scoped = scope_observation(step_result.observation, ["sat_1"])
    local_state = local.encode_observation(scoped)

    assert local.select_action(
        type("Context", (), {"state": local_state})
    )["sat_1"]["mode"] == "charging"


def test_ssa_symbolic_runner_tolerates_removed_oracle_before_policy_rewrite(tmp_path) -> None:
    # The committed ssa.yaml is real geometry (an 8-step run sees no RSO), so
    # pin the toy fixture through the env-config override path for this test.
    toy_blocks = {
        "targets": {
            "fixed_positions_km": {
                "rso_0": [0.0, 0.0, 530.0],
                "rso_1": [0.0, 1.0, 530.0],
            },
            "fov_half_angle_deg": 5.0,
            "r_cap_km": 52.7,
            "boresight_pitch_deg": 90.0,
            "m_lim": 100.0,
        },
        "satellite_positions_km": {
            "sat_0": [0.0, 0.0, 500.0],
            "sat_1": [0.4, 0.0, 500.0],
            "sat_2": [0.0, 0.4, 500.0],
        },
        "ground_station": {"always_visible": True},
    }
    sas_cfg = apply_overrides(
        load_config("configs/experiments/ssa_sas_ao_symb_n3.yaml"),
        episodes=1,
        steps=8,
        output_dir=str(tmp_path / "sas"),
    )
    imas_cfg = apply_overrides(
        load_config("configs/experiments/ssa_imas_ao_symb_n3.yaml"),
        episodes=1,
        steps=8,
        output_dir=str(tmp_path / "imas"),
    )
    sas_cfg.environment.scenario_config.update(toy_blocks)
    imas_cfg.environment.scenario_config.update(toy_blocks)

    sas = ExperimentRunner(config=sas_cfg).run()
    imas = ExperimentRunner(config=imas_cfg).run()

    sas_dupes = sas["experiment_statistics"].mean["duplicate_observation_rate"]
    imas_dupes = imas["experiment_statistics"].mean["duplicate_observation_rate"]
    # WP-B intentionally removes the old true-visibility task. The policy is
    # rewritten against knowledge-derived cues in WP-F; until then both
    # organizations must remain safe and make no oracle-driven detections.
    assert sas_dupes == 0.0
    assert imas_dupes == 0.0
    assert sas["experiment_statistics"].mean["ssa_onboard_coverage"] == 0.0
    assert imas["experiment_statistics"].mean["ssa_onboard_coverage"] == 0.0


@pytest.mark.parametrize("organization", ["sas", "independent_mas"])
def test_ssa_rejects_ground_paradigms_for_every_organization(
    organization: str,
) -> None:
    with pytest.raises(ValueError, match="SSA is AO-only"):
        ExperimentConfig(
            experiment_id="ssa_invalid_ground_imas",
            agent_organization=organization,
            decision_procedure="sda",
            representation="symbolic",
            behaviour="hand_designed",
            operations_paradigm="autonomous_ground",
            behaviour_config={"mode": "hand_designed"},
            environment={
                "constellation_size": 3,
                "timestep_seconds": 60,
                "max_steps": 10,
                "scenario": "ssa",
                "scenario_config": {"scenario_file": "configs/scenarios/ssa.yaml"},
            },
            num_episodes=1,
            max_steps=10,
        )


# -----------------------------------------------------------------
# SSA subsymbolic RL representation (mock; PPO checkpoints owner-gated)
# -----------------------------------------------------------------

def test_subsymbolic_ssa_mock_safely_handles_removed_oracle_and_can_relay() -> None:
    from src.ssa.rl import SubsymbolicSSA
    from src.ssa.rl_features import SSA_OBS_DIM

    env = SSAEnvironment(_ssa_env_config())
    observation = env.reset(seed=1)
    rep = SubsymbolicSSA({"rl_mock": True, "target_count": 2})

    state = rep.encode_observation(observation)
    assert set(state["_obs_vectors"]) == {"sat_0", "sat_1"}
    assert state["_obs_vectors"]["sat_0"].shape == (SSA_OBS_DIM,)

    actions = rep.select_action(type("Context", (), {"state": state}))
    # The old mock policy receives no true-visibility oracle. It remains safe
    # until WP-F rewrites it against the legal cue fields.
    assert actions["sat_0"]["mode"] == "charging"
    assert actions["sat_1"]["mode"] == "charging"
    assert rep.get_rationale()

    # After observing and leaving the pass, undelivered records with high SoC
    # should propose isl_share under the coordinated grounding rule.
    env.step(actions)
    followup = env.step({"sat_0": {"mode": "charging"}, "sat_1": {"mode": "charging"}})
    state2 = rep.encode_observation(followup.observation)
    sat0 = dict(state2["satellites"]["sat_0"])
    sat0["ground_pass_active"] = False
    sat0["visible_new_rso_ids"] = []
    sat0["undelivered_records"] = 1
    grounded = rep._ground_mode("isl_share", sat0, coordinated=True)
    assert grounded == "isl_share"


def test_ssa_rl_mock_runner_smoke_without_visibility_oracle(tmp_path) -> None:
    cfg = apply_overrides(
        load_config("configs/experiments/ssa_sas_ao_rl_n3.yaml"),
        episodes=1,
        steps=8,
        output_dir=str(tmp_path / "rl"),
    )
    cfg.environment.scenario_config.update({
        "targets": {
            "fixed_positions_km": {
                "rso_0": [0.0, 0.0, 530.0],
                "rso_1": [0.0, 1.0, 530.0],
            },
            "fov_half_angle_deg": 5.0,
            "r_cap_km": 52.7,
            "boresight_pitch_deg": 90.0,
            "m_lim": 100.0,
        },
        "satellite_positions_km": {
            "sat_0": [0.0, 0.0, 500.0],
            "sat_1": [0.4, 0.0, 500.0],
            "sat_2": [0.0, 0.4, 500.0],
        },
        "ground_station": {"always_visible": True},
    })
    result = ExperimentRunner(config=cfg).run()
    mean = result["experiment_statistics"].mean
    # The WP-F policy rewrite will consume knowledge-derived cues. Until then,
    # the legacy mock completes safely without oracle-driven observations.
    assert mean["ssa_onboard_coverage"] == 0.0
    assert "utility" in mean
