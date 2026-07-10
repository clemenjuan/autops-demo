"""Independent boundary tests for the canonical EventSat pipeline contract."""
from __future__ import annotations

import pytest

from src.eventsat.transitions import (
    PipelineParameters,
    apply_can_transfer,
    apply_compress,
    apply_detect,
    apply_downlink,
    apply_observe,
)
from src.eventsat.env import EventSatEnvironment
from src.orbital.context import OrbitalContext
from src.orbital.ground_access import GroundPass


def _params(**overrides: float) -> PipelineParameters:
    values = {
        "observation_size_mb": 9.41,
        "compression_ratio": 5.11,
        "jetson_capacity_mb": 100.0,
        "obc_capacity_mb": 1.0,
        "detection_metadata_mb": 0.01,
        "jetson_to_obc_rate_kbps": 8000.0,
        "downlink_rate_kbps": 50.0,
        "step_duration_s": 60.0,
    }
    values.update(overrides)
    return PipelineParameters(**values)


@pytest.mark.parametrize("epsilon,accepted", [(0.0, True), (1e-9, False)])
def test_observation_admission_is_atomic_at_shared_jetson_boundary(
    epsilon: float, accepted: bool
) -> None:
    params = _params()
    state = {
        "jetson_raw_mb": 80.0,
        "jetson_compressed_mb": 100.0 - 80.0 - params.observation_size_mb + epsilon,
        "uncompressed_observations": 2.0,
        "total_raw_captured_mb": 20.0,
        "total_observation_s": 120.0,
    }

    outcome = apply_observe(state, params)

    assert outcome.accepted is accepted
    if accepted:
        assert outcome.state["jetson_raw_mb"] == pytest.approx(89.41)
        assert outcome.state["uncompressed_observations"] == pytest.approx(3.0)
        assert outcome.state["total_raw_captured_mb"] == pytest.approx(29.41)
        assert outcome.state["total_observation_s"] == pytest.approx(180.0)
    else:
        assert outcome.state == state


def test_compression_completion_moves_one_product_without_creating_data() -> None:
    params = _params()
    state = {
        "jetson_raw_mb": 9.41,
        "jetson_compressed_mb": 1.0,
        "uncompressed_observations": 1.0,
        "undetected_observations": 0.0,
    }

    outcome = apply_compress(state, params)

    assert outcome.accepted is True
    assert outcome.state["jetson_raw_mb"] == pytest.approx(0.0)
    assert outcome.state["jetson_compressed_mb"] == pytest.approx(
        1.0 + 9.41 / 5.11
    )
    assert outcome.state["uncompressed_observations"] == pytest.approx(0.0)
    assert outcome.state["undetected_observations"] == pytest.approx(1.0)


@pytest.mark.parametrize("epsilon,accepted", [(0.0, True), (1e-9, False)])
def test_compression_admission_uses_projected_shared_capacity(
    epsilon: float, accepted: bool
) -> None:
    params = _params(jetson_capacity_mb=100.0)
    state = {
        "jetson_raw_mb": params.observation_size_mb,
        "jetson_compressed_mb": (
            params.jetson_capacity_mb
            - params.compressed_observation_mb
            + epsilon
        ),
        "uncompressed_observations": 1.0,
        "undetected_observations": 0.0,
    }

    outcome = apply_compress(state, params)

    assert outcome.accepted is accepted
    if accepted:
        occupancy = (
            outcome.state["jetson_raw_mb"]
            + outcome.state["jetson_compressed_mb"]
        )
        assert occupancy == pytest.approx(params.jetson_capacity_mb)
    else:
        assert outcome.state == state


@pytest.mark.parametrize("epsilon,accepted", [(0.0, True), (1e-9, False)])
def test_detection_metadata_is_atomic_at_obc_boundary(
    epsilon: float, accepted: bool
) -> None:
    params = _params()
    state = {
        "obc_data_mb": params.obc_capacity_mb - params.detection_metadata_mb + epsilon,
        "undetected_observations": 1.0,
        "total_detections": 4.0,
    }

    outcome = apply_detect(state, params)

    assert outcome.accepted is accepted
    if accepted:
        assert outcome.state["obc_data_mb"] == pytest.approx(1.0)
        assert outcome.state["undetected_observations"] == pytest.approx(0.0)
        assert outcome.state["total_detections"] == pytest.approx(5.0)
    else:
        assert outcome.state == state


@pytest.mark.parametrize(
    ("source", "headroom", "duration_s", "expected"),
    [
        (0.25, 1.0, 60.0, 0.25),
        (10.0, 0.2, 60.0, 0.2),
        (10.0, 1.0, 0.0004, 0.0004),
    ],
)
def test_can_transfer_is_bounded_by_source_headroom_and_rate(
    source: float, headroom: float, duration_s: float, expected: float
) -> None:
    params = _params(obc_capacity_mb=1.0)
    state = {
        "jetson_compressed_mb": source,
        "obc_data_mb": 1.0 - headroom,
        "obc_raw_equivalent_mb": 0.0,
    }

    outcome = apply_can_transfer(state, params, duration_s=duration_s)

    assert outcome.transferred_mb == pytest.approx(expected)
    assert outcome.state["jetson_compressed_mb"] == pytest.approx(source - expected)
    assert outcome.state["obc_data_mb"] == pytest.approx(1.0 - headroom + expected)


@pytest.mark.parametrize("epsilon", [0.0, 1e-9])
def test_can_transfer_exact_fit_or_epsilon_source_never_overfills_obc(
    epsilon: float,
) -> None:
    params = _params(obc_capacity_mb=1.0)
    state = {
        "jetson_compressed_mb": 0.25 + epsilon,
        "obc_data_mb": 0.75,
        "obc_raw_equivalent_mb": 0.0,
    }

    outcome = apply_can_transfer(state, params)

    assert outcome.accepted is True
    assert outcome.transferred_mb == pytest.approx(0.25)
    assert outcome.state["obc_data_mb"] == pytest.approx(1.0)
    assert outcome.state["jetson_compressed_mb"] == pytest.approx(epsilon)


def test_downlink_uses_actual_contact_seconds_and_source_backlog() -> None:
    params = _params()
    state = {
        "obc_data_mb": 1.0,
        "obc_raw_equivalent_mb": 5.11,
        "data_downlinked_mb": 2.0,
        "downlink_raw_equivalent_mb": 1.0,
    }

    outcome = apply_downlink(state, params, contact_seconds=30.0)

    assert outcome.transferred_mb == pytest.approx(0.1875)
    assert outcome.state["obc_data_mb"] == pytest.approx(0.8125)
    assert outcome.state["data_downlinked_mb"] == pytest.approx(2.1875)
    assert outcome.state["downlink_raw_equivalent_mb"] == pytest.approx(
        1.0 + 0.1875 * 5.11
    )


def test_downlink_without_contact_is_a_noop() -> None:
    state = {"obc_data_mb": 1.0, "data_downlinked_mb": 0.0}
    outcome = apply_downlink(state, _params(), contact_seconds=0.0)
    assert outcome.accepted is False
    assert outcome.reason == "no_contact"
    assert outcome.state == state


@pytest.mark.parametrize("epsilon", [0.0, 1e-9])
def test_downlink_exact_rate_fit_or_epsilon_backlog_conserves_bytes(
    epsilon: float,
) -> None:
    params = _params()
    step_capacity_mb = 0.375
    state = {
        "obc_data_mb": step_capacity_mb + epsilon,
        "obc_raw_equivalent_mb": step_capacity_mb + epsilon,
        "data_downlinked_mb": 0.0,
        "downlink_raw_equivalent_mb": 0.0,
    }

    outcome = apply_downlink(state, params, contact_seconds=60.0)

    assert outcome.accepted is True
    assert outcome.transferred_mb == pytest.approx(step_capacity_mb)
    assert outcome.state["obc_data_mb"] == pytest.approx(epsilon)
    assert (
        outcome.state["obc_data_mb"]
        + outcome.state["data_downlinked_mb"]
    ) == pytest.approx(state["obc_data_mb"])


def test_real_environment_observation_matches_pure_transition_projection() -> None:
    config = {
        "max_steps": 2,
        "step_duration_s": 60.0,
        "anomaly_prob": 0.0,
        "scenario_params": {
            "orbit": {},
            "communications": {"sband": {"downlink_rate_kbps": 50.0}},
            "storage": {
                "obc_capacity_mb": 4096.0,
                "jetson_capacity_mb": 100.0,
                "observation_size_mb": 9.41,
                "compression_ratio": 5.11,
            },
            "power": {
                "solar_panels": {"generation_peak_w": 0.0},
                "battery": {"capacity_wh": 70.0, "initial_soc": 0.8},
                "consumption": {
                    "payload_observe": {"sun_w": 1.0, "eclipse_w": 1.0}
                },
            },
            "modes": {"transition_overhead": {"settling_time_s": 0.0}},
            "payload": {
                "compression_time_factor": 2.0,
                "detection_time_s": 300.0,
            },
        },
    }
    env = EventSatEnvironment(config)
    env.reset(seed=42)
    expected = apply_observe(env._pipeline_state(), env._pipeline_parameters())

    result = env.step({"eventsat_0": {"mode": "payload_observe"}})

    assert result.info["observation_accepted"] is True
    assert env.jetson_raw_mb == pytest.approx(expected.state["jetson_raw_mb"])
    assert env.jetson_compressed_mb == pytest.approx(
        expected.state.get("jetson_compressed_mb", 0.0)
    )
    assert env.uncompressed_observations == int(
        expected.state["uncompressed_observations"]
    )
    assert env.total_raw_captured_mb == pytest.approx(
        expected.state["total_raw_captured_mb"]
    )
    assert env.total_observation_s == pytest.approx(
        expected.state["total_observation_s"]
    )


def test_seed_42_environment_matches_canonical_pipeline_for_200_steps() -> None:
    """Exercise every nominal pipeline stage through the real environment.

    This bounded wiring probe protects the pre-remediation nominal trajectory:
    the environment must commit exactly the state projected by the pure helper
    over repeated observe/compress/detect/CAN/downlink cycles.
    """

    config = {
        "max_steps": 200,
        "step_duration_s": 60.0,
        "anomaly_prob": 0.0,
        "scenario_params": {
            "orbit": {},
            "communications": {"sband": {"downlink_rate_kbps": 50.0}},
            "storage": {
                "obc_capacity_mb": 100.0,
                "jetson_capacity_mb": 100.0,
                "observation_size_mb": 9.41,
                "compression_ratio": 5.11,
                "jetson_to_obc_rate_kbps": 8000.0,
            },
            "power": {
                "solar_panels": {"generation_peak_w": 100.0},
                "battery": {"capacity_wh": 70.0, "initial_soc": 0.8},
            },
            "modes": {"transition_overhead": {"settling_time_s": 0.0}},
            "payload": {
                "compression_time_factor": 2.0,
                "detection_time_s": 300.0,
            },
        },
    }
    env = EventSatEnvironment(config)
    env.reset(seed=42)
    communication_steps = tuple(range(9, 200, 10))
    env._orbital_ctx = OrbitalContext(
        ground_passes=[
            GroundPass(
                step,
                step + 1,
                start_s=step * 60.0,
                end_s=(step + 1) * 60.0,
            )
            for step in communication_steps
        ],
        step_s=60.0,
    )

    cycle = (
        "payload_observe",
        "payload_compress",
        "payload_compress",
        "payload_detect",
        "payload_detect",
        "payload_detect",
        "payload_detect",
        "payload_detect",
        "payload_send",
        "communication",
    )
    expected = env._pipeline_state()
    compression_progress = 0
    detection_progress = 0
    pipeline_fields = tuple(expected)

    for step in range(200):
        mode = cycle[step % len(cycle)]
        if mode == "payload_observe":
            outcome = apply_observe(expected, env._pipeline_parameters())
            assert outcome.accepted
            expected = outcome.state
        elif mode == "payload_compress":
            compression_progress += 1
            if compression_progress >= env.compression_time_factor:
                outcome = apply_compress(expected, env._pipeline_parameters())
                assert outcome.accepted
                expected = outcome.state
                compression_progress = 0
        elif mode == "payload_detect":
            detection_progress += 1
            if detection_progress >= env.detection_steps:
                outcome = apply_detect(expected, env._pipeline_parameters())
                assert outcome.accepted
                expected = outcome.state
                detection_progress = 0
        elif mode == "payload_send":
            outcome = apply_can_transfer(
                expected,
                env._pipeline_parameters(),
                duration_s=env.step_duration_s,
            )
            assert outcome.accepted
            expected = outcome.state
        else:
            contact_s = env._contact_seconds()
            assert contact_s == 60.0
            outcome = apply_downlink(
                expected,
                env._pipeline_parameters(),
                contact_seconds=contact_s,
            )
            assert outcome.accepted
            expected = outcome.state

        result = env.step({"eventsat_0": {"mode": mode}})
        assert result.info["resolved_mode"] == mode
        actual = env._pipeline_state()
        for field in pipeline_fields:
            assert actual[field] == pytest.approx(expected[field]), (step, field)
        assert env.compression_progress == compression_progress
        assert env.detection_progress == detection_progress

    assert result.done is True
