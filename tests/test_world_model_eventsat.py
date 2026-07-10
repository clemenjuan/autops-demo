from __future__ import annotations

import numpy as np

from src.core.config_loader import ExperimentConfig
from src.core.experiment_runner import ExperimentRunner


def _wm_config(tmp_path, repr_type: str) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=f"test_{repr_type}",
        seed=3,
        agent_organization="sas",
        decision_procedure="sda",
        representation="rl",
        representation_config={
            "type": repr_type,
            "horizon": 3,
            "samples": 8,
            "elites": 2,
            "cem_iterations": 2,
            "mission_mode": "science",
        },
        behaviour="hand_designed",
        behaviour_config={"mode": "hand_designed"},
        operations_paradigm="autonomous_onboard",
        environment={
            "constellation_size": 1,
            "timestep_seconds": 60,
            "max_steps": 6,
            "scenario": "eventsat",
            "scenario_config": {"scenario_file": "configs/scenarios/eventsat.yaml"},
        },
        num_episodes=1,
        max_steps=6,
        output_dir=str(tmp_path / repr_type),
    )


def test_world_model_representations_register_and_smoke_run(tmp_path):
    import src.eventsat.world_model  # noqa: F401
    from src.core.behaviour.controller import _REPRESENTATION_REGISTRY

    assert "lewm_cem_eventsat" in _REPRESENTATION_REGISTRY
    assert "dreamerv3_eventsat" in _REPRESENTATION_REGISTRY

    cfg = _wm_config(tmp_path, "lewm_cem_eventsat")
    results = ExperimentRunner(config=cfg).run()
    mean = results["experiment_statistics"].mean
    assert results["num_episodes"] == 1
    assert mean["candidate_count"] == 8.0
    assert mean["cem_iterations"] == 2.0
    assert mean["planner_latency_s"] >= 0.0


def test_analytic_backend_provenance_and_cost_survive_real_runner(tmp_path):
    cfg = _wm_config(tmp_path, "lewm_cem_eventsat")
    cfg.representation_config.update(
        {
            "planner_backend": "analytic",
            "planner_pricing": "obc",
            "planner_power_w": 0.5,
            "plan_hold": 3,
        }
    )
    results = ExperimentRunner(config=cfg).run()
    mean = results["experiment_statistics"].mean

    assert results["rollout_backend"] == "analytic"
    assert results["experiment_statistics"].metadata["rollout_backend"] == "analytic"
    assert mean["artifact_loaded"] == 0.0
    assert "artifact_fallback" not in mean
    assert mean["planner_ms_per_event"] > 0.0
    assert mean["planner_event_count"] == 2.0
    assert mean["planner_energy_wh"] == 0.5 * 2.0 / 60.0


def test_eventsat_world_model_trace_schema(tmp_path):
    from src.eventsat.env import EventSatEnvironment
    from src.eventsat.trace import (
        ACTION_NAMES,
        OBS25_NAMES,
        STATE_NAMES,
        WorldModelTraceEpisode,
    )

    env = EventSatEnvironment(
        {
            "scenario_file": "configs/scenarios/eventsat.yaml",
            "max_steps": 4,
            "step_duration_s": 60,
            "anomaly_prob": 0.0,
        }
    )
    obs = env.reset(seed=11)
    action = {"eventsat_0": {"mode": "charging"}}
    result = env.step(action)

    trace = WorldModelTraceEpisode(episode_id=0, seed=11)
    trace.record(step=0, observation=obs, env_actions=action, rewards=result.rewards, info=result.info)
    arrays = trace.as_arrays()

    assert arrays["obs"].shape == (1, len(OBS25_NAMES))
    assert arrays["action"].shape == (1, len(ACTION_NAMES))
    assert arrays["state"].shape == (1, len(STATE_NAMES))
    assert len(STATE_NAMES) == 25  # published space-world-models v1 contract
    legacy_budget_idx = STATE_NAMES.index("daily_downlink_budget_mb")
    assert arrays["state"][0, legacy_budget_idx] == 0.0
    assert STATE_NAMES[legacy_budget_idx + 1] == "achievable_downlink_mb"
    assert STATE_NAMES[legacy_budget_idx + 2] == "health_nominal"
    assert np.isfinite(arrays["obs"]).all()
    assert np.isclose(arrays["action"].sum(), 1.0)

    out = tmp_path / "episode.npz"
    trace.write_npz(out)
    loaded = np.load(out)
    assert loaded["obs"].shape == arrays["obs"].shape


def test_plan_hold_clamped_to_horizon(caplog):
    """A hold longer than the horizon is impossible (the held tail is sliced from
    the planned sequence); it must clamp loudly, not truncate silently."""
    import logging

    from src.eventsat.world_model import _WorldModelPlanner

    with caplog.at_level(logging.WARNING, logger="src.eventsat.world_model"):
        planner = _WorldModelPlanner(
            {"horizon": 12, "plan_hold": 48, "samples": 8, "elites": 2, "cem_iterations": 1},
            method="cem",
        )
    assert planner.plan_hold == 12
    assert any("plan_hold" in rec.message for rec in caplog.records)

    # hold <= horizon passes through untouched (receding-horizon overlap is valid).
    ok = _WorldModelPlanner(
        {"horizon": 12, "plan_hold": 6, "samples": 8, "elites": 2, "cem_iterations": 1},
        method="cem",
    )
    assert ok.plan_hold == 6


def test_empty_mission_weights_uses_named_preset():
    """An empty config dict means no explicit override; use mission_mode preset.

    The A7 retargeting configs carry mission_weights: {}, so treating that as an
    explicit zero vector silently collapsed downlink/safe back to science.
    """
    from src.eventsat.world_model import _WorldModelPlanner

    downlink = _WorldModelPlanner(
        {
            "horizon": 3,
            "samples": 4,
            "elites": 1,
            "cem_iterations": 1,
            "mission_mode": "downlink",
            "mission_weights": {},
        },
        method="cem",
    )
    assert np.isclose(downlink.mode_weights["downlink_progress"], 0.38)
    assert np.isclose(downlink.mode_weights["science_progress"], 0.05)

    artifact_safe = {
        "battery_margin": 0.45,
        "storage_margin": 0.20,
        "downlink_progress": 0.0,
        "science_progress": 0.0,
        "detection_progress": 0.0,
        "communication_opportunity": 0.0,
        "forced_mode_risk": -0.20,
        "anomaly_safe": 0.15,
    }
    safe = _WorldModelPlanner(
        {
            "horizon": 3,
            "samples": 4,
            "elites": 1,
            "cem_iterations": 1,
            "mission_mode": "safe",
            "mission_weights": {},
            "mission_weight_presets": {"safe": artifact_safe},
        },
        method="cem",
    )
    assert safe.mode_weights["downlink_progress"] == 0.0
    assert safe.mode_weights["forced_mode_risk"] < 0.0


def _bare_latent_backend(attribute_names, attribute_scale=None, normalize=False):
    """Construct an _ArtifactLatentBackend without loading torch/a checkpoint --
    only the pure-numpy scoring path (_score_from_attrs) is exercised."""
    from src.eventsat.world_model import _ArtifactLatentBackend

    backend = object.__new__(_ArtifactLatentBackend)
    backend.attribute_names = list(attribute_names)
    if attribute_scale is None:
        attribute_scale = np.ones(len(attribute_names), dtype=np.float32)
    backend.attribute_scale = np.asarray(attribute_scale, dtype=np.float32)
    backend.normalize_attribute_scale = normalize
    return backend


def test_artifact_backend_loads_target_std_with_fallback():
    """probe.normalization.target_std is read when present and shaped correctly;
    a missing/mismatched field falls back to a no-op (all-ones) scale rather
    than crashing, so older artifacts keep loading."""
    from src.eventsat.world_model import _ArtifactLatentBackend

    names = ["a", "b", "c"]

    def scale_for(probe_extra):
        probe = {"W": np.zeros((3, 4), dtype=np.float32), "b": np.zeros(3, dtype=np.float32),
                 "attribute_names": names, **probe_extra}
        backend = object.__new__(_ArtifactLatentBackend)
        # Replicate exactly the loading logic from __init__ (the piece under test).
        probe_normalization = probe.get("normalization")
        target_std = probe_normalization.get("target_std") if isinstance(probe_normalization, dict) else None
        if target_std is not None and len(target_std) == len(names):
            scale = np.asarray(target_std, dtype=np.float32)
            scale[scale < 1e-8] = 1.0
        else:
            scale = np.ones(len(names), dtype=np.float32)
        return scale

    np.testing.assert_array_equal(scale_for({}), [1.0, 1.0, 1.0])  # no normalization key
    np.testing.assert_array_equal(
        scale_for({"normalization": {"target_std": [2.0, 0.0, 4.0]}}), [2.0, 1.0, 4.0]
    )  # near-zero std clamped to 1.0 (avoid div-by-zero)
    np.testing.assert_array_equal(
        scale_for({"normalization": {"target_std": [2.0, 3.0]}}), [1.0, 1.0, 1.0]
    )  # length mismatch -> fallback, not a crash


def test_attribute_scale_normalization_fixes_downlink_dominance():
    """Reproduces the 2026-07-09 E-A7 diagnosis on synthetic data shaped like the
    real one: one attribute (downlink_progress) has ~150x the candidate variance
    of the rest. With normalize_attribute_scale off (the historical default),
    two presets that both weight it positively must collapse to near-identical
    rankings; with it on, they must diverge meaningfully -- while a preset that
    assigns zero weight to the dominant attribute (safe) should differ from both
    regardless of the flag, since it never depended on that attribute's scale."""
    names = ["battery_margin", "downlink_progress", "science_progress"]
    rng = np.random.default_rng(0)
    n = 200
    attrs = np.stack([
        rng.normal(0.98, 0.03, n),      # battery_margin: tiny variance
        rng.normal(14.0, 8.5, n),       # downlink_progress: raw MB, huge variance
        rng.normal(0.27, 0.14, n),      # science_progress: tiny variance
    ], axis=1).astype(np.float32)

    science = {"battery_margin": 0.2, "downlink_progress": 0.25, "science_progress": 0.4}
    downlink = {"battery_margin": 0.158, "downlink_progress": 0.474}
    safe = {"battery_margin": 0.45}  # zero weight on downlink_progress

    def rho_argmax(backend, w1, w2):
        s1, s2 = backend._score_from_attrs(attrs, w1), backend._score_from_attrs(attrs, w2)
        return float(np.corrcoef(s1, s2)[0, 1]), int(np.argmax(s1)) == int(np.argmax(s2))

    raw = _bare_latent_backend(names, normalize=False)
    rho_raw, same_raw = rho_argmax(raw, science, downlink)
    assert rho_raw > 0.999 and same_raw, "raw scoring should reproduce the diagnosed dominance"

    normalized = _bare_latent_backend(names, attribute_scale=[0.03, 8.5, 0.14], normalize=True)
    rho_norm, _ = rho_argmax(normalized, science, downlink)
    assert rho_norm < 0.95, "normalized scoring must meaningfully diverge science vs downlink"

    # safe excludes the dominant attribute entirely -> should differ from science
    # under BOTH raw and normalized scoring (sanity: the flag doesn't just move
    # noise around, it fixes a real dominance problem without breaking the case
    # that already worked).
    rho_safe_raw, _ = rho_argmax(raw, science, safe)
    rho_safe_norm, _ = rho_argmax(normalized, science, safe)
    assert rho_safe_raw < 0.9
    assert rho_safe_norm < 0.9



def test_plan_hold_warm_start_advances_by_held_steps():
    from src.eventsat.world_model import MODE_TO_IDX, _WorldModelPlanner

    planner = _WorldModelPlanner(
        {"horizon": 5, "plan_hold": 3, "samples": 8, "elites": 2, "cem_iterations": 1},
        method="cem",
    )
    planner.previous_solution = np.asarray(
        [
            MODE_TO_IDX["charging"],
            MODE_TO_IDX["communication"],
            MODE_TO_IDX["payload_observe"],
            MODE_TO_IDX["payload_compress"],
            MODE_TO_IDX["payload_detect"],
        ],
        dtype=np.int64,
    )

    probs = planner._initial_probs(5)

    assert int(np.argmax(probs[0])) == MODE_TO_IDX["payload_compress"]
    assert int(np.argmax(probs[1])) == MODE_TO_IDX["payload_detect"]
    assert int(np.argmax(probs[2])) == MODE_TO_IDX["payload_detect"]


def _surrogate_state(**overrides):
    state = {
        "battery_soc": 0.8,
        "health_status": "nominal",
        "current_mode": "charging",
        "in_sunlight": True,
        "ground_pass_active": False,
        "remaining_pass_duration": 0.0,
        "time_to_next_pass": 50.0,
        "following_gap_steps": 94.0,
        "orbital_period_steps": 94.0,
        "obc_data_mb": 0.0,
        "data_downlinked_mb": 0.0,
        "jetson_raw_mb": 0.0,
        "jetson_compressed_mb": 0.0,
        "storage_capacity_mb": 4096.0,
        "jetson_capacity_mb": 249036.8,
        "uncompressed_observations": 0.0,
        "compression_progress": 0.0,
        "undetected_observations": 0.0,
        "detection_progress": 0.0,
        "total_observation_s": 0.0,
        "total_detections": 0.0,
    }
    state.update(overrides)
    return state


def test_episode_reset_matches_fresh_world_model_planner_state():
    from src.eventsat.world_model import _WorldModelPlanner, action_from_mode

    config = {
        "horizon": 4,
        "plan_hold": 3,
        "samples": 12,
        "elites": 3,
        "cem_iterations": 2,
    }
    state = _surrogate_state(battery_soc=0.75, jetson_raw_mb=2.0)

    reused = _WorldModelPlanner(config, method="cem")
    reused.seed(17)
    reused.select(state)
    # Explicitly dirty every history/counter family covered by reset().
    reused._obs_history = [np.ones(25, dtype=np.float32)]
    reused._action_history = [action_from_mode("payload_observe")]
    reused._contact_reflex_overrides = 4
    reused._held_plan_mask_repairs = 3

    reused.reset()
    reused.seed(29)
    reused_mode, _ = reused.select(state)

    fresh = _WorldModelPlanner(config, method="cem")
    fresh.reset()
    fresh.seed(29)
    fresh_mode, _ = fresh.select(state)

    assert reused_mode == fresh_mode
    np.testing.assert_array_equal(reused.previous_solution, fresh.previous_solution)
    assert reused._obs_history == fresh._obs_history == []
    assert reused._action_history == fresh._action_history == []
    assert reused._contact_reflex_overrides == 0
    assert reused._held_plan_mask_repairs == 0


def test_surrogate_downlinks_current_active_pass_step_with_scenario_rate():
    from src.eventsat.world_model import MODE_TO_IDX, _WorldModelPlanner

    planner = _WorldModelPlanner(
        {"horizon": 1, "samples": 4, "elites": 1, "cem_iterations": 1},
        method="cem",
    )
    state = _surrogate_state(
        current_mode="communication",
        ground_pass_active=True,
        remaining_pass_duration=1.0,
        obc_data_mb=1.0,
        data_stored_mb=1.0,
    )

    final, penalty = planner._rollout_surrogate(state, [MODE_TO_IDX["communication"]])

    assert penalty == 0.0
    assert np.isclose(final["data_downlinked_mb"], 0.375)
    assert np.isclose(final["obc_data_mb"], 0.625)


def test_surrogate_uses_second_accurate_contact_across_partial_los():
    from src.eventsat.world_model import MODE_TO_IDX, _WorldModelPlanner

    planner = _WorldModelPlanner({}, method="cem")
    state = _surrogate_state(
        current_mode="communication",
        ground_pass_active=True,
        remaining_pass_duration=95.0 / 60.0,
        remaining_pass_duration_s=95.0,
        contact_window_seconds=60.0,
        obc_data_mb=1.0,
        data_stored_mb=1.0,
    )

    final, _ = planner._rollout_surrogate(
        state,
        [MODE_TO_IDX["communication"], MODE_TO_IDX["communication"]],
    )

    assert np.isclose(final["data_downlinked_mb"], 50.0 / 8.0 * 95.0 / 1000.0)
    assert np.isclose(final["obc_data_mb"], 1.0 - final["data_downlinked_mb"])
    assert final["contact_window_seconds"] == 0.0
    assert final["remaining_pass_duration_s"] == 0.0
    assert final["ground_pass_active"] is False


def test_surrogate_uses_declared_pipeline_product_size_and_rate():
    from src.eventsat.world_model import _WorldModelPlanner

    planner = _WorldModelPlanner({}, method="cem")
    state = _surrogate_state(
        observation_size_mb=2.0,
        compression_ratio=4.0,
        jetson_capacity_mb=10.0,
        jetson_raw_mb=7.0,
        downlink_rate_kbps=100.0,
        step_duration_s=30.0,
    )

    planner._advance_pipeline(state, "payload_observe")
    assert state["jetson_raw_mb"] == 9.0
    assert state["uncompressed_observations"] == 1.0
    assert state["total_raw_captured_mb"] == 2.0
    assert state["total_observation_s"] == 30.0

    state.update({
        "ground_pass_active": True,
        "contact_window_seconds": 30.0,
        "obc_data_mb": 1.0,
    })
    planner._advance_pipeline(state, "communication")
    assert state["data_downlinked_mb"] == 0.375


def test_surrogate_resets_interrupted_compression_progress_like_environment():
    from src.eventsat.world_model import MODE_TO_IDX, _WorldModelPlanner

    planner = _WorldModelPlanner({}, method="cem")
    state = _surrogate_state(
        current_mode="charging",
        observation_size_mb=9.41,
        compression_ratio=5.11,
        compression_time_factor=2.0,
        jetson_raw_mb=9.41,
        uncompressed_observations=1.0,
    )

    final, _ = planner._rollout_surrogate(
        state,
        [
            MODE_TO_IDX["payload_compress"],
            MODE_TO_IDX["charging"],
            MODE_TO_IDX["payload_compress"],
        ],
    )

    assert final["compression_progress"] == 1.0
    assert final["uncompressed_observations"] == 1.0
    assert final["jetson_compressed_mb"] == 0.0


def test_world_model_encoding_carries_environment_pipeline_contract():
    from src.eventsat.env import EventSatEnvironment
    from src.eventsat.world_model import eventsat_observation_to_vector

    env = EventSatEnvironment({
        "max_steps": 4,
        "step_duration_s": 30.0,
        "anomaly_prob": 0.0,
        "scenario_params": {
            "orbit": {"orbital_period_s": 3000.0},
            "communications": {"sband": {"downlink_rate_kbps": 25.0}},
            "storage": {
                "obc_capacity_mb": 3.0,
                "jetson_capacity_mb": 10.0,
                "observation_size_mb": 2.0,
                "compression_ratio": 4.0,
                "jetson_to_obc_rate_kbps": 4000.0,
            },
            "payload": {
                "compression_time_factor": 3.0,
                "detection_time_s": 90.0,
            },
            "modes": {
                "constraints": {
                    "payload_observe": {"min_battery_soc": 0.45},
                    "payload_compress": {"min_battery_soc": 0.35},
                    "payload_detect": {"min_battery_soc": 0.35},
                    "payload_send": {"min_battery_soc": 0.35},
                },
                "transition_overhead": {
                    "settling_time_s": 60.0,
                    "attitude_maneuver_modes": [
                        "payload_observe",
                        "communication",
                    ],
                },
            },
            "power": {
                "battery": {"min_soc": 0.25},
            },
        },
    })
    encoded = eventsat_observation_to_vector(env.reset(seed=1)).raw

    assert encoded["observation_size_mb"] == 2.0
    assert encoded["compression_ratio"] == 4.0
    assert encoded["jetson_to_obc_rate_kbps"] == 4000.0
    assert encoded["downlink_rate_kbps"] == 25.0
    assert encoded["step_duration_s"] == 30.0
    assert encoded["compression_time_factor"] == 3.0
    assert encoded["detection_steps"] == 3.0
    assert encoded["battery_min_soc"] == 0.25
    assert encoded["mode_min_battery_soc"]["payload_observe"] == 0.45
    assert "contact_window_seconds" in encoded
    assert encoded["settling_time_steps"] == 2
    assert encoded["transition_steps_remaining"] == 0
    assert encoded["attitude_maneuver_modes"] == [
        "communication",
        "payload_observe",
    ]
    assert encoded["previous_mode"] == "charging"


def test_surrogate_live_pipeline_telemetry_beats_stale_planner_config():
    from src.eventsat.world_model import _WorldModelPlanner

    planner = _WorldModelPlanner(
        {
            "observation_size_mb": 9.0,
            "compression_ratio": 9.0,
            "step_duration_s": 30.0,
            "downlink_rate_kbps": 100.0,
            "downlink_rate_mb_per_step": 9.0,
            "jetson_to_obc_rate_kbps": 800.0,
            "jetson_to_obc_mb_per_step": 9.0,
            "detection_metadata_mb": 0.1,
        },
        method="cem",
    )
    telemetry = {
        "observation_size_mb": 2.0,
        "compression_ratio": 4.0,
        "step_duration_s": 60.0,
        "downlink_rate_kbps": 50.0,
        "jetson_to_obc_rate_kbps": 80.0,
        "detection_metadata_mb": 0.01,
    }

    observe = _surrogate_state(**telemetry)
    planner._advance_pipeline(observe, "payload_observe")
    assert observe["jetson_raw_mb"] == 2.0
    assert observe["total_observation_s"] == 60.0

    compress = _surrogate_state(
        **telemetry,
        jetson_raw_mb=2.0,
        uncompressed_observations=1.0,
        compression_progress=1.0,
        compression_time_factor=2.0,
    )
    planner._advance_pipeline(compress, "payload_compress")
    assert compress["jetson_raw_mb"] == 0.0
    assert compress["jetson_compressed_mb"] == 0.5

    detect = _surrogate_state(
        **telemetry,
        undetected_observations=1.0,
        detection_steps=1.0,
    )
    planner._advance_pipeline(detect, "payload_detect")
    assert detect["obc_data_mb"] == 0.01

    send = _surrogate_state(**telemetry, jetson_compressed_mb=1.0)
    planner._advance_pipeline(send, "payload_send")
    assert np.isclose(send["obc_data_mb"], 0.6)
    assert np.isclose(send["jetson_compressed_mb"], 0.4)

    downlink = _surrogate_state(
        **telemetry,
        ground_pass_active=True,
        contact_window_seconds=60.0,
        obc_data_mb=1.0,
    )
    planner._advance_pipeline(downlink, "communication")
    assert np.isclose(downlink["data_downlinked_mb"], 0.375)
    assert np.isclose(downlink["obc_data_mb"], 0.625)


def test_surrogate_settling_blocks_observe_and_communication_until_completion():
    from src.eventsat.world_model import MODE_TO_IDX, _WorldModelPlanner

    planner = _WorldModelPlanner(
        {"battery_penalty": 0.0, "pass_stage_reward": 0.0}, method="cem"
    )
    base = {
        "settling_time_steps": 2,
        "transition_steps_remaining": 0,
        "attitude_maneuver_modes": ["payload_observe", "communication"],
        "previous_mode": "charging",
        "observation_size_mb": 2.0,
        "step_duration_s": 60.0,
        "ground_pass_active": True,
        "remaining_pass_duration": 4.0,
        "remaining_pass_duration_s": 240.0,
        "contact_window_seconds": 60.0,
        "obc_data_mb": 1.0,
        "data_stored_mb": 1.0,
    }

    for mode in ("payload_observe", "communication"):
        state = _surrogate_state(**base)
        start_raw = state["jetson_raw_mb"]
        start_obc = state["obc_data_mb"]
        start_downlinked = state["data_downlinked_mb"]

        first, _ = planner._rollout_surrogate(state, [MODE_TO_IDX[mode]])
        assert first["current_mode"] == "charging"
        assert first["in_transition"] is True
        assert first["transition_steps_remaining"] == 1
        assert first["jetson_raw_mb"] == start_raw
        assert first["obc_data_mb"] == start_obc
        assert first["data_downlinked_mb"] == start_downlinked

        second, _ = planner._rollout_surrogate(first, [MODE_TO_IDX[mode]])
        assert second["current_mode"] == "charging"
        assert second["in_transition"] is True
        assert second["transition_steps_remaining"] == 0
        assert second["previous_mode"] == mode
        assert second["jetson_raw_mb"] == start_raw
        assert second["obc_data_mb"] == start_obc
        assert second["data_downlinked_mb"] == start_downlinked

        third, _ = planner._rollout_surrogate(second, [MODE_TO_IDX[mode]])
        assert third["current_mode"] == mode
        assert third["in_transition"] is False
        if mode == "payload_observe":
            assert third["jetson_raw_mb"] == start_raw + 2.0
            assert third["data_downlinked_mb"] == start_downlinked
        else:
            assert third["jetson_raw_mb"] == start_raw
            assert np.isclose(
                third["data_downlinked_mb"] - start_downlinked, 0.375
            )
            assert np.isclose(third["obc_data_mb"], start_obc - 0.375)

    communication = MODE_TO_IDX["communication"]
    two_settling_steps = np.asarray([[communication, communication]])
    completed_transition = np.asarray(
        [[communication, communication, communication]]
    )
    assert planner._shaping_scores(
        _surrogate_state(**base), two_settling_steps
    )[0] == 0.0
    assert planner._shaping_scores(
        _surrogate_state(**base), completed_transition
    )[0] > 0.0


def test_surrogate_one_step_settling_does_not_restart_forever():
    from src.eventsat.world_model import MODE_TO_IDX, _WorldModelPlanner

    planner = _WorldModelPlanner({}, method="cem")
    state = _surrogate_state(
        settling_time_steps=1,
        transition_steps_remaining=0,
        attitude_maneuver_modes=["payload_observe"],
        previous_mode="charging",
        observation_size_mb=2.0,
    )

    final, _ = planner._rollout_surrogate(
        state,
        [MODE_TO_IDX["payload_observe"], MODE_TO_IDX["payload_observe"]],
    )

    assert final["previous_mode"] == "payload_observe"
    assert final["in_transition"] is False
    assert final["current_mode"] == "payload_observe"
    assert final["jetson_raw_mb"] == 2.0


def test_surrogate_payload_send_uses_can_rate_and_never_negative_transfer():
    from src.eventsat.world_model import MODE_TO_IDX, _WorldModelPlanner

    planner = _WorldModelPlanner(
        {"horizon": 1, "samples": 4, "elites": 1, "cem_iterations": 1},
        method="cem",
    )

    can_state = _surrogate_state(jetson_compressed_mb=2.0)
    final, _ = planner._rollout_surrogate(can_state, [MODE_TO_IDX["payload_send"]])
    assert np.isclose(final["jetson_compressed_mb"], 0.0)
    assert np.isclose(final["obc_data_mb"], 2.0)

    full_obc = _surrogate_state(
        jetson_compressed_mb=2.0,
        obc_data_mb=4096.0,
        data_stored_mb=4098.0,
    )
    final, _ = planner._rollout_surrogate(full_obc, [MODE_TO_IDX["payload_send"]])
    assert np.isclose(final["jetson_compressed_mb"], 2.0)
    assert np.isclose(final["obc_data_mb"], 4096.0)


def test_surrogate_rejects_observation_that_overflows_shared_jetson_capacity():
    from src.eventsat.world_model import _WorldModelPlanner

    planner = _WorldModelPlanner(
        {"observation_size_mb": 9.41}, method="cem"
    )
    state = _surrogate_state(
        jetson_capacity_mb=100.0,
        jetson_raw_mb=90.0,
        jetson_compressed_mb=1.5,
        uncompressed_observations=2.0,
        total_observation_s=120.0,
    )

    planner._advance_pipeline(state, "payload_observe")

    assert state["jetson_raw_mb"] == 90.0
    assert state["jetson_compressed_mb"] == 1.5
    assert state["uncompressed_observations"] == 2.0
    assert state["total_observation_s"] == 120.0
    assert state["jetson_raw_mb"] + state["jetson_compressed_mb"] <= 100.0


def test_surrogate_detection_metadata_rejection_preserves_backlog_and_value():
    from src.eventsat.world_model import _WorldModelPlanner

    planner = _WorldModelPlanner(
        {"detection_steps": 1, "detection_metadata_mb": 0.01}, method="cem"
    )
    state = _surrogate_state(
        storage_capacity_mb=1.0,
        obc_data_mb=0.995,
        undetected_observations=1.0,
        detection_progress=0.0,
        total_detections=4.0,
    )

    planner._advance_pipeline(state, "payload_detect")

    assert state["obc_data_mb"] == 0.995
    assert state["undetected_observations"] == 1.0
    assert state["total_detections"] == 4.0
    assert state["detection_progress"] == 1.0


def test_plan_hold_and_jetson_planned_survive_metrics_aggregation():
    from src.eventsat.metrics import EventSatMetricsCollector

    collector = EventSatMetricsCollector({"max_steps": 2, "step_duration_s": 60.0})
    step = collector.collect_step_metrics(
        timestep=0,
        wall_clock_seconds=0.01,
        env_state=None,
        actions={},
        rewards={"total": 0.0},
        info={
            "battery_soc": 0.8,
            "prev_battery_soc": 0.8,
            "data_downlinked_mb": 0.0,
            "max_achievable_downlink_mb": 1.0,
        },
        decision_metrics={
            "inference_allowed": 1.0,
            "decision_latency_s": 0.01,
            "plan_hold": 12.0,
            "jetson_planned": 0.0,
        },
    )

    assert step.metrics["plan_hold"] == 12.0
    assert step.metrics["jetson_planned"] == 0.0
    episode = collector.aggregate_episode_metrics([step])
    assert episode.aggregated["plan_hold"] == 12.0
    assert episode.aggregated["jetson_planned"] == 0.0


def test_analytic_rollout_matches_real_environment_fixed_sequence():
    """The intentional analytic backend is the real deterministic plant, not
    the approximate artifact fallback under a new label."""
    from src.eventsat.env import EventSatEnvironment
    from src.eventsat.world_model import (
        MODE_TO_IDX,
        _WorldModelPlanner,
        eventsat_observation_to_vector,
    )
    from src.orbital.context import OrbitalContext
    from src.orbital.eclipse import EclipseInterval
    from src.orbital.ground_access import GroundPass

    env = EventSatEnvironment(
        {
            "scenario_file": "configs/scenarios/eventsat.yaml",
            "scenario_overrides": {
                "modes": {"transition_overhead": {"settling_time_s": 0.0}}
            },
            "max_steps": 12,
            "step_duration_s": 60.0,
            "anomaly_prob": 0.0,
        }
    )
    env.reset(seed=42)
    env._orbital_ctx = OrbitalContext(
        eclipses=[EclipseInterval(start_step=0, end_step=4)],
        ground_passes=[
            GroundPass(
                start_step=9,
                end_step=10,
                start_s=540.0,
                end_s=630.0,
            )
        ],
        mode="test",
        step_s=60.0,
    )
    state = dict(eventsat_observation_to_vector(env.get_observation()).raw)
    sequence = [
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
        "communication",
        "communication",
    ]
    planner = _WorldModelPlanner(
        {
            "planner_backend": "analytic",
            "planner_pricing": "obc",
            "planner_power_w": 0.0,
            "horizon": len(sequence),
            "plan_hold": len(sequence),
            "samples": 2,
            "elites": 1,
            "cem_iterations": 1,
        },
        method="cem",
    )
    encoded_sequence = [MODE_TO_IDX[mode] for mode in sequence]
    predicted = None
    for index, mode in enumerate(sequence):
        env.step(
            {
                "eventsat_0": {
                    "mode": mode,
                    "jetson_planned": True,
                    "planner_pricing": "obc",
                    "planner_power_w": 0.0,
                }
            }
        )
        predicted, _ = planner._rollout_analytic(
            state, encoded_sequence[: index + 1]
        )
        actual = eventsat_observation_to_vector(env.get_observation()).raw
        exact_fields = (
            "battery_soc",
            "jetson_raw_mb",
            "jetson_compressed_mb",
            "obc_data_mb",
            "data_downlinked_mb",
            "total_raw_captured_mb",
            "obc_raw_equivalent_mb",
            "downlink_raw_equivalent_mb",
            "uncompressed_observations",
            "compression_progress",
            "undetected_observations",
            "detection_progress",
            "total_observation_s",
            "total_detections",
            "total_pass_duration_s",
            "contact_window_seconds",
        )
        for field in exact_fields:
            if field == "battery_soc":
                assert np.isclose(
                    predicted[field], actual[field], rtol=0.0, atol=2e-7
                ), (index, field)
            else:
                assert predicted[field] == actual[field], (index, field)
        assert predicted["current_mode"] == actual["current_mode"]
        assert predicted["in_sunlight"] == actual["in_sunlight"]
        assert predicted["ground_pass_active"] == actual["ground_pass_active"]

    assert predicted is not None
    assert predicted["data_downlinked_mb"] == 0.5625
    assert predicted["total_pass_duration_s"] == 90.0
    assert env.total_detections == 1
    assert (
        predicted["obc_data_mb"] + predicted["data_downlinked_mb"]
        == 9.41 / 5.11 + env.detection_metadata_mb
    )


def test_analytic_backend_identity_never_sets_artifact_fallback():
    from src.eventsat.world_model import _WorldModelPlanner

    analytic = _WorldModelPlanner(
        {
            "planner_backend": "analytic",
            "planner_artifact": "/definitely/not/consulted.json",
            "samples": 2,
            "elites": 1,
            "cem_iterations": 1,
        },
        method="cem",
    )
    metrics = analytic.get_metrics()
    assert analytic.backend == "analytic"
    assert metrics["rollout_backend"] == "analytic"
    assert metrics["artifact_loaded"] == 0.0
    assert "artifact_fallback" not in metrics

    unintended = _WorldModelPlanner(
        {
            "planner_backend": "latent",
            "samples": 2,
            "elites": 1,
            "cem_iterations": 1,
        },
        method="cem",
    )
    assert unintended.get_metrics()["rollout_backend"] == "fallback"
    assert unintended.get_metrics()["artifact_fallback"] == 1.0


def test_planner_event_metrics_use_events_not_held_steps():
    from src.eventsat.metrics import EventSatMetricsCollector

    collector = EventSatMetricsCollector({"max_steps": 2, "step_duration_s": 60.0})
    steps = []
    for timestep, decision in enumerate(
        (
            {
                "planner_event": 1.0,
                "planner_event_latency_s": 0.002,
                "planner_step_energy_wh": 0.125,
            },
            {
                "planner_event": 0.0,
                "planner_event_latency_s": 0.0,
                "planner_step_energy_wh": 0.0,
            },
        )
    ):
        steps.append(
            collector.collect_step_metrics(
                timestep=timestep,
                wall_clock_seconds=0.01,
                env_state=None,
                actions={},
                rewards={"total": 0.0},
                info={
                    "battery_soc": 0.8,
                    "prev_battery_soc": 0.8,
                    "data_downlinked_mb": 0.0,
                    "max_achievable_downlink_mb": 1.0,
                },
                decision_metrics=decision,
            )
        )
    episode = collector.aggregate_episode_metrics(steps)
    assert episode.aggregated["planner_event_count"] == 1.0
    assert episode.aggregated["planner_ms_per_event"] == 2.0
    assert episode.aggregated["planner_energy_wh"] == 0.125


def test_both_cem_backends_record_positive_latency_on_50_step_smoke():
    from src.eventsat.env import EventSatEnvironment
    from src.eventsat.world_model import _WorldModelPlanner, eventsat_observation_to_vector

    class _FixedLatentBackend:
        history_size = 3

        @staticmethod
        def score_sequences(history, sequences, mode_weights):
            return np.zeros(sequences.shape[0], dtype=np.float64)

    for backend, pricing, power_w in (
        ("analytic", "obc", 0.5),
        ("latent", "jetson", 7.0),
    ):
        env = EventSatEnvironment(
            {
                "scenario_file": "configs/scenarios/eventsat.yaml",
                "max_steps": 50,
                "step_duration_s": 60.0,
                "anomaly_prob": 0.0,
            }
        )
        observation = env.reset(seed=9)
        planner = _WorldModelPlanner(
            {
                "planner_backend": backend,
                "planner_pricing": pricing,
                "planner_power_w": power_w,
                "horizon": 5,
                "plan_hold": 5,
                "samples": 4,
                "elites": 1,
                "cem_iterations": 1,
                "seed": 9,
            },
            method="cem",
        )
        if backend == "latent":
            planner.latent_backend = _FixedLatentBackend()
            planner.rollout_backend = "latent"
            planner._last_metrics["rollout_backend"] = "latent"

        for _ in range(50):
            encoded = eventsat_observation_to_vector(observation)
            state = dict(encoded.raw)
            state["obs25"] = encoded.obs25
            mode, metrics = planner.select(state)
            observation = env.step(
                {
                    "eventsat_0": {
                        "mode": mode,
                        "jetson_planned": metrics["jetson_planned"] >= 0.5,
                        "planner_pricing": pricing,
                        "planner_power_w": power_w,
                    }
                }
            ).observation

        metrics = planner.get_metrics()
        assert planner._planning_event_count == 10
        assert metrics["planner_ms_per_event"] > 0.0
        assert metrics["planner_energy_wh"] == power_w * 10.0 / 60.0



def test_paper_a7_harvest_uses_only_normalized_retargeting_runs():
    from scripts.build_paper_a_figures import a7_norm_cells, is_a7_norm_run

    def run(mode="science", normalize=False, hold=12):
        return {
            "config": {
                "representation_config": {
                    "type": "lewm_cem_eventsat",
                    "mission_mode": mode,
                    "plan_hold": hold,
                    "normalize_attribute_scale": normalize,
                }
            }
        }

    runs = {
        "eventsat_sas_ao_lewm-cem-h12": run("science", normalize=False),
        "eventsat_sas_ao_lewm-cem-h12-mmdl": run("downlink", normalize=False),
        "eventsat_sas_ao_lewm-cem-h12-norm": run("science", normalize=True),
        "eventsat_sas_ao_lewm-cem-h12-mmdl-norm": run("downlink", normalize=True),
        "eventsat_sas_ao_lewm-cem-h12-mmsafe-norm": run("safe", normalize=True),
        "eventsat_sas_ao_lewm-cem-h6-mmdl-norm": run("downlink", normalize=True, hold=6),
    }

    assert not is_a7_norm_run(runs["eventsat_sas_ao_lewm-cem-h12-mmdl"])
    assert [rid for rid, _ in a7_norm_cells(runs)] == [
        "eventsat_sas_ao_lewm-cem-h12-norm",
        "eventsat_sas_ao_lewm-cem-h12-mmdl-norm",
        "eventsat_sas_ao_lewm-cem-h12-mmsafe-norm",
    ]



def test_latent_shaping_downlink_reward_respects_mode_weight():
    from src.eventsat.world_model import MODE_TO_IDX, _WorldModelPlanner

    planner = _WorldModelPlanner(
        {"horizon": 1, "samples": 4, "elites": 1, "cem_iterations": 1},
        method="cem",
    )
    state = _surrogate_state(
        current_mode="communication",
        ground_pass_active=True,
        remaining_pass_duration=1.0,
        obc_data_mb=1.0,
        data_stored_mb=1.0,
    )
    seq = np.asarray(
        [[MODE_TO_IDX["communication"]], [MODE_TO_IDX["charging"]]],
        dtype=np.int64,
    )

    planner.mode_weights = {"downlink_progress": 0.0}
    safe_scores = planner._shaping_scores(state, seq)
    np.testing.assert_allclose(safe_scores, [0.0, 0.0])

    planner.mode_weights = {"downlink_progress": 0.25}
    science_scores = planner._shaping_scores(state, seq)
    assert science_scores[0] > science_scores[1] > 0.0


def test_held_plan_does_not_force_downlink_without_explicit_reflex():
    from src.eventsat.world_model import MODE_TO_IDX, _WorldModelPlanner

    state = _surrogate_state(
        battery_soc=0.8,
        ground_pass_active=True,
        obc_data_mb=5.0,
        data_stored_mb=5.0,
    )

    planner = _WorldModelPlanner(
        {"horizon": 2, "samples": 4, "elites": 1, "cem_iterations": 1},
        method="cem",
    )
    planner._plan_queue = [MODE_TO_IDX["charging"]]

    mode, metrics = planner.select(state)

    assert mode == "charging"
    assert metrics["contact_reflex_enabled"] == 0.0
    assert metrics["contact_reflex_overrides"] == 0.0


def test_explicit_contact_reflex_is_counted_when_enabled():
    from src.eventsat.world_model import MODE_TO_IDX, _WorldModelPlanner

    state = _surrogate_state(
        battery_soc=0.8,
        ground_pass_active=True,
        obc_data_mb=5.0,
        data_stored_mb=5.0,
    )

    planner = _WorldModelPlanner(
        {
            "horizon": 2,
            "samples": 4,
            "elites": 1,
            "cem_iterations": 1,
            "contact_reflex_enabled": True,
        },
        method="cem",
    )
    planner._plan_queue = [MODE_TO_IDX["charging"]]

    mode, metrics = planner.select(state)

    assert mode == "communication"
    assert metrics["contact_reflex_enabled"] == 1.0
    assert metrics["contact_reflex_overrides"] == 1.0
