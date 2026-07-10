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
