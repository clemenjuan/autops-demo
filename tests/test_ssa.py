"""Tests for SSA physics primitives."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.core.config_loader import ExperimentConfig, apply_overrides, load_config
from src.core.experiment_runner import ExperimentRunner
from src.core.satellite_env import scope_observation
from src.orbital.isl import effective_data_rate_bps, is_isl_feasible, vector_range_km
from src.ssa.env import SSAEnvironment, SSA_MODES
from src.ssa.metrics import SSAMetricsCollector
from src.ssa.rewards import SSARewardFunction
from src.ssa.symbolic import RuleBasedSSA
from src.ssa.targets import (
    detect_targets_in_fov,
    diffraction_limited_range_km,
    generate_sso_catalog,
    propagate_rso_position_km,
)


def test_diffraction_limited_range_matches_autops_rl_optic_payload() -> None:
    assert diffraction_limited_range_km() == pytest.approx(52.7, rel=2e-3)


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


def test_anti_nadir_fov_returns_multiple_targets_without_target_action() -> None:
    observer = (7000.0, 0.0, 0.0)
    angle_rad = math.radians(3.0)
    target_positions = {
        "rso_a": (7020.0, 0.0, 0.0),
        "rso_b": (7000.0 + 20.0 * math.cos(angle_rad), 20.0 * math.sin(angle_rad), 0.0),
        "too_wide": (7000.0 + 20.0 * math.cos(math.radians(8.0)), 20.0 * math.sin(math.radians(8.0)), 0.0),
        "too_far": (7060.0, 0.0, 0.0),
    }

    detections = detect_targets_in_fov(observer, target_positions)

    assert [d.object_id for d in detections] == ["rso_a", "rso_b"]
    assert all(d.quality > 0.0 for d in detections)


def test_synthetic_sso_catalog_is_seeded_and_fixed_size() -> None:
    first = generate_sso_catalog(5, seed=7)
    second = generate_sso_catalog(5, seed=7)
    third = generate_sso_catalog(5, seed=8)

    assert len(first) == 5
    assert first == second
    assert first != third
    assert all(6971.0 <= target.semi_major_axis_km <= 7271.0 for target in first)


def test_target_two_body_propagation_returns_finite_position() -> None:
    target = generate_sso_catalog(1, seed=3)[0]
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
            "max_range_km": 52.7,
        },
        "ground_station": {"always_visible": True},
        "reward_config": {"local_weight": 1.0, "team_weight": 0.0, "collective_weight": 1.0},
    }


def test_ssa_observe_updates_fixed_binary_detection_matrix() -> None:
    env = SSAEnvironment(_ssa_env_config())
    env.reset(seed=1)

    result = env.step({"sat_0": {"mode": "payload_observe"}, "sat_1": {"mode": "charging"}})

    assert env.detection_matrix == [[1, 1], [0, 0]]
    assert result.info["ssa_onboard_coverage"] == pytest.approx(1.0)
    assert result.info["ssa_delivered_coverage"] == 0.0
    row = result.observation.constellation_state.satellites["sat_0"].metadata["ssa_detection_row"]
    assert row == [1, 1]


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


def test_onboard_keeps_best_estimate_while_ground_archives_all_records() -> None:
    env = SSAEnvironment(_ssa_env_config())
    env.reset(seed=1)

    env.step({"sat_0": {"mode": "payload_observe"}, "sat_1": {"mode": "charging"}})
    env.step({"sat_0": {"mode": "payload_observe"}, "sat_1": {"mode": "charging"}})
    env.step({"sat_0": {"mode": "communication"}, "sat_1": {"mode": "charging"}})

    assert set(env.onboard_estimates["sat_0"]) == {"rso_0", "rso_1"}
    assert len(env.onboard_estimates["sat_0"]) == 2
    assert len(env.ground_archive["rso_0"]) == 2
    assert len(env.ground_archive["rso_1"]) == 2


def test_isl_merge_ors_matrix_and_keeps_higher_quality_estimate() -> None:
    env = SSAEnvironment(_ssa_env_config())
    env.reset(seed=1)
    env.step({"sat_0": {"mode": "payload_observe"}, "sat_1": {"mode": "charging"}})

    env.step({"sat_0": {"mode": "isl_share"}, "sat_1": {"mode": "charging"}})

    assert env.detection_matrix == [[1, 1], [1, 1]]
    assert set(env.onboard_estimates["sat_1"]) == {"rso_0", "rso_1"}
    assert env.get_metrics()["isl_connectivity"] > 0.0


def test_delivered_utility_credits_only_downlinked_objects() -> None:
    env = SSAEnvironment(_ssa_env_config())
    env.reset(seed=1)

    observe = env.step({"sat_0": {"mode": "payload_observe"}, "sat_1": {"mode": "charging"}})
    downlink = env.step({"sat_0": {"mode": "communication"}, "sat_1": {"mode": "charging"}})

    assert observe.info["ssa_delivered_coverage"] == 0.0
    assert downlink.info["ssa_delivered_coverage"] == pytest.approx(1.0)
    assert downlink.rewards["sat_0"] > observe.rewards["sat_0"]


def test_ssa_reward_collective_negative_uses_delivered_coverage() -> None:
    rf = SSARewardFunction({"collective_weight": 1.0, "mission_scale": 2.0})

    empty = rf.compute_rewards({"sat_0": 0.0}, {"_global": {"delivered_coverage": 0.0}})
    delivered = rf.compute_rewards({"sat_0": 0.0}, {"_global": {"delivered_coverage": 1.0}})

    assert empty["sat_0"] == pytest.approx(-2.0)
    assert delivered["sat_0"] == pytest.approx(0.0)


def test_ssa_metrics_adds_coverage_duplicate_connectivity_and_m10() -> None:
    collector = SSAMetricsCollector({
        "max_steps": 2,
        "step_duration_s": 60,
        "constellation_size": 2,
        "baseline_utility_n1": 0.5,
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


def test_rule_based_ssa_deconflicts_full_scope_but_local_scope_observes() -> None:
    env = SSAEnvironment({
        **_ssa_env_config(n=2),
        "satellite_positions_km": {
            "sat_0": [0.0, 0.0, 500.0],
            "sat_1": [0.4, 0.0, 500.0],
        },
        "targets": {
            "fixed_positions_km": {"rso_0": [0.0, 0.0, 530.0]},
            "fov_half_angle_deg": 5.0,
            "max_range_km": 52.7,
        },
    })
    observation = env.reset(seed=1)
    central = RuleBasedSSA({})

    central_action = central.select_action(type("Context", (), {"state": central.encode_observation(observation)}))

    assert central_action["sat_0"]["mode"] == "payload_observe"
    assert central_action["sat_1"]["mode"] != "payload_observe"

    step_result = env.step(central_action)
    local = RuleBasedSSA({"satellite_id": "sat_1"})
    scoped = scope_observation(step_result.observation, ["sat_1"])
    local_state = local.encode_observation(scoped)

    assert local.select_action(type("Context", (), {"state": local_state}))["sat_1"]["mode"] == "payload_observe"


def test_ssa_symbolic_runner_sas_deconflicts_imas_duplicates(tmp_path) -> None:
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

    sas = ExperimentRunner(config=sas_cfg).run()
    imas = ExperimentRunner(config=imas_cfg).run()

    sas_dupes = sas["experiment_statistics"].mean["duplicate_observation_rate"]
    imas_dupes = imas["experiment_statistics"].mean["duplicate_observation_rate"]
    assert sas_dupes == 0.0
    assert imas_dupes > 0.0


def test_ssa_ground_paradigms_reject_distributed_organizations() -> None:
    with pytest.raises(ValueError, match="SSA ground paradigms"):
        ExperimentConfig(
            experiment_id="ssa_invalid_ground_imas",
            agent_organization="independent_mas",
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


def test_config_rejects_divergent_max_steps_sources() -> None:
    with pytest.raises(ValueError, match="max_steps must be declared once"):
        ExperimentConfig(
            experiment_id="ssa_bad_horizon",
            agent_organization="sas",
            decision_procedure="sda",
            representation="rl",
            behaviour="emergent",
            operations_paradigm="autonomous_onboard",
            representation_config={
                "type": "subsymbolic_eventsat",
                "rl_mock": True,
                "max_steps": 10080,
            },
            behaviour_config={"mode": "emergent", "mechanism": "ppo"},
            environment={
                "constellation_size": 3,
                "timestep_seconds": 60,
                "max_steps": 10080,
                "scenario": "ssa",
                "scenario_config": {"scenario_file": "configs/scenarios/ssa.yaml"},
            },
            num_episodes=1,
            max_steps=100,
        )


def test_apply_overrides_keeps_representation_max_steps_in_sync() -> None:
    cfg = apply_overrides(
        load_config("configs/experiments/ssa_sas_ao_rl_n3.yaml"),
        steps=2,
    )

    assert cfg.max_steps == 2
    assert cfg.environment.max_steps == 2
    assert cfg.representation_config["max_steps"] == 2


# --- SSA RL contract (registry-driven adapter + representation) --------------

def _obs_with(satellite_id: str, metadata: dict) -> "EnvironmentObservation":
    from src.core.satellite_env import (
        ConstellationState,
        EnvironmentObservation,
        SatelliteState,
    )

    sat = SatelliteState(
        satellite_id=satellite_id,
        resources={"battery_soc": 0.6, "obc_data_mb": 100.0, "data_downlinked_mb": 5.0},
        status="charging",
        metadata=metadata,
    )
    return EnvironmentObservation(
        constellation_state=ConstellationState(
            timestep=3, epoch_seconds=180.0, satellites={satellite_id: sat}
        )
    )


def _multi_obs(metadata_by_sat: dict[str, dict]) -> "EnvironmentObservation":
    from src.core.satellite_env import (
        ConstellationState,
        EnvironmentObservation,
        SatelliteState,
    )

    sats = {
        sat_id: SatelliteState(
            satellite_id=sat_id,
            resources={"battery_soc": 0.6, "obc_data_mb": 100.0, "data_downlinked_mb": 5.0},
            status=metadata.get("status", "charging"),
            metadata=metadata,
        )
        for sat_id, metadata in metadata_by_sat.items()
    }
    return EnvironmentObservation(
        constellation_state=ConstellationState(
            timestep=3, epoch_seconds=180.0, satellites=sats
        )
    )


def test_ssa_rl_spec_declares_eight_modes_and_extended_obs() -> None:
    from src.rl.space_adapters import get_rl_spec

    spec = get_rl_spec("ssa")
    assert spec is not None
    assert "isl_share" in spec.mode_list
    assert len(spec.mode_list) == 8
    assert spec.action_dims == [8, 2, 2]
    assert spec.obs_dim == 29


def test_eventsat_rl_spec_unchanged() -> None:
    from src.rl.space_adapters import get_rl_spec

    spec = get_rl_spec("eventsat")
    assert len(spec.mode_list) == 7 and "isl_share" not in spec.mode_list
    assert spec.action_dims == [7, 2, 2]
    assert spec.obs_dim == 25
    # multieventsat reuses the same RL contract.
    assert get_rl_spec("multieventsat") is spec


def test_ssa_adapter_action_and_observation_space() -> None:
    pytest.importorskip("gymnasium")
    from src.rl.space_adapters import make_space_adapter

    adapter = make_space_adapter("ssa", config={"satellite_id": "sat_0"})
    assert list(adapter.action_space.nvec) == [8, 2, 2]
    assert adapter.observation_space.shape == (29,)


def test_rl_id_defaults_use_legacy_satellite_when_act_ids_absent() -> None:
    pytest.importorskip("gymnasium")
    from src.eventsat.rl import SubsymbolicEventSat
    from src.rl.space_adapters import make_space_adapter

    adapter = make_space_adapter("eventsat", config={"satellite_id": "eventsat_7"})
    rep = SubsymbolicEventSat(
        config={"rl_mock": True, "satellite_id": "eventsat_7", "scenario": "eventsat"}
    )

    assert adapter.act_ids == ["eventsat_7"]
    assert adapter.observe_ids == ["eventsat_7"]
    assert rep._act_ids == ["eventsat_7"]
    assert rep._observe_ids == ["eventsat_7"]


def test_rl_id_config_preserves_explicit_empty_act_ids() -> None:
    pytest.importorskip("gymnasium")
    from src.core.decision_procedure.context import DecisionContext
    from src.eventsat.rl import SubsymbolicEventSat
    from src.rl.space_adapters import make_space_adapter

    config = {
        "rl_mock": True,
        "scenario": "ssa",
        "satellite_id": "sat_0",
        "act_ids": [],
        "observe_ids": ["sat_0"],
    }
    adapter = make_space_adapter("ssa", config=config)
    rep = SubsymbolicEventSat(config=config)
    obs = _obs_with("sat_0", {})

    assert adapter.act_ids == []
    assert adapter.observe_ids == ["sat_0"]
    assert list(adapter.action_space.nvec) == [1]
    assert adapter.decode_action([0]) == {}
    assert rep._act_ids == []
    assert rep._observe_ids == ["sat_0"]
    assert rep._action_dims == [1]

    state = rep.encode_observation(obs)
    action = rep.select_action(
        DecisionContext(
            state=state,
            loop_type="sda",
            memory=None,
            enrichments={},
            loop_metadata={},
        )
    )
    assert action == {}
    assert "no controlled satellites" in (rep.get_rationale() or "")


def test_ssa_joint_adapter_stacks_obs_and_decodes_all_controlled_sats() -> None:
    pytest.importorskip("gymnasium")
    from src.rl.space_adapters import SSA_MODE_LIST, make_space_adapter

    adapter = make_space_adapter(
        "ssa",
        config={
            "observe_ids": ["sat_0", "sat_1", "sat_2"],
            "act_ids": ["sat_0", "sat_1", "sat_2"],
        },
    )
    assert adapter.observation_space.shape == (87,)
    assert list(adapter.action_space.nvec) == [8, 2, 2] * 3

    obs = _multi_obs({
        "sat_0": {"ssa_detection_row": [1, 0]},
        "sat_1": {"ssa_detection_row": [0, 1]},
        "sat_2": {"ssa_detection_row": [1, 1]},
    })
    vec = adapter.encode_observation(obs)
    assert vec.shape == (87,)

    decoded = adapter.decode_action([
        SSA_MODE_LIST.index("isl_share"), 0, 0,
        SSA_MODE_LIST.index("communication"), 1, 0,
        SSA_MODE_LIST.index("payload_observe"), 0, 1,
    ])
    assert set(decoded) == {"sat_0", "sat_1", "sat_2"}
    assert decoded["sat_0"]["mode"] == "isl_share"
    assert decoded["sat_1"]["mode"] == "communication"
    assert decoded["sat_2"]["mode"] == "payload_observe"


def test_ssa_adapter_decodes_isl_share() -> None:
    pytest.importorskip("gymnasium")
    from src.rl.space_adapters import SSA_MODE_LIST, make_space_adapter

    adapter = make_space_adapter("ssa", config={"satellite_id": "sat_0"})
    idx = SSA_MODE_LIST.index("isl_share")
    decoded = adapter.decode_action([idx, 0, 0])
    assert decoded["sat_0"]["mode"] == "isl_share"


def test_ssa_adapter_encodes_coordination_features() -> None:
    pytest.importorskip("gymnasium")
    from src.rl.space_adapters import make_space_adapter

    adapter = make_space_adapter("ssa", config={"satellite_id": "sat_0"})
    obs = _obs_with(
        "sat_0",
        {
            "storage_capacity_mb": 4096.0,
            "ssa_onboard_coverage": 0.4,
            "ssa_delivered_coverage": 0.2,
            "visible_rso_count": 3,
            "ssa_detection_row": [1, 0, 1, 0],
        },
    )
    vec = adapter.encode_observation(obs)
    assert vec.shape == (29,)
    # SSA features occupy the 4 slots after the 25D EventSat base.
    assert vec[25] == pytest.approx(0.4)  # onboard coverage
    assert vec[26] == pytest.approx(0.2)  # delivered coverage
    assert vec[27] == pytest.approx(0.3)  # visible RSO count 3/10
    assert vec[28] == pytest.approx(0.5)  # own known fraction 2/4


def test_ssa_adapter_encodes_peer_message_modes() -> None:
    pytest.importorskip("gymnasium")
    from src.core.organization.base import AgentObservation
    from src.rl.space_adapters import SSA_MODE_LIST, make_space_adapter

    adapter = make_space_adapter(
        "ssa",
        config={
            "observe_ids": ["sat_0", "sat_1", "sat_2"],
            "act_ids": ["sat_0"],
            "include_peer_messages": True,
        },
    )
    obs = _multi_obs({
        "sat_0": {},
        "sat_1": {},
        "sat_2": {},
    })
    agent_obs = AgentObservation(
        agent_id="sat_agent_0",
        local_state={"full_observation": obs},
        messages=[
            {"from": "sat_agent_1", "proposal": {"sat_1": {"mode": "isl_share"}}},
            {"from": "sat_agent_2", "proposal": {"sat_2": {"mode": "communication"}}},
        ],
    )

    vec = adapter.encode_observation(agent_obs)
    assert vec.shape == (87 + 24,)
    offset = 87
    assert vec[offset + 1 * 8 + SSA_MODE_LIST.index("isl_share")] == 1.0
    assert vec[offset + 2 * 8 + SSA_MODE_LIST.index("communication")] == 1.0


def test_subsymbolic_sda_loop_preserves_peer_message_modes() -> None:
    from src.core.decision_procedure.sda_loop import SDALoop
    from src.core.organization.base import AgentObservation
    from src.eventsat.rl import SubsymbolicEventSat
    from src.rl.space_adapters import SSA_MODE_LIST

    rep = SubsymbolicEventSat(
        config={
            "rl_mock": True,
            "scenario": "ssa",
            "observe_ids": ["sat_0", "sat_1", "sat_2"],
            "act_ids": ["sat_0"],
            "include_peer_messages": True,
        }
    )
    loop = SDALoop(config={}, representation=rep)
    obs = _multi_obs({
        "sat_0": {},
        "sat_1": {},
        "sat_2": {},
    })
    agent_obs = AgentObservation(
        agent_id="sat_agent_0",
        local_state={"full_observation": obs},
        messages=[
            {"from": "sat_agent_1", "proposal": {"sat_1": {"mode": "isl_share"}}},
            {"from": "sat_agent_2", "proposal": {"sat_2": {"mode": "communication"}}},
        ],
    )

    loop.process(agent_obs, memory=None)
    step_data = rep.get_last_step_data()
    assert step_data is not None
    vec = step_data["obs_vec"]
    offset = 87
    assert vec.shape == (87 + 24,)
    assert vec[offset + 1 * 8 + SSA_MODE_LIST.index("isl_share")] == 1.0
    assert vec[offset + 2 * 8 + SSA_MODE_LIST.index("communication")] == 1.0


def test_eventsat_encoder_parity_adapter_vs_representation() -> None:
    """Training (adapter) and inference (representation) must vectorise identically."""
    pytest.importorskip("gymnasium")
    pytest.importorskip("torch")
    from src.eventsat.rl import SubsymbolicEventSat
    from src.rl.space_adapters import make_space_adapter

    metadata = {
        "storage_capacity_mb": 4096.0,
        "in_sunlight": True,
        "ground_pass_active": False,
        "orbital_phase": 0.3,
        "uncompressed_observations": 2,
        "undetected_observations": 1,
    }
    obs = _obs_with("eventsat_0", metadata)

    adapter = make_space_adapter("eventsat", config={"satellite_id": "eventsat_0"})
    rep = SubsymbolicEventSat(
        config={"rl_mock": True, "satellite_id": "eventsat_0", "scenario": "eventsat"}
    )
    adapter_vec = adapter.encode_observation(obs)
    rep_vec = rep.encode_observation(obs)["_obs_vector"]
    assert adapter_vec.shape == rep_vec.shape == (25,)
    assert np.allclose(adapter_vec, rep_vec)


def test_ssa_representation_uses_eight_mode_contract() -> None:
    pytest.importorskip("torch")
    from src.eventsat.rl import SubsymbolicEventSat

    rep = SubsymbolicEventSat(
        config={"rl_mock": True, "satellite_id": "sat_0", "scenario": "ssa"}
    )
    assert len(rep._mode_list) == 8 and "isl_share" in rep._mode_list
    assert rep._action_dims == [8, 2, 2]
    assert rep._obs_dim == 29


def test_ssa_representation_joint_contract_scales_dims() -> None:
    pytest.importorskip("torch")
    from src.eventsat.rl import SubsymbolicEventSat

    rep = SubsymbolicEventSat(
        config={
            "rl_mock": True,
            "scenario": "ssa",
            "observe_ids": ["sat_0", "sat_1", "sat_2"],
            "act_ids": ["sat_0", "sat_1", "sat_2"],
        }
    )
    assert rep._action_dims == [8, 2, 2] * 3
    assert rep._obs_dim == 87


def test_ssa_sas_rllib_env_uses_joint_action_space_and_reward_sum() -> None:
    pytest.importorskip("gymnasium")
    from src.core.config_loader import apply_overrides, load_config
    from src.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

    cfg = apply_overrides(
        load_config("configs/experiments/ssa_sas_ao_rl_n3.yaml"),
        episodes=1,
        steps=2,
    )
    env = AUTOPSRLLibMultiAgentEnv({"experiment_config": cfg.model_dump()})

    assert env.possible_agents == ["central_agent"]
    assert env.observation_spaces["central_agent"].shape == (87,)
    assert list(env.action_spaces["central_agent"].nvec) == [8, 2, 2] * 3
    assert env._resolve_agent_reward(
        "central_agent", {"sat_0": 1.0, "sat_1": 2.0, "sat_2": 3.0}
    ) == pytest.approx(6.0)


def test_dmas_collect_actions_merges_disjoint_rl_proposals_and_keeps_metrics() -> None:
    from src.core.organization.base import AgentAction
    from src.core.organization.decentralized_mas import DecentralizedMAS

    org = DecentralizedMAS(config={"satellite_prefix": "sat"})
    org.initialize(constellation_size=3)
    merged = org.collect_actions({
        "sat_agent_0": AgentAction("sat_agent_0", {"sat_0": {"mode": "charging"}}),
        "sat_agent_1": AgentAction("sat_agent_1", {"sat_1": {"mode": "isl_share"}}),
        "sat_agent_2": AgentAction("sat_agent_2", {"sat_2": {"mode": "communication"}}),
    })

    assert set(merged) == {"sat_0", "sat_1", "sat_2"}
    assert org.get_metrics()["coordination_messages"] == 6.0
    assert org.get_metrics()["consensus_rounds"] == 1.0


def test_ssa_config_generator_sets_hmas_policy_and_dmas_messages() -> None:
    from scripts.generate_ssa_configs import build_matrix

    configs = build_matrix()
    assert (
        configs["ssa_hmas_ao_rl_n5"]["behaviour_config"]["policy_sharing"]["mode"]
        == "independent_per_agent"
    )
    assert (
        configs["ssa_dmas_ao_rl_n3"]["representation_config"]["include_peer_messages"]
        is True
    )
    cfg = configs["ssa_sas_ao_rl_n3"]
    assert cfg["max_steps"] == 100
    assert cfg["environment"]["max_steps"] == 100
    assert cfg["representation_config"]["max_steps"] == 100
