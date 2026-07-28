"""
Tests for Agent Organization base class and implementations.

Kim et al. (2025) [FVFQ73RF] "Towards a Science of Scaling Agent Systems"
taxonomy: SAS, CentralizedMAS, DecentralizedMAS, IndependentMAS, HybridMAS.
"""

from __future__ import annotations

import pytest

from src.core.organization.base import (
    AgentAction,
    AgentObservation,
    AgentOrganization,
    bind_communication_topology,
    derive_authorized_satellite_links,
    validate_agent_satellite_mapping,
)
from src.core.organization.single_agent_system import SingleAgentSystem
from src.core.organization.centralized_mas import CentralizedMAS
from src.core.organization.decentralized_mas import DecentralizedMAS
from src.core.organization.independent_mas import IndependentMAS
from src.core.organization.hybrid_mas import HybridMAS
from src.core.satellite_env import (
    ConstellationState,
    EnvironmentObservation,
    SatelliteState,
)


def _make_obs(
    satellite_ids: list[str],
    tasks: list[dict] | None = None,
) -> EnvironmentObservation:
    """Minimal EnvironmentObservation with the given satellites (for org tests)."""
    return EnvironmentObservation(
        constellation_state=ConstellationState(
            timestep=0,
            epoch_seconds=0.0,
            satellites={s: SatelliteState(satellite_id=s) for s in satellite_ids},
        ),
        tasks=tasks or [],
    )


class _StaticEnv:
    def __init__(self, satellite_ids: list[str]) -> None:
        self._obs = _make_obs(satellite_ids)
        self.communication_links = "unconfigured"

    def get_observation(self) -> EnvironmentObservation:
        return self._obs

    def configure_communication_links(
        self,
        links: set[tuple[str, str]] | None,
    ) -> None:
        self.communication_links = links


# ======================================================================
# Data structure tests
# ======================================================================


class TestAgentObservation:
    def test_default(self) -> None:
        obs = AgentObservation(agent_id="a1")
        assert obs.agent_id == "a1"
        assert obs.local_state == {}
        assert obs.messages == []


class TestAgentAction:
    def test_default(self) -> None:
        act = AgentAction(agent_id="a1", action="noop")
        assert act.action == "noop"


# ======================================================================
# ABC contract
# ======================================================================


class TestAgentOrganizationABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            AgentOrganization(config={})  # type: ignore[abstract]


# ======================================================================
# SingleAgentSystem (SAS) — Kim et al. 2025 |A|=1
# ======================================================================


class TestSingleAgentSystem:
    def test_single_agent(self) -> None:
        org = SingleAgentSystem(config={})
        org.initialize(constellation_size=5)
        agents = org.get_agents()
        assert len(agents) == 1
        assert agents[0] == "central_agent"

    def test_scopes_cover_full_constellation(self) -> None:
        org = SingleAgentSystem(config={"satellite_prefix": "sat"})
        org.initialize(constellation_size=3)
        assert org.observed_satellites_for_agent("central_agent") == [
            "sat_0",
            "sat_1",
            "sat_2",
        ]
        assert org.satellites_for_agent("central_agent") == [
            "sat_0",
            "sat_1",
            "sat_2",
        ]

    def test_distribute_observation(self) -> None:
        org = SingleAgentSystem(config={})
        org.initialize(constellation_size=3)
        obs = org.distribute_observation({"some": "data"})
        assert "central_agent" in obs
        assert obs["central_agent"].agent_id == "central_agent"

    def test_collect_actions(self) -> None:
        org = SingleAgentSystem(config={})
        org.initialize(constellation_size=2)
        actions = {
            "central_agent": AgentAction(
                agent_id="central_agent",
                action={"sat_0": "fire_thruster", "sat_1": "noop"},
            )
        }
        env_actions = org.collect_actions(actions)
        assert env_actions["sat_0"] == "fire_thruster"


# ======================================================================
# CentralizedMAS — Kim et al. 2025 Centralized MAS (star topology)
# ======================================================================


class TestCentralizedMAS:
    def test_agents_include_manager(self) -> None:
        org = CentralizedMAS(config={})
        org.initialize(constellation_size=3)
        agents = org.get_agents()
        assert "mission_manager" in agents
        assert len(agents) == 4  # 1 manager + 3 local

    def test_distribute_observation_no_prior_directive(self) -> None:
        org = CentralizedMAS(config={})
        org.initialize(constellation_size=1)
        obs = _make_obs(["eventsat_0"])
        result = org.distribute_observation(obs)
        assert "mission_manager" in result
        assert "sat_agent_0" in result
        # Centralized MAS: manager AND local both receive the full observation
        # (full observability + hierarchical directive); local has no directive yet.
        assert result["mission_manager"].local_state["full_observation"] is obs
        assert result["mission_manager"].messages == []
        assert result["sat_agent_0"].local_state["full_observation"] is obs
        assert result["sat_agent_0"].messages == []

    def test_distribute_observation_with_prior_directive(self) -> None:
        org = CentralizedMAS(config={})
        org.initialize(constellation_size=1)
        # Simulate a prior collect_actions that stored a directive
        org._last_manager_directive = {"eventsat_0": {"mode": "charging"}}
        result = org.distribute_observation(_make_obs(["eventsat_0"]))
        # Local agent now receives directive as message
        assert len(result["sat_agent_0"].messages) == 1
        assert result["sat_agent_0"].messages[0]["from"] == "mission_manager"
        assert result["sat_agent_0"].messages[0]["directive"] == {"eventsat_0": {"mode": "charging"}}
        # Manager still has no messages
        assert result["mission_manager"].messages == []

    def test_collect_actions_stores_directive_and_uses_local(self) -> None:
        org = CentralizedMAS(config={})
        org.initialize(constellation_size=1)
        actions = {
            "mission_manager": AgentAction(
                agent_id="mission_manager",
                action={"eventsat_0": {"mode": "charging"}},
            ),
            "sat_agent_0": AgentAction(
                agent_id="sat_agent_0",
                action={"eventsat_0": {"mode": "payload_observe"}},
            ),
        }
        env_actions = org.collect_actions(actions)
        # Local agent's action is used as env action
        assert env_actions == {"eventsat_0": {"mode": "payload_observe"}}
        # Manager's action stored as directive for next step
        assert org._last_manager_directive == {"eventsat_0": {"mode": "charging"}}

    def test_collect_actions_fallback_to_manager(self) -> None:
        org = CentralizedMAS(config={})
        org.initialize(constellation_size=1)
        # Only manager action, no local agent action
        actions = {
            "mission_manager": AgentAction(
                agent_id="mission_manager",
                action={"eventsat_0": {"mode": "safe"}},
            ),
        }
        env_actions = org.collect_actions(actions)
        assert env_actions == {"eventsat_0": {"mode": "safe"}}

    def test_initialize_resets_directive(self) -> None:
        org = CentralizedMAS(config={})
        org.initialize(constellation_size=1)
        org._last_manager_directive = {"some": "directive"}
        org.initialize(constellation_size=1)
        assert org._last_manager_directive is None

    def test_manager_safety_directive_reaches_context_and_vetoes_payload(self) -> None:
        from src.core.decision_procedure.sda_loop import SDALoop

        class CaptureRepresentation:
            def __init__(self) -> None:
                self.context = None

            def encode_observation(self, observation):
                return {"encoded": observation}

            def select_action(self, context):
                self.context = context
                return {"eventsat_0": {"mode": "payload_observe", "priority": 2}}

        representation = CaptureRepresentation()
        loop = SDALoop(config={}, representation=representation)
        observation = AgentObservation(
            agent_id="sat_agent_0",
            local_state={"full_observation": {"state": "nominal"}},
            messages=[
                {
                    "from": "mission_manager",
                    "directive": {"eventsat_0": {"mode": "charging"}},
                }
            ],
            metadata={"organization_role": "local"},
        )

        action, _ = loop.process(observation, memory=None)

        assert action == {
            "eventsat_0": {"mode": "charging", "priority": 2}
        }
        assert representation.context.enrichments["organization_messages"] == observation.messages
        assert representation.context.enrichments["organization_metadata"] == observation.metadata
        assert representation.context.loop_metadata["agent_id"] == "sat_agent_0"

    def test_scopes_manager_observes_all_locals_control_own(self) -> None:
        org = CentralizedMAS(config={"satellite_prefix": "sat"})
        org.initialize(constellation_size=3)
        assert org.observed_satellites_for_agent("mission_manager") == [
            "sat_0",
            "sat_1",
            "sat_2",
        ]
        assert org.satellites_for_agent("mission_manager") == []
        assert org.observed_satellites_for_agent("sat_agent_1") == [
            "sat_0",
            "sat_1",
            "sat_2",
        ]
        assert org.satellites_for_agent("sat_agent_1") == ["sat_1"]


# ======================================================================
# DecentralizedMAS — SSA organisation implementation
# ======================================================================


class TestDecentralizedMAS:
    def test_agents(self) -> None:
        org = DecentralizedMAS(config={})
        org.initialize(constellation_size=4)
        agents = org.get_agents()
        assert len(agents) == 4

    def test_scopes_peers_observe_and_control_only_own_satellite(self) -> None:
        org = DecentralizedMAS(config={"satellite_prefix": "sat"})
        org.initialize(constellation_size=3)
        assert org.observed_satellites_for_agent("sat_agent_1") == ["sat_1"]
        assert org.satellites_for_agent("sat_agent_1") == ["sat_1"]

    def test_distribute_strict_local_copied_observation(self) -> None:
        org = DecentralizedMAS(config={"satellite_prefix": "sat"})
        org.initialize(constellation_size=3)
        source = _make_obs(
            ["sat_0", "sat_1", "sat_2"],
            tasks=[
                {"satellite_id": "sat_0", "task": "own"},
                {"satellite_id": "sat_1", "task": "peer"},
                {"task": "unaddressed"},
            ],
        )
        source.constellation_state.global_info["team_coverage"] = 0.5
        source.constellation_state.satellites["sat_0"].metadata["known"] = ["rso_0"]
        source.events = [
            {"satellite_id": "sat_0", "event": "own"},
            {"satellite_id": "sat_2", "event": "peer"},
            {"event": "unaddressed"},
        ]

        result = org.distribute_observation(source)

        assert set(result.keys()) == {"sat_agent_0", "sat_agent_1", "sat_agent_2"}
        local = result["sat_agent_0"]
        view = local.local_state["full_observation"]
        assert set(view.constellation_state.satellites) == {"sat_0"}
        assert view.constellation_state.global_info == {}
        assert view.tasks == [{"satellite_id": "sat_0", "task": "own"}]
        assert view.events == [{"satellite_id": "sat_0", "event": "own"}]
        assert local.messages == []
        view.constellation_state.satellites["sat_0"].metadata["known"].append("rso_1")
        assert source.constellation_state.satellites["sat_0"].metadata["known"] == [
            "rso_0"
        ]

    def test_logical_topology_is_directed_all_to_all(self) -> None:
        org = DecentralizedMAS(config={"satellite_prefix": "sat"})
        org.initialize(constellation_size=3)
        assert org.logical_communication_edges() == {
            ("sat_agent_0", "sat_agent_1"),
            ("sat_agent_0", "sat_agent_2"),
            ("sat_agent_1", "sat_agent_0"),
            ("sat_agent_1", "sat_agent_2"),
            ("sat_agent_2", "sat_agent_0"),
            ("sat_agent_2", "sat_agent_1"),
        }

    def test_collect_merges_disjoint_owned_actions(self) -> None:
        org = DecentralizedMAS(config={"satellite_prefix": "sat"})
        org.initialize(constellation_size=3)
        actions = {
            "sat_agent_0": AgentAction(
                agent_id="sat_agent_0",
                action={"sat_0": {"mode": "charging"}},
            ),
            "sat_agent_1": AgentAction(
                agent_id="sat_agent_1",
                action={"sat_1": {"mode": "isl_share"}},
            ),
        }
        assert org.collect_actions(actions) == {
            "sat_0": {"mode": "charging"},
            "sat_1": {"mode": "isl_share"},
        }
        assert org.get_metrics() == {}

    def test_collect_rejects_foreign_satellite_action(self) -> None:
        org = DecentralizedMAS(config={"satellite_prefix": "sat"})
        org.initialize(constellation_size=2)
        with pytest.raises(ValueError, match="outside its action scope"):
            org.collect_actions(
                {
                    "sat_agent_0": AgentAction(
                        agent_id="sat_agent_0",
                        action={"sat_1": {"mode": "safe"}},
                    )
                }
            )

    def test_collect_rejects_unknown_agent(self) -> None:
        org = DecentralizedMAS(config={"satellite_prefix": "sat"})
        org.initialize(constellation_size=2)
        with pytest.raises(ValueError, match="Unknown DMAS agents"):
            org.collect_actions(
                {
                    "intruder": AgentAction(
                        agent_id="intruder",
                        action={"sat_0": {"mode": "safe"}},
                    )
                }
            )


# ======================================================================
# IndependentMAS — local per-satellite views, no deconfliction
# ======================================================================


class TestIndependentMAS:
    def test_agents(self) -> None:
        org = IndependentMAS(config={})
        org.initialize(constellation_size=3)
        assert org.get_agents() == ["sat_agent_0", "sat_agent_1", "sat_agent_2"]

    def test_satellite_for_agent_maps_multieventsat_default(self) -> None:
        org = IndependentMAS(config={})
        org.initialize(constellation_size=3)
        assert org.satellite_for_agent("sat_agent_0") == "sat_0"
        assert org.satellite_for_agent("sat_agent_2") == "sat_2"

    def test_satellite_for_agent_allows_scenario_prefix(self) -> None:
        org = IndependentMAS(config={"satellite_prefix": "demo"})
        org.initialize(constellation_size=2)
        assert org.satellite_for_agent("sat_agent_1") == "demo_1"

    def test_scopes_are_local(self) -> None:
        org = IndependentMAS(config={})
        org.initialize(constellation_size=3)
        assert org.observed_satellites_for_agent("sat_agent_2") == ["sat_2"]
        assert org.satellites_for_agent("sat_agent_2") == ["sat_2"]

    def test_distribute_gives_each_agent_only_its_own_satellite(self) -> None:
        env_obs = _make_obs(
            ["sat_0", "sat_1"],
            tasks=[
                {"satellite_id": "sat_0", "target_id": "rso_0", "priority": 3.0},
                {"satellite_id": "sat_1", "target_id": "rso_1", "priority": 2.0},
            ],
        )

        org = IndependentMAS(config={"satellite_prefix": "demo"})
        org.initialize(constellation_size=2)
        result = org.distribute_observation(env_obs)

        local0 = result["sat_agent_0"].local_state["full_observation"]
        assert list(local0.constellation_state.satellites.keys()) == ["sat_0"]
        assert [t["target_id"] for t in local0.tasks] == ["rso_0"]
        assert result["sat_agent_0"].metadata["satellite_id"] == "sat_0"

        local1 = result["sat_agent_1"].local_state["full_observation"]
        assert list(local1.constellation_state.satellites.keys()) == ["sat_1"]
        assert [t["target_id"] for t in local1.tasks] == ["rso_1"]
        assert result["sat_agent_1"].messages == []

    def test_distribute_falls_back_to_observation_order_for_unknown_prefix(self) -> None:
        org = IndependentMAS(config={})
        org.initialize(constellation_size=2)
        result = org.distribute_observation(_make_obs(["demo_0", "demo_1"]))
        view = result["sat_agent_0"].local_state["full_observation"].constellation_state.satellites
        assert set(view) == {"demo_0"}
        assert result["sat_agent_0"].metadata["satellite_id"] == "demo_0"

    def test_collect_merges_without_deconfliction(self) -> None:
        org = IndependentMAS(config={})
        org.initialize(constellation_size=2)
        actions = {
            "sat_agent_0": AgentAction(
                agent_id="sat_agent_0", action={"sat_0": {"target_id": "rso_0"}}
            ),
            "sat_agent_1": AgentAction(
                agent_id="sat_agent_1", action={"sat_1": {"target_id": "rso_0"}}
            ),
        }
        env_actions = org.collect_actions(actions)
        assert env_actions["sat_0"] == {"target_id": "rso_0"}
        assert env_actions["sat_1"] == {"target_id": "rso_0"}


# ======================================================================
# HybridMAS — clustered: coordinate within, independent across
# ======================================================================


class TestHybridMAS:
    def test_one_cluster_head_per_cluster(self) -> None:
        org = HybridMAS(config={"num_clusters": 2})
        org.initialize(constellation_size=5)
        # 5 satellites into 2 contiguous near-equal clusters -> 2 head agents.
        assert org.get_agents() == ["cluster_agent_0", "cluster_agent_1"]

    def test_explicit_clusters_partition(self) -> None:
        org = HybridMAS(config={"clusters": [[0, 1, 2], [3, 4]]})
        org.initialize(constellation_size=5)
        assert len(org.get_agents()) == 2

    def test_scopes_are_cluster_local(self) -> None:
        org = HybridMAS(config={"num_clusters": 2, "satellite_prefix": "sat"})
        org.initialize(constellation_size=5)
        assert org.observed_satellites_for_agent("cluster_agent_0") == [
            "sat_0",
            "sat_1",
            "sat_2",
        ]
        assert org.satellites_for_agent("cluster_agent_1") == ["sat_3", "sat_4"]

    def test_distribute_gives_each_head_only_its_cluster(self) -> None:
        from src.core.satellite_env import (
            ConstellationState,
            EnvironmentObservation,
            SatelliteState,
        )

        env_obs = EnvironmentObservation(
            constellation_state=ConstellationState(
                timestep=0,
                epoch_seconds=0.0,
                satellites={
                    f"sat_{i}": SatelliteState(satellite_id=f"sat_{i}")
                    for i in range(3)
                },
            ),
            tasks=[
                {"satellite_id": f"sat_{i}", "target_id": f"rso_{i}", "priority": 1.0}
                for i in range(3)
            ],
        )
        org = HybridMAS(config={"num_clusters": 2})
        org.initialize(constellation_size=3)
        result = org.distribute_observation(env_obs)
        # Cluster 0 = {sat_0, sat_1}, cluster 1 = {sat_2}.
        head0 = result["cluster_agent_0"].local_state["full_observation"]
        assert set(head0.constellation_state.satellites.keys()) == {
            "sat_0",
            "sat_1",
        }
        head1 = result["cluster_agent_1"].local_state["full_observation"]
        assert set(head1.constellation_state.satellites.keys()) == {"sat_2"}

    def test_collect_merges_clusters_and_reports_localised_cost(self) -> None:
        org = HybridMAS(config={"num_clusters": 2})
        org.initialize(constellation_size=3)
        actions = {
            "cluster_agent_0": AgentAction(
                agent_id="cluster_agent_0",
                action={"sat_0": {"target_id": "rso_0"},
                        "sat_1": {"target_id": "rso_1"}},
            ),
            "cluster_agent_1": AgentAction(
                agent_id="cluster_agent_1",
                action={"sat_2": {"target_id": "rso_0"}},
            ),
        }
        merged = org.collect_actions(actions)
        assert set(merged.keys()) == {"sat_0", "sat_1", "sat_2"}
        # Localised cost: clusters of size 2 and 1 -> 2*1 + 1*0 = 2 messages.
        assert org.get_metrics()["coordination_messages"] == 2.0

    def test_num_clusters_spans_the_spectrum(self) -> None:
        # One cluster -> SAS-like all-to-all cost; singletons -> IMAS-like (zero).
        one = HybridMAS(config={"num_clusters": 1})
        one.initialize(constellation_size=4)
        one.collect_actions({})
        assert one.get_metrics()["coordination_messages"] == 12.0  # 4*3
        singletons = HybridMAS(config={"num_clusters": 4})
        singletons.initialize(constellation_size=4)
        singletons.collect_actions({})
        assert singletons.get_metrics()["coordination_messages"] == 0.0


def test_validate_agent_satellite_mapping_accepts_plural_scopes() -> None:
    org = SingleAgentSystem(config={"satellite_prefix": "sat"})
    org.initialize(constellation_size=3)

    validate_agent_satellite_mapping(org, _StaticEnv(["sat_0", "sat_1", "sat_2"]), 3, "ssa")


def test_validate_agent_satellite_mapping_rejects_overlapping_controls() -> None:
    org = IndependentMAS(config={"satellite_ids": ["sat_0", "sat_0"]})
    org.initialize(constellation_size=2)

    with pytest.raises(ValueError, match="disjoint"):
        validate_agent_satellite_mapping(org, _StaticEnv(["sat_0", "sat_1"]), 2, "ssa")


def test_validate_agent_satellite_mapping_rejects_incomplete_controls() -> None:
    org = HybridMAS(config={"clusters": [[0]], "satellite_prefix": "sat"})
    org.initialize(constellation_size=2)

    with pytest.raises(ValueError, match="cover"):
        validate_agent_satellite_mapping(org, _StaticEnv(["sat_0", "sat_1"]), 2, "ssa")


def test_undeclared_topology_preserves_unbound_environment() -> None:
    org = IndependentMAS(config={"satellite_prefix": "sat"})
    org.initialize(constellation_size=2)
    env = _StaticEnv(["sat_0", "sat_1"])

    assert derive_authorized_satellite_links(org, env) is None
    assert bind_communication_topology(org, env) is None
    assert env.communication_links is None


def test_dmas_topology_maps_agents_to_physical_endpoints() -> None:
    org = DecentralizedMAS(config={"satellite_prefix": "sat"})
    org.initialize(constellation_size=3)
    env = _StaticEnv(["sat_0", "sat_1", "sat_2"])

    expected = {
        (src, dst)
        for src in ("sat_0", "sat_1", "sat_2")
        for dst in ("sat_0", "sat_1", "sat_2")
        if src != dst
    }
    assert derive_authorized_satellite_links(org, env) == expected
    assert bind_communication_topology(org, env) == expected
    assert env.communication_links == expected
