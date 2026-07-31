"""Decentralized peer organisation with local views and all-to-all links.

Kim et al. (2025) [FVFQ73RF] supplies the equal-peer logical topology. In SSA,
physical knowledge exchange is performed by the environment when a peer selects
``isl_share``; this class neither duplicates that data plane nor runs a
full-plan plurality protocol.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.core.organization.base import (
    AgentAction,
    AgentObservation,
    AgentOrganization,
    satellite_id_for_index,
    satellite_ids_for_constellation,
)
from src.core.satellite_env import scope_observation


class DecentralizedMAS(AgentOrganization):
    """One equal reasoning peer per satellite with disjoint action ownership."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._agent_ids: List[str] = []
        self._satellite_ids: List[str] = []

    def initialize(self, constellation_size: int, **kwargs: Any) -> None:
        self._agent_ids = [f"sat_agent_{i}" for i in range(constellation_size)]
        self._satellite_ids = satellite_ids_for_constellation(
            self.config, constellation_size
        )

    def satellite_for_agent(self, agent_id: str) -> str:
        return satellite_id_for_index(self.config, self._agent_index(agent_id))

    def satellites_for_agent(self, agent_id: str) -> List[str]:
        return [self.satellite_for_agent(agent_id)]

    def observed_satellites_for_agent(self, agent_id: str) -> List[str]:
        return [self.satellite_for_agent(agent_id)]

    def logical_communication_edges(self) -> set[tuple[str, str]]:
        """Return the authoritative all-to-all directed peer graph."""
        return {
            (src, dst)
            for src in self._agent_ids
            for dst in self._agent_ids
            if src != dst
        }

    def distribute_observation(
        self,
        env_observation: Any,
    ) -> Dict[str, AgentObservation]:
        """Project one copied, strict local environment view per peer."""
        result: Dict[str, AgentObservation] = {}
        for agent_id in self._agent_ids:
            sat_id = self.satellite_for_agent(agent_id)
            result[agent_id] = AgentObservation(
                agent_id=agent_id,
                local_state={
                    "full_observation": scope_observation(
                        env_observation,
                        [sat_id],
                        copy_satellite_states=True,
                        include_global_info=False,
                        strict_addressed_records=True,
                    )
                },
                messages=[],
                metadata={"satellite_id": sat_id, "organization_role": "peer"},
            )
        return result

    def collect_actions(
        self,
        agent_actions: Dict[str, AgentAction],
    ) -> Dict[str, Any]:
        """Validate peer ownership and compose disjoint local actions."""
        unknown_agents = sorted(set(agent_actions) - set(self._agent_ids))
        if unknown_agents:
            raise ValueError(f"Unknown DMAS agents: {unknown_agents}")

        merged: Dict[str, Any] = {}
        for agent_id in self._agent_ids:
            agent_action = agent_actions.get(agent_id)
            if agent_action is None or not isinstance(agent_action.action, dict):
                continue
            allowed = set(self.satellites_for_agent(agent_id))
            foreign = sorted(set(agent_action.action) - allowed)
            if foreign:
                raise ValueError(
                    f"DMAS peer '{agent_id}' emitted satellites {foreign} "
                    f"outside its action scope {sorted(allowed)}"
                )
            overlap = sorted(set(agent_action.action) & set(merged))
            if overlap:
                raise ValueError(
                    f"DMAS peer actions overlap on satellites: {overlap}"
                )
            merged.update(agent_action.action)
        return merged

    def get_agents(self) -> List[str]:
        return list(self._agent_ids)

    @staticmethod
    def _agent_index(agent_id: str) -> int:
        prefix = "sat_agent_"
        if not agent_id.startswith(prefix):
            raise ValueError(
                f"DecentralizedMAS expects 'sat_agent_i' agent ids, got '{agent_id}'"
            )
        try:
            return int(agent_id[len(prefix):])
        except ValueError as exc:
            raise ValueError(
                f"DecentralizedMAS expects 'sat_agent_i' agent ids, got '{agent_id}'"
            ) from exc
