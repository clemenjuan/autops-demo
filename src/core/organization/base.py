"""
Agent Organization — Abstract Base Class.

Defines abstract coordination patterns between agents in a constellation.
Controls how observations are distributed to individual agents and how
their actions are aggregated before being sent to the environment.

Full Kim et al. (2025) [FVFQ73RF] taxonomy — "Towards a Science of Scaling
Agent Systems":

Implementations:
- ``SingleAgentSystem``:  |A|=1, single agent controls entire constellation.
- ``CentralizedMAS``:     Orchestrator + local satellite agents, star topology.
- ``DecentralizedMAS``:   Peer-to-peer multi-agent with all-to-all topology.
- ``IndependentMAS``:     Multiple agents with no inter-agent communication.
- ``HybridMAS``:          Heterogeneous mixed-topology multi-agent organization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentObservation:
    """Observation tailored for a single agent.

    Attributes:
        agent_id: Unique identifier for this agent.
        local_state: State information visible to this agent.
        messages: Messages received from other agents (if any).
        metadata: Additional context.
    """

    agent_id: str
    local_state: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentAction:
    """Action produced by a single agent.

    Attributes:
        agent_id: Identifier of the acting agent.
        action: The action payload (type depends on scenario).
        messages: Messages to send to other agents (if any).
        metadata: Additional context / diagnostics.
    """

    agent_id: str
    action: Any = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentOrganization(ABC):
    """Abstract base class for agent coordination patterns.

    The organization layer sits between the environment and the individual
    decision loops. It defines *who sees what* and *how individual actions
    compose* into the environment action dictionary.

    Attributes:
        config: Organization-specific configuration section from YAML.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise the organization.

        Args:
            config: Configuration dictionary for this organization type.
        """
        self.config = config

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self, constellation_size: int, **kwargs: Any) -> None:
        """Set up the organization for a given constellation size.

        Called once at the beginning of each episode after environment reset.

        Args:
            constellation_size: Number of satellites in the constellation.
            **kwargs: Additional scenario-specific initialization parameters.
        """
        ...

    @abstractmethod
    def distribute_observation(
        self,
        env_observation: Any,
    ) -> Dict[str, AgentObservation]:
        """Map a global environment observation to per-agent observations.

        Args:
            env_observation: The full :class:`EnvironmentObservation` from
                the environment.

        Returns:
            Mapping of agent_id → :class:`AgentObservation`.
        """
        ...

    @abstractmethod
    def collect_actions(
        self,
        agent_actions: Dict[str, AgentAction],
    ) -> Dict[str, Any]:
        """Aggregate individual agent actions into an environment action dict.

        Args:
            agent_actions: Mapping of agent_id → :class:`AgentAction`.

        Returns:
            Action dictionary suitable for ``SatelliteEnvironment.step()``.
        """
        ...

    @abstractmethod
    def get_agents(self) -> List[str]:
        """Return list of all agent identifiers in this organization.

        Returns:
            List of agent_id strings.
        """
        ...

    # ------------------------------------------------------------------
    # Agent ↔ satellite mapping
    # ------------------------------------------------------------------

    def satellite_for_agent(self, agent_id: str) -> str:
        """Return the ``satellite_id`` that ``agent_id`` observes and controls.

        Agents (decision-making entities, e.g. ``central_agent``,
        ``mission_manager``, ``sat_agent_0``) and satellites (physical objects,
        e.g. ``eventsat_0``, ``sat_0``) live in two separate namespaces. The
        RLlib bridge needs this mapping to encode the right satellite into each
        agent's observation, decode its action onto the right satellite, and
        assign it the right per-satellite reward.

        Default: single-satellite organizations map every agent to one
        canonical satellite (overridable via ``agent_organization_config``
        key ``satellite_id``; defaults to ``"eventsat_0"`` to match legacy
        single-satellite behaviour). Multi-satellite organizations override
        this -- see :class:`IndependentMAS`.
        """
        return str(self.config.get("satellite_id", "eventsat_0"))

    def satellites_for_agent(self, agent_id: str) -> List[str]:
        """Return satellites whose actions may be emitted by ``agent_id``.

        This is the organisation-level actuator scope. Single-satellite and
        one-agent-per-satellite organisations inherit the legacy singular
        mapping unchanged. Multi-satellite organisations override this method to
        expose central, clustered, peer, or manager/local control scopes without
        forcing model-specific branches into RL/LLM/symbolic bridges.
        """
        return [self.satellite_for_agent(agent_id)]

    def observed_satellites_for_agent(self, agent_id: str) -> List[str]:
        """Return satellites visible to ``agent_id``.

        The default matches the actuator scope. Organisations whose information
        topology differs from their actuator topology (for example DMAS peers
        that observe the full constellation but act locally) override this
        method. This contract is model-agnostic: representations consume these
        scopes, they do not define them.
        """
        return self.satellites_for_agent(agent_id)

    # ------------------------------------------------------------------
    # Optional hooks (override as needed)
    # ------------------------------------------------------------------

    def pre_step(self) -> None:
        """Hook called before each environment step. Override if needed."""

    def post_step(self, step_result: Any) -> None:
        """Hook called after each environment step. Override if needed."""

    def get_metrics(self) -> Dict[str, float]:
        """Return organization-level metrics (e.g. communication overhead)."""
        return {}


def validate_agent_satellite_mapping(
    organization: "AgentOrganization",
    environment: Any,
    constellation_size: int,
    scenario: str,
) -> None:
    """Ensure organisation scopes are valid for the environment satellites.

    For multi-satellite scenarios, action scopes must form a disjoint cover of
    the environment satellites. Empty action scopes are allowed for non-actuator
    agents such as managers. Observation scopes are allowed to overlap, but all
    referenced satellite ids must exist.
    """
    if (
        environment is None
        or constellation_size <= 1
        or scenario not in {"multieventsat", "ssa"}
    ):
        return
    env_sats = set(environment.get_observation().constellation_state.satellites)
    agents = organization.get_agents()

    control_by_sat: Dict[str, str] = {}
    duplicate_controls: Dict[str, List[str]] = {}
    unknown: set[str] = set()
    observed_unknown: set[str] = set()

    for agent_id in agents:
        control_scope = list(organization.satellites_for_agent(agent_id))
        observed_scope = list(organization.observed_satellites_for_agent(agent_id))
        unknown.update(sid for sid in control_scope if sid not in env_sats)
        observed_unknown.update(sid for sid in observed_scope if sid not in env_sats)
        for sat_id in control_scope:
            owner = control_by_sat.get(sat_id)
            if owner is not None:
                duplicate_controls.setdefault(sat_id, [owner]).append(agent_id)
            else:
                control_by_sat[sat_id] = agent_id

    if unknown:
        raise ValueError(
            f"Organization maps action scopes to satellites {sorted(unknown)} not "
            f"present in scenario '{scenario}' (has {sorted(env_sats)}). "
            "Multi-satellite organizations must expose valid satellites_for_agent scopes."
        )
    if observed_unknown:
        raise ValueError(
            f"Organization maps observation scopes to satellites {sorted(observed_unknown)} not "
            f"present in scenario '{scenario}' (has {sorted(env_sats)}). "
            "Multi-satellite organizations must expose valid observed_satellites_for_agent scopes."
        )
    if duplicate_controls:
        details = {
            sat_id: owners for sat_id, owners in sorted(duplicate_controls.items())
        }
        raise ValueError(
            "Organization action scopes must be disjoint; overlapping controls: "
            f"{details}"
        )
    controlled = set(control_by_sat)
    missing = env_sats - controlled
    extra = controlled - env_sats
    if missing or extra:
        raise ValueError(
            f"Organization action scopes must cover scenario '{scenario}' satellites exactly. "
            f"Missing={sorted(missing)}, extra={sorted(extra)}, env={sorted(env_sats)}."
        )


def satellite_id_for_index(config: Dict[str, Any], idx: int) -> str:
    """Return the configured satellite id for a constellation index."""
    explicit = config.get("satellite_ids")
    if explicit is not None:
        try:
            return str(explicit[idx])
        except IndexError as exc:
            raise ValueError(
                f"No satellite_ids[{idx}] configured for agent/satellite index {idx}"
            ) from exc
    prefix = str(config.get("satellite_prefix", "sat"))
    return f"{prefix}_{idx}"


def satellite_ids_for_constellation(
    config: Dict[str, Any],
    constellation_size: int,
) -> List[str]:
    """Return configured satellite ids for all constellation indices."""
    return [
        satellite_id_for_index(config, idx)
        for idx in range(max(0, constellation_size))
    ]
