"""
Satellite Environment — Abstract Base Class.

Unified environment for all experiments. Handles orbital mechanics,
task generation, and constraint management. Must be operational-scenario-agnostic
at the base level.

Specific task types, rewards, and constraints depend on the chosen operational
scenario and live in the scenario-owned packages such as ``src/eventsat`` and
``src/ssa``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SatelliteState:
    """State of a single satellite at a given timestep.

    Attributes:
        satellite_id: Unique identifier for the satellite.
        position: Position vector [x, y, z] in chosen reference frame (km).
        velocity: Velocity vector [vx, vy, vz] (km/s).
        resources: Dictionary of resource budgets (e.g. power_w, data_storage_mb).
        status: Operational status string (e.g. "nominal", "safe_mode").
        metadata: Additional satellite-specific information.
    """

    satellite_id: str
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    resources: Dict[str, float] = field(default_factory=dict)
    status: str = "nominal"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstellationState:
    """Aggregate state of the full constellation.

    Attributes:
        timestep: Current simulation timestep index.
        epoch_seconds: Simulation epoch in seconds since reference time.
        satellites: Mapping from satellite_id to its state.
        global_info: Environment-wide information (e.g. pending tasks).
    """

    timestep: int
    epoch_seconds: float
    satellites: Dict[str, SatelliteState] = field(default_factory=dict)
    global_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentObservation:
    """Observation returned by the environment each step.

    Attributes:
        constellation_state: Full constellation state snapshot.
        tasks: List of currently active / pending tasks.
        events: Any events that occurred since the last step.
    """

    constellation_state: ConstellationState
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StepResult:
    """Result of executing one environment step.

    Attributes:
        observation: New observation after actions were applied.
        rewards: Mapping of reward components to their values.
        done: Whether the episode has terminated.
        truncated: Whether the episode was truncated (e.g. max steps).
        info: Additional diagnostic information.
    """

    observation: EnvironmentObservation
    rewards: Dict[str, float] = field(default_factory=dict)
    done: bool = False
    truncated: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


def scope_observation(
    env_observation: EnvironmentObservation,
    satellite_ids: List[str],
    *,
    copy_satellite_states: bool = False,
    include_global_info: bool = True,
    strict_addressed_records: bool = False,
) -> EnvironmentObservation:
    """Return a copy of ``env_observation`` restricted to ``satellite_ids``.

    A pure, organization-agnostic helper used by ``AgentOrganization`` subclasses
    to build per-agent partial views (the *who-sees-what* decision lives in each
    organization's ``distribute_observation``; this only performs the mechanical
    slice). Defaults retain the legacy permissive behaviour. Strict local
    organisations may copy selected states, omit dynamic global information,
    and keep only tasks/events explicitly addressed to their satellites.
    """
    constellation = env_observation.constellation_state
    satellite_set = set(satellite_ids)
    scoped = {}
    for sat_id in satellite_ids:
        if sat_id not in constellation.satellites:
            continue
        state = constellation.satellites[sat_id]
        scoped[sat_id] = deepcopy(state) if copy_satellite_states else state

    if strict_addressed_records:
        scoped_tasks = [
            deepcopy(task)
            for task in (env_observation.tasks or [])
            if task.get("satellite_id") in satellite_set
        ]
        scoped_events = [
            deepcopy(event)
            for event in (env_observation.events or [])
            if event.get("satellite_id") in satellite_set
        ]
    else:
        scoped_tasks = [
            task
            for task in (env_observation.tasks or [])
            if task.get("satellite_id") is None
            or task.get("satellite_id") in satellite_set
        ]
        scoped_events = env_observation.events
    return EnvironmentObservation(
        constellation_state=ConstellationState(
            timestep=constellation.timestep,
            epoch_seconds=constellation.epoch_seconds,
            satellites=scoped,
            global_info=(
                constellation.global_info
                if include_global_info
                else {}
            ),
        ),
        tasks=scoped_tasks,
        events=scoped_events,
    )


class SatelliteEnvironment(ABC):
    """Abstract base class for satellite constellation environments.

    Subclasses implement scenario-specific logic (task generation, reward
    computation, constraint checking) while this base enforces the common
    interface used by :class:`ExperimentRunner`.

    Attributes:
        config: Scenario configuration dictionary loaded from YAML.
        constellation_size: Number of satellites in the constellation.
        timestep_seconds: Duration of one simulation timestep in seconds.
        current_step: Current step counter within the episode.
        max_steps: Maximum number of steps per episode.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise the environment from a configuration dictionary.

        Args:
            config: Full environment configuration section from the
                experiment YAML. Must contain at least ``constellation_size``
                and ``timestep_seconds``.
        """
        self.config = config
        self.constellation_size: int = config.get("constellation_size", 1)
        self.timestep_seconds: int = config.get("timestep_seconds", 60)
        self.max_steps: int = config.get("max_steps", 1440)
        self.current_step: int = 0

    def configure_communication_links(
        self,
        links: set[tuple[str, str]] | None,
    ) -> None:
        """Optionally bind organisation-authorised communication endpoints.

        Scenario environments without a physical communication data plane keep
        this no-op. ``None`` means that no organisation topology is imposed.
        """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> EnvironmentObservation:
        """Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Initial observation of the constellation.
        """
        ...

    @abstractmethod
    def step(self, actions: Dict[str, Any]) -> StepResult:
        """Execute one simulation timestep.

        Args:
            actions: Mapping of satellite_id → action to execute.

        Returns:
            A :class:`StepResult` containing the new observation, rewards,
            termination flags, and diagnostic info.
        """
        ...

    @abstractmethod
    def get_observation(self) -> EnvironmentObservation:
        """Return the current observation without advancing the simulation.

        Returns:
            Current observation snapshot.
        """
        ...

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Return current environment-level performance metrics.

        Returns:
            Dictionary of metric name → value.
        """
        ...

    # ------------------------------------------------------------------
    # Common helpers (non-abstract)
    # ------------------------------------------------------------------

    def is_done(self) -> bool:
        """Check whether the episode is complete."""
        return self.current_step >= self.max_steps

    def get_config(self) -> Dict[str, Any]:
        """Return the environment configuration."""
        return dict(self.config)
