"""Single source of truth: scenario name -> model-agnostic components.

Maps a scenario to its environment and metrics-collector classes, so the
experiment runner and the RLlib bridge stop re-declaring (and drifting on) the
mapping. This is the same idea the codebase already uses for representations via
``@register`` in the behaviour controller, applied to scenarios.

Scope: **model-agnostic only** (env + metrics). RL-specific contracts -- action
space, observation vectorisation -- live in the RL adapter layer
(``src/rl/space_adapters.py``), not here. Scenarios must stay usable by symbolic,
LLM and RL models alike.

Loaders are lazy so importing this module does not pull in every scenario (and
avoids import cycles with ``src.core``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass(frozen=True)
class ScenarioSpec:
    """Model-agnostic components for one scenario."""

    name: str
    env_loader: Callable[[], type]
    metrics_loader: Callable[[], type]


def _eventsat_env() -> type:
    from src.eventsat.env import EventSatEnvironment

    return EventSatEnvironment


def _multieventsat_env() -> type:
    from src.eventsat.multieventsat_env import MultiEventsatEnv

    return MultiEventsatEnv


def _ssa_env() -> type:
    from src.ssa.env import SSAEnvironment

    return SSAEnvironment


def _eventsat_metrics() -> type:
    from src.eventsat.metrics import EventSatMetricsCollector

    return EventSatMetricsCollector


def _ssa_metrics() -> type:
    from src.ssa.metrics import SSAMetricsCollector

    return SSAMetricsCollector


SCENARIOS: Dict[str, ScenarioSpec] = {
    # MultiEventsat reuses the EventSat metrics collector (EventSat-compatible
    # aggregate telemetry); SSA has its own.
    "eventsat": ScenarioSpec("eventsat", _eventsat_env, _eventsat_metrics),
    "multieventsat": ScenarioSpec("multieventsat", _multieventsat_env, _eventsat_metrics),
    "ssa": ScenarioSpec("ssa", _ssa_env, _ssa_metrics),
}


def get_scenario_spec(name: str) -> Optional[ScenarioSpec]:
    """Return the spec for ``name``, or ``None`` if the scenario is unknown."""
    return SCENARIOS.get(name)
