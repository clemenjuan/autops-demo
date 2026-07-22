"""Policy-sharing helpers for RLlib multi-agent training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List

import numpy as np


@dataclass(frozen=True)
class PolicySharingConfig:
    """Declarative policy sharing strategy for RLlib.

    This is not a morphological-matrix dimension.  It only controls whether
    multiple RLlib agents share neural weights during training.
    """

    mode: str = "shared_all"

    @classmethod
    def from_config(cls, config: Dict[str, Any] | str | None) -> "PolicySharingConfig":
        if config is None:
            return cls()
        if isinstance(config, str):
            return cls(mode=config)
        return cls(mode=str(config.get("mode", "shared_all")))

    def policy_id_for(self, agent_id: str) -> str:
        if self.mode == "shared_all":
            return "shared_policy"
        if self.mode == "independent_per_agent":
            return f"policy_{agent_id}"
        if self.mode == "shared_by_role":
            if agent_id == "mission_manager":
                return "manager_policy"
            if agent_id.startswith("sat_agent_"):
                return "satellite_policy"
            if agent_id == "central_agent":
                return "central_policy"
            return "shared_policy"
        raise ValueError(
            "policy_sharing.mode must be one of "
            "{'shared_all', 'shared_by_role', 'independent_per_agent'}, "
            f"got '{self.mode}'"
        )

    def policy_ids(self, agent_ids: Iterable[str]) -> List[str]:
        return sorted({self.policy_id_for(agent_id) for agent_id in agent_ids})

    def mapping_fn(self) -> Callable[..., str]:
        def _map(agent_id: str, *args: Any, **kwargs: Any) -> str:
            return self.policy_id_for(agent_id)

        return _map


def build_policy_specs(
    agent_ids: Iterable[str],
    observation_space: Any,
    action_space: Any,
    sharing: PolicySharingConfig,
) -> Dict[str, Any]:
    """Build RLlib PolicySpec objects for the selected sharing strategy.

    ``observation_space`` and ``action_space`` may be either single spaces
    (legacy uniform case) or ``agent_id -> space`` dictionaries. Agents sharing
    one policy must have compatible spaces; otherwise weight sharing is
    physically impossible and the caller should use ``independent_per_agent`` or
    another grouping that preserves shape compatibility.
    """
    try:
        from ray.rllib.policy.policy import PolicySpec
    except ImportError as exc:
        raise ImportError("ray[rllib] is required to build RLlib policies") from exc

    agent_ids = list(agent_ids)
    policies: Dict[str, Any] = {}
    for policy_id in sharing.policy_ids(agent_ids):
        policy_agents = [
            agent_id
            for agent_id in agent_ids
            if sharing.policy_id_for(agent_id) == policy_id
        ]
        if not policy_agents:
            continue
        obs_space = _space_for_agent(observation_space, policy_agents[0])
        act_space = _space_for_agent(action_space, policy_agents[0])
        for other_agent in policy_agents[1:]:
            other_obs = _space_for_agent(observation_space, other_agent)
            other_act = _space_for_agent(action_space, other_agent)
            if not _spaces_compatible(obs_space, other_obs) or not _spaces_compatible(
                act_space, other_act
            ):
                raise ValueError(
                    f"Agents {policy_agents[0]} and {other_agent} cannot share "
                    f"policy '{policy_id}' because their RL spaces differ. "
                    "Use policy_sharing.mode: independent_per_agent or another "
                    "shape-compatible sharing strategy."
                )
        policies[policy_id] = PolicySpec(
            policy_class=None,
            observation_space=obs_space,
            action_space=act_space,
            config={},
        )
    return policies


def _space_for_agent(space_or_map: Any, agent_id: str) -> Any:
    if isinstance(space_or_map, dict):
        return space_or_map[agent_id]
    return space_or_map


def _spaces_compatible(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if hasattr(left, "shape") or hasattr(right, "shape"):
        if tuple(getattr(left, "shape", ())) != tuple(getattr(right, "shape", ())):
            return False
    if hasattr(left, "dtype") or hasattr(right, "dtype"):
        if getattr(left, "dtype", None) != getattr(right, "dtype", None):
            return False
    if hasattr(left, "nvec") or hasattr(right, "nvec"):
        return np.array_equal(getattr(left, "nvec", None), getattr(right, "nvec", None))
    if hasattr(left, "n") or hasattr(right, "n"):
        return getattr(left, "n", None) == getattr(right, "n", None)
    return repr(left) == repr(right)
