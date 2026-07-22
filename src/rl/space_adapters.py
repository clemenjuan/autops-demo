"""Observation/action adapters for RL training backends.

RLlib expects vector observations and Gymnasium spaces, while AUTOPS scenarios
work with rich domain objects and satellite-keyed action dictionaries. Adapters
own that translation and apply each logical agent's observation and actuation
scopes consistently in training and evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np

from src.eventsat.rl_obs_encoder import (
    ACTION_DIMS,
    MODE_LIST,
    OBS_DIM,
    _DEFAULT_JETSON_CAPACITY_MB,
    encode_eventsat_rl_obs,
)
from src.rl import observation_bounds
from src.ssa.rl_features import (
    SSA_ACTION_DIMS,
    SSA_MODE_LIST,
    SSA_OBS_DIM,
    build_ssa_obs_vector,
    mode_from_action,
)

try:
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:
    spaces = None  # type: ignore[assignment]
    GYMNASIUM_AVAILABLE = False


@dataclass(frozen=True)
class RLSpec:
    """Stable observation/action contract for one RL-enabled scenario."""

    scenario: str
    mode_list: List[str]
    obs_dim: int
    action_dims: List[int]
    obs_encoder: Callable[..., np.ndarray]


_EVENTSAT_RL_SPEC = RLSpec(
    "eventsat",
    list(MODE_LIST),
    OBS_DIM,
    list(ACTION_DIMS),
    encode_eventsat_rl_obs,
)
_SSA_RL_SPEC = RLSpec(
    "ssa",
    list(SSA_MODE_LIST),
    SSA_OBS_DIM,
    list(SSA_ACTION_DIMS),
    build_ssa_obs_vector,
)

RL_SPECS: Dict[str, RLSpec] = {
    "eventsat": _EVENTSAT_RL_SPEC,
    "multieventsat": _EVENTSAT_RL_SPEC,
    "ssa": _SSA_RL_SPEC,
}


def get_rl_spec(scenario: str) -> RLSpec | None:
    """Return the RL contract for ``scenario``, if one is registered."""

    return RL_SPECS.get(scenario)


def _coerce_id_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _configured_id_source(
    config: Dict[str, Any], *keys: str, default: Any
) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _configured_scopes(
    config: Dict[str, Any], default_satellite_id: str
) -> tuple[List[str], List[str]]:
    act_ids = _coerce_id_list(
        _configured_id_source(
            config,
            "act_ids",
            "satellite_ids",
            default=[default_satellite_id],
        )
    )
    observe_ids = _coerce_id_list(
        _configured_id_source(
            config,
            "observe_ids",
            "observed_satellite_ids",
            default=act_ids,
        )
    )
    return act_ids, observe_ids


def _message_vector(
    messages: List[Dict[str, Any]],
    observe_ids: List[str],
    mode_list: List[str],
) -> np.ndarray:
    mode_count = len(mode_list)
    vector = np.zeros(len(observe_ids) * mode_count, dtype=np.float32)
    satellite_slots = {satellite_id: idx for idx, satellite_id in enumerate(observe_ids)}
    mode_indices = {mode: idx for idx, mode in enumerate(mode_list)}
    for message in messages:
        if not isinstance(message, dict):
            continue
        proposal = message.get("proposal") or message.get("action") or {}
        if not isinstance(proposal, dict):
            continue
        for satellite_id, action in proposal.items():
            slot = satellite_slots.get(str(satellite_id))
            if slot is None or not isinstance(action, dict):
                continue
            mode_idx = mode_indices.get(str(action.get("mode", "")))
            if mode_idx is not None:
                vector[slot * mode_count + mode_idx] = 1.0
    return vector


@dataclass(frozen=True)
class RLSpaceAdapter:
    """Base adapter contract for scenario-specific RL spaces."""

    scenario: str

    @property
    def observation_space(self) -> Any:
        raise NotImplementedError

    @property
    def action_space(self) -> Any:
        raise NotImplementedError

    @property
    def observe_ids(self) -> List[str]:
        raise NotImplementedError

    @property
    def act_ids(self) -> List[str]:
        raise NotImplementedError

    def encode_observation(self, observation: Any) -> np.ndarray:
        raise NotImplementedError

    def decode_action(self, action: Any, agent_id: str | None = None) -> Dict[str, Any]:
        raise NotImplementedError

    def scalar_reward(self, rewards: Dict[str, float]) -> float:
        if not rewards:
            return 0.0
        act_ids = self.act_ids
        if not act_ids:
            return 0.0
        scoped = [float(rewards[sat_id]) for sat_id in act_ids if sat_id in rewards]
        return float(sum(scoped)) if scoped else float(sum(rewards.values()))


class EventSatSpaceAdapter(RLSpaceAdapter):
    """Joint EventSat observation and factored action adapter."""

    def __init__(
        self,
        config: Dict[str, Any] | None = None,
        env: Any | None = None,
        *,
        scenario: str = "eventsat",
    ) -> None:
        super().__init__(scenario=scenario)
        if not GYMNASIUM_AVAILABLE:
            raise ImportError(
                "gymnasium is required for RL spaces. Install with: uv sync --extra rl"
            )
        self.config = config or {}
        self.env = env
        legacy_id = str(self.config.get("satellite_id", "eventsat_0"))
        self._act_ids, self._observe_ids = _configured_scopes(self.config, legacy_id)
        self.satellite_id = (
            self._act_ids[0]
            if self._act_ids
            else (self._observe_ids[0] if self._observe_ids else legacy_id)
        )
        self._include_messages = bool(self.config.get("include_peer_messages", False))

        base_low, base_high = observation_bounds(
            OBS_DIM, signed_indices=(4, 5)
        )
        low_parts = [base_low.copy() for _ in self._observe_ids]
        high_parts = [base_high.copy() for _ in self._observe_ids]
        if self._include_messages:
            message_dim = len(self._observe_ids) * len(MODE_LIST)
            low_parts.append(np.zeros(message_dim, dtype=np.float32))
            high_parts.append(np.ones(message_dim, dtype=np.float32))
        low = np.concatenate(low_parts) if low_parts else np.zeros(1, dtype=np.float32)
        high = np.concatenate(high_parts) if high_parts else np.ones(1, dtype=np.float32)
        self._observation_space = spaces.Box(low=low, high=high, dtype=np.float32)  # type: ignore[union-attr]
        action_dims = list(ACTION_DIMS) * len(self._act_ids)
        self._action_space = spaces.MultiDiscrete(action_dims or [1])  # type: ignore[union-attr]

    @property
    def observation_space(self) -> Any:
        return self._observation_space

    @property
    def action_space(self) -> Any:
        return self._action_space

    @property
    def observe_ids(self) -> List[str]:
        return list(getattr(self, "_observe_ids", [self.satellite_id]))

    @property
    def act_ids(self) -> List[str]:
        return list(getattr(self, "_act_ids", [self.satellite_id]))

    def encode_observation(self, observation: Any) -> np.ndarray:
        raw_observation = observation
        messages: List[Dict[str, Any]] = []
        if hasattr(observation, "local_state") and isinstance(observation.local_state, dict):
            raw_observation = observation.local_state.get("full_observation", observation)
            messages = list(getattr(observation, "messages", []) or [])

        observe_ids = self.observe_ids
        expected_dim = len(observe_ids) * OBS_DIM
        if getattr(self, "_include_messages", False):
            expected_dim += len(observe_ids) * len(MODE_LIST)
        if not hasattr(raw_observation, "constellation_state"):
            return np.zeros(expected_dim or 1, dtype=np.float32)

        parts = [
            self._encode_satellite(raw_observation, satellite_id)
            for satellite_id in observe_ids
        ]
        if getattr(self, "_include_messages", False):
            parts.append(_message_vector(messages, observe_ids, list(MODE_LIST)))
        return (
            np.concatenate(parts).astype(np.float32)
            if parts
            else np.zeros(1, dtype=np.float32)
        )

    def decode_action(self, action: Any, agent_id: str | None = None) -> Dict[str, Any]:
        action_vector = np.asarray(action, dtype=int).reshape(-1)
        width = len(ACTION_DIMS)
        decoded: Dict[str, Any] = {}
        for satellite_idx, satellite_id in enumerate(self.act_ids):
            start = satellite_idx * width
            mode_idx = int(action_vector[start]) if action_vector.size > start else 0
            data_idx = start + 1
            routing_idx = start + 2
            data_priority = int(action_vector[data_idx]) if action_vector.size > data_idx else 0
            pipeline_routing = (
                int(action_vector[routing_idx]) if action_vector.size > routing_idx else 0
            )
            mode_idx = max(0, min(mode_idx, len(MODE_LIST) - 1))
            decoded[satellite_id] = {
                "mode": MODE_LIST[mode_idx],
                "data_priority": max(0, min(data_priority, 1)),
                "pipeline_routing": max(0, min(pipeline_routing, 1)),
            }
        return decoded

    def _env_or_config(self, name: str, default: float) -> float:
        if self.env is not None and hasattr(self.env, name):
            return float(getattr(self.env, name))
        return float(self.config.get(name, default))

    def _encode_satellite(self, observation: Any, satellite_id: str) -> np.ndarray:
        constellation = observation.constellation_state
        satellite = constellation.satellites.get(satellite_id)
        if satellite is None:
            return np.zeros(OBS_DIM, dtype=np.float32)
        resources = satellite.resources or {}
        metadata = satellite.metadata or {}
        current_step = int(
            getattr(constellation, "timestep", getattr(self.env, "current_step", 0))
        )
        detection_progress = float(
            getattr(
                self.env,
                "detection_progress",
                metadata.get("detection_progress", 0.0),
            )
        )
        return encode_eventsat_rl_obs(
            resources,
            metadata,
            str(satellite.status or "charging"),
            obc_cap=self._env_or_config(
                "storage_capacity_mb",
                metadata.get("storage_capacity_mb", 512.0),
            ),
            jetson_cap=self._env_or_config(
                "jetson_capacity_mb", _DEFAULT_JETSON_CAPACITY_MB
            ),
            orbital_period=self._env_or_config("orbital_period_steps", 94.0),
            max_steps=self._env_or_config("max_steps", 10080.0),
            compression_time=self._env_or_config("compression_time_factor", 2.0),
            detection_steps=self._env_or_config("detection_steps", 5.0),
            current_step=current_step,
            detection_progress=detection_progress,
        )


class SSASpaceAdapter(RLSpaceAdapter):
    """Joint 32D-per-satellite SSA observation and 8-mode action adapter."""

    def __init__(self, config: Dict[str, Any] | None = None, env: Any | None = None) -> None:
        super().__init__(scenario="ssa")
        if not GYMNASIUM_AVAILABLE:
            raise ImportError(
                "gymnasium is required for RL spaces. Install with: uv sync --extra rl"
            )
        self.config = config or {}
        self.env = env
        legacy_id = str(self.config.get("satellite_id", "sat_0"))
        self._act_ids, self._observe_ids = _configured_scopes(self.config, legacy_id)
        self.satellite_id = (
            self._act_ids[0]
            if self._act_ids
            else (self._observe_ids[0] if self._observe_ids else legacy_id)
        )
        self._include_messages = bool(self.config.get("include_peer_messages", False))

        base_low, base_high = observation_bounds(SSA_OBS_DIM)
        low_parts = [base_low.copy() for _ in self._observe_ids]
        high_parts = [base_high.copy() for _ in self._observe_ids]
        if self._include_messages:
            message_dim = len(self._observe_ids) * len(SSA_MODE_LIST)
            low_parts.append(np.zeros(message_dim, dtype=np.float32))
            high_parts.append(np.ones(message_dim, dtype=np.float32))
        low = np.concatenate(low_parts) if low_parts else np.zeros(1, dtype=np.float32)
        high = np.concatenate(high_parts) if high_parts else np.ones(1, dtype=np.float32)
        self._observation_space = spaces.Box(low=low, high=high, dtype=np.float32)  # type: ignore[union-attr]
        action_dims = list(SSA_ACTION_DIMS) * len(self._act_ids)
        self._action_space = spaces.MultiDiscrete(action_dims or [1])  # type: ignore[union-attr]

    @property
    def observation_space(self) -> Any:
        return self._observation_space

    @property
    def action_space(self) -> Any:
        return self._action_space

    @property
    def observe_ids(self) -> List[str]:
        return list(getattr(self, "_observe_ids", [self.satellite_id]))

    @property
    def act_ids(self) -> List[str]:
        return list(getattr(self, "_act_ids", [self.satellite_id]))

    def encode_observation(self, observation: Any) -> np.ndarray:
        raw_observation = observation
        messages: List[Dict[str, Any]] = []
        if hasattr(observation, "local_state") and isinstance(observation.local_state, dict):
            raw_observation = observation.local_state.get("full_observation", observation)
            messages = list(getattr(observation, "messages", []) or [])

        observe_ids = self.observe_ids
        expected_dim = len(observe_ids) * SSA_OBS_DIM
        if getattr(self, "_include_messages", False):
            expected_dim += len(observe_ids) * len(SSA_MODE_LIST)
        if not hasattr(raw_observation, "constellation_state"):
            return np.zeros(expected_dim or 1, dtype=np.float32)

        constellation = raw_observation.constellation_state
        global_info = dict(getattr(constellation, "global_info", {}) or {})
        configured_target_count = int(global_info.get("ssa_target_count", 0) or 0)
        parts: List[np.ndarray] = []
        for satellite_id in observe_ids:
            satellite = constellation.satellites.get(satellite_id)
            if satellite is None:
                parts.append(np.zeros(SSA_OBS_DIM, dtype=np.float32))
                continue
            target_count = configured_target_count
            if target_count <= 0:
                target_count = (
                    len((satellite.metadata or {}).get("ssa_detection_row", []) or [])
                    or 1
                )
            parts.append(
                build_ssa_obs_vector(
                    sat=satellite,
                    constellation=constellation,
                    target_count=target_count,
                    max_steps=int(self.config.get("max_steps", 10080) or 10080),
                    config=self.config,
                )
            )
        if getattr(self, "_include_messages", False):
            parts.append(_message_vector(messages, observe_ids, list(SSA_MODE_LIST)))
        return (
            np.concatenate(parts).astype(np.float32)
            if parts
            else np.zeros(1, dtype=np.float32)
        )

    def decode_action(self, action: Any, agent_id: str | None = None) -> Dict[str, Any]:
        action_vector = np.asarray(action, dtype=int).reshape(-1)
        decoded: Dict[str, Any] = {}
        for satellite_idx, satellite_id in enumerate(self.act_ids):
            value = (
                action_vector[satellite_idx]
                if action_vector.size > satellite_idx
                else np.array([0])
            )
            decoded[satellite_id] = {"mode": mode_from_action(value)}
        return decoded


def make_space_adapter(
    scenario: str,
    config: Dict[str, Any] | None = None,
    env: Any | None = None,
) -> RLSpaceAdapter:
    """Create the scoped RL adapter registered for ``scenario``."""

    if scenario in ("eventsat", "multieventsat"):
        return EventSatSpaceAdapter(config=config, env=env, scenario=scenario)
    if scenario == "ssa":
        return SSASpaceAdapter(config=config, env=env)
    raise ValueError(f"No RL space adapter registered for scenario '{scenario}'")
