"""Observation/action adapters for RL training backends.

RLlib expects vector observations and Gymnasium spaces, while AUTOPS scenarios
work with rich domain objects and action dictionaries.  Adapters keep that
scenario-specific translation out of the training pipeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from src.rl import (
    bound_observation_vector,
    bounded_ratio,
    downlink_utilization,
    observation_bounds,
)

from src.ssa.rl_features import (
    SSA_ACTION_DIMS,
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


MODE_LIST = [
    "charging",
    "communication",
    "payload_observe",
    "payload_compress",
    "payload_detect",
    "payload_send",
    "safe",
]
MODE_TO_IDX = {mode: idx for idx, mode in enumerate(MODE_LIST)}
OBS_DIM = 25
ACTION_DIMS = [len(MODE_LIST), 2, 2]
_DEFAULT_JETSON_CAPACITY_MB = 249036.8
_DEFAULT_MAX_PASS_STEPS = 10.0


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

    def encode_observation(self, observation: Any) -> np.ndarray:
        raise NotImplementedError

    def decode_action(self, action: Any, agent_id: str | None = None) -> Dict[str, Any]:
        raise NotImplementedError

    def scalar_reward(self, rewards: Dict[str, float]) -> float:
        return float(sum(rewards.values())) if rewards else 0.0


class EventSatSpaceAdapter(RLSpaceAdapter):
    """25D EventSat observation and MultiDiscrete([7, 2, 2]) action adapter."""

    def __init__(self, config: Dict[str, Any] | None = None, env: Any | None = None) -> None:
        super().__init__(scenario="eventsat")
        if not GYMNASIUM_AVAILABLE:
            raise ImportError("gymnasium is required for RL spaces. Install with: uv sync --extra rl")
        self.config = config or {}
        self.env = env
        self.satellite_id = str(self.config.get("satellite_id", "eventsat_0"))

        low, high = observation_bounds(OBS_DIM, signed_indices=(4, 5))
        self._observation_space = spaces.Box(  # type: ignore[union-attr]
            low=low,
            high=high,
            dtype=np.float32,
        )
        self._action_space = spaces.MultiDiscrete(ACTION_DIMS)  # type: ignore[union-attr]

    @property
    def observation_space(self) -> Any:
        return self._observation_space

    @property
    def action_space(self) -> Any:
        return self._action_space

    def encode_observation(self, observation: Any) -> np.ndarray:
        """Encode AUTOPS or AgentObservation objects into a 25D vector."""
        raw_observation = observation
        if hasattr(observation, "local_state") and isinstance(observation.local_state, dict):
            raw_observation = observation.local_state.get("full_observation", observation)

        vec = np.zeros(OBS_DIM, dtype=np.float32)
        if not hasattr(raw_observation, "constellation_state"):
            return vec

        constellation = raw_observation.constellation_state
        sat = constellation.satellites.get(self.satellite_id)
        if sat is None:
            return vec

        res = sat.resources or {}
        meta = sat.metadata or {}

        vec[0] = float(res.get("battery_soc", 0.5))
        obc_cap = self._env_or_config("storage_capacity_mb", meta.get("storage_capacity_mb", 512.0))
        vec[1] = bounded_ratio(
            res.get("obc_data_mb", meta.get("obc_data_mb", 0.0)), obc_cap
        )
        jetson_cap = self._env_or_config("jetson_capacity_mb", _DEFAULT_JETSON_CAPACITY_MB)
        vec[2] = bounded_ratio(meta.get("jetson_raw_mb", 0.0), jetson_cap)
        vec[3] = bounded_ratio(meta.get("jetson_compressed_mb", 0.0), jetson_cap)

        orbital_phase = float(meta.get("orbital_phase", 0.0))
        vec[4] = math.sin(orbital_phase * 2 * math.pi)
        vec[5] = math.cos(orbital_phase * 2 * math.pi)

        orbital_period = self._env_or_config("orbital_period_steps", 94.0)
        vec[6] = min(float(meta.get("time_to_next_eclipse", orbital_period)) / (orbital_period or 1.0), 1.0)
        vec[7] = min(float(meta.get("time_to_next_pass", orbital_period)) / (orbital_period or 1.0), 1.0)
        vec[8] = min(float(meta.get("remaining_pass_duration", 0.0)) / _DEFAULT_MAX_PASS_STEPS, 1.0)
        current_step = int(getattr(constellation, "timestep", getattr(self.env, "current_step", 0)))
        max_steps = self._env_or_config("max_steps", 10080.0)
        vec[9] = float(current_step) / (max_steps or 1.0)

        vec[10] = 1.0 if meta.get("in_sunlight", False) else 0.0
        vec[11] = 1.0 if meta.get("contact_window_active", meta.get("ground_pass_active", False)) else 0.0
        vec[12] = 1.0 if meta.get("health_status", "nominal") == "nominal" else 0.0

        vec[13] = min(float(meta.get("uncompressed_observations", 0)) / 10.0, 1.0)
        compression_time = self._env_or_config("compression_time_factor", 2.0)
        vec[14] = min(float(meta.get("compression_progress", 0)) / (compression_time or 1.0), 1.0)
        vec[15] = min(float(meta.get("undetected_observations", 0)) / 10.0, 1.0)
        detection_steps = self._env_or_config("detection_steps", 5.0)
        detection_progress = float(getattr(self.env, "detection_progress", meta.get("detection_progress", 0.0)))
        vec[16] = min(detection_progress / (detection_steps or 1.0), 1.0)
        vec[17] = downlink_utilization(res, meta, float(obc_cap))

        mode_idx = MODE_TO_IDX.get(str(sat.status or "charging"), 0)
        vec[18 + mode_idx] = 1.0

        return bound_observation_vector(vec, signed_indices=(4, 5))

    def decode_action(self, action: Any, agent_id: str | None = None) -> Dict[str, Any]:
        action_vec = np.asarray(action, dtype=int).reshape(-1)
        mode_idx = int(action_vec[0]) if action_vec.size > 0 else 0
        data_priority = int(action_vec[1]) if action_vec.size > 1 else 0
        pipeline_routing = int(action_vec[2]) if action_vec.size > 2 else 0
        mode_idx = max(0, min(mode_idx, len(MODE_LIST) - 1))
        return {
            self.satellite_id: {
                "mode": MODE_LIST[mode_idx],
                "data_priority": max(0, min(data_priority, 1)),
                "pipeline_routing": max(0, min(pipeline_routing, 1)),
            }
        }

    def _env_or_config(self, name: str, default: float) -> float:
        if self.env is not None and hasattr(self.env, name):
            return float(getattr(self.env, name))
        return float(self.config.get(name, default))


class SSASpaceAdapter(RLSpaceAdapter):
    """32D SSA observation and MultiDiscrete([8]) action adapter."""

    def __init__(self, config: Dict[str, Any] | None = None, env: Any | None = None) -> None:
        super().__init__(scenario="ssa")
        if not GYMNASIUM_AVAILABLE:
            raise ImportError("gymnasium is required for RL spaces. Install with: uv sync --extra rl")
        self.config = config or {}
        self.env = env
        self.satellite_id = str(self.config.get("satellite_id", "sat_0"))

        low, high = observation_bounds(SSA_OBS_DIM)
        self._observation_space = spaces.Box(  # type: ignore[union-attr]
            low=low,
            high=high,
            dtype=np.float32,
        )
        self._action_space = spaces.MultiDiscrete([8])  # type: ignore[union-attr]

    @property
    def observation_space(self) -> Any:
        return self._observation_space

    @property
    def action_space(self) -> Any:
        return self._action_space

    def encode_observation(self, observation: Any) -> np.ndarray:
        raw_observation = observation
        if hasattr(observation, "local_state") and isinstance(observation.local_state, dict):
            raw_observation = observation.local_state.get("full_observation", observation)

        vec = np.zeros(SSA_OBS_DIM, dtype=np.float32)
        if not hasattr(raw_observation, "constellation_state"):
            return vec

        constellation = raw_observation.constellation_state
        sat = constellation.satellites.get(self.satellite_id)
        if sat is None:
            return vec
        global_info = dict(getattr(constellation, "global_info", {}) or {})
        target_count = int(global_info.get("ssa_target_count", 0) or 0)
        if target_count <= 0:
            target_count = len((sat.metadata or {}).get("ssa_detection_row", []) or []) or 1
        return build_ssa_obs_vector(
            sat=sat,
            constellation=constellation,
            target_count=target_count,
            max_steps=int(self.config.get("max_steps", 10080) or 10080),
            config=self.config,
        )

    def decode_action(self, action: Any, agent_id: str | None = None) -> Dict[str, Any]:
        return {self.satellite_id: {"mode": mode_from_action(action)}}


def make_space_adapter(
    scenario: str,
    config: Dict[str, Any] | None = None,
    env: Any | None = None,
) -> RLSpaceAdapter:
    """Create an RL adapter for a scenario."""
    if scenario in ("eventsat", "multieventsat"):
        # multieventsat reuses the EventSat observation/action contract
        # (25D obs, MultiDiscrete([7, 2, 2])); per-agent adapters bind to a
        # specific satellite via config["satellite_id"].
        return EventSatSpaceAdapter(config=config, env=env)
    if scenario == "ssa":
        return SSASpaceAdapter(config=config, env=env)
    raise ValueError(f"No RL space adapter registered for scenario '{scenario}'")
