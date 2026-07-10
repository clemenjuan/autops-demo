"""Observation/action adapters for RL training backends.

RLlib expects vector observations and Gymnasium spaces, while AUTOPS scenarios
work with rich domain objects and action dictionaries.  Adapters keep that
scenario-specific translation out of the training pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np

# RL contract constants + the shared observation encoder live in one place
# (re-exported here so existing importers of ``ACTION_DIMS`` etc. keep working).
from src.eventsat.rl_obs_encoder import (  # noqa: F401
    ACTION_DIMS,
    MODE_LIST,
    MODE_TO_IDX,
    OBS_DIM,
    _DEFAULT_JETSON_CAPACITY_MB,
    _DEFAULT_MAX_PASS_STEPS,
    encode_eventsat_rl_obs,
)

try:
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:
    spaces = None  # type: ignore[assignment]
    GYMNASIUM_AVAILABLE = False


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
        return float(sum(rewards.values())) if rewards else 0.0


# --- RL contract per scenario (RL-side; scenarios stay model-agnostic) --------
#
# Action space and observation vectorisation are RL-specific. Each scenario
# declares its (mode_list, obs_dim, obs_encoder) here; the generic adapter and
# the subsymbolic representation read from it, so adding an RL scenario is one
# RLSpec entry -- no adapter/representation subclass.

# SSA operational modes -- must match the strings SSAEnvironment reads by name
# (its SSA_MODES); ``isl_share`` is the 8th mode the policy can now choose.
SSA_MODE_LIST = [
    "charging",
    "payload_observe",
    "payload_compress",
    "payload_detect",
    "payload_send",
    "communication",
    "isl_share",
    "safe",
]
_SSA_EXTRA_FEATURES = 4
SSA_OBS_DIM = OBS_DIM + _SSA_EXTRA_FEATURES


def encode_ssa_rl_obs(
    res: Dict[str, Any], meta: Dict[str, Any], status: str, **consts: Any
) -> np.ndarray:
    """SSA RL observation: EventSat base 25D + 4 SSA coordination features.

    Features (all in [0, 1]) are read from the per-satellite metadata the SSA
    env already publishes (no scenario change): onboard coverage, delivered
    coverage, visible-RSO count (capped), and this satellite's own
    known-target fraction.
    """
    base = encode_eventsat_rl_obs(res, meta, status, **consts)
    detection_row = meta.get("ssa_detection_row", []) or []
    known_fraction = (sum(detection_row) / len(detection_row)) if detection_row else 0.0
    extra = np.array(
        [
            float(meta.get("ssa_onboard_coverage", 0.0)),
            float(meta.get("ssa_delivered_coverage", 0.0)),
            min(float(meta.get("visible_rso_count", 0)) / 10.0, 1.0),
            float(known_fraction),
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, extra])


@dataclass(frozen=True)
class RLSpec:
    """RL contract for one scenario: action modes + observation vectorisation."""

    scenario: str
    mode_list: List[str]
    obs_dim: int
    obs_encoder: Callable[..., np.ndarray]

    @property
    def action_dims(self) -> List[int]:
        return [len(self.mode_list), 2, 2]


_EVENTSAT_RLSPEC = RLSpec("eventsat", MODE_LIST, OBS_DIM, encode_eventsat_rl_obs)
_SSA_RLSPEC = RLSpec("ssa", SSA_MODE_LIST, SSA_OBS_DIM, encode_ssa_rl_obs)

RL_SPECS: Dict[str, RLSpec] = {
    "eventsat": _EVENTSAT_RLSPEC,
    "multieventsat": _EVENTSAT_RLSPEC,  # reuses the EventSat RL contract
    "ssa": _SSA_RLSPEC,
}


def get_rl_spec(scenario: str) -> "RLSpec | None":
    """Return the RL contract for ``scenario``, or ``None`` if not RL-enabled."""
    return RL_SPECS.get(scenario)


class ScenarioSpaceAdapter(RLSpaceAdapter):
    """Generic RL space adapter, parametrised by a scenario's :class:`RLSpec`.

    EventSat/MultiEventsat use the default 25D / MultiDiscrete([7,2,2]) contract;
    SSA passes the 8-mode + extended-obs spec. No per-scenario subclass.
    """

    def __init__(
        self,
        config: Dict[str, Any] | None = None,
        env: Any | None = None,
        spec: "RLSpec | None" = None,
    ) -> None:
        self._spec = spec or _EVENTSAT_RLSPEC
        super().__init__(scenario=self._spec.scenario)
        if not GYMNASIUM_AVAILABLE:
            raise ImportError("gymnasium is required for RL spaces. Install with: uv sync --extra rl")
        self.config = config or {}
        self.env = env
        legacy_satellite_id = str(self.config.get("satellite_id", "eventsat_0"))
        self._act_ids = self._coerce_id_list(
            self._configured_id_source(
                "act_ids",
                "satellite_ids",
                default=[legacy_satellite_id],
            )
        )
        self._observe_ids = self._coerce_id_list(
            self._configured_id_source(
                "observe_ids",
                "observed_satellite_ids",
                default=self._act_ids,
            )
        )
        self.satellite_id = (
            self._act_ids[0]
            if self._act_ids
            else (self._observe_ids[0] if self._observe_ids else legacy_satellite_id)
        )
        self._include_messages = bool(self.config.get("include_peer_messages", False))
        self._message_dim = (
            len(self._observe_ids) * len(self._spec.mode_list)
            if self._include_messages
            else 0
        )

        base_low = np.zeros(self._spec.obs_dim, dtype=np.float32)
        base_high = np.ones(self._spec.obs_dim, dtype=np.float32) * 2.0
        base_low[4:6] = -1.0  # sin/cos orbital phase
        low_parts = [base_low.copy() for _ in self._observe_ids]
        high_parts = [base_high.copy() for _ in self._observe_ids]
        if self._message_dim:
            low_parts.append(np.zeros(self._message_dim, dtype=np.float32))
            high_parts.append(np.ones(self._message_dim, dtype=np.float32))
        low = (
            np.concatenate(low_parts)
            if low_parts
            else np.zeros(1, dtype=np.float32)
        )
        high = (
            np.concatenate(high_parts)
            if high_parts
            else np.ones(1, dtype=np.float32)
        )
        self._observation_space = spaces.Box(  # type: ignore[union-attr]
            low=low,
            high=high,
            dtype=np.float32,
        )
        action_dims = self._spec.action_dims * len(self._act_ids)
        self._action_space = spaces.MultiDiscrete(action_dims or [1])  # type: ignore[union-attr]

    @property
    def observation_space(self) -> Any:
        return self._observation_space

    @property
    def action_space(self) -> Any:
        return self._action_space

    @property
    def observe_ids(self) -> List[str]:
        return list(self._observe_ids)

    @property
    def act_ids(self) -> List[str]:
        return list(self._act_ids)

    def encode_observation(self, observation: Any) -> np.ndarray:
        """Encode AUTOPS or AgentObservation objects into the scenario's obs vector."""
        raw_observation = observation
        messages = []
        if hasattr(observation, "local_state") and isinstance(observation.local_state, dict):
            raw_observation = observation.local_state.get("full_observation", observation)
            messages = list(getattr(observation, "messages", []) or [])

        if not hasattr(raw_observation, "constellation_state"):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        constellation = raw_observation.constellation_state
        parts = [
            self._encode_satellite(raw_observation, sat_id)
            for sat_id in self._observe_ids
        ]
        if self._message_dim:
            parts.append(self._encode_messages(messages))
        if not parts:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)

    def decode_action(self, action: Any, agent_id: str | None = None) -> Dict[str, Any]:
        mode_list = self._spec.mode_list
        action_vec = np.asarray(action, dtype=int).reshape(-1)
        decoded: Dict[str, Any] = {}
        width = len(self._spec.action_dims)
        for sat_idx, satellite_id in enumerate(self._act_ids):
            start = sat_idx * width
            mode_idx = int(action_vec[start]) if action_vec.size > start else 0
            data_idx = start + 1
            routing_idx = start + 2
            data_priority = int(action_vec[data_idx]) if action_vec.size > data_idx else 0
            pipeline_routing = (
                int(action_vec[routing_idx]) if action_vec.size > routing_idx else 0
            )
            mode_idx = max(0, min(mode_idx, len(mode_list) - 1))
            decoded[satellite_id] = {
                "mode": mode_list[mode_idx],
                "data_priority": max(0, min(data_priority, 1)),
                "pipeline_routing": max(0, min(pipeline_routing, 1)),
            }
        return decoded

    def _env_or_config(self, name: str, default: float) -> float:
        if self.env is not None and hasattr(self.env, name):
            return float(getattr(self.env, name))
        return float(self.config.get(name, default))

    def _encode_satellite(self, raw_observation: Any, satellite_id: str) -> np.ndarray:
        constellation = raw_observation.constellation_state
        sat = constellation.satellites.get(satellite_id)
        if sat is None:
            return np.zeros(self._spec.obs_dim, dtype=np.float32)

        res = sat.resources or {}
        meta = sat.metadata or {}
        current_step = int(
            getattr(constellation, "timestep", getattr(self.env, "current_step", 0))
        )
        detection_progress = float(
            getattr(self.env, "detection_progress", meta.get("detection_progress", 0.0))
        )
        return self._spec.obs_encoder(
            res,
            meta,
            str(sat.status or "charging"),
            obc_cap=self._env_or_config(
                "storage_capacity_mb", meta.get("storage_capacity_mb", 512.0)
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

    def _encode_messages(self, messages: List[Dict[str, Any]]) -> np.ndarray:
        vec = np.zeros(self._message_dim, dtype=np.float32)
        if not self._message_dim:
            return vec
        sat_to_slot = {sat_id: idx for idx, sat_id in enumerate(self._observe_ids)}
        mode_to_idx = {mode: idx for idx, mode in enumerate(self._spec.mode_list)}
        mode_count = len(self._spec.mode_list)
        for message in messages:
            proposal = message.get("proposal") or message.get("action") or {}
            if not isinstance(proposal, dict):
                continue
            for sat_id, sat_action in proposal.items():
                slot = sat_to_slot.get(str(sat_id))
                if slot is None or not isinstance(sat_action, dict):
                    continue
                mode_idx = mode_to_idx.get(str(sat_action.get("mode", "")))
                if mode_idx is None:
                    continue
                vec[slot * mode_count + mode_idx] = 1.0
        return vec

    @staticmethod
    def _coerce_id_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    def _configured_id_source(self, *keys: str, default: Any) -> Any:
        for key in keys:
            if key in self.config and self.config[key] is not None:
                return self.config[key]
        return default


# Backwards-compatible alias: the adapter used to be EventSat-specific.
EventSatSpaceAdapter = ScenarioSpaceAdapter


def make_space_adapter(
    scenario: str,
    config: Dict[str, Any] | None = None,
    env: Any | None = None,
) -> RLSpaceAdapter:
    """Create an RL adapter for a scenario from its :class:`RLSpec`.

    Per-agent adapters bind to a specific satellite via ``config["satellite_id"]``.
    """
    spec = get_rl_spec(scenario)
    if spec is None:
        raise ValueError(f"No RL space adapter registered for scenario '{scenario}'")
    return ScenarioSpaceAdapter(config=config, env=env, spec=spec)
