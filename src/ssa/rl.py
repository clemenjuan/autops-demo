"""Subsymbolic RL representation for SSA constellations."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from src.core.behaviour.controller import register
from src.core.representation import Representation
from src.eventsat.neural_policy import RandomPolicy
from src.ssa.rl_features import (
    SSA_ACTION_DIMS,
    SSA_MODE_LIST,
    SSA_OBS_DIM,
    SSA_OBS_SCHEMA_ID,
    build_ssa_obs_vector,
)

if TYPE_CHECKING:
    from src.core.decision_procedure.context import DecisionContext

logger = logging.getLogger(__name__)


@register("subsymbolic_ssa")
class SubsymbolicSSA(Representation):
    """RL-backed SSA mode selector with symbolic safety/resource grounding.

    The action space is one SSA mode over ``charging``, payload pipeline modes,
    ``communication``, ``isl_share``, and ``safe``. ``rl_mock`` uses a deterministic
    SSA heuristic by default so local smokes exercise the RL surface without
    degrading into uniformly random constellation actions.
    """

    supported_scenarios = frozenset({"ssa"})
    action_key_schema = "native_satellites"
    uses_agent_observation = True

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._deterministic = bool(self.config.get("deterministic", True))
        self._mock_uses_heuristic = bool(self.config.get("mock_uses_heuristic", True))
        self._policy_id = str(self.config.get("policy_id", "shared_policy"))
        self._act_scope_explicit = "act_ids" in self.config
        self._observe_scope_explicit = "observe_ids" in self.config
        legacy_satellite_id = self.config.get("satellite_id")
        self._act_ids = (
            self._coerce_id_list(self.config.get("act_ids"))
            if self._act_scope_explicit
            else (
                [str(legacy_satellite_id)]
                if legacy_satellite_id is not None
                else None
            )
        )
        self._observe_ids = (
            self._coerce_id_list(self.config.get("observe_ids"))
            if self._observe_scope_explicit
            else (list(self._act_ids) if self._act_ids is not None else None)
        )
        self._satellite_id = (
            self._act_ids[0]
            if self._act_ids
            else (
                self._observe_ids[0]
                if self._observe_ids
                else legacy_satellite_id
            )
        )
        action_scope_size = len(self._act_ids) if self._act_ids is not None else 1
        observe_scope_size = (
            len(self._observe_ids) if self._observe_ids is not None else 1
        )
        self._action_dims = list(SSA_ACTION_DIMS) * action_scope_size or [1]
        self._obs_dim = observe_scope_size * SSA_OBS_DIM
        self._target_count = int(self.config.get("target_count", 0) or 0)
        self._max_steps = int(self.config.get("max_steps", 10080) or 10080)
        self._low_soc = float(self.config.get("battery_threshold_low", 0.3))
        self._observe_soc = float(self.config.get("observe_soc", 0.6))
        self._storage_high = float(self.config.get("storage_threshold_high", 0.7))

        checkpoint_path = self.config.get("checkpoint_path") or self._find_default_checkpoint()
        self._policy: Any
        self._mock = True
        if bool(self.config.get("rl_mock", False)):
            self._policy = RandomPolicy(
                obs_dim=self._obs_dim,
                action_dims=self._action_dims,
            )
        elif checkpoint_path:
            self._validate_checkpoint_manifest(checkpoint_path)
            try:
                from src.eventsat.rllib_policy_adapter import RLLibPolicyAdapter

                self._policy = RLLibPolicyAdapter(
                    checkpoint_path=checkpoint_path,
                    policy_id=self._policy_id,
                    action_dims=self._action_dims,
                )
                self._mock = False
                logger.info("Loaded SSA RLlib checkpoint from %s", checkpoint_path)
            except (ImportError, FileNotFoundError, OSError) as exc:
                raise RuntimeError(
                    "SSA RL cell integrity violation: configured RLlib checkpoint "
                    f"could not be loaded from '{checkpoint_path}'. Install the RL "
                    "extra or provide a valid checkpoint; use rl_mock: true only "
                    "for CI/smoke runs."
                ) from exc
        elif bool(self.config.get("allow_untrained", False)):
            self._policy = RandomPolicy(
                obs_dim=self._obs_dim,
                action_dims=self._action_dims,
            )
        else:
            raise RuntimeError(
                "SSA RL cell integrity violation: no checkpoint_path configured and no "
                "default RLlib checkpoint manifest found. Provide a checkpoint, or set "
                "allow_untrained: true for training setup / rl_mock: true for CI."
            )

        self._last_rationale: Optional[str] = None
        self._last_action_vec: Optional[np.ndarray] = None
        self._last_obs_vec: Optional[np.ndarray] = None
        self._last_mode_probs: Optional[np.ndarray] = None
        self._last_value = 0.0
        self._last_log_prob = 0.0
        self._last_inference_latency_s = 0.0
        self._grounding_overrides = 0
        self._total_steps = 0

    def encode_observation(self, observation: Any) -> Dict[str, Any]:
        raw_observation = observation
        if hasattr(observation, "local_state") and isinstance(observation.local_state, dict):
            raw_observation = observation.local_state.get(
                "full_observation", observation
            )
        if not hasattr(raw_observation, "constellation_state"):
            return {
                "satellites": {},
                "global": {},
                "_obs_vectors": {},
                "_obs_vector": np.zeros(self._obs_dim, dtype=np.float32),
            }

        cstate = raw_observation.constellation_state
        global_info = dict(getattr(cstate, "global_info", {}) or {})
        target_count = int(global_info.get("ssa_target_count", 0) or 0)
        if target_count <= 0:
            row_target_count = max(
                (
                    len((sat.metadata or {}).get("ssa_detection_row", []) or [])
                    for sat in cstate.satellites.values()
                ),
                default=0,
            )
            target_count = (
                row_target_count
                if row_target_count > 0
                else max(1, self._target_count)
            )

        available_ids = list(cstate.satellites)
        observe_ids = (
            list(self._observe_ids)
            if self._observe_ids is not None
            else available_ids
        )
        act_ids = (
            list(self._act_ids)
            if self._act_ids is not None
            else available_ids
        )
        state_ids = list(dict.fromkeys([*observe_ids, *act_ids]))

        satellites: Dict[str, Dict[str, Any]] = {}
        vectors: Dict[str, np.ndarray] = {}
        scoped_known = set()
        scoped_delivered = set(
            str(oid)
            for oid in (global_info.get("ssa_delivered_objects", []) or [])
        )
        for sat_id in observe_ids:
            sat = cstate.satellites.get(sat_id)
            if sat is None:
                continue
            meta = sat.metadata or {}
            scoped_known.update(str(oid) for oid in (meta.get("ssa_known_objects", []) or []))
            scoped_delivered.update(str(oid) for oid in (meta.get("ssa_delivered_objects", []) or []))

        for sat_id in state_ids:
            sat = cstate.satellites.get(sat_id)
            if sat is None:
                continue
            res = sat.resources or {}
            meta = sat.metadata or {}
            cap_mb = float(meta.get("storage_capacity_mb", 4096.0) or 4096.0)
            data_mb = float(res.get("data_stored_mb", 0.0) or 0.0)
            visible = [str(oid) for oid in (meta.get("visible_rso_ids", []) or [])]
            if sat_id in observe_ids:
                vectors[sat_id] = build_ssa_obs_vector(
                    sat=sat,
                    constellation=cstate,
                    target_count=target_count,
                    max_steps=self._max_steps,
                    config=self.config,
                )
            satellites[sat_id] = {
                "battery_soc": float(res.get("battery_soc", 0.5) or 0.0),
                "current_mode": sat.status,
                "health_status": meta.get("health_status", "nominal"),
                "ground_pass_active": bool(
                    meta.get("contact_window_active", meta.get("ground_pass_active", False))
                ),
                "storage_used_fraction": data_mb / cap_mb if cap_mb > 0 else 0.0,
                "obc_data_mb": float(res.get("obc_data_mb", meta.get("obc_data_mb", 0.0)) or 0.0),
                "jetson_compressed_mb": float(meta.get("jetson_compressed_mb", 0.0) or 0.0),
                "uncompressed_observations": int(meta.get("uncompressed_observations", 0) or 0),
                "undetected_observations": int(meta.get("undetected_observations", 0) or 0),
                "undelivered_records": int(meta.get("ssa_undelivered_records", 0) or 0),
                "visible_new_rso_ids": [
                    oid for oid in visible
                    if oid not in scoped_known and oid not in scoped_delivered
                ],
                "known_objects": list(meta.get("ssa_known_objects", []) or []),
                "has_isl_peer": bool(meta.get("has_isl_peer", False)),
            }

        joint_parts = [
            vectors.get(sat_id, np.zeros(SSA_OBS_DIM, dtype=np.float32))
            for sat_id in observe_ids
        ]
        joint_vector = (
            np.concatenate(joint_parts).astype(np.float32)
            if joint_parts
            else np.zeros(self._obs_dim, dtype=np.float32)
        )
        return {
            "satellites": satellites,
            "global": global_info,
            "_obs_vectors": vectors,
            "_obs_vector": joint_vector,
            "_act_ids": act_ids,
        }

    def select_action(self, context: "DecisionContext") -> Dict[str, Any]:
        state = context.state or {}
        satellites: Dict[str, Dict[str, Any]] = state.get("satellites", {}) or {}
        if not satellites:
            self._last_rationale = "SSA RL: no state available; charging."
            return {}

        sat_ids = [
            str(sat_id)
            for sat_id in state.get("_act_ids", [])
            if str(sat_id) in satellites
        ]
        if not sat_ids:
            self._last_rationale = "SSA RL: no controlled satellites."
            return {}

        joint_obs = np.asarray(
            state.get("_obs_vector", np.zeros(self._obs_dim, dtype=np.float32)),
            dtype=np.float32,
        )
        use_heuristic = self._mock and self._mock_uses_heuristic and self._deterministic
        action_arr = np.zeros(len(sat_ids), dtype=int)
        if not use_heuristic:
            t0 = time.perf_counter()
            raw_action, log_prob, value = self._policy.get_action(
                joint_obs,
                deterministic=self._deterministic,
            )
            action_arr = np.asarray(raw_action, dtype=int).reshape(-1)
            mode_probs = np.asarray(
                self._policy.get_mode_probs(joint_obs), dtype=np.float32
            )
            self._last_inference_latency_s = time.perf_counter() - t0
            self._last_mode_probs = self._normalise_probs(mode_probs)
            self._last_value = (
                float(value.item()) if hasattr(value, "item") else float(value)
            )
            self._last_log_prob = (
                float(log_prob.item())
                if hasattr(log_prob, "item")
                else float(log_prob)
            )

        claimed: set[str] = set()
        actions: Dict[str, Dict[str, str]] = {}
        rationale_bits = []
        selected_indices: List[int] = []
        for sat_idx, sat_id in enumerate(sat_ids):
            sat_state = satellites[sat_id]
            if use_heuristic:
                proposed = self._heuristic_mode(sat_state, claimed)
                mode_idx = SSA_MODE_LIST.index(proposed)
            else:
                mode_idx = self._clip_action_component(
                    action_arr, sat_idx, len(SSA_MODE_LIST) - 1
                )
                proposed = SSA_MODE_LIST[mode_idx]
            selected_indices.append(mode_idx)
            grounded = self._ground_mode(
                proposed,
                sat_state,
                coordinated=(
                    len(sat_ids) > 1
                    or bool(sat_state.get("has_isl_peer", False))
                ),
            )
            if grounded != proposed:
                self._grounding_overrides += 1
            if grounded == "payload_observe":
                claimed.update(str(oid) for oid in sat_state.get("visible_new_rso_ids", []))
            actions[sat_id] = {"mode": grounded}
            rationale_bits.append(f"{sat_id}={grounded}")

        self._last_obs_vec = joint_obs
        self._last_action_vec = np.asarray(selected_indices, dtype=int)
        if use_heuristic:
            self._last_mode_probs = (
                np.ones(len(SSA_MODE_LIST), dtype=np.float32)
                / len(SSA_MODE_LIST)
            )
            self._last_value = 0.0
            self._last_log_prob = 0.0
            self._last_inference_latency_s = 0.0
        self._total_steps += 1
        source = "SSA RL mock heuristic" if use_heuristic else (
            "SSA RL mock policy" if self._mock else "SSA RLlib PPO"
        )
        self._last_rationale = f"{source}: " + ", ".join(rationale_bits)
        return actions

    def reason(self, state: Dict[str, Any], memory: Any) -> list[Dict[str, Any]]:
        satellites = (state or {}).get("satellites", {}) if isinstance(state, dict) else {}
        return [
            {
                "check": "ssa_rl_policy",
                "satellites": len(satellites),
                "implication": "mode_from_policy_then_symbolic_grounding",
            }
        ]

    def get_rationale(self) -> Optional[str]:
        return self._last_rationale

    def get_last_step_data(self) -> Optional[Dict[str, Any]]:
        if self._last_obs_vec is None or self._last_action_vec is None:
            return None
        return {
            "obs_vec": self._last_obs_vec,
            "action_vec": self._last_action_vec,
            "log_prob": self._last_log_prob,
            "value": self._last_value,
        }

    def get_metrics(self) -> Dict[str, float]:
        metrics = {
            "rl_inference_latency_s": self._last_inference_latency_s,
            "rl_value_estimate": self._last_value,
            "rl_grounding_overrides": float(self._grounding_overrides),
            "rl_total_steps": float(self._total_steps),
        }
        if self._last_mode_probs is not None:
            top = np.argsort(self._last_mode_probs)[::-1][:3]
            for rank, idx in enumerate(top):
                metrics[f"rl_mode_prob_{rank + 1}_{SSA_MODE_LIST[idx]}"] = float(
                    self._last_mode_probs[idx]
                )
        return metrics

    def reset(self) -> None:
        """Clear inference diagnostics and counters for a new episode."""
        self._last_rationale = None
        self._last_action_vec = None
        self._last_obs_vec = None
        self._last_mode_probs = None
        self._last_value = 0.0
        self._last_log_prob = 0.0
        self._last_inference_latency_s = 0.0
        self._grounding_overrides = 0
        self._total_steps = 0
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def seed(self, seed: int) -> None:
        if hasattr(self._policy, "seed"):
            self._policy.seed(int(seed))

    def close(self) -> None:
        if hasattr(self._policy, "close"):
            self._policy.close()

    def _heuristic_mode(self, sat: Dict[str, Any], claimed: set[str]) -> str:
        if sat.get("health_status", "nominal") != "nominal":
            return "safe"
        soc = float(sat.get("battery_soc", 0.5))
        if soc < self._low_soc:
            return "charging"
        undelivered = int(sat.get("undelivered_records", 0) or 0)
        obc_mb = float(sat.get("obc_data_mb", 0.0) or 0.0)
        if sat.get("ground_pass_active") and (obc_mb > 0.0 or undelivered > 0):
            return "communication"
        if int(sat.get("uncompressed_observations", 0) or 0) > 0:
            return "payload_compress"
        if int(sat.get("undetected_observations", 0) or 0) > 0:
            return "payload_detect"
        if float(sat.get("jetson_compressed_mb", 0.0) or 0.0) > 0.0:
            return "payload_send"
        visible = [
            str(oid) for oid in sat.get("visible_new_rso_ids", [])
            if str(oid) not in claimed
        ]
        if (
            visible
            and soc > self._observe_soc
            and float(sat.get("storage_used_fraction", 0.0)) < self._storage_high
        ):
            return "payload_observe"
        if undelivered > 0 and soc > self._observe_soc:
            return "isl_share"
        if sat.get("known_objects") and soc > self._observe_soc:
            return "isl_share"
        return "charging"

    def _ground_mode(self, mode: str, sat: Dict[str, Any], *, coordinated: bool) -> str:
        if sat.get("health_status", "nominal") != "nominal":
            return "safe"
        soc = float(sat.get("battery_soc", 0.5))
        if soc < self._low_soc and mode != "charging":
            return "charging"
        if mode == "communication":
            has_records = (
                float(sat.get("obc_data_mb", 0.0) or 0.0) > 0.0
                or int(sat.get("undelivered_records", 0) or 0) > 0
            )
            return mode if sat.get("ground_pass_active") and has_records else "charging"
        if mode == "payload_observe":
            return mode if sat.get("visible_new_rso_ids") and soc > self._observe_soc else "charging"
        if mode == "payload_compress":
            return mode if int(sat.get("uncompressed_observations", 0) or 0) > 0 else "charging"
        if mode == "payload_detect":
            return mode if int(sat.get("undetected_observations", 0) or 0) > 0 else "charging"
        if mode == "payload_send":
            return mode if float(sat.get("jetson_compressed_mb", 0.0) or 0.0) > 0.0 else "charging"
        if mode == "isl_share":
            useful = int(sat.get("undelivered_records", 0) or 0) > 0 or bool(sat.get("known_objects"))
            return mode if coordinated and useful and soc > self._observe_soc else "charging"
        return mode if mode in SSA_MODE_LIST else "charging"

    def _validate_checkpoint_manifest(self, checkpoint_path: str | Path) -> None:
        """Fail clearly when an SSA checkpoint uses another observation MDP."""
        path = Path(checkpoint_path).expanduser()
        candidates = [
            path / "manifest.json",
            path.parent / "manifest.json",
        ]
        trained_model_dir = self.config.get("trained_model_dir")
        if trained_model_dir:
            candidates.append(
                Path(str(trained_model_dir)).expanduser() / "manifest.json"
            )
        manifest_path = next(
            (candidate for candidate in candidates if candidate.is_file()),
            None,
        )
        if manifest_path is None:
            raise RuntimeError(
                "SSA RL checkpoint manifest is missing; checkpoints must declare "
                f"observation schema '{SSA_OBS_SCHEMA_ID}' and policy shape."
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"SSA RL checkpoint manifest is unreadable: {manifest_path}"
            ) from exc

        actual_schema = manifest.get("observation_schema_id")
        if actual_schema != SSA_OBS_SCHEMA_ID:
            raise RuntimeError(
                "SSA RL checkpoint observation schema mismatch: "
                f"expected '{SSA_OBS_SCHEMA_ID}', got {actual_schema!r}."
            )
        shapes = manifest.get("policy_observation_shapes", {}) or {}
        actual_shape = shapes.get(self._policy_id)
        expected_shape = [self._obs_dim]
        if list(actual_shape or []) != expected_shape:
            raise RuntimeError(
                "SSA RL checkpoint observation shape mismatch for policy "
                f"'{self._policy_id}': expected {expected_shape}, "
                f"got {actual_shape!r}."
            )

    def _find_default_checkpoint(self) -> Optional[str]:
        experiment_id = self.config.get("experiment_id")
        if not experiment_id:
            return None
        root = Path(
            self.config.get("trained_model_dir", f"data/trained_models/{experiment_id}")
        )
        manifest = root / "manifest.json"
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                checkpoint_path = data.get("checkpoint_path")
                if checkpoint_path and Path(checkpoint_path).exists():
                    return str(checkpoint_path)
            except (OSError, ValueError):
                pass
        if root.exists():
            candidates = sorted(root.glob("checkpoint_*"), key=lambda path: path.stat().st_mtime)
            return str(candidates[-1]) if candidates else None
        return None

    @staticmethod
    def _clip_action_component(
        action: np.ndarray, index: int, max_value: int
    ) -> int:
        value = int(action[index]) if action.size > index else 0
        return max(0, min(value, max_value))

    @staticmethod
    def _coerce_id_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @staticmethod
    def _normalise_probs(probs: np.ndarray) -> np.ndarray:
        if probs.shape[0] != len(SSA_MODE_LIST) or not np.all(np.isfinite(probs)):
            return np.ones(len(SSA_MODE_LIST), dtype=np.float32) / len(SSA_MODE_LIST)
        total = float(np.sum(probs))
        if total <= 0.0:
            return np.ones(len(SSA_MODE_LIST), dtype=np.float32) / len(SSA_MODE_LIST)
        return probs / total
