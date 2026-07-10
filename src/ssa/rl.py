"""Subsymbolic RL representation for SSA constellations."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from src.core.behaviour.controller import register
from src.core.representation import Representation
from src.eventsat.neural_policy import RandomPolicy
from src.ssa.rl_features import (
    SSA_ACTION_DIMS,
    SSA_MODE_LIST,
    SSA_OBS_DIM,
    build_ssa_obs_vector,
    mode_from_action,
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

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._deterministic = bool(self.config.get("deterministic", True))
        self._mock_uses_heuristic = bool(self.config.get("mock_uses_heuristic", True))
        self._policy_id = str(self.config.get("policy_id", "shared_policy"))
        self._satellite_id = self.config.get("satellite_id")
        self._target_count = int(self.config.get("target_count", 0) or 0)
        self._max_steps = int(self.config.get("max_steps", 10080) or 10080)
        self._low_soc = float(self.config.get("battery_threshold_low", 0.3))
        self._observe_soc = float(self.config.get("observe_soc", 0.6))
        self._storage_high = float(self.config.get("storage_threshold_high", 0.7))

        checkpoint_path = self.config.get("checkpoint_path") or self._find_default_checkpoint()
        self._policy: Any
        self._mock = True
        if bool(self.config.get("rl_mock", False)):
            self._policy = RandomPolicy(obs_dim=SSA_OBS_DIM, action_dims=SSA_ACTION_DIMS)
        elif checkpoint_path:
            try:
                from src.eventsat.rllib_policy_adapter import RLLibPolicyAdapter

                self._policy = RLLibPolicyAdapter(
                    checkpoint_path=checkpoint_path,
                    policy_id=self._policy_id,
                    action_dims=SSA_ACTION_DIMS,
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
            self._policy = RandomPolicy(obs_dim=SSA_OBS_DIM, action_dims=SSA_ACTION_DIMS)
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
        if not hasattr(observation, "constellation_state"):
            return {"satellites": {}, "global": {}, "_obs_vectors": {}}

        cstate = observation.constellation_state
        global_info = dict(getattr(cstate, "global_info", {}) or {})
        target_count = int(global_info.get("ssa_target_count", self._target_count) or 0)
        if target_count <= 0:
            target_count = max(
                1,
                len(global_info.get("ssa_detection_matrix", []) or []),
                max(
                    (
                        len((sat.metadata or {}).get("ssa_detection_row", []) or [])
                        for sat in cstate.satellites.values()
                    ),
                    default=1,
                ),
            )

        satellites: Dict[str, Dict[str, Any]] = {}
        vectors: Dict[str, np.ndarray] = {}
        scoped_known = set()
        scoped_delivered = set()
        for sat in cstate.satellites.values():
            meta = sat.metadata or {}
            scoped_known.update(str(oid) for oid in (meta.get("ssa_known_objects", []) or []))
            scoped_delivered.update(str(oid) for oid in (meta.get("ssa_delivered_objects", []) or []))

        for sat_id, sat in cstate.satellites.items():
            res = sat.resources or {}
            meta = sat.metadata or {}
            cap_mb = float(meta.get("storage_capacity_mb", 4096.0) or 4096.0)
            data_mb = float(res.get("data_stored_mb", 0.0) or 0.0)
            visible = [str(oid) for oid in (meta.get("visible_rso_ids", []) or [])]
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
            }

        return {
            "satellites": satellites,
            "global": global_info,
            "_obs_vectors": vectors,
        }

    def select_action(self, context: "DecisionContext") -> Dict[str, Any]:
        state = context.state or {}
        satellites: Dict[str, Dict[str, Any]] = state.get("satellites", {}) or {}
        vectors: Dict[str, np.ndarray] = state.get("_obs_vectors", {}) or {}
        if not satellites:
            self._last_rationale = "SSA RL: no state available; charging."
            return {}

        sat_ids = sorted(satellites)
        if self._satellite_id in satellites:
            sat_ids = [str(self._satellite_id)]

        claimed: set[str] = set()
        actions: Dict[str, Dict[str, str]] = {}
        rationale_bits = []
        for sat_id in sat_ids:
            sat_state = satellites[sat_id]
            obs_vec = vectors.get(sat_id, np.zeros(SSA_OBS_DIM, dtype=np.float32))
            proposed = self._propose_mode(sat_id, sat_state, obs_vec, claimed)
            grounded = self._ground_mode(proposed, sat_state, coordinated=len(sat_ids) > 1)
            if grounded != proposed:
                self._grounding_overrides += 1
            if grounded == "payload_observe":
                claimed.update(str(oid) for oid in sat_state.get("visible_new_rso_ids", []))
            actions[sat_id] = {"mode": grounded}
            rationale_bits.append(f"{sat_id}={grounded}")

        source = "SSA RL mock heuristic" if self._mock and self._mock_uses_heuristic else (
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

    def _propose_mode(
        self,
        sat_id: str,
        sat_state: Dict[str, Any],
        obs_vec: np.ndarray,
        claimed: set[str],
    ) -> str:
        if self._mock and self._mock_uses_heuristic and self._deterministic:
            mode = self._heuristic_mode(sat_state, claimed)
            self._last_obs_vec = np.asarray(obs_vec, dtype=np.float32)
            self._last_action_vec = np.array([SSA_MODE_LIST.index(mode)], dtype=int)
            self._last_mode_probs = np.ones(len(SSA_MODE_LIST), dtype=np.float32) / len(SSA_MODE_LIST)
            self._last_value = 0.0
            self._last_log_prob = 0.0
            self._last_inference_latency_s = 0.0
            self._total_steps += 1
            return mode

        t0 = time.perf_counter()
        action_vec, log_prob, value = self._policy.get_action(
            obs_vec,
            deterministic=self._deterministic,
        )
        mode_probs = np.asarray(self._policy.get_mode_probs(obs_vec), dtype=np.float32)
        self._last_inference_latency_s = time.perf_counter() - t0
        self._total_steps += 1
        self._last_action_vec = np.asarray(action_vec, dtype=int).reshape(-1)
        self._last_obs_vec = np.asarray(obs_vec, dtype=np.float32)
        self._last_mode_probs = self._normalise_probs(mode_probs)
        self._last_value = float(value.item()) if hasattr(value, "item") else float(value)
        self._last_log_prob = (
            float(log_prob.item()) if hasattr(log_prob, "item") else float(log_prob)
        )
        return mode_from_action(action_vec)

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
    def _normalise_probs(probs: np.ndarray) -> np.ndarray:
        if probs.shape[0] != len(SSA_MODE_LIST) or not np.all(np.isfinite(probs)):
            return np.ones(len(SSA_MODE_LIST), dtype=np.float32) / len(SSA_MODE_LIST)
        total = float(np.sum(probs))
        if total <= 0.0:
            return np.ones(len(SSA_MODE_LIST), dtype=np.float32) / len(SSA_MODE_LIST)
        return probs / total
