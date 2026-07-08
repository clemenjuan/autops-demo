"""Shared SSA RL observation/action feature definitions."""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

SSA_MODE_LIST = (
    "charging",
    "payload_observe",
    "payload_compress",
    "payload_detect",
    "payload_send",
    "communication",
    "isl_share",
    "safe",
)
SSA_MODE_TO_IDX = {mode: idx for idx, mode in enumerate(SSA_MODE_LIST)}
SSA_ACTION_DIMS = [len(SSA_MODE_LIST)]
SSA_OBS_DIM = 32

_DEFAULT_JETSON_CAPACITY_MB = 249036.8
_DEFAULT_ORBITAL_PERIOD_STEPS = 94.0
_DEFAULT_MAX_PASS_STEPS = 10.0


def build_ssa_obs_vector(
    *,
    sat: Any,
    constellation: Any,
    target_count: int,
    max_steps: int,
    config: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Encode one SSA satellite state as a bounded 32D float vector."""
    cfg = config or {}
    res = getattr(sat, "resources", {}) or {}
    meta = getattr(sat, "metadata", {}) or {}
    target_scale = float(max(1, target_count))
    vec = np.zeros(SSA_OBS_DIM, dtype=np.float32)

    storage_cap = float(meta.get("storage_capacity_mb", 4096.0) or 4096.0)
    obc_mb = float(res.get("obc_data_mb", meta.get("obc_data_mb", 0.0)) or 0.0)
    jetson_cap = float(cfg.get("jetson_capacity_mb", _DEFAULT_JETSON_CAPACITY_MB) or 1.0)
    orbital_period = float(cfg.get("orbital_period_steps", _DEFAULT_ORBITAL_PERIOD_STEPS) or 1.0)
    compression_time = float(cfg.get("compression_time_factor", 2.0) or 1.0)
    detection_steps = float(cfg.get("detection_steps", 5.0) or 1.0)
    daily_budget = float(meta.get("daily_downlink_budget_mb", 27.0) or 1.0)

    visible = list(meta.get("visible_rso_ids", []) or [])
    known = set(str(oid) for oid in (meta.get("ssa_known_objects", []) or []))
    delivered = set(str(oid) for oid in (meta.get("ssa_delivered_objects", []) or []))
    visible_new = [oid for oid in visible if str(oid) not in known and str(oid) not in delivered]

    vec[0] = float(res.get("battery_soc", 0.5) or 0.0)
    vec[1] = obc_mb / storage_cap
    vec[2] = float(res.get("data_stored_mb", 0.0) or 0.0) / storage_cap
    vec[3] = float(meta.get("jetson_raw_mb", 0.0) or 0.0) / jetson_cap
    vec[4] = float(meta.get("jetson_compressed_mb", 0.0) or 0.0) / jetson_cap
    vec[5] = 1.0 if meta.get("ground_pass_active", False) else 0.0
    vec[6] = 1.0 if meta.get("health_status", "nominal") == "nominal" else 0.0
    vec[7] = 1.0 if meta.get("in_sunlight", False) else 0.0
    vec[8] = min(len(visible) / target_scale, 1.0)
    vec[9] = min(len(visible_new) / target_scale, 1.0)
    vec[10] = min(len(known) / target_scale, 1.0)
    vec[11] = min(len(delivered) / target_scale, 1.0)
    vec[12] = min(float(meta.get("ssa_undelivered_records", 0) or 0.0) / target_scale, 1.0)
    vec[13] = float(meta.get("ssa_onboard_coverage", 0.0) or 0.0)
    vec[14] = float(meta.get("ssa_delivered_coverage", 0.0) or 0.0)
    vec[15] = min(float(meta.get("remaining_pass_duration", 0.0) or 0.0) / _DEFAULT_MAX_PASS_STEPS, 1.0)
    vec[16] = min(float(meta.get("time_to_next_pass", orbital_period) or orbital_period) / orbital_period, 1.0)
    vec[17] = min(float(meta.get("time_to_next_eclipse", orbital_period) or orbital_period) / orbital_period, 1.0)
    vec[18] = float(getattr(constellation, "timestep", 0) or 0) / float(max(1, max_steps))
    vec[19] = min(float(meta.get("uncompressed_observations", 0) or 0.0) / 10.0, 1.0)
    vec[20] = min(float(meta.get("undetected_observations", 0) or 0.0) / 10.0, 1.0)
    vec[21] = min(float(meta.get("compression_progress", 0) or 0.0) / compression_time, 1.0)
    vec[22] = min(float(meta.get("detection_progress", 0) or 0.0) / detection_steps, 1.0)
    vec[23] = float(res.get("data_downlinked_mb", 0.0) or 0.0) / daily_budget

    mode_idx = SSA_MODE_TO_IDX.get(str(getattr(sat, "status", "charging")), 0)
    vec[24 + mode_idx] = 1.0
    return np.nan_to_num(vec, copy=False, nan=0.0, posinf=2.0, neginf=-1.0)


def mode_from_action(action: Any) -> str:
    """Decode an SSA mode from a scalar, vector, or one-hot-like action."""
    arr = np.asarray(action, dtype=int).reshape(-1)
    if arr.size == len(SSA_MODE_LIST) and np.sum(arr == 1) == 1:
        idx = int(np.argmax(arr))
    else:
        idx = int(arr[0]) if arr.size else 0
    idx = max(0, min(idx, len(SSA_MODE_LIST) - 1))
    return SSA_MODE_LIST[idx]


def cyclic_phase(value: float) -> tuple[float, float]:
    angle = float(value) * 2.0 * math.pi
    return math.sin(angle), math.cos(angle)
