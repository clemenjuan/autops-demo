"""Pure EventSat data-pipeline transitions.

The environment, learned-planner surrogate, and agentic what-if tools must
project the same physical state change.  Discrete products (an observation or
detection metadata record) are admitted atomically: exact fits succeed and an
epsilon overflow leaves every counter and buffer untouched.  Flow operations
(CAN and S-band) remain physically partial and are bounded by rate, source
backlog, destination headroom, and contact duration.

Archived results produced before this shared contract are not directly
comparable at capacity boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


CAPACITY_EPSILON_MB = 1e-12


@dataclass(frozen=True)
class PipelineParameters:
    """Physical constants required by the data-pipeline transitions."""

    observation_size_mb: float = 9.41
    compression_ratio: float = 5.11
    jetson_capacity_mb: float = 249036.8
    obc_capacity_mb: float = 4096.0
    detection_metadata_mb: float = 0.01
    jetson_to_obc_rate_kbps: float = 8000.0
    downlink_rate_kbps: float = 50.0
    step_duration_s: float = 60.0

    @property
    def compressed_observation_mb(self) -> float:
        return self.observation_size_mb / max(self.compression_ratio, 1e-12)


@dataclass(frozen=True)
class TransitionOutcome:
    """Result of a side-effect-free transition projection."""

    state: dict[str, Any]
    accepted: bool
    reason: str | None = None
    transferred_mb: float = 0.0
    raw_equivalent_mb: float = 0.0


def _copy_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return dict(state)


def _number(state: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = state.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _fits(projected_mb: float, capacity_mb: float) -> bool:
    return projected_mb <= capacity_mb + CAPACITY_EPSILON_MB


def jetson_occupancy_mb(state: Mapping[str, Any]) -> float:
    return max(0.0, _number(state, "jetson_raw_mb")) + max(
        0.0, _number(state, "jetson_compressed_mb")
    )


def can_observe(state: Mapping[str, Any], params: PipelineParameters) -> bool:
    return _fits(
        jetson_occupancy_mb(state) + params.observation_size_mb,
        params.jetson_capacity_mb,
    )


def apply_observe(
    state: Mapping[str, Any],
    params: PipelineParameters,
    *,
    observation_duration_s: float | None = None,
) -> TransitionOutcome:
    """Atomically admit one raw observation product."""

    projected = _copy_state(state)
    if not can_observe(state, params):
        return TransitionOutcome(projected, False, "jetson_capacity")

    duration_s = (
        params.step_duration_s
        if observation_duration_s is None
        else max(0.0, float(observation_duration_s))
    )
    projected["jetson_raw_mb"] = _number(state, "jetson_raw_mb") + params.observation_size_mb
    projected["uncompressed_observations"] = (
        _number(state, "uncompressed_observations") + 1.0
    )
    projected["total_raw_captured_mb"] = (
        _number(state, "total_raw_captured_mb") + params.observation_size_mb
    )
    projected["total_observation_s"] = _number(state, "total_observation_s") + duration_s
    return TransitionOutcome(projected, True)


def can_compress(state: Mapping[str, Any], params: PipelineParameters) -> bool:
    if _number(state, "uncompressed_observations") < 1.0:
        return False
    if _number(state, "jetson_raw_mb") + CAPACITY_EPSILON_MB < params.observation_size_mb:
        return False
    projected_occupancy = (
        jetson_occupancy_mb(state)
        - params.observation_size_mb
        + params.compressed_observation_mb
    )
    return _fits(projected_occupancy, params.jetson_capacity_mb)


def apply_compress(
    state: Mapping[str, Any], params: PipelineParameters
) -> TransitionOutcome:
    """Atomically complete compression of one raw product."""

    projected = _copy_state(state)
    if not can_compress(state, params):
        reason = (
            "no_raw_product"
            if _number(state, "uncompressed_observations") < 1.0
            else "jetson_capacity"
        )
        return TransitionOutcome(projected, False, reason)

    projected["uncompressed_observations"] = max(
        0.0, _number(state, "uncompressed_observations") - 1.0
    )
    projected["jetson_raw_mb"] = max(
        0.0, _number(state, "jetson_raw_mb") - params.observation_size_mb
    )
    projected["jetson_compressed_mb"] = (
        _number(state, "jetson_compressed_mb") + params.compressed_observation_mb
    )
    projected["undetected_observations"] = (
        _number(state, "undetected_observations") + 1.0
    )
    return TransitionOutcome(projected, True)


def can_detect(state: Mapping[str, Any], params: PipelineParameters) -> bool:
    return _number(state, "undetected_observations") >= 1.0 and _fits(
        _number(state, "obc_data_mb") + params.detection_metadata_mb,
        params.obc_capacity_mb,
    )


def apply_detect(
    state: Mapping[str, Any], params: PipelineParameters
) -> TransitionOutcome:
    """Atomically complete detection and write its OBC metadata."""

    projected = _copy_state(state)
    if _number(state, "undetected_observations") < 1.0:
        return TransitionOutcome(projected, False, "no_undetected_product")
    if not can_detect(state, params):
        return TransitionOutcome(projected, False, "obc_capacity")

    projected["undetected_observations"] = max(
        0.0, _number(state, "undetected_observations") - 1.0
    )
    projected["obc_data_mb"] = _number(state, "obc_data_mb") + params.detection_metadata_mb
    projected["total_detections"] = _number(state, "total_detections") + 1.0
    return TransitionOutcome(projected, True)


def apply_can_transfer(
    state: Mapping[str, Any],
    params: PipelineParameters,
    *,
    duration_s: float | None = None,
) -> TransitionOutcome:
    """Move the physically transferable compressed backlog to the OBC."""

    projected = _copy_state(state)
    seconds = params.step_duration_s if duration_s is None else max(0.0, float(duration_s))
    source_mb = max(0.0, _number(state, "jetson_compressed_mb"))
    headroom_mb = max(0.0, params.obc_capacity_mb - _number(state, "obc_data_mb"))
    rate_limit_mb = max(0.0, params.jetson_to_obc_rate_kbps / 8.0 * seconds / 1000.0)
    transfer_mb = min(source_mb, headroom_mb, rate_limit_mb)
    if transfer_mb <= CAPACITY_EPSILON_MB:
        reason = "no_source_data" if source_mb <= CAPACITY_EPSILON_MB else "obc_capacity"
        return TransitionOutcome(projected, False, reason)

    projected["jetson_compressed_mb"] = source_mb - transfer_mb
    projected["obc_data_mb"] = _number(state, "obc_data_mb") + transfer_mb
    raw_equivalent_mb = transfer_mb * params.compression_ratio
    projected["obc_raw_equivalent_mb"] = (
        _number(state, "obc_raw_equivalent_mb") + raw_equivalent_mb
    )
    return TransitionOutcome(
        projected,
        True,
        transferred_mb=transfer_mb,
        raw_equivalent_mb=raw_equivalent_mb,
    )


def apply_downlink(
    state: Mapping[str, Any],
    params: PipelineParameters,
    *,
    contact_seconds: float,
) -> TransitionOutcome:
    """Downlink bytes bounded by actual contact duration and OBC backlog."""

    projected = _copy_state(state)
    seconds = max(0.0, float(contact_seconds))
    source_mb = max(0.0, _number(state, "obc_data_mb"))
    rate_limit_mb = max(0.0, params.downlink_rate_kbps / 8.0 * seconds / 1000.0)
    transfer_mb = min(source_mb, rate_limit_mb)
    if transfer_mb <= CAPACITY_EPSILON_MB:
        reason = "no_contact" if seconds <= 0.0 else "no_source_data"
        return TransitionOutcome(projected, False, reason)

    raw_backlog_mb = max(0.0, _number(state, "obc_raw_equivalent_mb"))
    raw_equivalent_mb = 0.0
    if source_mb > 0.0:
        raw_equivalent_mb = min(raw_backlog_mb, transfer_mb * raw_backlog_mb / source_mb)
    projected["obc_data_mb"] = source_mb - transfer_mb
    projected["obc_raw_equivalent_mb"] = max(0.0, raw_backlog_mb - raw_equivalent_mb)
    projected["downlink_raw_equivalent_mb"] = (
        _number(state, "downlink_raw_equivalent_mb") + raw_equivalent_mb
    )
    projected["data_downlinked_mb"] = _number(state, "data_downlinked_mb") + transfer_mb
    return TransitionOutcome(
        projected,
        True,
        transferred_mb=transfer_mb,
        raw_equivalent_mb=raw_equivalent_mb,
    )


def with_total_storage(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with the backward-compatible total storage field updated."""

    projected = _copy_state(state)
    projected["data_stored_mb"] = (
        max(0.0, _number(state, "jetson_raw_mb"))
        + max(0.0, _number(state, "jetson_compressed_mb"))
        + max(0.0, _number(state, "obc_data_mb"))
    )
    return projected
