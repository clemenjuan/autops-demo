"""RL backend integration helpers.

The scientific emergence mechanism remains ``ppo``.  This package contains
the technical RLlib bridge used to train and load that mechanism.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np


# Every declared EventSat/SSA RL space uses this upper feature bound. Changing
# the saturation semantic changes learned inputs, so existing artifacts must be
# retrained before they are compared with post-change policies.
RL_FEATURE_MAX = 2.0


def observation_bounds(
    size: int,
    *,
    signed_indices: Iterable[int] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Return the canonical bounds used by both adapters and regressions."""
    low = np.zeros(int(size), dtype=np.float32)
    high = np.full(int(size), RL_FEATURE_MAX, dtype=np.float32)
    for index in signed_indices:
        if 0 <= index < low.size:
            low[index] = -1.0
    return low, high


def observation_within_bounds(
    vector: Any,
    *,
    size: int,
    signed_indices: Iterable[int] = (),
) -> bool:
    """Dependency-free equivalent of ``Box.contains`` for the base install."""
    arr = np.asarray(vector, dtype=np.float32)
    low, high = observation_bounds(size, signed_indices=signed_indices)
    return bool(
        arr.shape == low.shape
        and np.isfinite(arr).all()
        and np.greater_equal(arr, low).all()
        and np.less_equal(arr, high).all()
    )


def bounded_ratio(value: Any, scale: Any, upper: float = RL_FEATURE_MAX) -> float:
    """Return a finite, non-negative ratio bounded by the declared space."""
    try:
        numerator = float(value)
    except (TypeError, ValueError):
        numerator = 0.0
    try:
        denominator = float(scale)
    except (TypeError, ValueError):
        denominator = 0.0
    if not math.isfinite(numerator):
        numerator = 0.0
    if not math.isfinite(denominator) or denominator <= 0.0:
        return float(upper if numerator > 0.0 else 0.0)
    return float(min(max(numerator / denominator, 0.0), upper))


def downlink_utilization(
    resources: Mapping[str, Any],
    metadata: Mapping[str, Any],
    storage_capacity_mb: float,
) -> float:
    """Bound cumulative downlink against the explicitly available capacity."""
    scale = metadata.get("max_achievable_downlink_mb")
    if scale is None:
        scale = metadata.get("achievable_downlink_mb")
    if scale is None:
        scale = storage_capacity_mb
    return bounded_ratio(resources.get("data_downlinked_mb", 0.0), scale)


def bound_observation_vector(
    vector: Any,
    *,
    signed_indices: Iterable[int] = (),
) -> np.ndarray:
    """Make a feature vector satisfy the shared [0, 2] / signed bounds."""
    original = np.asarray(vector, dtype=np.float32)
    arr = np.nan_to_num(
        original.copy(),
        copy=False,
        nan=0.0,
        posinf=RL_FEATURE_MAX,
        neginf=0.0,
    )
    np.clip(arr, 0.0, RL_FEATURE_MAX, out=arr)
    flat_original = original.reshape(-1)
    flat_bounded = arr.reshape(-1)
    for index in signed_indices:
        if 0 <= index < flat_bounded.size:
            value = float(flat_original[index])
            flat_bounded[index] = value if math.isfinite(value) else 0.0
            flat_bounded[index] = np.clip(flat_bounded[index], -1.0, RL_FEATURE_MAX)
    return arr
