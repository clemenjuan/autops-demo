"""SSA metrics collector."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Dict, List

from src.core.metrics_collector import EpisodeMetrics, StepMetrics
from src.eventsat.metrics import EventSatMetricsCollector


def geometric_utility_ceiling(
    pass_windows: Iterable[Mapping[str, Any]],
    visibility_timeline: Iterable[Mapping[str, Any] | tuple[int, Iterable[str]]],
    target_count: int,
) -> float:
    """Fraction of targets observable early enough for a final-pass delivery.

    ``visibility_timeline`` is agent-free geometry: per step, the RSO IDs that
    physically enter at least one satellite FOV/range. A target contributes only
    if its first observable step is strictly before the final ground-pass start.
    """
    catalog_size = int(target_count)
    if catalog_size <= 0:
        return 0.0

    last_pass_start: int | None = None
    for window in pass_windows:
        raw_start = window.get("start_step") if isinstance(window, Mapping) else None
        if raw_start is None:
            continue
        try:
            start = int(raw_start)
        except (TypeError, ValueError):
            continue
        if last_pass_start is None or start > last_pass_start:
            last_pass_start = start

    if last_pass_start is None:
        return 0.0

    observable_before_last_pass: set[str] = set()
    for item in visibility_timeline:
        if isinstance(item, Mapping):
            raw_step = item.get("step", item.get("timestep", 0))
            visible = (
                item.get("visible_target_ids")
                or item.get("visible_rso_ids")
                or item.get("target_ids")
                or ()
            )
        else:
            raw_step, visible = item
        try:
            step = int(raw_step)
        except (TypeError, ValueError):
            continue
        if step >= last_pass_start:
            break
        if isinstance(visible, str):
            observable_before_last_pass.add(visible)
        else:
            observable_before_last_pass.update(str(target_id) for target_id in visible)

    return min(len(observable_before_last_pass), catalog_size) / catalog_size


class SSAMetricsCollector(EventSatMetricsCollector):
    """EventSat metrics plus SSA coverage, duplication, ISL, and M-10 hooks."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._baseline_utility_n1 = float(self.config.get("baseline_utility_n1", 0.0))
        self._constellation_size = int(self.config.get("constellation_size", 1))

    def collect_step_metrics(
        self,
        timestep: int,
        wall_clock_seconds: float,
        env_state: Any,
        actions: Any,
        rewards: Dict[str, float],
        info: Dict[str, Any],
        decision_metrics: Dict[str, Any],
    ) -> StepMetrics:
        step = super().collect_step_metrics(
            timestep=timestep,
            wall_clock_seconds=wall_clock_seconds,
            env_state=env_state,
            actions=actions,
            rewards=rewards,
            info=info,
            decision_metrics=decision_metrics,
        )
        step.metrics.update({
            "ssa_onboard_coverage": float(info.get("ssa_onboard_coverage", 0.0)),
            "ssa_delivered_coverage": float(info.get("ssa_delivered_coverage", 0.0)),
            "ssa_delivered_coverage_auc": float(info.get("ssa_delivered_coverage_auc", 0.0)),
            "duplicate_observation_rate": float(info.get("duplicate_observation_rate", 0.0)),
            "mean_revisit_steps": float(info.get("mean_revisit_steps", 0.0)),
            "mean_staleness_steps": float(info.get("mean_staleness_steps", 0.0)),
            "mean_delivery_latency_steps": float(info.get("mean_delivery_latency_steps", 0.0)),
            "relayed_delivery_fraction": float(info.get("relayed_delivery_fraction", 0.0)),
            "isl_connectivity": float(info.get("isl_connectivity", 0.0)),
            "isl_records_relayed": float(info.get("isl_records_relayed", 0.0)),
            "isl_bytes_transferred": float(info.get("isl_bytes_transferred", 0.0)),
            "ssa_delivered_objects": float(info.get("ssa_delivered_objects", 0.0)),
            "ssa_known_objects": float(info.get("ssa_known_objects", 0.0)),
            "physical_utility_ceiling": float(info.get("physical_utility_ceiling", 0.0)),
        })
        return step

    def aggregate_episode_metrics(self, step_metrics: List[StepMetrics]) -> EpisodeMetrics:
        episode = super().aggregate_episode_metrics(step_metrics)
        if not step_metrics:
            return episode
        last = step_metrics[-1].metrics
        delivered_coverage = float(last.get("ssa_delivered_coverage", 0.0))
        utility = delivered_coverage
        if self._baseline_utility_n1 > 0.0 and self._constellation_size > 0:
            eta_scale = (utility / self._constellation_size) / self._baseline_utility_n1
        else:
            eta_scale = 0.0
        total_energy = float(episode.aggregated.get("total_energy_consumed_wh", 0.0))
        resource_efficiency = utility / total_energy if total_energy > 0.0 else 0.0
        physical_utility_ceiling = float(last.get("physical_utility_ceiling", 0.0))
        utility_fraction_of_physical_ceiling = (
            utility / physical_utility_ceiling
            if physical_utility_ceiling > 0.0 else 0.0
        )
        eta_scale_vs_1overN = utility
        episode.aggregated.update({
            "utility": utility,
            "resource_efficiency": resource_efficiency,
            "mission_goal_utility": 1.0,
            "physical_utility_ceiling": physical_utility_ceiling,
            "utility_fraction_of_physical_ceiling": utility_fraction_of_physical_ceiling,
            "ssa_onboard_coverage": float(last.get("ssa_onboard_coverage", 0.0)),
            "ssa_delivered_coverage": delivered_coverage,
            "ssa_delivered_coverage_auc": float(last.get("ssa_delivered_coverage_auc", 0.0)),
            "duplicate_observation_rate": float(last.get("duplicate_observation_rate", 0.0)),
            "mean_revisit_steps": float(last.get("mean_revisit_steps", 0.0)),
            "mean_staleness_steps": float(last.get("mean_staleness_steps", 0.0)),
            "mean_delivery_latency_steps": float(last.get("mean_delivery_latency_steps", 0.0)),
            "relayed_delivery_fraction": float(last.get("relayed_delivery_fraction", 0.0)),
            "isl_connectivity": float(last.get("isl_connectivity", 0.0)),
            "isl_records_relayed": float(last.get("isl_records_relayed", 0.0)),
            "isl_bytes_transferred": float(last.get("isl_bytes_transferred", 0.0)),
            "ssa_delivered_objects": float(last.get("ssa_delivered_objects", 0.0)),
            "ssa_known_objects": float(last.get("ssa_known_objects", 0.0)),
            "eta_scale": eta_scale,
            "eta_scale_vs_1overN": eta_scale_vs_1overN,
        })
        return episode
