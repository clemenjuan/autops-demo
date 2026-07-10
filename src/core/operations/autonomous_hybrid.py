"""Autonomous Hybrid operations paradigm (dual-slot).

Two distinct cores (morphological_matrix.md §3 — operational paradigm):
- a **ground planner** that produces a whole-pass uplinked schedule (the *same*
  artifact AutonomousGround uses), refreshed during ground passes from stale
  telemetry, and
- an **onboard** per-step core that runs every step on full real-time state and
  can **override** the uplinked plan.

Between passes the satellite follows the uplinked plan by default; the onboard
core overrides only when triggered (a safety mode — e.g. the onboard core forces
charging/safe on low SoC / anomaly). During passes the onboard action is still
the env-facing action, so communication is a controller decision rather than a
paradigm rewrite. Onboard compute (Jetson) is powered every step
(``has_onboard_autonomy() = True``).

The runner wires the two cores: it runs the onboard loop every step (full state)
and, at ground passes, runs the ground planner on the stale view and calls
``set_uplinked_plan``. ``process_action`` then arbitrates.

In EventSat no LLM runs onboard; the onboard core is rules / a small RL policy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.operations.base import OperationsParadigm


class AutonomousHybrid(OperationsParadigm):
    """Dual-slot onboard + ground-planner operations with plan-default override."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        super().__init__(config)
        # Uplinked plan (mutable list of [mode, remaining_steps]) from the ground planner.
        self._uplinked_schedule: List[List] = []
        self._schedule_index: int = 0
        self._upload_candidate: Optional[List[List]] = None
        # Onboard overrides the plan only when its mode is one of these (safety).
        self._override_modes = set(
            self.config.get("onboard_override_modes", ["charging", "safe"])
        )
        # If True, onboard always wins (closed-loop); for ablation experiments.
        self._onboard_authoritative: bool = self.config.get("onboard_authoritative", False)
        self._onboard_overrides: int = 0
        self._plan_steps: int = 0

    # -- observation views ------------------------------------------------
    def filter_observation(self, full_observation: Any, step: int) -> Any:
        """Onboard sees full real-time state."""
        return full_observation

    def ground_planner_view(self, full_observation: Any, step: int) -> Any:
        """Stale ground-segment view the ground planner plans from (last downlink)."""
        return self._stale_ground_observation(full_observation, step)

    # -- timing / autonomy ------------------------------------------------
    def can_act(self, step: int, ground_pass_active: bool) -> bool:
        return True

    def should_allow_inference(self, step: int, ground_pass_active: bool) -> bool:
        """Onboard inference runs every step (real-time)."""
        return True

    def can_self_recover_anomaly(self) -> bool:
        """Onboard FDIR can clear anomalies without ground contact."""
        return True

    def has_onboard_autonomy(self) -> bool:
        """AH runs the onboard per-step core — Jetson powered every step."""
        return True

    # -- ground plan ------------------------------------------------------
    def set_uplinked_plan(self, ground_action: Dict[str, Any]) -> None:
        """Stage a ground plan for uplink at an established contact.

        The name is retained for the runner API, but this method cannot prove
        that the spacecraft reached communication mode. Promotion therefore
        occurs only in update_ground_knowledge().
        """
        sat_action = ground_action.get("eventsat_0", {})
        schedule = sat_action.get("schedule")
        if schedule:
            self._upload_candidate = [[mode, steps] for mode, steps in schedule]

    def update_ground_knowledge(self, full_observation: Any, step: int) -> None:
        """Promote a staged ground plan after resolved bidirectional contact."""
        super().update_ground_knowledge(full_observation, step)
        if self._upload_candidate is not None:
            self._uplinked_schedule = self._upload_candidate
            self._schedule_index = 0
            self._upload_candidate = None

    # -- arbitration ------------------------------------------------------
    def process_action(
        self,
        action: Dict[str, Any],
        step: int,
        ground_pass_active: bool,
    ) -> Dict[str, Any]:
        """Arbitrate the onboard action against the uplinked plan.

        During a pass: execute the onboard (real-time) action. Between passes:
        follow the uplinked plan unless the onboard core triggers an override
        (its mode is a configured safety mode and differs from the plan).
        """
        onboard_mode = action.get("eventsat_0", {}).get("mode", self._default_mode)

        if ground_pass_active:
            return action

        # No / exhausted plan → onboard takes over (closed-loop fallback).
        if self._schedule_index >= len(self._uplinked_schedule):
            return action

        plan_action, self._schedule_index = self._consume_schedule_list(
            self._uplinked_schedule, self._schedule_index
        )
        self._plan_steps += 1
        plan_mode = plan_action.get("eventsat_0", {}).get("mode", self._default_mode)

        override = self._onboard_authoritative or (
            onboard_mode != plan_mode and onboard_mode in self._override_modes
        )
        if override:
            self._onboard_overrides += 1
            return action
        return plan_action

    def get_metrics(self) -> Dict[str, float]:
        """Override accounting (between-pass steps where onboard overrode the plan)."""
        rate = (self._onboard_overrides / self._plan_steps) if self._plan_steps else 0.0
        return {
            "onboard_overrides": float(self._onboard_overrides),
            "onboard_override_rate": rate,
        }

    def reset(self) -> None:
        super().reset()
        self._uplinked_schedule = []
        self._schedule_index = 0
        self._upload_candidate = None
        self._onboard_overrides = 0
        self._plan_steps = 0
