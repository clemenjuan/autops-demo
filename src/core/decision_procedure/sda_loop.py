"""
SDA (Sense-Decide-Act) Decision Loop.

Simplest possible decision loop: one pass through sense -> decide -> act
with no iteration, reflection, or memory update.

Grounded in the classical situated-action / reactive-agent tradition:
  - Agre, P. E. & Chapman, D. (1990), "What are plans for?", Robotics and
    Autonomous Systems 6(1-2):17-34 [Zotero: BERWVN2V]. Plan-as-communication
    view motivates treating each cycle as a fresh sensing-then-action step
    rather than execution of a pre-computed program.

The SDA variant is the fixed benchmark driver: no memory read, no orient step,
and no iteration.
"""
from __future__ import annotations
import time
from typing import Any, Dict, Tuple
from src.core.decision_procedure.base import DecisionProcedure
from src.core.decision_procedure.context import DecisionContext
from src.core.representation import Representation


class SDALoop(DecisionProcedure):
    """Sense-Decide-Act: single-pass decision making."""

    def __init__(self, config: Dict[str, Any], representation: Representation) -> None:
        self.config = config
        self.representation = representation
        self._last_latency: float = 0.0
        self._total_steps: int = 0
        self._last_has_rationale: bool = False

    def process(
        self, observation: Any, memory: Any
    ) -> Tuple[Dict[str, Any], Any]:
        t0 = time.perf_counter()
        # Unwrap AgentObservation unless the representation explicitly consumes
        # organisation-level metadata such as messages.
        raw_obs = observation
        messages = list(getattr(observation, "messages", []) or [])
        organization_metadata = dict(getattr(observation, "metadata", {}) or {})
        agent_id = getattr(observation, "agent_id", None)
        if (
            not getattr(self.representation, "uses_agent_observation", False)
            and hasattr(observation, "local_state")
            and isinstance(observation.local_state, dict)
        ):
            raw_obs = observation.local_state.get("full_observation", observation)
        encoded = self.representation.encode_observation(raw_obs)
        context = DecisionContext(
            state=encoded,
            loop_type="sda",
            memory=memory,
            enrichments={
                "organization_messages": messages,
                "organization_metadata": organization_metadata,
            },
            loop_metadata={"agent_id": agent_id} if agent_id is not None else {},
        )
        action = self.representation.select_action(context)
        action = self._apply_safety_directives(action, messages)
        self._last_latency = time.perf_counter() - t0
        self._total_steps += 1
        # Check if representation provides a rationale (explainability)
        self._last_has_rationale = (
            hasattr(self.representation, "get_rationale")
            and self.representation.get_rationale() is not None
        )
        return action, memory

    @staticmethod
    def _apply_safety_directives(
        action: Dict[str, Any], messages: list[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply the minimal organization-wide hierarchy contract.

        CMAS manager directives are strategic vetoes, not replacement tactical
        policies. ``safe`` always overrides a local action; ``charging`` vetoes
        payload work while leaving ordinary communication/safe choices local.
        Other manager/peer messages remain representation context only.
        """
        if not isinstance(action, dict) or not messages:
            return action

        directive_by_satellite: Dict[str, str] = {}
        global_directive = None
        for message in messages:
            if not isinstance(message, dict) or "directive" not in message:
                continue
            directive = message["directive"]
            if isinstance(directive, str) and directive in {"safe", "charging"}:
                global_directive = directive
            elif isinstance(directive, dict):
                mode = directive.get("mode")
                if mode in {"safe", "charging"}:
                    global_directive = str(mode)
                for satellite_id, payload in directive.items():
                    if isinstance(payload, dict) and payload.get("mode") in {
                        "safe",
                        "charging",
                    }:
                        directive_by_satellite[str(satellite_id)] = str(payload["mode"])

        if global_directive is None and not directive_by_satellite:
            return action

        # Direct single-action payload (rather than satellite -> payload).
        if "mode" in action:
            mode = global_directive
            current = str(action.get("mode", ""))
            if mode == "safe" or (mode == "charging" and current.startswith("payload_")):
                return {**action, "mode": mode}
            return action

        rewritten = dict(action)
        for satellite_id, payload in action.items():
            if not isinstance(payload, dict):
                continue
            directive_mode = directive_by_satellite.get(str(satellite_id), global_directive)
            current = str(payload.get("mode", ""))
            if directive_mode == "safe" or (
                directive_mode == "charging" and current.startswith("payload_")
            ):
                rewritten[satellite_id] = {**payload, "mode": directive_mode}
        return rewritten

    def get_metrics(self) -> Dict[str, Any]:
        rationale = ""
        if hasattr(self.representation, "get_rationale"):
            rationale = self.representation.get_rationale() or ""
        metrics = {
            "decision_latency_s": self._last_latency,
            "total_decisions": float(self._total_steps),
            "has_rationale": float(self._last_has_rationale),
            "rationale": rationale,
        }
        if hasattr(self.representation, "get_metrics"):
            for key, value in self.representation.get_metrics().items():
                if key not in metrics:
                    metrics[key] = value
        return metrics

    def reset(self) -> None:
        """Clear per-episode driver metrics."""
        self._last_latency = 0.0
        self._total_steps = 0
        self._last_has_rationale = False
