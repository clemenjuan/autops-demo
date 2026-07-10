"""Representation factory.

Controls whether representation logic is hand-designed or learned from
experience, then returns the configured representation class.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from src.core.representation import Representation


# Registry of known representation classes, populated by ``register()``.
_REPRESENTATION_REGISTRY: Dict[str, Type[Representation]] = {}


def register(name: str) -> Any:
    """Decorator to register a representation class in the factory.

    Usage::

        @register("symbolic_rule_based")
        class RuleBasedRepresentation(Representation):
            ...

    Args:
        name: Unique identifier used in YAML configuration files.

    Returns:
        The original class, unmodified.
    """

    def _decorator(cls: Type[Representation]) -> Type[Representation]:
        _REPRESENTATION_REGISTRY[name] = cls
        return cls

    return _decorator


class BehaviourController:
    """Factory for creating representation modules based on behaviour mode.

    The controller reads the experiment configuration and returns the
    appropriate representation instance — either a hand-designed variant
    or a learned variant.

    Attributes:
        config: Behaviour-specific configuration from experiment YAML.
        mode: Behaviour (``"hand_designed"`` or ``"emergent"``).
    """

    VALID_MODES = {"hand_designed", "emergent"}
    _SCENARIO_ACTION_SCHEMAS = {
        "eventsat": "eventsat_single",
        "multieventsat": "native_satellites",
        "ssa": "native_satellites",
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise the behaviour controller.

        Args:
            config: Behaviour configuration section from experiment YAML.
                Must contain at least ``mode``.

        Raises:
            ValueError: If the mode is not recognised.
        """
        self.config = config
        self.mode: str = config.get("mode", "hand_designed")
        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown behaviour mode '{self.mode}'. "
                f"Valid modes: {self.VALID_MODES}"
            )

    def get_representation(
        self,
        repr_type: str,
        repr_config: Dict[str, Any] | None = None,
    ) -> Representation:
        """Create and return a configured representation instance.

        Args:
            repr_type: Registered name of the representation class
                (e.g. ``"symbolic_rule_based"``).
            repr_config: Optional extra configuration for the representation.

        Returns:
            An initialised :class:`Representation` instance.

        Raises:
            KeyError: If ``repr_type`` is not found in the registry.
        """
        if repr_type not in _REPRESENTATION_REGISTRY:
            available = ", ".join(sorted(_REPRESENTATION_REGISTRY)) or "(none registered)"
            raise KeyError(
                f"Representation '{repr_type}' not found in registry. "
                f"Available: {available}"
            )

        representation_cls = _REPRESENTATION_REGISTRY[repr_type]
        merged_config = {**self.config, **(repr_config or {})}
        return representation_cls(config=merged_config)

    @staticmethod
    def validate_representation(repr_type: str, scenario: str) -> None:
        """Fail before reset when a core cannot control the chosen scenario."""
        if repr_type not in _REPRESENTATION_REGISTRY:
            available = ", ".join(sorted(_REPRESENTATION_REGISTRY)) or "(none registered)"
            raise KeyError(
                f"Representation '{repr_type}' not found in registry. "
                f"Available: {available}"
            )

        representation_cls = _REPRESENTATION_REGISTRY[repr_type]
        supported = frozenset(getattr(representation_cls, "supported_scenarios", ()))
        declared_schema = getattr(representation_cls, "action_key_schema", "")
        schema = (
            str(declared_schema.get(scenario, ""))
            if isinstance(declared_schema, dict)
            else str(declared_schema)
        )
        expected_schema = BehaviourController._SCENARIO_ACTION_SCHEMAS.get(scenario)
        # ``to_be_defined`` is retained for the environment-less orchestration
        # smoke harness. Every executable benchmark scenario is in the mapping
        # above and therefore receives strict support/schema validation.
        if expected_schema is None:
            return
        if scenario not in supported or schema != expected_schema:
            raise ValueError(
                f"Representation '{repr_type}' is incompatible with scenario "
                f"'{scenario}': it supports {sorted(supported)} with action-key "
                f"schema '{schema}', while this scenario requires "
                f"'{expected_schema or 'an explicitly declared schema'}'."
            )

    @staticmethod
    def list_registered() -> list[str]:
        """Return names of all registered representation classes.

        Returns:
            Sorted list of registered representation names.
        """
        return sorted(_REPRESENTATION_REGISTRY)
