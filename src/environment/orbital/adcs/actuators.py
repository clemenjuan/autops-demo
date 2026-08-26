"""ADCS actuator models.

The apply functions simulate one actuator each: they take the true state and
environment, the actuator's config, and the command issued to it, and return the
body-frame torque it produces. 
ControlCommand is the bundle the controller emits
and simulation.step() unpacks into the per-actuator commands.

All functions are currently place-holders, so the loop runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from src.environment.orbital.adcs.configs import MagnetorquerConfig, ReactionWheelConfig
from src.environment.orbital.adcs.state import SatState
from src.environment.orbital.propagator import EnvironmentData
from src.environment.orbital.adcs.dynamics import dcm_eci_to_body

import logging

logger = logging.getLogger(__name__)


def apply_reaction_wheel(
    state: SatState,
    config: ReactionWheelConfig,
    command: float,
    wheel_index: int,
    dt: float,
) -> float:

    """Motor torque actually delivered to one flywheel [N·m], scalar.
    Args:
        state: True satellite state; supplies this wheel's current speed.
        config: This wheel's configuration.
        command: Commanded motor torque along the spin axis [N·m].
        wheel_index: Index of this wheel in ``state.wheel_speeds``.
        dt: Length of the step this torque will be held over [s].
    Returns:
        The achieved motor torque along ``config.spin_axis_body`` [N·m].
    """

    torque = float(np.clip(command, -config.max_torque, config.max_torque)) # Torque Clamping
    speed = float(state.wheel_speeds[wheel_index])
    
    torque -= config.friction_coulomb * np.sign(speed)  # Friction (Paluszek Eq. 10.14)
    torque -= config.friction_viscous * speed           # No stiction included, but will think about it

    momentum = config.wheel_inertia * speed

    # Saturation/Momentum limit
    lower_m = min(0.0, (-config.max_momentum - momentum) / dt)
    upper_m = max(0.0, (config.max_momentum - momentum) / dt)
    torque = float(np.clip(torque, lower_m, upper_m))

    return torque


def apply_magnetorquer(
    state: SatState,
    env: EnvironmentData,
    config: MagnetorquerConfig,
    command: float,
) -> np.ndarray:
    """Body-frame torque produced by one magnetorquer [N·m], shape (3,).

    Args:
        state: True satellite state; supplies the attitude for the field
            rotation.
        env: Environment at this instant; supplies the geomagnetic field.
        config: This rod's configuration.
        command: Commanded dipole magnitude along the rod axis [A·m²].

    Returns:
        Body-frame torque [N·m], shape (3,).
    """
    dipole_magnitude = float(
        np.clip(command, -config.max_dipole, config.max_dipole)
    )
    dipole_body = dipole_magnitude * config.axis_body

    b_body = dcm_eci_to_body(state.q_eci_body) @ env.b_field_eci

    return np.cross(dipole_body, b_body)


@dataclass
class ControlCommand:
    """Actuator commands for one timestep, produced by the controller.

    Attributes:
        wheel_commands: Torque command per reaction wheel [N·m], in the order of
            ActuatorSuite.reaction_wheels.
        mtq_commands: Dipole command per magnetorquer [A·m²], in the order of
            ActuatorSuite.magnetorquers.
    """

    wheel_commands: List[float]
    mtq_commands: List[float]