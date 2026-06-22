"""Closed-loop ADCS simulation.

step() runs the full control loop once
run() drives it over a range of steps

Data flow:

    sense -> estimate -> control -> actuate -> disturb -> integrate -> reconcile

The controller sees only the estimator state, never.
The dynamics integrator and the propagator are the only writers of the true state.

Time is step-based. 
run() works in integer step indices (start_step, end_step) with a fixed
step length (step_s [s]).

Continuous time (state.t) is carried in seconds for the physics that needs it.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np

from src.environment.orbital.adcs.actuators import apply_magnetorquer, apply_reaction_wheel
from src.environment.orbital.adcs.configs import ActuatorSuite, SensorSuite, SatelliteConfig, OrbitConfig
from src.environment.orbital.adcs.control import Setpoint, compute_control, initial_setpoint
from src.environment.orbital.adcs.dynamics import disturbance_torque, integrate
from src.environment.orbital.adcs.estimator import (
    EstimatorState,
    initial_estimator_state,
    update_estimator,
)
from src.environment.orbital.adcs.sensors import (
    SensorMeasurements,
    read_coarse_sun_sensor,
    read_earth_horizon,
    read_fine_sun_sensor,
    read_magnetometer,
    read_star_tracker,
)
from src.environment.orbital.adcs.state import SatState
from src.environment.orbital.propagator import get_environment, configure

logger = logging.getLogger(__name__)


def initial_state(t: float = 0.0, n_wheels: int = 4) -> SatState:
    """Create the initial true state.

    Args:
        t: Initial time since the simulation epoch [s].
        n_wheels: Number of reaction wheels (sizes the wheel-speed vector).
    """
    return SatState(
        t=t,
        q_eci_body=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_body=np.zeros(3),
        wheel_speeds=np.zeros(n_wheels),
        r_eci=np.zeros(3),
        v_eci=np.zeros(3),
    )


def step(
    state: SatState,
    estimator: EstimatorState,
    sensors: SensorSuite,
    actuators: ActuatorSuite,
    satellite: SatelliteConfig,
    setpoint: Setpoint,
    step_s: float,
) -> Tuple[SatState, EstimatorState]:
    """Advance the simulation by one timestep, running the full closed loop.

    Args:
        state: Current true satellite state.
        estimator: Current estimator state.
        sensors: The sensor suite.
        actuators: The actuator suite.
        satellite: Satellite physical parameters (mass properties + wheel
            geometry) used by the rigid-body integrator.
        setpoint: The target the controller tracks.
        step_s: Timestep length [s].

    Returns:
        A tuple of (new true state, updated estimator state).
    """
    env = get_environment(state.t)

    # Sense: every sensor reads the true state and the true environment.
    measurements = SensorMeasurements(
        magnetometers=[read_magnetometer(state, env, mag) for mag in sensors.magnetometers],
        fine_sun_sensors=[
            read_fine_sun_sensor(state, env, fss) for fss in sensors.fine_sun_sensors
        ],
        coarse_sun=read_coarse_sun_sensor(state, env, sensors.coarse_sun_sensor),
        earth_horizon=read_earth_horizon(state, env, sensors.earth_horizon_sensor),
        star_trackers=[read_star_tracker(state, env, st) for st in sensors.star_trackers],
    )

    # Estimate: fuse the measurements into the estimated state.
    estimator = update_estimator(estimator, measurements, step_s)

    # Control: the controller sees only the estimate, never the truth.
    command = compute_control(estimator, setpoint, actuators, step_s)

    # Actuate: magnetorquers produce a body-frame torque directly; the reaction
    # wheels' effect is folded into the gyrostat inside integrate.
    # INTERIM: until apply_reaction_wheel returns a per-wheel motor torque, the
    # wheels are left unforced (wheel_torque = 0) and their 3-vector outputs are
    # summed into the body torque. Runs the loop; not the final gyrostat wiring.
    body_torque = np.zeros(3)
    for wheel, cmd in zip(actuators.reaction_wheels, command.wheel_commands):
        body_torque = body_torque + apply_reaction_wheel(state, env, wheel, cmd)
    for rod, cmd in zip(actuators.magnetorquers, command.mtq_commands):
        body_torque = body_torque + apply_magnetorquer(state, env, rod, cmd)

    # Disturb: environmental disturbance torque adds into the body torque.
    body_torque = body_torque + disturbance_torque(state, env, satellite)

    wheel_torque = np.zeros(len(actuators.reaction_wheels))

    # Integrate: the sole writer of the true rotational state advances it.
    new_state = integrate(state, body_torque, wheel_torque, satellite, step_s)

    # Reconcile: the orbital state comes from the propagator at the new time.
    env_next = get_environment(new_state.t)
    new_state = replace(new_state, r_eci=env_next.r_eci, v_eci=env_next.v_eci)

    return new_state, estimator


def run(
    sensors: SensorSuite,
    actuators: ActuatorSuite,
    satellite: SatelliteConfig,
    step_s: float,
    start_step: int,
    end_step: int,
    setpoint: Optional[Setpoint] = None,
    orbit: Optional[OrbitConfig] = None
) -> List[SatState]:
    if orbit is not None:
        configure(orbit)
    if setpoint is None:
        setpoint = initial_setpoint()
    """Run the closed-loop simulation over a range of steps.

    Args:
        sensors: The sensor suite.
        actuators: The actuator suite.
        satellite: Satellite physical parameters passed to the integrator.
        step_s: Timestep length [s].
        start_step: First step index (inclusive).
        end_step: Final step index (exclusive); the run executes
            ``end_step - start_step`` steps.
        setpoint: Target for the controller; defaults to holding identity
            attitude at zero rate.

    Returns:
        The state history: the initial state followed by the state after each
        step (length ``end_step - start_step + 1``).
    """
    if setpoint is None:
        setpoint = initial_setpoint()

    state = initial_state(start_step * step_s, len(actuators.reaction_wheels))
    estimator = initial_estimator_state()
    history = [state]

    logger.info(
        "Running ADCS simulation: steps %d..%d at %g s/step",
        start_step,
        end_step,
        step_s,
    )
    for _ in range(start_step, end_step):
        state, estimator = step(
            state, estimator, sensors, actuators, satellite, setpoint, step_s
        )
        history.append(state)

    return history