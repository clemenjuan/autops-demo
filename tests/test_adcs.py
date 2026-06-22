"""Test for the ADCS simulation skeleton.
Confirms the closed loop runs through all modules and produces a state history
of the expected shape. 
It checks no physics (every dummy function returns zeros)
Its job is to catch wiring and interface breakage as real implementations replace 
the dummy ones, so it asserts structure correctness. 
"""
from __future__ import annotations
from typing import List
import pytest
from src.environment.orbital.adcs.control import initial_setpoint
from src.environment.orbital.adcs.estimator import initial_estimator_state
from src.environment.orbital.adcs.eventsat import actuators, satellite, sensors
from src.environment.orbital.adcs.simulation import initial_state, run, step
from src.environment.orbital.adcs.state import SatState
from src.environment.orbital import propagator as P
from src.environment.orbital.adcs.eventsat import actuators, satellite, sensors, orbit

requires_orekit = pytest.mark.skipif(
    not P.is_available(), reason="Orekit unavailable; skipping physics checks."
)

STEP_S = 1.0
START_STEP = 0
END_STEP = 10
@pytest.fixture
def history() -> List[SatState]:
    """Run the skeleton once and share the resulting state history."""
    return run(
        sensors, actuators, satellite, step_s=STEP_S, start_step=START_STEP, end_step=END_STEP
    )
def test_eventsat_config_counts() -> None:
    """The EventSat suite has the expected instrument counts."""
    assert len(sensors.magnetometers) == 2
    assert len(sensors.fine_sun_sensors) == 2
    assert len(sensors.star_trackers) == 0
    assert len(actuators.reaction_wheels) == 4
    assert len(actuators.magnetorquers) == 3
def test_run_executes_end_to_end(history: List[SatState]) -> None:
    """run() completes and returns one state per step boundary."""
    assert len(history) == END_STEP - START_STEP + 1
    assert all(isinstance(s, SatState) for s in history)
def test_run_advances_time(history: List[SatState]) -> None:
    """Time runs from start_step * step_s to end_step * step_s."""
    assert history[0].t == START_STEP * STEP_S
    assert history[-1].t == END_STEP * STEP_S
def test_final_state_shapes(history: List[SatState]) -> None:
    """The final state has the expected vector shapes."""
    final = history[-1]
    assert final.q_eci_body.shape == (4,)
    assert final.omega_body.shape == (3,)
    assert final.wheel_speeds.shape == (len(actuators.reaction_wheels),)
    assert final.r_eci.shape == (3,)
    assert final.v_eci.shape == (3,)
def test_single_step_returns_state_and_estimator() -> None:
    """One step returns an advanced state and an estimator with a 6x6 covariance."""
    state = initial_state(0.0, len(actuators.reaction_wheels))
    estimator = initial_estimator_state()
    new_state, new_estimator = step(
        state, estimator, sensors, actuators, satellite, initial_setpoint(), STEP_S
    )
    assert isinstance(new_state, SatState)
    assert new_state.t == STEP_S
    assert new_estimator.covariance.shape == (6, 6)

@requires_orekit
def test_orbit_propagation_physics() -> None:
    """With an orbit configured, the propagator yields a physically correct SSO.

    Checks (tolerances loose, only catching gross errors):
      - |r|, |v| match a circular orbit at the configured altitude
      - r . v ~ 0 (circular: position perpendicular to velocity)
      - position roughly reverses after half a period
      - RAAN drifts at the sun-synchronous rate (~+0.986 deg/day) -> J2 is active
    """
    import numpy as np

    P.configure(orbit)

    # Expectations derived from the same config, not hard-coded.
    mu = 3.986004418e14  # WGS84 Earth mu [m^3/s^2]
    r_e = 6378137.0      # WGS84 equatorial radius [m]
    a = r_e + orbit.altitude_km * 1000.0
    v_circular = np.sqrt(mu / a)
    period = 2.0 * np.pi * np.sqrt(a**3 / mu)

    def raan_deg(r: np.ndarray, v: np.ndarray) -> float:
        h = np.cross(r, v)                    # orbit normal
        node = np.array([-h[1], h[0], 0.0])   # z x h -> ascending-node direction
        return float(np.degrees(np.arctan2(node[1], node[0])))

    e0 = P.get_environment(0.0)
    r0 = np.linalg.norm(e0.r_eci)
    v0 = np.linalg.norm(e0.v_eci)

    # Magnitudes within ~10 km / ~10 m/s of the circular ideal.
    assert abs(r0 - a) < 1.0e4
    assert abs(v0 - v_circular) < 1.0e1
    # Circular: r perpendicular to v (dot product small relative to r*v).
    assert abs(np.dot(e0.r_eci, e0.v_eci)) < 1.0e-3 * r0 * v0

    # Half a period later, position points roughly the opposite way.
    eh = P.get_environment(period / 2.0)
    cos_ang = np.dot(e0.r_eci, eh.r_eci) / (r0 * np.linalg.norm(eh.r_eci))
    assert cos_ang < -0.99

    # RAAN drift over one day ~ +0.986 deg/day confirms J2 precession is on.
    # (A Keplerian fallback would give ~0 and fail this.)
    ed = P.get_environment(86400.0)
    drift = (raan_deg(ed.r_eci, ed.v_eci) - raan_deg(e0.r_eci, e0.v_eci) + 180.0) % 360.0 - 180.0
    assert 0.9 < drift < 1.1