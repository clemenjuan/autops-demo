"""Rigid-body attitude dynamics and environmental disturbances.

integrate is the only writer of the true rotational state: given the net torque
on the body, it advances attitude, angular velocity, and wheel speeds over one
timestep.

disturbance_torque returns the net environmental torque that feeds into it.

All functions are currently place-holders, so the loop runs.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.environment.orbital.adcs.state import SatState
from src.environment.orbital.propagator import EnvironmentData
from src.environment.orbital.adcs.configs import SatelliteConfig

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product ``q1 ⊗ q2`` (scalar-first ``[w, x, y, z]``).
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_kinematics(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Quaternion time-derivative ``q̇`` for a body-frame angular velocity.

    Locked convention (Solà, arXiv:1711.02508): ``q`` is the scalar-first
    ECI->body attitude quaternion, ``omega`` is the body-frame angular
    velocity [rad/s], and

        q̇ = ½ q ⊗ [0, ω].

    ``omega`` is promoted to the pure quaternion ``[0, ωx, ωy, ωz]`` before the
    product. This is the kinematics half of the convention; the matching
    ECI->body rotation matrix (built from the conjugate of ``q``) arrives with
    the gravity-gradient disturbance, and a single-axis spin test will lock the
    two halves together.
    """
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
    return 0.5 * quat_multiply(q, omega_quat)

def _omega_dot(
    omega: np.ndarray,
    wheel_speeds: np.ndarray,
    body_torque: np.ndarray,
    wheel_torque: np.ndarray,
    params: SatelliteConfig,
) -> np.ndarray:
    """Body angular acceleration ω̇ for the full reaction-wheel gyrostat.

    Spec body equation, reduced-inertia form:

        h  = I·ω + W·J_w·(Ω + Wᵀ·ω)      # total system momentum (body frame)
        ω̇ = I⁻¹·[ τ_ext − ω×h − W·u_w ]

    with I the reduced inertia (cached on ``params``), τ_ext = ``body_torque``
    (magnetorquer + disturbances), and u_w = ``wheel_torque`` (per-wheel motor
    torque). The ``Wᵀ·ω`` inside h is each wheel's *absolute* spin — its speed
    relative to the body plus the body's own rotation along that wheel's axis —
    and ``W·u_w`` is the motor reaction on the body.
    """
    W = params.wheel_axes            # (3,4)
    Jw = params.wheel_inertia_mat     # (4,4)
    h = params.inertia @ omega + W @ Jw @ (wheel_speeds + W.T @ omega)
    return params.inertia_inv @ (body_torque - np.cross(omega, h) - W @ wheel_torque)


def integrate(state: SatState, total_torque: np.ndarray, dt: float) -> SatState:
    """Advance the true rotational state by one timestep.

    Args:
        state: Current true satellite state.
        total_torque: Net body-frame torque acting on the satellite [N·m],
            shape (3,) — the sum of actuator and disturbance torques.
        dt: Timestep [s].

    Returns:
        The state advanced by dt.
    """
    return replace(state, t=state.t + dt)


def disturbance_torque(state: SatState, env: EnvironmentData) -> np.ndarray:
    """Net environmental disturbance torque in the body frame [N·m], shape (3,).

    Args:
        state: Current true satellite state.
        env: Environment at this instant.

    Returns:
        The summed disturbance torque.
    """
    return np.zeros(3)