"""Fragmentation-family targets, propagation, and optical access for SSA.

Catalog draws use paired episode seeds, breakup-scale velocity dispersions, and
an aged near-co-planar debris torus. Orekit propagation is used when available;
a deterministic two-body fallback keeps local tests cheap and offline.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
import random
from typing import Iterable, Mapping, Sequence

from src.orbital import propagator


_EARTH_RADIUS_KM = 6371.0
_MU_EARTH_KM3_S2 = propagator.MU_EARTH / 1.0e9
_DEFAULT_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class RSOTarget:
    """Synthetic resident-space-object orbit definition."""

    object_id: str
    semi_major_axis_km: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    arg_perigee_deg: float
    true_anomaly_deg: float
    size_m: float = 1.0
    priority: float = 1.0
    epoch: datetime = field(default_factory=lambda: _DEFAULT_EPOCH)


@dataclass(frozen=True)
class DetectionAccess:
    """A target passing the deterministic optical-access gates."""

    object_id: str
    position_km: tuple[float, float, float]
    range_km: float
    angle_deg: float
    m: float
    p_detect: float
    quality: float


def phase_function(phi_rad: float) -> float:
    """Diffuse Lambertian-sphere phase function."""

    phi = min(math.pi, max(0.0, float(phi_rad)))
    if phi >= math.pi:
        return 0.0
    value = (2.0 / (3.0 * math.pi ** 2)) * (
        (math.pi - phi) * math.cos(phi) + math.sin(phi)
    )
    return max(0.0, value)


def apparent_magnitude(
    size_m: float,
    range_km: float,
    phase_rad: float,
    albedo: float = 0.13,
) -> float:
    """Return apparent visual magnitude for a diffuse spherical target."""

    if size_m <= 0.0 or range_km <= 0.0 or albedo <= 0.0:
        return math.inf
    phase = phase_function(phase_rad)
    flux_ratio = (
        float(albedo)
        * float(size_m) ** 2
        / (4.0 * (float(range_km) * 1000.0) ** 2)
        * phase
    )
    if flux_ratio <= 0.0:
        return math.inf
    return -26.74 - 2.5 * math.log10(flux_ratio)


def sun_unit_eci(
    epoch_seconds: float,
    epoch_datetime: datetime,
) -> tuple[float, float, float]:
    """Low-precision analytic Earth-to-Sun unit vector in mean-equator ECI.

    The mean-longitude/mean-anomaly approximation is the deterministic
    Vallado-style solar ephemeris appropriate to the cylindrical shadow gate.
    """

    epoch = epoch_datetime
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=timezone.utc)
    else:
        epoch = epoch.astimezone(timezone.utc)
    when = epoch + timedelta(seconds=float(epoch_seconds))
    j2000 = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
    days = (when - j2000).total_seconds() / 86400.0
    mean_longitude = math.radians((280.460 + 0.9856474 * days) % 360.0)
    mean_anomaly = math.radians((357.528 + 0.9856003 * days) % 360.0)
    ecliptic_longitude = mean_longitude + math.radians(
        1.915 * math.sin(mean_anomaly) + 0.020 * math.sin(2.0 * mean_anomaly)
    )
    obliquity = math.radians(23.439 - 0.0000004 * days)
    return _unit((
        math.cos(ecliptic_longitude),
        math.cos(obliquity) * math.sin(ecliptic_longitude),
        math.sin(obliquity) * math.sin(ecliptic_longitude),
    ))


def target_sunlit(
    target_pos_km: Sequence[float],
    sun_hat: Sequence[float],
) -> bool:
    """Return whether a target lies outside the cylindrical Earth shadow."""

    r = tuple(float(value) for value in target_pos_km)
    s = _unit(sun_hat)
    projection = _dot(r, s)
    perpendicular = tuple(rv - projection * sv for rv, sv in zip(r, s))
    return projection > 0.0 or _norm(perpendicular) > _EARTH_RADIUS_KM


def detection_probability(
    m: float,
    m_lim: float = 15.0,
    sigma_m: float = 0.5,
) -> float:
    """Logistic probability of detection about the limiting magnitude."""

    if sigma_m <= 0.0:
        raise ValueError("sigma_m must be positive")
    scaled = (float(m_lim) - float(m)) / float(sigma_m)
    if scaled >= 0.0:
        return 1.0 / (1.0 + math.exp(-scaled))
    exp_scaled = math.exp(scaled)
    return exp_scaled / (1.0 + exp_scaled)


def detection_draw(
    episode_seed: int,
    object_id: str,
    sat_id: str,
    step: int,
) -> float:
    """Pure paired-seed detection draw for one object/satellite/step tuple."""

    key = (
        f"ssa-detection-v1|{int(episode_seed)}|{str(object_id)}|"
        f"{str(sat_id)}|{int(step)}"
    )
    raw = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    return raw / float(1 << 64)


def optical_accesses(
    observer_pos_km: Sequence[float],
    observer_vel_unit: Sequence[float],
    targets: Iterable[RSOTarget],
    target_positions: Mapping[str, Sequence[float]],
    sun_hat: Sequence[float],
    *,
    fov_half_angle_deg: float,
    boresight_pitch_deg: float = 12.0,
    r_cap_km: float = 150.0,
    m_lim: float = 15.0,
    sigma_m: float = 0.5,
    albedo: float = 0.13,
) -> list[DetectionAccess]:
    """Return targets passing range, pitched-FOV, and sunlight gates."""

    observer = tuple(float(value) for value in observer_pos_km)
    r_hat = _unit(observer)
    v_hat = _unit(observer_vel_unit)
    pitch = math.radians(float(boresight_pitch_deg))
    boresight = _unit(tuple(
        math.cos(pitch) * vv + math.sin(pitch) * rr
        for vv, rr in zip(v_hat, r_hat)
    ))
    sun = _unit(sun_hat)
    sizes = {target.object_id: float(target.size_m) for target in targets}
    accesses: list[DetectionAccess] = []
    for object_id in sorted(target_positions):
        target_position = tuple(float(value) for value in target_positions[object_id])
        relative = tuple(
            target - origin for target, origin in zip(target_position, observer)
        )
        range_km = _norm(relative)
        if range_km <= 0.0 or range_km > float(r_cap_km):
            continue
        relative_hat = _unit(relative)
        angle_deg = math.degrees(math.acos(
            max(-1.0, min(1.0, _dot(boresight, relative_hat)))
        ))
        if angle_deg > float(fov_half_angle_deg):
            continue
        if not target_sunlit(target_position, sun):
            continue
        target_to_observer = tuple(-value for value in relative_hat)
        phase_rad = math.acos(
            max(-1.0, min(1.0, _dot(sun, target_to_observer)))
        )
        size_m = sizes.get(str(object_id))
        if size_m is None:
            continue
        magnitude = apparent_magnitude(
            size_m,
            range_km,
            phase_rad,
            albedo=float(albedo),
        )
        accesses.append(DetectionAccess(
            object_id=str(object_id),
            position_km=target_position,
            range_km=range_km,
            angle_deg=angle_deg,
            m=magnitude,
            p_detect=detection_probability(magnitude, m_lim, sigma_m),
            quality=float(m_lim) - magnitude,
        ))
    return accesses


def generate_family_catalog(
    count: int,
    seed: int | None,
    *,
    parent_altitude_km: float = 805.0,
    parent_inclination_deg: float = 98.6,
    raan_center_deg: float,
    raan_spread_deg: float = 0.3,
    sigma_dv_along_ms: float = 13.0,
    sigma_dv_normal_ms: float = 26.0,
    size_bounds_m: tuple[float, float] = (0.01, 0.10),
    epoch: datetime = _DEFAULT_EPOCH,
) -> list[RSOTarget]:
    """Sample a co-moving fragmentation-family catalog.

    The physical basis is that breakup delta-v is much smaller than orbital
    velocity, so fragments remain a co-moving family.  The along-track and
    normal dispersions use NASA standard-breakup-model scales (Johnson et al.
    2001); the dispersed angular torus follows Jehn (1991), and its aged-cloud
    interpretation follows Pardini & Anselmo (2011).

    First-order circular-orbit relations convert along-track delta-v to
    semi-major-axis offset and normal delta-v to inclination offset.  Fragment
    diameters follow the fixed cumulative law ``N(>d) ~ d^-2.5`` truncated to
    ``size_bounds_m``.
    """

    parent_a_km = _EARTH_RADIUS_KM + float(parent_altitude_km)
    if parent_a_km <= 0.0:
        raise ValueError("parent orbit semi-major axis must be positive")
    if float(raan_spread_deg) < 0.0:
        raise ValueError("raan_spread_deg must be non-negative")
    if float(sigma_dv_along_ms) < 0.0 or float(sigma_dv_normal_ms) < 0.0:
        raise ValueError("delta-v dispersions must be non-negative")

    dmin_m, dmax_m = (float(value) for value in size_bounds_m)
    if not 0.0 < dmin_m <= dmax_m:
        raise ValueError("size_bounds_m must satisfy 0 < dmin <= dmax")

    orbital_speed_ms = math.sqrt(_MU_EARTH_KM3_S2 / parent_a_km) * 1000.0
    size_span_factor = 1.0 - (dmax_m / dmin_m) ** -2.5
    rng = random.Random(seed)
    targets: list[RSOTarget] = []
    for idx in range(int(count)):
        dv_along_ms = rng.gauss(0.0, float(sigma_dv_along_ms))
        dv_normal_ms = rng.gauss(0.0, float(sigma_dv_normal_ms))
        delta_a_km = max(
            -25.0,
            min(25.0, 2.0 * parent_a_km * dv_along_ms / orbital_speed_ms),
        )
        delta_i_deg = max(
            -0.2,
            min(0.2, math.degrees(dv_normal_ms / orbital_speed_ms)),
        )
        raan_deg = (
            float(raan_center_deg)
            + rng.uniform(-float(raan_spread_deg), float(raan_spread_deg))
        ) % 360.0
        eccentricity = rng.uniform(0.0, 0.001)
        arg_perigee_deg = rng.uniform(0.0, 360.0)
        true_anomaly_deg = rng.uniform(0.0, 360.0)
        size_u = rng.random()
        size_m = dmin_m * (1.0 - size_u * size_span_factor) ** (-1.0 / 2.5)
        targets.append(
            RSOTarget(
                object_id=f"rso_{idx}",
                semi_major_axis_km=parent_a_km + delta_a_km,
                eccentricity=eccentricity,
                inclination_deg=round(float(parent_inclination_deg) + delta_i_deg, 12),
                raan_deg=raan_deg,
                arg_perigee_deg=arg_perigee_deg,
                true_anomaly_deg=true_anomaly_deg,
                size_m=size_m,
                priority=1.0,
                epoch=epoch,
            )
        )
    return targets


def propagate_rso_position_km(
    target: RSOTarget,
    epoch_seconds: float,
    *,
    prefer_orekit: bool = True,
) -> tuple[float, float, float]:
    """Propagate a target and return ECI position in km."""

    if prefer_orekit and propagator.is_available():
        try:
            prop = propagator.create_j2_propagator(
                a_km=target.semi_major_axis_km,
                e=target.eccentricity,
                i_deg=target.inclination_deg,
                raan_deg=target.raan_deg,
                argp_deg=target.arg_perigee_deg,
                ta_deg=target.true_anomaly_deg,
                epoch=target.epoch,
            )
            when = target.epoch + timedelta(seconds=float(epoch_seconds))
            return propagator.propagate_position_km(prop, when)
        except Exception:
            # Keep unit tests and cheap smoke runs deterministic even when Orekit
            # is installed but rejects a synthetic near-circular draw.
            pass
    return _propagate_two_body_position_km(target, epoch_seconds)


def propagated_catalog_positions_km(
    targets: Iterable[RSOTarget],
    epoch_seconds: float,
    *,
    prefer_orekit: bool = True,
) -> dict[str, tuple[float, float, float]]:
    """Propagate all targets into an object_id -> position map."""

    return {
        target.object_id: propagate_rso_position_km(
            target, epoch_seconds, prefer_orekit=prefer_orekit
        )
        for target in targets
    }


def _propagate_two_body_position_km(target: RSOTarget, epoch_seconds: float) -> tuple[float, float, float]:
    a = target.semi_major_axis_km
    e = target.eccentricity
    n = math.sqrt(_MU_EARTH_KM3_S2 / (a ** 3))
    m0 = _true_to_mean_anomaly(math.radians(target.true_anomaly_deg), e)
    mean_anomaly = (m0 + n * float(epoch_seconds)) % (2.0 * math.pi)
    eccentric_anomaly = _solve_kepler(mean_anomaly, e)
    x_p = a * (math.cos(eccentric_anomaly) - e)
    y_p = a * math.sqrt(1.0 - e * e) * math.sin(eccentric_anomaly)
    return _rotate_pqw_to_eci(
        x_p,
        y_p,
        math.radians(target.raan_deg),
        math.radians(target.inclination_deg),
        math.radians(target.arg_perigee_deg),
    )


def _true_to_mean_anomaly(true_anomaly_rad: float, eccentricity: float) -> float:
    if eccentricity <= 0.0:
        return true_anomaly_rad % (2.0 * math.pi)
    e_anomaly = 2.0 * math.atan2(
        math.sqrt(1.0 - eccentricity) * math.sin(true_anomaly_rad / 2.0),
        math.sqrt(1.0 + eccentricity) * math.cos(true_anomaly_rad / 2.0),
    )
    return (e_anomaly - eccentricity * math.sin(e_anomaly)) % (2.0 * math.pi)


def _solve_kepler(mean_anomaly: float, eccentricity: float) -> float:
    e_anomaly = mean_anomaly
    for _ in range(12):
        delta = (e_anomaly - eccentricity * math.sin(e_anomaly) - mean_anomaly) / (
            1.0 - eccentricity * math.cos(e_anomaly)
        )
        e_anomaly -= delta
        if abs(delta) < 1e-12:
            break
    return e_anomaly


def _rotate_pqw_to_eci(
    x_p: float,
    y_p: float,
    raan: float,
    inclination: float,
    arg_perigee: float,
) -> tuple[float, float, float]:
    cos_o, sin_o = math.cos(raan), math.sin(raan)
    cos_i, sin_i = math.cos(inclination), math.sin(inclination)
    cos_w, sin_w = math.cos(arg_perigee), math.sin(arg_perigee)
    r11 = cos_o * cos_w - sin_o * sin_w * cos_i
    r12 = -cos_o * sin_w - sin_o * cos_w * cos_i
    r21 = sin_o * cos_w + cos_o * sin_w * cos_i
    r22 = -sin_o * sin_w + cos_o * cos_w * cos_i
    r31 = sin_w * sin_i
    r32 = cos_w * sin_i
    return (r11 * x_p + r12 * y_p, r21 * x_p + r22 * y_p, r31 * x_p + r32 * y_p)


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) ** 2 for x in vector))


def _unit(vector: Sequence[float]) -> tuple[float, float, float]:
    norm = _norm(vector)
    if norm <= 0.0:
        raise ValueError("cannot normalize a zero vector")
    return tuple(float(x) / norm for x in vector)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))
