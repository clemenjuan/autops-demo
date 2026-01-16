"""
Orekit Propagation Tool

High-precision orbital propagation, maneuver computation, and conjunction analysis.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List
import math

try:
    from agent.data_pipeline.fetchers.orekit_setup import init_orekit, is_initialized
    init_orekit()
    
    from org.orekit.frames import FramesFactory
    from org.orekit.time import TimeScalesFactory, AbsoluteDate
    from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
    from org.orekit.orbits import KeplerianOrbit, PositionAngleType
    from org.orekit.propagation.analytical import KeplerianPropagator
    from org.orekit.propagation.analytical.tle import TLE, TLEPropagator
    from org.orekit.utils import Constants, IERSConventions
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    OREKIT_AVAILABLE = True
except ImportError:
    OREKIT_AVAILABLE = False


def get_frames():
    """Get commonly used reference frames."""
    if not OREKIT_AVAILABLE:
        return None
    return {
        'eme2000': FramesFactory.getEME2000(),
        'itrf': FramesFactory.getITRF(IERSConventions.IERS_2010, True),
        'gcrf': FramesFactory.getGCRF()
    }


def tle_to_orekit(tle_line1: str, tle_line2: str):
    """Convert TLE strings to Orekit TLE object."""
    if not OREKIT_AVAILABLE:
        return None
    return TLE(tle_line1, tle_line2)


def propagate_tle(tle_line1: str, tle_line2: str, target_time: datetime) -> Dict[str, Any]:
    """Propagate TLE to a target time using SGP4/SDP4."""
    if not OREKIT_AVAILABLE:
        return {'error': 'Orekit not available'}
    
    try:
        tle = TLE(tle_line1, tle_line2)
        propagator = TLEPropagator.selectExtrapolator(tle)
        utc = TimeScalesFactory.getUTC()
        target = AbsoluteDate(target_time.year, target_time.month, target_time.day,
                              target_time.hour, target_time.minute, float(target_time.second), utc)
        
        state = propagator.propagate(target)
        pos = state.getPVCoordinates().getPosition()
        vel = state.getPVCoordinates().getVelocity()
        
        return {
            'position_eci': {'x': pos.getX(), 'y': pos.getY(), 'z': pos.getZ()},
            'velocity_eci': {'x': vel.getX(), 'y': vel.getY(), 'z': vel.getZ()},
            'epoch': target_time.isoformat()
        }
    except Exception as e:
        return {'error': str(e)}


def propagate_numerical(tle_line1: str, tle_line2: str, duration_hours: float, step_seconds: float = 60) -> List[Dict]:
    """Propagate orbit numerically for a given duration."""
    if not OREKIT_AVAILABLE:
        return [{'error': 'Orekit not available'}]
    
    try:
        tle = TLE(tle_line1, tle_line2)
        propagator = TLEPropagator.selectExtrapolator(tle)
        utc = TimeScalesFactory.getUTC()
        
        start = propagator.getInitialState().getDate()
        end_offset = duration_hours * 3600
        
        trajectory = []
        t = 0
        while t <= end_offset:
            state = propagator.propagate(start.shiftedBy(t))
            pos = state.getPVCoordinates().getPosition()
            trajectory.append({
                'time_offset_sec': t,
                'position_eci': {'x': pos.getX(), 'y': pos.getY(), 'z': pos.getZ()}
            })
            t += step_seconds
        
        return trajectory
    except Exception as e:
        return [{'error': str(e)}]


def compute_hohmann_transfer(r1_km: float, r2_km: float) -> Dict[str, Any]:
    """Compute Hohmann transfer delta-v between circular orbits."""
    mu = 398600.4418
    
    v1 = math.sqrt(mu / r1_km)
    v2 = math.sqrt(mu / r2_km)
    
    a_transfer = (r1_km + r2_km) / 2
    v_transfer_perigee = math.sqrt(mu * (2/r1_km - 1/a_transfer))
    v_transfer_apogee = math.sqrt(mu * (2/r2_km - 1/a_transfer))
    
    dv1 = abs(v_transfer_perigee - v1)
    dv2 = abs(v2 - v_transfer_apogee)
    
    transfer_period = 2 * math.pi * math.sqrt(a_transfer**3 / mu)
    transfer_time = transfer_period / 2
    
    return {
        'dv1_km_s': dv1,
        'dv2_km_s': dv2,
        'total_dv_km_s': dv1 + dv2,
        'transfer_time_sec': transfer_time,
        'transfer_sma_km': a_transfer
    }


def predict_conjunction(tle1_line1: str, tle1_line2: str, tle2_line1: str, tle2_line2: str, 
                        hours_ahead: float = 24, threshold_km: float = 10) -> List[Dict]:
    """Screen for close approaches between two objects."""
    if not OREKIT_AVAILABLE:
        return [{'error': 'Orekit not available'}]
    
    try:
        prop1 = TLEPropagator.selectExtrapolator(TLE(tle1_line1, tle1_line2))
        prop2 = TLEPropagator.selectExtrapolator(TLE(tle2_line1, tle2_line2))
        
        start = prop1.getInitialState().getDate()
        step = 60.0
        conjunctions = []
        
        t = 0
        while t <= hours_ahead * 3600:
            date = start.shiftedBy(t)
            pos1 = prop1.propagate(date).getPVCoordinates().getPosition()
            pos2 = prop2.propagate(date).getPVCoordinates().getPosition()
            
            distance = Vector3D.distance(pos1, pos2) / 1000.0
            
            if distance < threshold_km:
                conjunctions.append({
                    'time_offset_sec': t,
                    'distance_km': distance
                })
            t += step
        
        return conjunctions
    except Exception as e:
        return [{'error': str(e)}]


async def execute(params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute orekit propagation tool actions."""
    action = params.get('action', 'propagate')
    
    if not OREKIT_AVAILABLE:
        return {'error': 'Orekit not installed. Run: pip install orekit'}
    
    if action == 'propagate':
        tle1 = params.get('tle_line1')
        tle2 = params.get('tle_line2')
        target = params.get('target_time')
        if not tle1 or not tle2:
            return {'error': 'tle_line1 and tle_line2 required'}
        if isinstance(target, str):
            target = datetime.fromisoformat(target)
        elif target is None:
            target = datetime.utcnow()
        return propagate_tle(tle1, tle2, target)
    
    elif action == 'propagate_trajectory':
        tle1 = params.get('tle_line1')
        tle2 = params.get('tle_line2')
        hours = params.get('duration_hours', 2)
        step = params.get('step_seconds', 60)
        if not tle1 or not tle2:
            return {'error': 'tle_line1 and tle_line2 required'}
        trajectory = propagate_numerical(tle1, tle2, hours, step)
        return {'status': 'success', 'trajectory': trajectory}
    
    elif action == 'compute_maneuver':
        r1 = params.get('initial_radius_km')
        r2 = params.get('target_radius_km')
        if not r1 or not r2:
            return {'error': 'initial_radius_km and target_radius_km required'}
        return compute_hohmann_transfer(r1, r2)
    
    elif action == 'predict_conjunction':
        tle1_l1 = params.get('object1_tle_line1')
        tle1_l2 = params.get('object1_tle_line2')
        tle2_l1 = params.get('object2_tle_line1')
        tle2_l2 = params.get('object2_tle_line2')
        hours = params.get('hours_ahead', 24)
        threshold = params.get('threshold_km', 10)
        if not all([tle1_l1, tle1_l2, tle2_l1, tle2_l2]):
            return {'error': 'Both objects TLE lines required'}
        conjunctions = predict_conjunction(tle1_l1, tle1_l2, tle2_l1, tle2_l2, hours, threshold)
        return {'status': 'success', 'conjunctions': conjunctions}
    
    elif action == 'compare_tle':
        tle1 = params.get('tle_line1')
        tle2 = params.get('tle_line2')
        hours = params.get('hours_ahead', 24)
        if not tle1 or not tle2:
            return {'error': 'tle_line1 and tle_line2 required'}
        trajectory = propagate_numerical(tle1, tle2, hours, 300)
        return {'status': 'success', 'comparison_points': len(trajectory), 'trajectory': trajectory}
    
    else:
        return {'error': f'Unknown action: {action}'}
