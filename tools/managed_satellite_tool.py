"""
Managed Satellite Tool

Operations for managed/owned satellites with precise telemetry,
propulsion modeling, and maneuver planning.
"""

import os
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from utils.toon_formatter import ToonFormatter

# Configuration file path
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'managed_satellites.toon')

# Cache for configuration
_config_cache = None
_config_mtime = 0


def _load_config() -> Dict[str, Any]:
    """Load managed satellites configuration from TOON file."""
    global _config_cache, _config_mtime
    
    config_path = os.path.abspath(CONFIG_PATH)
    
    if not os.path.exists(config_path):
        return {'satellites': [], 'ground_stations': {}}
    
    mtime = os.path.getmtime(config_path)
    if _config_cache is not None and mtime == _config_mtime:
        return _config_cache
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
        _config_cache = ToonFormatter.loads(content)
        _config_mtime = mtime
    
    return _config_cache


def _save_config(config: Dict[str, Any]):
    """Save configuration back to TOON file."""
    global _config_cache, _config_mtime
    
    config_path = os.path.abspath(CONFIG_PATH)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(ToonFormatter.dumps(config))
    
    _config_mtime = os.path.getmtime(config_path)
    _config_cache = config


def _get_satellite_by_id(sat_id: str) -> Optional[Dict[str, Any]]:
    """Get satellite configuration by ID."""
    config = _load_config()
    for sat in config.get('satellites', []):
        if sat.get('id') == sat_id:
            return sat
    return None


def _get_satellite_by_norad(norad_id: int) -> Optional[Dict[str, Any]]:
    """Get satellite configuration by NORAD ID."""
    config = _load_config()
    for sat in config.get('satellites', []):
        if sat.get('norad_id') == norad_id:
            return sat
    return None


def calculate_delta_v_budget(propulsion: Dict[str, Any], spacecraft: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate remaining delta-v budget using Tsiolkovsky equation.
    
    delta_v = Isp * g0 * ln(m_wet / m_dry)
    """
    g0 = 9.80665  # m/s^2
    
    isp = propulsion.get('isp_s', 0)
    fuel_remaining = propulsion.get('fuel_remaining_kg', 0)
    dry_mass = spacecraft.get('dry_mass_kg', 0)
    
    if dry_mass <= 0 or isp <= 0:
        return {'error': 'Invalid spacecraft or propulsion parameters'}
    
    wet_mass = dry_mass + fuel_remaining
    
    if fuel_remaining <= 0:
        return {
            'delta_v_remaining_m_s': 0,
            'fuel_remaining_kg': 0,
            'status': 'depleted'
        }
    
    delta_v = isp * g0 * math.log(wet_mass / dry_mass)
    
    return {
        'delta_v_remaining_m_s': round(delta_v, 2),
        'fuel_remaining_kg': fuel_remaining,
        'dry_mass_kg': dry_mass,
        'wet_mass_kg': wet_mass,
        'isp_s': isp,
        'status': 'operational' if delta_v > 10 else 'low_fuel'
    }


def calculate_fuel_for_maneuver(delta_v_m_s: float, propulsion: Dict, spacecraft: Dict) -> Dict[str, Any]:
    """Calculate fuel required for a given delta-v."""
    g0 = 9.80665
    
    isp = propulsion.get('isp_s', 0)
    fuel_remaining = propulsion.get('fuel_remaining_kg', 0)
    dry_mass = spacecraft.get('dry_mass_kg', 0)
    
    if dry_mass <= 0 or isp <= 0:
        return {'error': 'Invalid parameters'}
    
    wet_mass = dry_mass + fuel_remaining
    
    # m_final = m_initial * exp(-delta_v / (Isp * g0))
    mass_ratio = math.exp(-delta_v_m_s / (isp * g0))
    final_mass = wet_mass * mass_ratio
    fuel_required = wet_mass - final_mass
    
    feasible = fuel_required <= fuel_remaining
    
    return {
        'delta_v_m_s': delta_v_m_s,
        'fuel_required_kg': round(fuel_required, 4),
        'fuel_remaining_after_kg': round(fuel_remaining - fuel_required, 4) if feasible else None,
        'feasible': feasible,
        'margin_kg': round(fuel_remaining - fuel_required, 4) if feasible else None
    }


def calculate_burn_duration(delta_v_m_s: float, propulsion: Dict, spacecraft: Dict) -> Dict[str, Any]:
    """Calculate burn duration for a maneuver."""
    thrust = propulsion.get('thrust_n', 0)
    fuel_remaining = propulsion.get('fuel_remaining_kg', 0)
    dry_mass = spacecraft.get('dry_mass_kg', 0)
    
    if thrust <= 0:
        return {'error': 'No thrust available'}
    
    wet_mass = dry_mass + fuel_remaining
    
    # a = F / m (average)
    # delta_v = a * t -> t = delta_v * m / F
    # Using average mass for approximation
    fuel_calc = calculate_fuel_for_maneuver(delta_v_m_s, propulsion, spacecraft)
    if not fuel_calc.get('feasible', False):
        return {'error': 'Insufficient fuel', 'details': fuel_calc}
    
    avg_mass = wet_mass - fuel_calc['fuel_required_kg'] / 2
    duration_s = delta_v_m_s * avg_mass / thrust
    
    min_burn = propulsion.get('min_burn_duration_s', 0)
    max_burn = propulsion.get('max_burn_duration_s', float('inf'))
    
    return {
        'duration_s': round(duration_s, 1),
        'duration_min': round(duration_s / 60, 2),
        'within_limits': min_burn <= duration_s <= max_burn,
        'min_burn_s': min_burn,
        'max_burn_s': max_burn
    }


async def list_managed(params: Dict) -> Dict[str, Any]:
    """List all managed satellites."""
    config = _load_config()
    satellites = config.get('satellites', [])
    
    result = []
    for sat in satellites:
        budget = calculate_delta_v_budget(
            sat.get('propulsion', {}),
            sat.get('spacecraft', {})
        )
        
        result.append({
            'id': sat.get('id'),
            'name': sat.get('name'),
            'norad_id': sat.get('norad_id'),
            'active': sat.get('operations', {}).get('active', False),
            'propulsion_type': sat.get('propulsion', {}).get('type'),
            'delta_v_remaining_m_s': budget.get('delta_v_remaining_m_s'),
            'fuel_remaining_kg': budget.get('fuel_remaining_kg'),
            'fuel_status': budget.get('status')
        })
    
    return {
        'status': 'success',
        'count': len(result),
        'satellites': result
    }


async def get_satellite(params: Dict) -> Dict[str, Any]:
    """Get detailed satellite configuration and current state."""
    sat_id = params.get('id') or params.get('satellite_id')
    norad_id = params.get('norad_id')
    
    if sat_id:
        sat = _get_satellite_by_id(sat_id)
    elif norad_id:
        sat = _get_satellite_by_norad(norad_id)
    else:
        return {'error': 'id or norad_id required'}
    
    if not sat:
        return {'error': f'Satellite not found'}
    
    budget = calculate_delta_v_budget(
        sat.get('propulsion', {}),
        sat.get('spacecraft', {})
    )
    
    config = _load_config()
    ground_stations = config.get('ground_stations', {})
    sat_stations = sat.get('operations', {}).get('ground_stations', [])
    station_details = {s: ground_stations.get(s, {}) for s in sat_stations}
    
    return {
        'status': 'success',
        'satellite': sat,
        'delta_v_budget': budget,
        'ground_stations': station_details
    }


async def get_delta_v_budget(params: Dict) -> Dict[str, Any]:
    """Calculate remaining delta-v capacity."""
    sat_id = params.get('id') or params.get('satellite_id')
    norad_id = params.get('norad_id')
    
    if sat_id:
        sat = _get_satellite_by_id(sat_id)
    elif norad_id:
        sat = _get_satellite_by_norad(norad_id)
    else:
        return {'error': 'id or norad_id required'}
    
    if not sat:
        return {'error': 'Satellite not found'}
    
    budget = calculate_delta_v_budget(
        sat.get('propulsion', {}),
        sat.get('spacecraft', {})
    )
    
    budget['satellite_id'] = sat.get('id')
    budget['satellite_name'] = sat.get('name')
    
    return {'status': 'success', **budget}


async def compute_maneuver(params: Dict) -> Dict[str, Any]:
    """Plan a maneuver using Orekit integration."""
    sat_id = params.get('id') or params.get('satellite_id')
    maneuver_type = params.get('maneuver_type', 'hohmann')
    
    sat = _get_satellite_by_id(sat_id) if sat_id else None
    if not sat:
        return {'error': 'Satellite not found'}
    
    propulsion = sat.get('propulsion', {})
    spacecraft = sat.get('spacecraft', {})
    initial_orbit = sat.get('initial_orbit', {})
    
    try:
        from tools.orekit_propagation_tool import (
            compute_hohmann_transfer,
            compute_bielliptic_transfer,
            compute_station_keeping
        )
        
        if maneuver_type == 'hohmann':
            r1 = params.get('initial_radius_km') or initial_orbit.get('semi_major_axis_km')
            r2 = params.get('target_radius_km')
            if not r2:
                return {'error': 'target_radius_km required for Hohmann transfer'}
            
            transfer = compute_hohmann_transfer(r1, r2)
            
        elif maneuver_type == 'bielliptic':
            r1 = params.get('initial_radius_km') or initial_orbit.get('semi_major_axis_km')
            r2 = params.get('target_radius_km')
            rb = params.get('intermediate_radius_km')
            if not r2 or not rb:
                return {'error': 'target_radius_km and intermediate_radius_km required'}
            
            transfer = compute_bielliptic_transfer(r1, r2, rb)
            
        elif maneuver_type == 'altitude_raise':
            delta_alt_km = params.get('delta_altitude_km', 0)
            r1 = initial_orbit.get('semi_major_axis_km')
            r2 = r1 + delta_alt_km
            transfer = compute_hohmann_transfer(r1, r2)
            
        elif maneuver_type == 'station_keeping':
            sma = initial_orbit.get('semi_major_axis_km')
            ecc = initial_orbit.get('eccentricity', 0)
            inc = initial_orbit.get('inclination_deg', 0)
            days = params.get('duration_days', 365)
            
            transfer = compute_station_keeping(sma, ecc, inc, days)
            
        else:
            return {'error': f'Unknown maneuver type: {maneuver_type}'}
        
        # Calculate fuel requirements
        total_dv = transfer.get('total_dv_km_s', 0) * 1000  # Convert to m/s
        fuel_calc = calculate_fuel_for_maneuver(total_dv, propulsion, spacecraft)
        burn_calc = calculate_burn_duration(total_dv, propulsion, spacecraft)
        
        return {
            'status': 'success',
            'satellite_id': sat_id,
            'satellite_name': sat.get('name'),
            'maneuver_type': maneuver_type,
            'orbital_mechanics': transfer,
            'fuel_requirements': fuel_calc,
            'burn_parameters': burn_calc,
            'feasible': fuel_calc.get('feasible', False)
        }
        
    except ImportError:
        return {'error': 'Orekit not available for maneuver computation'}
    except Exception as e:
        return {'error': f'Maneuver computation failed: {str(e)}'}


async def record_maneuver(params: Dict) -> Dict[str, Any]:
    """Log an executed maneuver and update fuel state."""
    sat_id = params.get('id') or params.get('satellite_id')
    fuel_consumed = params.get('fuel_consumed_kg')
    delta_v_achieved = params.get('delta_v_achieved_m_s')
    
    if not sat_id:
        return {'error': 'satellite_id required'}
    
    config = _load_config()
    sat_idx = None
    
    for i, sat in enumerate(config.get('satellites', [])):
        if sat.get('id') == sat_id:
            sat_idx = i
            break
    
    if sat_idx is None:
        return {'error': 'Satellite not found'}
    
    sat = config['satellites'][sat_idx]
    propulsion = sat.get('propulsion', {})
    
    old_fuel = propulsion.get('fuel_remaining_kg', 0)
    
    if fuel_consumed is not None:
        new_fuel = max(0, old_fuel - fuel_consumed)
        config['satellites'][sat_idx]['propulsion']['fuel_remaining_kg'] = new_fuel
        _save_config(config)
        
        new_budget = calculate_delta_v_budget(
            config['satellites'][sat_idx]['propulsion'],
            config['satellites'][sat_idx].get('spacecraft', {})
        )
        
        return {
            'status': 'success',
            'satellite_id': sat_id,
            'maneuver_recorded': True,
            'fuel_consumed_kg': fuel_consumed,
            'delta_v_achieved_m_s': delta_v_achieved,
            'previous_fuel_kg': old_fuel,
            'new_fuel_kg': new_fuel,
            'new_delta_v_budget': new_budget
        }
    
    return {'error': 'fuel_consumed_kg required'}


async def update_state(params: Dict) -> Dict[str, Any]:
    """Store a new state vector from telemetry."""
    sat_id = params.get('id') or params.get('satellite_id')
    position = params.get('position_m') or params.get('position_km')
    velocity = params.get('velocity_m_s') or params.get('velocity_km_s')
    epoch = params.get('epoch')
    
    if not sat_id:
        return {'error': 'satellite_id required'}
    if not position or not velocity:
        return {'error': 'position and velocity required'}
    
    sat = _get_satellite_by_id(sat_id)
    if not sat:
        return {'error': 'Satellite not found'}
    
    # Convert km to m if needed
    if params.get('position_km'):
        position = {k: v * 1000 for k, v in position.items()}
    if params.get('velocity_km_s'):
        velocity = {k: v * 1000 for k, v in velocity.items()}
    
    state_record = {
        'satellite_id': sat_id,
        'epoch': epoch or datetime.now(timezone.utc).isoformat(),
        'position_m': position,
        'velocity_m_s': velocity,
        'frame': params.get('frame', 'EME2000'),
        'source': params.get('source', 'manual'),
        'covariance': params.get('covariance')
    }
    
    return {
        'status': 'success',
        'message': 'State vector recorded (DB storage requires running database)',
        'state': state_record
    }


async def predict_position(params: Dict) -> Dict[str, Any]:
    """High-fidelity position prediction using spacecraft-specific parameters."""
    sat_id = params.get('id') or params.get('satellite_id')
    target_time = params.get('target_time')
    duration_hours = params.get('duration_hours')
    
    sat = _get_satellite_by_id(sat_id) if sat_id else None
    if not sat:
        return {'error': 'Satellite not found'}
    
    spacecraft = sat.get('spacecraft', {})
    initial_orbit = sat.get('initial_orbit', {})
    
    try:
        from tools.orekit_propagation_tool import execute as orekit_execute
        
        # Build force models with spacecraft-specific parameters
        force_models = {
            'gravity_degree': 20,
            'gravity_order': 20,
            'moon': True,
            'sun': True,
            'drag': True,
            'srp': True
        }
        
        # If we have TLE, use it for propagation
        # Otherwise, would need to convert Keplerian to TLE or use numerical prop from state
        tle_line1 = params.get('tle_line1')
        tle_line2 = params.get('tle_line2')
        
        if tle_line1 and tle_line2:
            if duration_hours:
                result = await orekit_execute({
                    'action': 'propagate_numerical',
                    'tle_line1': tle_line1,
                    'tle_line2': tle_line2,
                    'duration_hours': duration_hours,
                    'force_models': force_models
                })
            else:
                result = await orekit_execute({
                    'action': 'propagate',
                    'tle_line1': tle_line1,
                    'tle_line2': tle_line2,
                    'target_time': target_time
                })
            
            result['satellite_id'] = sat_id
            result['satellite_name'] = sat.get('name')
            result['spacecraft_params'] = spacecraft
            return result
        else:
            return {
                'status': 'partial',
                'satellite_id': sat_id,
                'initial_orbit': initial_orbit,
                'message': 'TLE required for propagation. Use update_state to provide current state.'
            }
            
    except ImportError:
        return {'error': 'Orekit not available'}
    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}'}


async def get_state_history(params: Dict) -> Dict[str, Any]:
    """Retrieve state vector history (placeholder for DB integration)."""
    sat_id = params.get('id') or params.get('satellite_id')
    limit = params.get('limit', 100)
    
    if not sat_id:
        return {'error': 'satellite_id required'}
    
    sat = _get_satellite_by_id(sat_id)
    if not sat:
        return {'error': 'Satellite not found'}
    
    return {
        'status': 'success',
        'satellite_id': sat_id,
        'satellite_name': sat.get('name'),
        'history': [],
        'message': 'State history requires database connection. Initial orbit from config:',
        'initial_orbit': sat.get('initial_orbit', {})
    }


async def get_ground_stations(params: Dict) -> Dict[str, Any]:
    """Get configured ground stations."""
    config = _load_config()
    return {
        'status': 'success',
        'ground_stations': config.get('ground_stations', {})
    }


async def execute(params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute managed satellite tool."""
    action = params.get('action', 'list_managed')
    
    actions = {
        'list_managed': list_managed,
        'get_satellite': get_satellite,
        'get_delta_v_budget': get_delta_v_budget,
        'compute_maneuver': compute_maneuver,
        'record_maneuver': record_maneuver,
        'update_state': update_state,
        'predict_position': predict_position,
        'get_state_history': get_state_history,
        'get_ground_stations': get_ground_stations
    }
    
    if action not in actions:
        return {
            'error': f'Unknown action: {action}',
            'available_actions': list(actions.keys())
        }
    
    return await actions[action](params)
