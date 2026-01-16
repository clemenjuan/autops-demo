import requests
import os
from datetime import datetime, timedelta

API_BASE_URL = os.getenv('SATELLITE_API_URL', 'http://localhost:8000')

try:
    from tools.orekit_propagation_tool import propagate_tle, propagate_numerical, OREKIT_AVAILABLE
except ImportError:
    OREKIT_AVAILABLE = False

async def get_satellite(params):
    norad_id = params.get('norad_id')
    if not norad_id:
        return {'error': 'norad_id parameter required'}
    
    try:
        resp = requests.get(f"{API_BASE_URL}/satellites/{norad_id}", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        
        if 'error' in data:
            return {'error': data['error']}
        
        return {'status': 'success', 'satellite': data}
    except requests.exceptions.ConnectionError:
        return {
            'error': 'Satellite API server not running',
            'message': f'Cannot connect to {API_BASE_URL}. Start the server with: python run_satellite_api.py'
        }
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return {'error': f'Satellite with NORAD ID {norad_id} not found'}
        return {'error': f'Failed to fetch satellite data: {str(e)}'}
    except Exception as e:
        return {'error': f'Failed to fetch satellite data: {str(e)}'}

async def get_tle_history(params):
    norad_id = params.get('norad_id')
    days = params.get('days', 30)
    
    if not norad_id:
        return {'error': 'norad_id parameter required'}
    
    try:
        resp = requests.get(f"{API_BASE_URL}/tle/{norad_id}/history", params={'days': days}, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        return {'status': 'success', 'data': data}
    except requests.exceptions.ConnectionError:
        return {
            'error': 'Satellite API server not running',
            'message': f'Cannot connect to {API_BASE_URL}. Start the server with: python run_satellite_api.py'
        }
    except Exception as e:
        return {'error': f'Failed to fetch TLE history: {str(e)}'}

async def get_maneuvers(params):
    satellite_id = params.get('satellite_id')
    min_confidence = params.get('min_confidence', 0.0)
    days = params.get('days', 30)
    
    query_params = {'min_confidence': min_confidence, 'days': days}
    if satellite_id:
        query_params['satellite_id'] = satellite_id
    
    try:
        resp = requests.get(f"{API_BASE_URL}/maneuvers", params=query_params, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        return {'status': 'success', 'data': data}
    except requests.exceptions.ConnectionError:
        return {
            'error': 'Satellite API server not running',
            'message': f'Cannot connect to {API_BASE_URL}. Start the server with: python run_satellite_api.py'
        }
    except Exception as e:
        return {'error': f'Failed to fetch maneuvers: {str(e)}'}

async def predict_position(params):
    """Predict satellite position at a future time using Orekit."""
    norad_id = params.get('norad_id')
    target_time = params.get('target_time')
    
    if not norad_id:
        return {'error': 'norad_id parameter required'}
    
    if not OREKIT_AVAILABLE:
        return {'error': 'Orekit not available for high-precision prediction'}
    
    try:
        resp = requests.get(f"{API_BASE_URL}/tle/{norad_id}/history", params={'days': 1}, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        
        if not data.get('history'):
            return {'error': f'No TLE found for NORAD ID {norad_id}'}
        
        tle = data['history'][0]
        if isinstance(target_time, str):
            target_time = datetime.fromisoformat(target_time)
        elif target_time is None:
            target_time = datetime.utcnow() + timedelta(hours=1)
        
        result = propagate_tle(tle['tle_line1'], tle['tle_line2'], target_time)
        result['norad_id'] = norad_id
        return {'status': 'success', 'prediction': result}
    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}'}


async def calculate_passes(params):
    """Calculate sensor visibility passes for a satellite."""
    norad_id = params.get('norad_id')
    sensor_lat = params.get('sensor_lat')
    sensor_lon = params.get('sensor_lon')
    sensor_alt = params.get('sensor_alt', 0)
    hours_ahead = params.get('hours_ahead', 24)
    min_elevation = params.get('min_elevation', 10)
    
    if not norad_id:
        return {'error': 'norad_id parameter required'}
    if sensor_lat is None or sensor_lon is None:
        return {'error': 'sensor_lat and sensor_lon required'}
    
    # For now, return placeholder - full implementation requires ground station visibility calc
    return {
        'status': 'success',
        'norad_id': norad_id,
        'sensor': {'lat': sensor_lat, 'lon': sensor_lon, 'alt': sensor_alt},
        'passes': [],
        'message': 'Pass calculation requires Orekit EventDetector - use orekit_propagation_tool for full analysis'
    }


async def get_orbit_trajectory(params):
    """Get past and predicted orbit trajectory for visualization."""
    norad_id = params.get('norad_id')
    past_minutes = params.get('past_minutes', 45)
    prediction_minutes = params.get('prediction_minutes', 90)
    
    if not norad_id:
        return {'error': 'norad_id parameter required'}
    
    try:
        resp = requests.get(f"{API_BASE_URL}/tle/{norad_id}/history", params={'days': 1}, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        
        if not data.get('history'):
            return {'error': f'No TLE found for NORAD ID {norad_id}'}
        
        tle = data['history'][0]
        
        if OREKIT_AVAILABLE:
            total_hours = (past_minutes + prediction_minutes) / 60.0
            trajectory = propagate_numerical(tle['tle_line1'], tle['tle_line2'], total_hours, 60)
            
            past_count = past_minutes
            return {
                'status': 'success',
                'norad_id': norad_id,
                'past_trajectory': trajectory[:past_count],
                'prediction_trajectory': trajectory[past_count:]
            }
        else:
            return {
                'status': 'partial',
                'norad_id': norad_id,
                'tle': tle,
                'message': 'Orekit not available - use frontend satellite.js for propagation'
            }
    except Exception as e:
        return {'error': f'Trajectory fetch failed: {str(e)}'}


async def execute(params):
    action = params.get('action', 'get_satellite')
    
    if action == 'get_satellite':
        return await get_satellite(params)
    elif action == 'get_tle_history':
        return await get_tle_history(params)
    elif action == 'get_maneuvers':
        return await get_maneuvers(params)
    elif action == 'predict_position':
        return await predict_position(params)
    elif action == 'calculate_passes':
        return await calculate_passes(params)
    elif action == 'get_orbit_trajectory':
        return await get_orbit_trajectory(params)
    else:
        return {'error': f'Unknown action: {action}'}
