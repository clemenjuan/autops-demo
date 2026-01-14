import requests
import os

API_BASE_URL = os.getenv('SATELLITE_API_URL', 'http://localhost:8000')

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

async def execute(params):
    action = params.get('action', 'get_satellite')
    
    if action == 'get_satellite':
        return await get_satellite(params)
    elif action == 'get_tle_history':
        return await get_tle_history(params)
    elif action == 'get_maneuvers':
        return await get_maneuvers(params)
    else:
        return {'error': f'Unknown action: {action}'}
