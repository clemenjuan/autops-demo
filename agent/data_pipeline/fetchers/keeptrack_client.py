import httpx
from datetime import datetime, timedelta
from typing import List, Dict

class KeepTrackClient:
    API_URL = "https://api.keeptrack.space/v2/sats"
    TIMEOUT = 30.0
    
    async def fetch_all(self) -> List[Dict]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(self.API_URL, timeout=self.TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return data.get('sats', [])
    
    @staticmethod
    def parse_tle_epoch(line1: str) -> datetime:
        year = int(line1[18:20])
        day_of_year_fraction = float(line1[20:32])
        
        if year < 57:
            year += 2000
        else:
            year += 1900
        
        base = datetime(year, 1, 1)
        delta = timedelta(days=day_of_year_fraction - 1)
        return base + delta
    
    @staticmethod
    def normalize_satellite(raw_data: Dict) -> Dict:
        return {
            'norad_id': raw_data.get('satid'),
            'keeptrack_id': raw_data.get('satid'),
            'name': raw_data.get('name', ''),
            'country': raw_data.get('country', ''),
            'operator': raw_data.get('operator', ''),
            'orbit_type': determine_orbit_type(raw_data.get('semiMajorAxis')),
            'mission_type': raw_data.get('type', ''),
            'payload': raw_data.get('payload', ''),
            'launched': parse_date(raw_data.get('launchDate')),
            'decay_date': parse_date(raw_data.get('decayDate')),
            'last_updated': datetime.utcnow()
        }

def determine_orbit_type(semi_major_axis: float) -> str:
    if semi_major_axis is None:
        return 'UNKNOWN'
    if semi_major_axis < 6.75:
        return 'LEO'
    elif 6.6 <= semi_major_axis < 6.8:
        return 'GEO'
    elif semi_major_axis < 15:
        return 'MEO'
    else:
        return 'xGEO'

def parse_date(date_str: str) -> datetime:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except:
        return None
