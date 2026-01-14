import requests
from datetime import datetime, timedelta
from typing import List, Dict

class KeepTrackClient:
    API_URL = "https://api.keeptrack.space/v2/sats"
    TIMEOUT = 30.0
    
    def fetch_all(self) -> List[Dict]:
        resp = requests.get(self.API_URL, timeout=self.TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []
    
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
    def extract_norad_id(tle_line1: str) -> int:
        if not tle_line1 or len(tle_line1) < 7:
            return None
        try:
            return int(tle_line1[2:7].strip())
        except:
            return None
    
    @staticmethod
    def parse_tle_orbital_params(line1: str, line2: str) -> Dict:
        if not line1 or not line2 or len(line1) < 69 or len(line2) < 69:
            return {'a': None, 'e': None, 'i': None, 'raan': None, 'aop': None, 'mean_anomaly': None}
        
        try:
            mean_motion = float(line2[52:63])
            eccentricity_str = line2[26:33].strip()
            eccentricity = float('0.' + eccentricity_str) if eccentricity_str else 0.0
            inclination = float(line2[8:16])
            raan = float(line2[17:25])
            aop = float(line2[34:42])
            mean_anomaly = float(line2[43:51])
            
            mu = 398600.4418
            n_rad_per_sec = mean_motion * 2 * 3.141592653589793 / 86400.0
            semi_major_axis = (mu / (n_rad_per_sec ** 2)) ** (1/3)
            
            return {
                'a': semi_major_axis,
                'e': eccentricity,
                'i': inclination,
                'raan': raan,
                'aop': aop,
                'mean_anomaly': mean_anomaly
            }
        except:
            return {'a': None, 'e': None, 'i': None, 'raan': None, 'aop': None, 'mean_anomaly': None}
    
    @staticmethod
    def normalize_satellite(raw_data: Dict) -> Dict:
        tle1 = raw_data.get('tle1', '')
        norad_id = KeepTrackClient.extract_norad_id(tle1)
        
        mass = raw_data.get('Mass') or raw_data.get('launchMass')
        orbit_type = 'UNKNOWN'
        if raw_data.get('type') == 1:
            orbit_type = 'LEO'
        
        return {
            'norad_id': norad_id,
            'keeptrack_id': norad_id,
            'name': raw_data.get('name', ''),
            'country': raw_data.get('country', ''),
            'operator': raw_data.get('owner', ''),
            'orbit_type': orbit_type,
            'mission_type': raw_data.get('payload', ''),
            'payload': raw_data.get('payload', ''),
            'launched': parse_date(raw_data.get('launchDate')),
            'decay_date': None,
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
