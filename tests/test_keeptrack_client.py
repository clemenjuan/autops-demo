import pytest
from agent.data_pipeline.fetchers.keeptrack_client import KeepTrackClient, determine_orbit_type

@pytest.mark.asyncio
async def test_fetch_all():
    client = KeepTrackClient()
    sats = await client.fetch_all()
    
    assert isinstance(sats, list)
    assert len(sats) > 30000
    assert all('satid' in s and 'line1' in s and 'line2' in s for s in sats)

def test_tle_epoch_parsing():
    client = KeepTrackClient()
    line1 = "1 25544U 98067A   24015.50000000  .00012345  00000-0  12345-3 0  9999"
    epoch = client.parse_tle_epoch(line1)
    
    assert epoch.year == 2024
    assert epoch.month == 1
    assert 14 <= epoch.day <= 16

def test_orbit_type_classification():
    assert determine_orbit_type(6.65) == 'LEO'
    assert determine_orbit_type(6.61) == 'LEO'
    assert determine_orbit_type(6.66) == 'GEO'
    assert determine_orbit_type(10.0) == 'MEO'
    assert determine_orbit_type(20.0) == 'xGEO'
    assert determine_orbit_type(None) == 'UNKNOWN'

def test_normalize_satellite():
    client = KeepTrackClient()
    raw = {
        'satid': 25544,
        'name': 'ISS (ZARYA)',
        'country': 'RUS',
        'operator': 'RKA',
        'semiMajorAxis': 6.7436,
        'type': 'Space Station'
    }
    normalized = client.normalize_satellite(raw)
    
    assert normalized['norad_id'] == 25544
    assert normalized['name'] == 'ISS (ZARYA)'
    assert normalized['orbit_type'] == 'LEO'
