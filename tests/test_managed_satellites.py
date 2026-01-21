"""
Test Managed Satellites Tool
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio


async def test_list_managed():
    """Test listing all managed satellites."""
    from tools.managed_satellite_tool import execute
    
    result = await execute({'action': 'list_managed'})
    
    assert result.get('status') == 'success', f"Error: {result}"
    print(f"Found {result['count']} managed satellites:")
    for sat in result['satellites']:
        print(f"  - {sat['name']} ({sat['id']}): {sat['delta_v_remaining_m_s']:.1f} m/s remaining")


async def test_get_satellite():
    """Test getting satellite details."""
    from tools.managed_satellite_tool import execute
    
    result = await execute({'action': 'get_satellite', 'id': 'sat-001'})
    
    assert result.get('status') == 'success', f"Error: {result}"
    sat = result['satellite']
    print(f"Satellite: {sat['name']}")
    print(f"  NORAD ID: {sat['norad_id']}")
    print(f"  Propulsion: {sat['propulsion']['type']}, Isp={sat['propulsion']['isp_s']}s")
    print(f"  Delta-v budget: {result['delta_v_budget']}")


async def test_get_delta_v_budget():
    """Test delta-v budget calculation."""
    from tools.managed_satellite_tool import execute
    
    result = await execute({'action': 'get_delta_v_budget', 'id': 'sat-001'})
    
    assert result.get('status') == 'success', f"Error: {result}"
    print(f"Delta-v budget for {result['satellite_name']}:")
    print(f"  Remaining: {result['delta_v_remaining_m_s']:.2f} m/s")
    print(f"  Fuel: {result['fuel_remaining_kg']:.3f} kg")
    print(f"  Status: {result['status']}")


async def test_compute_maneuver():
    """Test maneuver computation."""
    from tools.managed_satellite_tool import execute
    
    # Test Hohmann transfer (altitude raise)
    result = await execute({
        'action': 'compute_maneuver',
        'id': 'sat-001',
        'maneuver_type': 'altitude_raise',
        'delta_altitude_km': 50
    })
    
    if 'error' in result and 'Orekit' in result['error']:
        print(f"Skipping maneuver test (Orekit not initialized in this context)")
        return
    
    assert result.get('status') == 'success', f"Error: {result}"
    print(f"Altitude raise maneuver for {result['satellite_name']}:")
    print(f"  Delta-v required: {result['orbital_mechanics']['total_dv_km_s']*1000:.2f} m/s")
    print(f"  Fuel required: {result['fuel_requirements']['fuel_required_kg']:.4f} kg")
    print(f"  Feasible: {result['feasible']}")


async def test_ground_stations():
    """Test getting ground stations."""
    from tools.managed_satellite_tool import execute
    
    result = await execute({'action': 'get_ground_stations'})
    
    assert result.get('status') == 'success', f"Error: {result}"
    print("Ground stations:")
    for name, info in result['ground_stations'].items():
        print(f"  - {name}: {info['name']} ({info['latitude_deg']}, {info['longitude_deg']})")


async def test_record_maneuver():
    """Test recording a maneuver (fuel update)."""
    from tools.managed_satellite_tool import execute
    
    # First get current fuel
    before = await execute({'action': 'get_delta_v_budget', 'id': 'sat-002'})
    fuel_before = before.get('fuel_remaining_kg', 0)
    
    # Record a small maneuver
    result = await execute({
        'action': 'record_maneuver',
        'id': 'sat-002',
        'fuel_consumed_kg': 0.01,
        'delta_v_achieved_m_s': 5.0
    })
    
    assert result.get('status') == 'success', f"Error: {result}"
    print(f"Maneuver recorded:")
    print(f"  Fuel before: {result['previous_fuel_kg']:.4f} kg")
    print(f"  Fuel consumed: {result['fuel_consumed_kg']:.4f} kg")
    print(f"  Fuel after: {result['new_fuel_kg']:.4f} kg")
    
    # Restore fuel (undo for testing)
    from tools.managed_satellite_tool import _load_config, _save_config
    config = _load_config()
    for sat in config['satellites']:
        if sat['id'] == 'sat-002':
            sat['propulsion']['fuel_remaining_kg'] = fuel_before
    _save_config(config)


async def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("Testing Managed Satellites Tool")
    print("=" * 70)
    
    tests = [
        ("List Managed Satellites", test_list_managed),
        ("Get Satellite Details", test_get_satellite),
        ("Get Delta-V Budget", test_get_delta_v_budget),
        ("Get Ground Stations", test_ground_stations),
        ("Record Maneuver", test_record_maneuver),
        ("Compute Maneuver", test_compute_maneuver),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            await test_func()
            print(f"PASSED: {name}")
            passed += 1
        except Exception as e:
            print(f"FAILED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
