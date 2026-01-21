"""
Test Orekit-JPype installation and configuration.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime


def test_orekit_initialization():
    """Test that Orekit JVM initializes correctly."""
    import orekit_jpype
    vm = orekit_jpype.initVM()
    # initVM returns None if already initialized, which is fine
    print(f"JVM initialized (or was already running)")


def test_orekit_data_setup():
    """Test that Orekit data loads from pip package."""
    import orekit_jpype
    orekit_jpype.initVM()
    
    from orekit_jpype.pyhelpers import setup_orekit_data
    setup_orekit_data(from_pip_library=True)
    print("Orekit data loaded successfully")


def test_time_scales():
    """Test basic Orekit functionality with time scales."""
    import orekit_jpype
    orekit_jpype.initVM()
    
    from orekit_jpype.pyhelpers import setup_orekit_data
    setup_orekit_data(from_pip_library=True)
    
    from org.orekit.time import TimeScalesFactory, AbsoluteDate
    
    utc = TimeScalesFactory.getUTC()
    now = AbsoluteDate(2024, 1, 15, 12, 0, 0.0, utc)
    print(f"Created AbsoluteDate: {now}")
    assert now is not None


def test_tle_propagation():
    """Test TLE propagation with ISS example."""
    import orekit_jpype
    orekit_jpype.initVM()
    
    from orekit_jpype.pyhelpers import setup_orekit_data
    setup_orekit_data(from_pip_library=True)
    
    from org.orekit.time import TimeScalesFactory, AbsoluteDate
    from org.orekit.propagation.analytical.tle import TLE, TLEPropagator
    
    # ISS TLE example
    tle_line1 = "1 25544U 98067A   24015.50000000  .00016717  00000-0  30000-3 0  9993"
    tle_line2 = "2 25544  51.6400 100.0000 0007000  90.0000 270.0000 15.50000000400000"
    
    tle = TLE(tle_line1, tle_line2)
    propagator = TLEPropagator.selectExtrapolator(tle)
    
    utc = TimeScalesFactory.getUTC()
    target = AbsoluteDate(2024, 1, 15, 12, 0, 0.0, utc)
    
    state = propagator.propagate(target)
    pos = state.getPVCoordinates().getPosition()
    
    print(f"ISS Position at {target}:")
    print(f"  X: {pos.getX()/1000:.2f} km")
    print(f"  Y: {pos.getY()/1000:.2f} km")
    print(f"  Z: {pos.getZ()/1000:.2f} km")
    
    assert pos.getX() != 0


def test_setup_module():
    """Test the project's orekit_setup module."""
    from agent.data_pipeline.fetchers.orekit_setup import init_orekit, is_initialized, get_vm
    
    result = init_orekit()
    assert result is True
    assert is_initialized() is True
    assert get_vm() is not None
    print("orekit_setup module works correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Orekit-JPype Installation")
    print("=" * 60)
    
    tests = [
        ("JVM Initialization", test_orekit_initialization),
        ("Orekit Data Setup", test_orekit_data_setup),
        ("Time Scales", test_time_scales),
        ("TLE Propagation", test_tle_propagation),
        ("Setup Module", test_setup_module),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            test_func()
            print(f"PASSED: {name}")
            passed += 1
        except Exception as e:
            print(f"FAILED: {name}")
            print(f"  Error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
