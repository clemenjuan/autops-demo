"""
Orekit Setup Module

Initializes Orekit JVM and configures data files for high-precision orbital calculations.
Uses orekit-jpype with data from orekit-data pip package.
"""

_orekit_initialized = False
_orekit_vm = None


def init_orekit():
    """Initialize Orekit JVM and load data files. Safe to call multiple times."""
    global _orekit_initialized, _orekit_vm
    
    if _orekit_initialized:
        return True
    
    try:
        import orekit_jpype
        import jpype
        
        # Check if JVM is already running
        if jpype.isJVMStarted():
            _orekit_vm = jpype.getDefaultJVMPath()
        else:
            _orekit_vm = orekit_jpype.initVM()
        
        from orekit_jpype.pyhelpers import setup_orekit_data
        setup_orekit_data(from_pip_library=True)
        
        _orekit_initialized = True
        return True
    except Exception as e:
        print(f"Orekit initialization failed: {e}")
        return False


def get_vm():
    """Get the Orekit JVM instance."""
    if not _orekit_initialized:
        init_orekit()
    return _orekit_vm


def is_initialized():
    """Check if Orekit is initialized."""
    return _orekit_initialized
