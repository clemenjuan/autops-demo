"""
Orekit Setup Module

Initializes Orekit JVM and configures data files for high-precision orbital calculations.
"""

import os
import orekit
from orekit.pyhelpers import setup_orekit_curdir, download_orekit_data_curdir

_orekit_initialized = False
_orekit_vm = None

OREKIT_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'orekit')

def init_orekit():
    """Initialize Orekit JVM and load data files. Safe to call multiple times."""
    global _orekit_initialized, _orekit_vm
    
    if _orekit_initialized:
        return True
    
    try:
        _orekit_vm = orekit.initVM()
        
        data_path = os.path.abspath(OREKIT_DATA_DIR)
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)
            download_orekit_data_curdir(data_path)
        
        setup_orekit_curdir(data_path)
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
