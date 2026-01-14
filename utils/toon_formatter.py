import json
import logging

logger = logging.getLogger(__name__)

class ToonFormatter:
    """
    Formatter for converting data to TOON format (https://github.com/toon-format/toon).
    Falls back to JSON if toon-python library is not installed or conversion fails.
    """
    
    _toon_encode = None
    _toon_decode = None
    _toon_available = False
    
    try:
        from toon_format import encode, decode
        _toon_encode = encode
        _toon_decode = decode
        _toon_available = True
    except ImportError:
        try:
            import toon_python
            _toon_encode = toon_python.encode
            _toon_available = True
        except ImportError:
            try:
                from pytoon import dumps, loads
                _toon_encode = dumps
                _toon_decode = loads
                _toon_available = True
            except ImportError:
                logger.warning("TOON library not found. Using JSON fallback.")
        
    @classmethod
    def dumps(cls, data, **kwargs) -> str:
        """
        Convert data to TOON string.
        
        Args:
            data: The data to convert (dict, list, etc.)
            **kwargs: Additional arguments passed to encode/dumps
            
        Returns:
            str: TOON formatted string (or JSON if fallback)
        """
        if cls._toon_available and cls._toon_encode:
            try:
                result = cls._toon_encode(data, **kwargs)
                if isinstance(result, bytes):
                    return result.decode('utf-8')
                return str(result)
            except Exception as e:
                logger.error(f"TOON conversion failed: {e}. Falling back to JSON.")
        
        # Fallback to JSON
        return json.dumps(data, indent=2, default=str)

    @classmethod
    def loads(cls, data: str, **kwargs):
        """
        Convert TOON string to data.
        
        Args:
            data: The TOON string to convert
            **kwargs: Additional arguments
            
        Returns:
            Data structure (dict, list, etc.)
        """
        if cls._toon_available and cls._toon_decode:
            try:
                return cls._toon_decode(data, **kwargs)
            except Exception as e:
                logger.error(f"TOON decoding failed: {e}. Falling back to JSON.")
        
        # Fallback to JSON
        return json.loads(data, **kwargs)

    @classmethod
    def is_available(cls) -> bool:
        """Check if TOON library is available."""
        return cls._toon_available
