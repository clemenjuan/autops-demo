"""
Convert JSON to TOON format. Also fixes .toon files that contain JSON.

Usage:
  uv run python utils/convert_metadata.py
"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from toon_format import encode, decode
    TOON_AVAILABLE = True
except ImportError:
    TOON_AVAILABLE = False
    print("Error: toon_format not available. Install with: uv pip install toon-format")
    sys.exit(1)

FILES = [
    'tools/tools_metadata',
    'data/memory/episodic_memory',
    'data/memory/semantic_memory',
    'data/memory/procedural_memory',
    'data/memory/working_memory',
]

def is_valid_toon(content):
    """Check if content is valid TOON (not JSON)."""
    try:
        decode(content)
        # If it starts with { it's JSON, not TOON
        return not content.strip().startswith('{')
    except:
        return False

def read_any(path):
    """Read file as TOON or JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    try:
        return decode(content)
    except:
        return json.loads(content)

def convert():
    converted = 0
    
    for base in FILES:
        json_path = base + '.json'
        toon_path = base + '.toon'
        
        # Priority 1: Convert .json to .toon
        if os.path.exists(json_path):
            print(f"Converting {json_path} -> {toon_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            with open(toon_path, 'w', encoding='utf-8') as f:
                f.write(encode(data))
            converted += 1
            continue
        
        # Priority 2: Fix .toon files that contain JSON
        if os.path.exists(toon_path):
            with open(toon_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                print(f"Skipping {toon_path} (empty)")
                continue
                
            if is_valid_toon(content):
                print(f"OK {toon_path} (already valid TOON)")
                continue
            
            # It's JSON or invalid, convert to TOON
            print(f"Fixing {toon_path} (converting JSON to TOON)")
            try:
                data = json.loads(content)
                with open(toon_path, 'w', encoding='utf-8') as f:
                    f.write(encode(data))
                converted += 1
            except Exception as e:
                print(f"Error fixing {toon_path}: {e}")
    
    print(f"\nDone. Converted {converted} file(s).")

if __name__ == "__main__":
    convert()

