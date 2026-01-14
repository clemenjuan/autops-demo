import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.toon_formatter import ToonFormatter

def convert_file(json_path, toon_path):
    if not os.path.exists(json_path):
        print(f"Skipping {json_path} (not found)")
        return False
    
    print(f"Converting {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    toon_content = ToonFormatter.dumps(data)
    
    with open(toon_path, 'w', encoding='utf-8') as f:
        f.write(toon_content)
    
    print(f"  -> {toon_path}")
    return True

def convert_metadata():
    files_to_convert = [
        ('tools/tools_metadata.json', 'tools/tools_metadata.toon'),
        ('data/memory/episodic_memory.json', 'data/memory/episodic_memory.toon'),
        ('data/memory/semantic_memory.json', 'data/memory/semantic_memory.toon'),
        ('data/memory/procedural_memory.json', 'data/memory/procedural_memory.toon'),
    ]
    
    converted = 0
    for json_path, toon_path in files_to_convert:
        if convert_file(json_path, toon_path):
            converted += 1
    
    print(f"\nConversion complete. Converted {converted} file(s).")

if __name__ == "__main__":
    convert_metadata()

