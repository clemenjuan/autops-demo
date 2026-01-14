import importlib
import os
from utils.toon_formatter import ToonFormatter

def load_tools(metadata_path='tools/tools_metadata.toon'):
    if not os.path.isabs(metadata_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        metadata_path = os.path.join(base_dir, metadata_path)
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Tools metadata file not found: {metadata_path}")
            
    with open(metadata_path, 'r', encoding='utf-8') as f:
        content = f.read()
        metadata = ToonFormatter.loads(content)
    
    tools = {}
    for tool_def in metadata['tools']:
        try:
            module = importlib.import_module(tool_def['module'])
            func = getattr(module, tool_def['function'])
            tools[tool_def['name']] = {
                'execute': func,
                'description': tool_def['description'],
                'tags': tool_def['tags'],
                'parameters': tool_def['parameters'],
                'examples': tool_def['examples']
            }
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not load tool '{tool_def['name']}': {e}")
    
    return tools, metadata

