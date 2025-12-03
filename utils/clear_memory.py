#!/usr/bin/env python3
"""
Clear all memory files manually.
This script deletes all TOON memory files to start fresh.

Note: Memory structure is maintained:
- SemanticMemory: Reinitializes with default domain knowledge (Bayern, Munich, etc.)
- ProceduralMemory: Reinitializes with default procedures (tool sequences, strategies)
- EpisodicMemory: Starts empty (no defaults)
- WorkingMemory: Starts empty (no defaults)

All memories maintain the same TOON file structure (memory_type, last_updated, entries).
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.memory import EpisodicMemory, SemanticMemory, ProceduralMemory, WorkingMemory

def clear_memory_files(memory_type: str = 'all'):
    """
    Clears specified memory files and reinitializes default content for semantic and procedural memories.
    """
    memory_paths = {
        'episodic': 'data/memory/episodic_memory.toon',
        'semantic': 'data/memory/semantic_memory.toon',
        'procedural': 'data/memory/procedural_memory.toon',
        'working': 'data/memory/working_memory.toon'
    }

    cleared_files = []
    reinitialized_memories = []

    for mem_name, file_path in memory_paths.items():
        if memory_type == 'all' or memory_type == mem_name:
            if os.path.exists(file_path):
                os.remove(file_path)
                cleared_files.append(file_path)
                print(f"Removed {file_path}")
            else:
                print(f"Skipping {file_path} (not found)")

            # Reinitialize memories that have default content
            if mem_name == 'semantic':
                sm = SemanticMemory()
                sm.save() # Save to ensure default content is written
                reinitialized_memories.append('SemanticMemory (default facts reinitialized)')
            elif mem_name == 'procedural':
                pm = ProceduralMemory()
                pm.save() # Save to ensure default content is written
                reinitialized_memories.append('ProceduralMemory (default procedures reinitialized)')
            elif mem_name == 'episodic':
                em = EpisodicMemory()
                em.save() # Ensure an empty TOON file is created
                reinitialized_memories.append('EpisodicMemory (empty)')
            elif mem_name == 'working':
                wm = WorkingMemory(persistent=True) # Ensure working memory is persistent for clearing
                wm.save() # Ensure an empty TOON file is created
                reinitialized_memories.append('WorkingMemory (empty)')

    print(f"\nMemory clearing complete.")
    if cleared_files:
        print(f"Cleared files: {', '.join(cleared_files)}")
    if reinitialized_memories:
        print(f"Reinitialized memories: {', '.join(reinitialized_memories)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clear memory modules.")
    parser.add_argument('--memory_type', type=str, default='all',
                        help="Specify memory type to clear: 'all', 'working', 'episodic', 'semantic', 'procedural'. Default is 'all'.")
    args = parser.parse_args()

    clear_memory_files(args.memory_type)

