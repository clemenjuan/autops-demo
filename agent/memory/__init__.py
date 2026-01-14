"""
CoALA Memory System for AUTOPS

Implements the four memory types from Cognitive Architectures for Language Agents:
- Working Memory: Short-term context for current reasoning
- Episodic Memory: Long-term storage of past episodes
- Semantic Memory: Long-term factual knowledge
- Procedural Memory: Learned strategies and skills
"""

from agent.memory.base_memory import BaseMemory
from agent.memory.working_memory import WorkingMemory
from agent.memory.episodic_memory import EpisodicMemory
from agent.memory.semantic_memory import SemanticMemory
from agent.memory.procedural_memory import ProceduralMemory

__all__ = [
    'BaseMemory',
    'WorkingMemory',
    'EpisodicMemory',
    'SemanticMemory',
    'ProceduralMemory'
]

