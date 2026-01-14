"""
Working Memory Module for CoALA Architecture

Short-term memory that holds the current task context, intermediate results,
and state during the active reasoning cycle.
"""

from typing import Dict, List, Any, Optional
from agent.memory.base_memory import BaseMemory


class WorkingMemory(BaseMemory):
    """
    Working Memory: Short-term context for current reasoning cycle.
    
    Stores:
    - Current task description
    - Discovered/available tools
    - Intermediate reasoning results
    - Confidence scores
    - Current cycle state
    
    Working memory is cleared/reset between major task cycles.
    Can be session-only (not persisted) or saved for debugging.
    """
    
    def __init__(self, file_path: str = "data/memory/working_memory.toon", persistent: bool = False):
        """
        Initialize working memory.
        
        Args:
            file_path: Path to TOON file for persistence
            persistent: Whether to persist (default False for session-only)
        """
        super().__init__(memory_type="working", file_path=file_path, persistent=persistent)
        self.current_state: Dict[str, Any] = {}
    
    def store(self, entry: Dict[str, Any]) -> str:
        """
        Store an entry in working memory.
        
        Args:
            entry: Dictionary with keys like 'task', 'tools', 'results', 'confidence'
            
        Returns:
            entry_id: Unique identifier for the stored entry
        """
        entry = self._add_timestamp(entry)
        if 'id' not in entry:
            entry['id'] = self._generate_id()
        
        self.data.append(entry)
        
        if self.persistent:
            self.save()
        
        return entry['id']
    
    def retrieve(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent entries from working memory.
        
        Args:
            query: Query parameters (e.g., {'type': 'reasoning_result'})
            limit: Maximum number of entries to return
            
        Returns:
            List of matching entries (most recent first)
        """
        results = []
        
        for entry in reversed(self.data):
            match = True
            for key, value in query.items():
                if key not in entry or entry[key] != value:
                    match = False
                    break
            
            if match:
                results.append(entry)
                if len(results) >= limit:
                    break
        
        return results
    
    def set_current_task(self, task: str) -> None:
        """Set the current task being worked on."""
        self.current_state['task'] = task
        self.current_state['task_start_time'] = self._add_timestamp({})['created_at']
    
    def set_available_tools(self, tools: List[str]) -> None:
        """Set the list of available/discovered tools."""
        self.current_state['available_tools'] = tools
    
    def add_intermediate_result(self, key: str, value: Any) -> None:
        """Add an intermediate result to current state."""
        if 'intermediate_results' not in self.current_state:
            self.current_state['intermediate_results'] = {}
        self.current_state['intermediate_results'][key] = value
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current working memory state."""
        return self.current_state.copy()
    
    def update_confidence(self, confidence: float) -> None:
        """Update confidence score for current task."""
        self.current_state['confidence'] = confidence
    
    def reset(self) -> None:
        """
        Reset working memory for a new task.
        Clears current state but keeps history for potential debugging.
        """
        if self.current_state:
            self.store({
                'type': 'completed_state',
                'state': self.current_state.copy()
            })
        
        self.current_state = {}
    
    def clear_all(self) -> None:
        """Completely clear working memory including history."""
        self.clear()
        self.current_state = {}

