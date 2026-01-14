"""
Base Memory Module for CoALA Architecture

Provides abstract base class for all memory types with TOON persistence.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import os
from pathlib import Path
from utils.toon_formatter import ToonFormatter


class BaseMemory(ABC):
    """
    Abstract base class for CoALA memory modules.
    
    All memory types (working, episodic, semantic, procedural) inherit from this
    and implement their specific storage/retrieval logic.
    
    Future: Migrate from JSON to SQL database for better performance and scalability.
    Consider PostgreSQL with pgvector extension for semantic similarity search.
    """
    
    def __init__(self, memory_type: str, file_path: str, persistent: bool = True):
        """
        Initialize a memory module.
        
        Args:
            memory_type: Type of memory (working, episodic, semantic, procedural)
            file_path: Path to TOON file for persistence
            persistent: Whether to persist to disk (False for session-only)
        """
        self.memory_type = memory_type
        self.file_path = file_path
        self.persistent = persistent
        self.data: List[Dict[str, Any]] = []
        
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        
        if self.persistent:
            self.load()
    
    @abstractmethod
    def store(self, entry: Dict[str, Any]) -> str:
        """
        Store an entry in memory.
        
        Args:
            entry: Dictionary containing memory entry data
            
        Returns:
            entry_id: Unique identifier for the stored entry
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve entries from memory based on query.
        
        Args:
            query: Query parameters for retrieval
            limit: Maximum number of entries to return
            
        Returns:
            List of matching memory entries
        """
        pass
    
    def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing memory entry.
        
        Args:
            entry_id: ID of entry to update
            updates: Dictionary of fields to update
            
        Returns:
            True if update successful, False otherwise
        """
        for entry in self.data:
            if entry.get('id') == entry_id:
                entry.update(updates)
                entry['updated_at'] = datetime.now(timezone.utc).isoformat()
                if self.persistent:
                    self.save()
                return True
        return False
    
    def clear(self) -> None:
        """Clear all entries from memory."""
        self.data = []
        if self.persistent:
            self.save()
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all entries in memory."""
        return self.data.copy()
    
    def get_by_id(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific entry by ID."""
        for entry in self.data:
            if entry.get('id') == entry_id:
                return entry.copy()
        return None
    
    def save(self) -> None:
        """Persist memory to file."""
        if not self.persistent:
            return
        
        try:
            data = {
                'memory_type': self.memory_type,
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'entries': self.data
            }
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                f.write(ToonFormatter.dumps(data))
        except Exception as e:
            print(f"[{self.memory_type}] ERROR saving to {self.file_path}: {e}")
    
    def load(self) -> None:
        """Load memory from file."""
        if not self.persistent:
            return
        
        if not os.path.exists(self.file_path):
            self.data = []
            return
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                data = ToonFormatter.loads(content)
                self.data = data.get('entries', [])
                print(f"[{self.memory_type}] Loaded {len(self.data)} entries from {self.file_path}")
        except Exception as e:
            print(f"[{self.memory_type}] WARNING: Could not parse {self.file_path} ({e}), starting fresh")
            self.data = []
    
    def size(self) -> int:
        """Get the number of entries in memory."""
        return len(self.data)
    
    def _generate_id(self) -> str:
        """Generate a unique ID for a memory entry."""
        import uuid
        return f"{self.memory_type}_{uuid.uuid4().hex[:12]}"
    
    def _add_timestamp(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Add timestamp metadata to an entry."""
        entry['created_at'] = datetime.now(timezone.utc).isoformat()
        entry['updated_at'] = entry['created_at']
        return entry

