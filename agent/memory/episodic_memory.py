"""
Episodic Memory Module for CoALA Architecture

Long-term storage of past reasoning episodes, including task history,
reasoning traces, tool execution results, and outcomes.
"""

from typing import Dict, List, Any, Optional
from agent.memory.base_memory import BaseMemory


class EpisodicMemory(BaseMemory):
    """
    Episodic Memory: Long-term storage of past reasoning episodes.
    
    Each episode includes:
    - Timestamp
    - Task description
    - Actions taken (tools used)
    - Results obtained
    - Outcome/confidence
    - Full reasoning trace
    
    Used to retrieve similar past experiences to inform current reasoning.
    Future: Semantic similarity search using embeddings instead of keyword matching.
    """
    
    def __init__(self, file_path: str = "data/memory/episodic_memory.toon"):
        """
        Initialize episodic memory.
        
        Args:
            file_path: Path to TOON file for persistence
        """
        super().__init__(memory_type="episodic", file_path=file_path, persistent=True)
    
    def store(self, entry: Dict[str, Any]) -> str:
        """
        Store a complete episode in memory.
        
        Args:
            entry: Episode dictionary with keys:
                - task: Task description
                - actions: List of actions taken
                - results: Results obtained
                - outcome: Final outcome
                - confidence: Confidence score
                - reasoning_trace: Full reasoning history
                - metadata: Additional context
            
        Returns:
            entry_id: Unique identifier for the stored episode
        """
        entry = self._add_timestamp(entry)
        if 'id' not in entry:
            entry['id'] = self._generate_id()
        
        if 'task' not in entry:
            raise ValueError("Episode must include 'task' field")
        
        self.data.append(entry)
        
        if self.persistent:
            self.save()
        
        print(f"[Episodic Memory] Stored episode {entry['id']}: {entry.get('task', '')[:80]}...")
        
        return entry['id']
    
    def retrieve(self, query: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve similar past episodes based on query.
        
        Args:
            query: Query parameters with keys like:
                - task_keywords: List of keywords to match in task description
                - tools_used: List of tool names to match
                - min_confidence: Minimum confidence threshold
                - time_range: Tuple of (start_time, end_time)
            limit: Maximum number of episodes to return
            
        Returns:
            List of matching episodes (most recent first)
            
        Note: Currently uses simple keyword matching.
        Future: Implement semantic similarity using embeddings (e.g., pgvector).
        """
        results = []
        task_keywords = query.get('task_keywords', [])
        tools_used = query.get('tools_used', [])
        min_confidence = query.get('min_confidence', 0.0)
        
        for episode in reversed(self.data):
            score = 0
            
            if min_confidence > 0 and episode.get('confidence', 0) < min_confidence:
                continue
            
            task = episode.get('task', '').lower()
            for keyword in task_keywords:
                if keyword.lower() in task:
                    score += 1
            
            episode_tools = episode.get('actions', [])
            if isinstance(episode_tools, list):
                episode_tool_names = [
                    action.get('tool', action.get('action', '')) 
                    for action in episode_tools 
                    if isinstance(action, dict)
                ]
                for tool in tools_used:
                    if tool in episode_tool_names:
                        score += 2
            
            if score > 0 or (not task_keywords and not tools_used):
                results.append({
                    'episode': episode,
                    'relevance_score': score
                })
                
                if len(results) >= limit:
                    break
        
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return [r['episode'] for r in results[:limit]]
    
    def get_recent_episodes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent episodes."""
        return list(reversed(self.data[-limit:]))
    
    def get_successful_episodes(self, min_confidence: float = 0.7, limit: int = 10) -> List[Dict[str, Any]]:
        """Get episodes that completed successfully with high confidence."""
        successful = [
            episode for episode in reversed(self.data)
            if episode.get('confidence', 0) >= min_confidence
            and episode.get('outcome') != 'failed'
        ]
        return successful[:limit]
    
    def get_episodes_by_tools(self, tools: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Get episodes that used specific tools."""
        matching = []
        for episode in reversed(self.data):
            episode_tools = episode.get('actions', [])
            if isinstance(episode_tools, list):
                episode_tool_names = [
                    action.get('tool', action.get('action', ''))
                    for action in episode_tools
                    if isinstance(action, dict)
                ]
                if any(tool in episode_tool_names for tool in tools):
                    matching.append(episode)
                    if len(matching) >= limit:
                        break
        return matching
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored episodes."""
        if not self.data:
            return {
                'total_episodes': 0,
                'avg_confidence': 0.0,
                'most_used_tools': []
            }
        
        confidences = [ep.get('confidence', 0) for ep in self.data]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        tool_counts = {}
        for episode in self.data:
            actions = episode.get('actions', [])
            if isinstance(actions, list):
                for action in actions:
                    if isinstance(action, dict):
                        tool = action.get('tool', action.get('action', ''))
                        if tool:
                            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        
        most_used = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_episodes': len(self.data),
            'avg_confidence': avg_confidence,
            'most_used_tools': [{'tool': tool, 'count': count} for tool, count in most_used]
        }

