"""
Procedural Memory Module for CoALA Architecture

Long-term storage of learned skills, effective strategies, prompt templates,
and successful tool usage patterns.
"""

from typing import Dict, List, Any, Optional
from agent.memory.base_memory import BaseMemory


class ProceduralMemory(BaseMemory):
    """
    Procedural Memory: Learned strategies and skills.
    
    Stores:
    - Successful tool combinations (e.g., "use region_mapper before object_detector")
    - Effective prompt templates
    - Learned workflows (e.g., "München query → bbox + detection workflow")
    - Strategy patterns that worked well
    
    Structure is in place for future learning algorithms.
    Currently stores patterns manually or from successful episodes.
    Future: Implement automatic pattern extraction and reinforcement learning.
    """
    
    def __init__(self, file_path: str = "data/memory/procedural_memory.toon"):
        """
        Initialize procedural memory.
        
        Args:
            file_path: Path to TOON file for persistence
        """
        super().__init__(memory_type="procedural", file_path=file_path, persistent=True)
        self._initialize_default_procedures()
    
    def _initialize_default_procedures(self):
        """Initialize with some default procedural knowledge if memory is empty."""
        if len(self.data) == 0:
            default_procedures = [
                {
                    'procedure_type': 'tool_sequence',
                    'name': 'region_to_detection',
                    'description': 'Standard workflow: Map region first, then detect objects',
                    'pattern': ['region_mapper', 'object_detector'],
                    'context': 'When task involves detecting objects in a named region',
                    'success_rate': 1.0,
                    'usage_count': 0,
                    'examples': ['Detect vehicles in München', 'Find ships near Starnberger See']
                },
                {
                    'procedure_type': 'tool_sequence',
                    'name': 'image_to_detection',
                    'description': 'Process image before running detection',
                    'pattern': ['image_processor', 'object_detector'],
                    'context': 'When working with raw satellite imagery',
                    'success_rate': 1.0,
                    'usage_count': 0,
                    'examples': ['Analyze raw satellite image', 'Process and detect objects']
                },
                {
                    'procedure_type': 'strategy',
                    'name': 'region_expansion',
                    'description': 'For large regions like Bayern, use moderate bbox expansion (0.3-0.5)',
                    'pattern': {'tool': 'region_mapper', 'parameter': 'expand_bbox', 'value_range': [0.3, 0.5]},
                    'context': 'Querying large geographic areas',
                    'success_rate': 1.0,
                    'usage_count': 0
                },
                {
                    'procedure_type': 'prompt_template',
                    'name': 'region_query_extraction',
                    'description': 'Effective prompt for extracting region names from user queries',
                    'template': 'Extract the geographic location from: {query}. Return region name or null if not found.',
                    'context': 'When user query mentions a location',
                    'success_rate': 1.0,
                    'usage_count': 0
                }
            ]
            
            for proc in default_procedures:
                self.store(proc)
    
    def store(self, entry: Dict[str, Any]) -> str:
        """
        Store a procedure in memory.
        
        Args:
            entry: Procedure dictionary with keys:
                - procedure_type: Type ('tool_sequence', 'strategy', 'prompt_template', 'skill')
                - name: Name of the procedure
                - description: What this procedure does
                - pattern: The actual pattern/template/sequence
                - context: When to use this procedure
                - success_rate: Success rate (0.0-1.0)
                - usage_count: How many times used
                - examples: Example use cases (optional)
            
        Returns:
            entry_id: Unique identifier for the stored procedure
        """
        entry = self._add_timestamp(entry)
        if 'id' not in entry:
            entry['id'] = self._generate_id()
        
        if 'procedure_type' not in entry or 'name' not in entry:
            raise ValueError("Procedural entry must include 'procedure_type' and 'name' fields")
        
        if 'success_rate' not in entry:
            entry['success_rate'] = 0.5
        if 'usage_count' not in entry:
            entry['usage_count'] = 0
        
        self.data.append(entry)
        
        if self.persistent:
            self.save()
        
        return entry['id']
    
    def retrieve(self, query: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve procedures based on type, context, or pattern.
        
        Args:
            query: Query parameters with keys like:
                - procedure_type: Type of procedure to find
                - context_keywords: Keywords to match in context
                - min_success_rate: Minimum success rate threshold
                - tools: List of tools to match in pattern
            limit: Maximum number of procedures to return
            
        Returns:
            List of matching procedures (sorted by success rate)
            
        Note: Future improvement - learn which procedures work best in which contexts.
        """
        results = []
        procedure_type = query.get('procedure_type', '').lower()
        context_keywords = [k.lower() for k in query.get('context_keywords', [])]
        min_success_rate = query.get('min_success_rate', 0.0)
        tools = query.get('tools', [])
        
        for proc in self.data:
            score = 0
            
            if min_success_rate > 0 and proc.get('success_rate', 0) < min_success_rate:
                continue
            
            if procedure_type and procedure_type == proc.get('procedure_type', '').lower():
                score += 3
            
            context = proc.get('context', '').lower()
            description = proc.get('description', '').lower()
            for keyword in context_keywords:
                if keyword in context or keyword in description:
                    score += 2
            
            pattern = proc.get('pattern', [])
            if isinstance(pattern, list):
                for tool in tools:
                    if tool in pattern:
                        score += 2
            
            success_bonus = proc.get('success_rate', 0.5) * 2
            score += success_bonus
            
            if score > 0 or (not procedure_type and not context_keywords and not tools):
                results.append({
                    'procedure': proc,
                    'relevance_score': score
                })
        
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return [r['procedure'] for r in results[:limit]]
    
    def get_tool_sequences(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get all learned tool sequences."""
        return self.retrieve({'procedure_type': 'tool_sequence'}, limit=limit)
    
    def get_strategies(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get all learned strategies."""
        return self.retrieve({'procedure_type': 'strategy'}, limit=limit)
    
    def get_prompt_templates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get all prompt templates."""
        return self.retrieve({'procedure_type': 'prompt_template'}, limit=limit)
    
    def increment_usage(self, entry_id: str) -> bool:
        """Increment the usage count for a procedure."""
        for proc in self.data:
            if proc.get('id') == entry_id:
                proc['usage_count'] = proc.get('usage_count', 0) + 1
                if self.persistent:
                    self.save()
                return True
        return False
    
    def update_success_rate(self, entry_id: str, success: bool) -> bool:
        """
        Update success rate for a procedure based on outcome.
        
        Uses exponential moving average to update success rate.
        Future: Implement more sophisticated reinforcement learning.
        
        Args:
            entry_id: ID of procedure
            success: Whether the procedure succeeded
            
        Returns:
            True if update successful
        """
        for proc in self.data:
            if proc.get('id') == entry_id:
                current_rate = proc.get('success_rate', 0.5)
                alpha = 0.2
                new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_rate
                proc['success_rate'] = new_rate
                
                if self.persistent:
                    self.save()
                return True
        return False
    
    def suggest_tool_sequence(self, context: str) -> Optional[List[str]]:
        """
        Suggest a tool sequence based on context.
        
        Args:
            context: Description of current task/context
            
        Returns:
            List of tool names in suggested order, or None if no suggestion
        """
        keywords = context.lower().split()
        sequences = self.retrieve({
            'procedure_type': 'tool_sequence',
            'context_keywords': keywords,
            'min_success_rate': 0.5
        }, limit=1)
        
        if sequences:
            pattern = sequences[0].get('pattern', [])
            if isinstance(pattern, list):
                return pattern
        
        return None
    
    def store_successful_sequence(self, tools_used: List[str], task_description: str, outcome: str) -> str:
        """
        Store a successful tool sequence from a completed task.
        
        Placeholder for future learning logic.
        Currently stores the sequence if outcome indicates success.
        
        Args:
            tools_used: List of tools used in order
            task_description: Description of the task
            outcome: Outcome of the task
            
        Returns:
            entry_id if stored, empty string otherwise
        """
        if outcome in ['completed', 'success'] and len(tools_used) > 1:
            return self.store({
                'procedure_type': 'tool_sequence',
                'name': f'learned_sequence_{len(self.data)}',
                'description': f'Learned from successful task: {task_description[:100]}',
                'pattern': tools_used,
                'context': task_description,
                'success_rate': 0.7,
                'usage_count': 1,
                'source': 'automatic_learning'
            })
        return ''
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about procedural memory."""
        if not self.data:
            return {
                'total_procedures': 0,
                'by_type': {},
                'avg_success_rate': 0.0,
                'most_used': []
            }
        
        by_type = {}
        total_success = 0
        
        for proc in self.data:
            proc_type = proc.get('procedure_type', 'unknown')
            by_type[proc_type] = by_type.get(proc_type, 0) + 1
            total_success += proc.get('success_rate', 0)
        
        avg_success = total_success / len(self.data) if self.data else 0.0
        
        most_used = sorted(self.data, key=lambda x: x.get('usage_count', 0), reverse=True)[:5]
        most_used_info = [{
            'name': proc.get('name'),
            'usage_count': proc.get('usage_count', 0),
            'success_rate': proc.get('success_rate', 0)
        } for proc in most_used]
        
        return {
            'total_procedures': len(self.data),
            'by_type': by_type,
            'avg_success_rate': avg_success,
            'most_used': most_used_info
        }

