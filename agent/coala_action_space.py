"""
CoALA Action Space Module

Defines the action space for the CoALA architecture, separating internal actions
(reasoning, retrieval, learning) from external actions (grounding via tools).

Reference: https://arxiv.org/abs/2309.02427
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass


class ActionType(Enum):
    """Types of actions in CoALA framework."""
    INTERNAL_REASONING = "internal_reasoning"
    INTERNAL_RETRIEVAL = "internal_retrieval"
    INTERNAL_LEARNING = "internal_learning"
    EXTERNAL_GROUNDING = "external_grounding"


class MemoryTarget(Enum):
    """Memory modules that can be targeted by internal actions."""
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


@dataclass
class Action:
    """
    Represents a single action in the CoALA framework.
    
    Actions can be internal (operating on memory) or external (interacting with environment).
    """
    name: str
    action_type: ActionType
    description: str
    preconditions: List[str]
    effects: List[str]
    parameters: Dict[str, Any]
    execute: Optional[Callable] = None
    
    def is_internal(self) -> bool:
        """Check if this is an internal action (memory operation)."""
        return self.action_type in [
            ActionType.INTERNAL_REASONING,
            ActionType.INTERNAL_RETRIEVAL,
            ActionType.INTERNAL_LEARNING
        ]
    
    def is_external(self) -> bool:
        """Check if this is an external action (grounding)."""
        return self.action_type == ActionType.EXTERNAL_GROUNDING
    
    def can_execute(self, current_state: Dict[str, Any]) -> bool:
        """
        Check if action's preconditions are met.
        
        Placeholder for future precondition checking logic.
        Currently always returns True.
        """
        return True


class CoALAActionSpace:
    """
    Manages the action space for CoALA agent.
    
    Provides registry of available actions and methods to query them.
    """
    
    def __init__(self, tools: Dict[str, Any], memory_modules: Dict[str, Any]):
        """
        Initialize action space.
        
        Args:
            tools: Dictionary of available grounding tools
            memory_modules: Dictionary of memory modules (working, episodic, semantic, procedural)
        """
        self.tools = tools
        self.memory_modules = memory_modules
        self.actions: Dict[str, Action] = {}
        
        self._register_internal_actions()
        self._register_external_actions()
    
    def _register_internal_actions(self):
        """Register internal actions for memory operations."""
        
        self.actions['reasoning'] = Action(
            name='reasoning',
            action_type=ActionType.INTERNAL_REASONING,
            description='Use LLM to analyze task and update working memory',
            preconditions=['task_defined'],
            effects=['working_memory_updated'],
            parameters={},
            execute=None
        )
        
        self.actions['retrieve_episodic'] = Action(
            name='retrieve_episodic',
            action_type=ActionType.INTERNAL_RETRIEVAL,
            description='Retrieve similar past episodes from episodic memory',
            preconditions=[],
            effects=['past_episodes_available'],
            parameters={'query': {}, 'limit': 5},
            execute=lambda params: self.memory_modules['episodic'].retrieve(
                params.get('query', {}),
                params.get('limit', 5)
            )
        )
        
        self.actions['retrieve_semantic'] = Action(
            name='retrieve_semantic',
            action_type=ActionType.INTERNAL_RETRIEVAL,
            description='Retrieve facts from semantic memory',
            preconditions=[],
            effects=['facts_available'],
            parameters={'query': {}, 'limit': 5},
            execute=lambda params: self.memory_modules['semantic'].retrieve(
                params.get('query', {}),
                params.get('limit', 5)
            )
        )
        
        self.actions['retrieve_procedural'] = Action(
            name='retrieve_procedural',
            action_type=ActionType.INTERNAL_RETRIEVAL,
            description='Retrieve learned strategies from procedural memory',
            preconditions=[],
            effects=['strategies_available'],
            parameters={'query': {}, 'limit': 5},
            execute=lambda params: self.memory_modules['procedural'].retrieve(
                params.get('query', {}),
                params.get('limit', 5)
            )
        )
        
        self.actions['store_episode'] = Action(
            name='store_episode',
            action_type=ActionType.INTERNAL_LEARNING,
            description='Store complete episode in episodic memory',
            preconditions=['episode_complete'],
            effects=['episode_stored'],
            parameters={'episode': {}},
            execute=lambda params: self.memory_modules['episodic'].store(
                params.get('episode', {})
            )
        )
        
        self.actions['store_fact'] = Action(
            name='store_fact',
            action_type=ActionType.INTERNAL_LEARNING,
            description='Store new fact in semantic memory',
            preconditions=[],
            effects=['fact_stored'],
            parameters={'fact': {}},
            execute=lambda params: self.memory_modules['semantic'].store(
                params.get('fact', {})
            )
        )
        
        self.actions['store_procedure'] = Action(
            name='store_procedure',
            action_type=ActionType.INTERNAL_LEARNING,
            description='Store learned strategy in procedural memory',
            preconditions=[],
            effects=['procedure_stored'],
            parameters={'procedure': {}},
            execute=lambda params: self.memory_modules['procedural'].store(
                params.get('procedure', {})
            )
        )
    
    def _register_external_actions(self):
        """Register external actions (grounding tools)."""
        for tool_name, tool_info in self.tools.items():
            self.actions[tool_name] = Action(
                name=tool_name,
                action_type=ActionType.EXTERNAL_GROUNDING,
                description=tool_info.get('description', f'Execute {tool_name}'),
                preconditions=[],
                effects=['environment_interaction'],
                parameters=tool_info.get('parameters', {}),
                execute=tool_info.get('execute')
            )
    
    def get_internal_actions(self) -> List[Action]:
        """Get all internal actions (reasoning, retrieval, learning)."""
        return [action for action in self.actions.values() if action.is_internal()]
    
    def get_external_actions(self) -> List[Action]:
        """Get all external actions (grounding tools)."""
        return [action for action in self.actions.values() if action.is_external()]
    
    def get_reasoning_actions(self) -> List[Action]:
        """Get reasoning actions."""
        return [action for action in self.actions.values() 
                if action.action_type == ActionType.INTERNAL_REASONING]
    
    def get_retrieval_actions(self) -> List[Action]:
        """Get retrieval actions."""
        return [action for action in self.actions.values()
                if action.action_type == ActionType.INTERNAL_RETRIEVAL]
    
    def get_learning_actions(self) -> List[Action]:
        """Get learning actions."""
        return [action for action in self.actions.values()
                if action.action_type == ActionType.INTERNAL_LEARNING]
    
    def get_grounding_actions(self) -> List[Action]:
        """Get grounding actions (tools)."""
        return self.get_external_actions()
    
    def get_action(self, action_name: str) -> Optional[Action]:
        """Get a specific action by name."""
        return self.actions.get(action_name)
    
    def list_actions(self, action_type: Optional[ActionType] = None) -> List[str]:
        """
        List available action names.
        
        Args:
            action_type: Filter by action type, or None for all actions
            
        Returns:
            List of action names
        """
        if action_type is None:
            return list(self.actions.keys())
        
        return [name for name, action in self.actions.items() 
                if action.action_type == action_type]
    
    def execute_action(self, action_name: str, parameters: Dict[str, Any]) -> Any:
        """
        Execute an action.
        
        Args:
            action_name: Name of action to execute
            parameters: Parameters for the action
            
        Returns:
            Result of action execution
            
        Raises:
            ValueError: If action not found or cannot be executed
        """
        action = self.get_action(action_name)
        if action is None:
            raise ValueError(f"Action '{action_name}' not found in action space")
        
        if action.execute is None:
            raise ValueError(f"Action '{action_name}' has no execute function")
        
        return action.execute(parameters)
    
    def get_action_summary(self) -> Dict[str, Any]:
        """Get summary of action space."""
        return {
            'total_actions': len(self.actions),
            'internal_actions': len(self.get_internal_actions()),
            'external_actions': len(self.get_external_actions()),
            'reasoning': len(self.get_reasoning_actions()),
            'retrieval': len(self.get_retrieval_actions()),
            'learning': len(self.get_learning_actions()),
            'grounding': len(self.get_grounding_actions()),
            'action_list': {
                'reasoning': self.list_actions(ActionType.INTERNAL_REASONING),
                'retrieval': self.list_actions(ActionType.INTERNAL_RETRIEVAL),
                'learning': self.list_actions(ActionType.INTERNAL_LEARNING),
                'grounding': self.list_actions(ActionType.EXTERNAL_GROUNDING)
            }
        }

