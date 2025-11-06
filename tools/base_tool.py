"""
Base classes for workflows and utilities in the hierarchical tool architecture
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from enum import Enum

class ToolType(Enum):
    WORKFLOW = "workflow"
    UTILITY = "utility"

class BaseWorkflow(ABC):
    """
    Base class for high-level workflow tools
    Workflows orchestrate multiple utilities to complete complex tasks
    """
    
    def __init__(self, name: Optional[str] = None, description: Optional[str] = None, 
                 parameters: Optional[Dict] = None):
        self.tool_type = ToolType.WORKFLOW
        self.name = name or self.__class__.__name__
        self.description = description or self.get_description()
        self.parameters = parameters or {}
        
    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the workflow with given parameters
        
        Args:
            params: Dictionary containing workflow parameters
            
        Returns:
            Dictionary containing workflow results
        """
        pass
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """
        Return visualization configuration for analytics dashboard
        Override this method to provide workflow-specific visualization settings
        
        Returns:
            Dictionary with visualization configuration:
            {
                "workflow_type": "workflow_name",
                "components": ["list", "of", "visualization", "components"],
                "primary_view": "main_visualization_component",
                "metrics": {
                    "metric_name": "Metric description"
                },
                "export_formats": ["json", "csv", "pdf"]
            }
        """
        return {
            "workflow_type": self.name.lower(),
            "components": ["metric_cards"],
            "primary_view": "metric_cards",
            "metrics": {},
            "export_formats": ["json"]
        }
    
    def get_description(self) -> str:
        """
        Return human-readable description of what this workflow does
        Override to provide specific description
        """
        return f"{self.name} workflow"
    
    def get_required_utilities(self) -> List[str]:
        """
        Return list of utility names that this workflow depends on
        Override to specify dependencies
        """
        return []


class BaseUtility(ABC):
    """
    Base class for low-level utility tools
    Utilities are atomic, reusable components that can be used by multiple workflows
    """
    
    def __init__(self, name: Optional[str] = None, description: Optional[str] = None,
                 parameters: Optional[Dict] = None):
        self.tool_type = ToolType.UTILITY
        self.name = name or self.__class__.__name__
        self.description = description or self.get_description()
        self.parameters = parameters or {}
        
    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the utility function with given parameters
        Note: Utilities should be synchronous or async based on their needs
        
        Args:
            params: Dictionary containing utility parameters
            
        Returns:
            Dictionary containing utility results
        """
        pass
    
    def get_description(self) -> str:
        """
        Return human-readable description of what this utility does
        Override to provide specific description
        """
        return f"{self.name} utility"
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """
        Return JSON schema describing the parameters this utility accepts
        Override to provide parameter documentation
        """
        return {
            "type": "object",
            "properties": {},
            "required": []
        }

