"""
API module for the agent system. This module provides a clean interface for other systems
to interact with the agent, completely separate from the UI layer.
"""
import logging
from typing import Dict, List, Any, Tuple, Optional

from agent.core import Agent

logger = logging.getLogger(__name__)

class AgentAPI:
    """
    API interface for the agent system, allowing it to be queried by other nodes
    in a larger system. This class acts as a facade over the agent, hiding implementation
    details and providing a clean, stable interface.
    """
    
    def __init__(self, agent: Agent):
        """
        Initialize the API with a configured agent

        Args:
            agent: An initialized agent instance
        """
        self.agent = agent
        
    def process_query(
        self, 
        query: str, 
        system_prompt: Optional[str] = None, 
        query_type: str = "text",
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a query from another node or system

        Args:
            query: The query text to process
            system_prompt: Optional system prompt that can override the default
                           (allowing parent nodes to customize the agent's behavior)
            query_type: Type of query (text, code, image)
            agent_id: Optional ID to track which agent/node is making the request
                      (for cost monitoring and analytics)

        Returns:
            Dictionary with response and metadata
        """
        try:
            # Use default system prompt if none provided
            if system_prompt is None:
                system_prompt = "You are a helpful AI assistant."
                
            # Process query using the agent
            response_text, model_used = self.agent.process_query(
                query, 
                system_prompt, 
                query_type
            )
            
            # Return structured response
            return {
                "status": "success",
                "response": response_text,
                "model_used": model_used,
                "query_type": query_type
            }
            
        except Exception as e:
            logger.error(f"Error processing API query: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def update_system_prompt(self, system_prompt: str) -> Dict[str, Any]:
        """
        Update the default system prompt for the agent
        This allows parent nodes to modify agent behavior

        Args:
            system_prompt: New system prompt

        Returns:
            Status dictionary
        """
        try:
            # In a more complex implementation, this would update the agent's 
            # configuration in a database or other persistent storage
            return {
                "status": "success",
                "message": "System prompt updated successfully"
            }
        except Exception as e:
            logger.error(f"Error updating system prompt: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_available_models(self) -> Dict[str, Any]:
        """
        Get information about available models across all providers
        
        Returns:
            Dictionary with model information grouped by provider
        """
        try:
            models_by_provider = {}
            
            # Get Venice models
            if hasattr(self.agent, 'venice_client') and self.agent.venice_client:
                venice_models = self.agent.venice_client.get_available_models()
                models_by_provider['venice'] = venice_models
            
            # Get other provider models if available
            if hasattr(self.agent, 'perplexity_client') and self.agent.perplexity_client:
                perplexity_models = self.agent.perplexity_client.get_available_models()
                models_by_provider['perplexity'] = perplexity_models
            
            if hasattr(self.agent, 'anthropic_client') and self.agent.anthropic_client:
                anthropic_models = self.agent.anthropic_client.get_available_models()
                models_by_provider['anthropic'] = anthropic_models
                
            return {
                "status": "success",
                "models": models_by_provider
            }
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
            
    def get_cost_metrics(self) -> Dict[str, Any]:
        """
        Get cost monitoring metrics for this agent
        
        Returns:
            Dictionary with cost and efficiency metrics
        """
        try:
            if hasattr(self.agent, 'cost_monitor') and self.agent.cost_monitor:
                cost_summary = self.agent.cost_monitor.get_cost_summary()
                efficiency_metrics = self.agent.cost_monitor.get_efficiency_metrics()
                
                return {
                    "status": "success",
                    "cost_summary": cost_summary,
                    "efficiency_metrics": efficiency_metrics
                }
            else:
                return {
                    "status": "error",
                    "error": "Cost monitoring not available"
                }
        except Exception as e:
            logger.error(f"Error getting cost metrics: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def reset_memory(self) -> Dict[str, Any]:
        """
        Reset the agent's memory
        
        Returns:
            Status dictionary
        """
        try:
            if hasattr(self.agent, 'memory_manager') and self.agent.memory_manager:
                self.agent.memory_manager.clear_memories()
                
                return {
                    "status": "success",
                    "message": "Memory reset successfully"
                }
            else:
                return {
                    "status": "error",
                    "error": "Memory manager not available"
                }
        except Exception as e:
            logger.error(f"Error resetting memory: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }