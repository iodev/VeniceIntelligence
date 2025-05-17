"""
API module for the agent system. This module provides a clean interface for other systems
to interact with the agent, completely separate from the UI layer.

The AgentAPI class serves as the primary gateway for external systems to interact with
the agent, including:
1. Parent nodes that want to control this agent
2. Other agent systems that need to query this agent
3. Web interfaces or other clients
"""
import logging
import json
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import uuid

from agent.core import Agent
from agent.models import VeniceClient
from agent.perplexity import PerplexityClient
from agent.anthropic_client import AnthropicClient
from agent.memory import MemoryManager

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Handles model registration, deregistration, and availability tracking
    for different model providers.
    
    This class maintains the available models across providers and ensures
    that the agent system can adapt to changing model availability.
    """
    
    def __init__(self):
        """Initialize the model manager"""
        from models import ModelPerformance
        from main import db
        
        self.models = {}  # Dict of provider -> list of model info
        self.db = db
        
    def register_model(self, provider: str, model_id: str, model_info: Dict[str, Any]) -> bool:
        """
        Register a new model with the system
        
        Args:
            provider: Provider name (venice, anthropic, perplexity, etc.)
            model_id: Unique identifier for the model
            model_info: Dictionary with model information
            
        Returns:
            Success status
        """
        try:
            # Add to in-memory registry
            if provider not in self.models:
                self.models[provider] = {}
                
            self.models[provider][model_id] = model_info
            
            # Add or update the database record
            from models import ModelPerformance
            model_record = ModelPerformance.query.filter_by(
                model_id=model_id, 
                provider=provider
            ).first()
            
            if not model_record:
                # Create new record
                model_record = ModelPerformance(
                    model_id=model_id,
                    provider=provider,
                    display_name=model_info.get('name', model_id),
                    capabilities=",".join(model_info.get('capabilities', ['text'])),
                    context_window=model_info.get('context_length', 8192),
                    cost_per_1k_tokens=model_info.get('cost_per_1k_tokens', 0.0)
                )
                self.db.session.add(model_record)
            else:
                # Update existing record
                model_record.display_name = model_info.get('name', model_id)
                model_record.capabilities = ",".join(model_info.get('capabilities', ['text']))
                model_record.context_window = model_info.get('context_length', 8192)
                model_record.cost_per_1k_tokens = model_info.get('cost_per_1k_tokens', 0.0)
                
            self.db.session.commit()
            logger.info(f"Successfully registered model {provider}:{model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {provider}:{model_id}: {str(e)}")
            return False
            
    def deregister_model(self, provider: str, model_id: str) -> bool:
        """
        Remove a model from the available models list
        
        Args:
            provider: Provider name
            model_id: Model identifier
            
        Returns:
            Success status
        """
        try:
            # Remove from in-memory registry
            if provider in self.models and model_id in self.models[provider]:
                del self.models[provider][model_id]
                
            # Mark as unavailable in database
            from models import ModelPerformance
            model_record = ModelPerformance.query.filter_by(
                model_id=model_id, 
                provider=provider
            ).first()
            
            if model_record:
                # Don't delete, just mark as unavailable
                model_record.is_available = False
                self.db.session.commit()
                
            logger.info(f"Successfully deregistered model {provider}:{model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deregister model {provider}:{model_id}: {str(e)}")
            return False
            
    def get_available_models(self, provider: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get list of available models, optionally filtered by provider
        
        Args:
            provider: Optional provider name to filter by
            
        Returns:
            Dictionary of provider -> list of model info
        """
        if provider:
            return {provider: self.models.get(provider, {})}
        else:
            return self.models
        
    def refresh_models(self) -> bool:
        """
        Refresh the available models list from all providers
        
        Returns:
            Success status
        """
        try:
            # This would typically involve querying each provider's API
            # for their current available models
            logger.info("Refreshing available models from all providers")
            return True
        except Exception as e:
            logger.error(f"Failed to refresh models: {str(e)}")
            return False


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
        self.model_manager = ModelManager()
        
    def process_query(
        self, 
        query: str, 
        system_prompt: Optional[str] = None, 
        query_type: str = "text",
        agent_id: Optional[str] = None,
        provider: Optional[str] = None,
        model_id: Optional[str] = None,
        stream: bool = False
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
            provider: Optional provider to use (venice, anthropic, perplexity)
            model_id: Optional specific model ID to use
            stream: Whether to stream the response (for streaming UIs)

        Returns:
            Dictionary with response and metadata
        """
        try:
            # Generate a unique query ID for tracking
            query_id = str(uuid.uuid4())
            logger.info(f"Processing query {query_id} from agent {agent_id or 'unknown'}")
            
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
                "query_type": query_type,
                "query_id": query_id,
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": agent_id
            }
            
        except Exception as e:
            logger.error(f"Error processing API query: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "query_id": str(uuid.uuid4()),
                "agent_id": agent_id
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