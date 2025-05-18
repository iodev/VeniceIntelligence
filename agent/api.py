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
                model_record = ModelPerformance()
                model_record.model_id = model_id
                model_record.provider = provider
                model_record.display_name = model_info.get('name', model_id)
                model_record.capabilities = ",".join(model_info.get('capabilities', ['text']))
                model_record.context_window = model_info.get('context_length', 8192)
                model_record.cost_per_1k_tokens = model_info.get('cost_per_1k_tokens', 0.0)
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
        
    def refresh_models(self, agent=None) -> bool:
        """
        Refresh the available models list from all providers
        
        Args:
            agent: Optional agent instance to access provider clients
            
        Returns:
            Success status
        """
        try:
            if agent is None:
                logger.warning("Cannot refresh models without agent instance")
                return False
                
            # Refresh Venice models
            if hasattr(agent, 'venice_client') and agent.venice_client:
                try:
                    venice_models = agent.venice_client.get_available_models()
                    for model_info in venice_models:
                        self.register_model("venice", model_info["id"], model_info)
                    logger.info(f"Refreshed {len(venice_models)} Venice models")
                except Exception as e:
                    logger.error(f"Error refreshing Venice models: {str(e)}")
            
            # Refresh Perplexity models
            if hasattr(agent, 'perplexity_client') and agent.perplexity_client:
                try:
                    perplexity_models = agent.perplexity_client.get_available_models()
                    for model_info in perplexity_models:
                        self.register_model("perplexity", model_info["id"], model_info)
                    logger.info(f"Refreshed {len(perplexity_models)} Perplexity models")
                except Exception as e:
                    logger.error(f"Error refreshing Perplexity models: {str(e)}")
            
            # Refresh Anthropic models
            if hasattr(agent, 'anthropic_client') and agent.anthropic_client:
                try:
                    anthropic_models = agent.anthropic_client.get_available_models()
                    for model_info in anthropic_models:
                        self.register_model("anthropic", model_info["id"], model_info)
                    logger.info(f"Refreshed {len(anthropic_models)} Anthropic models")
                except Exception as e:
                    logger.error(f"Error refreshing Anthropic models: {str(e)}")
                    
            # Check for deprecated models
            self._mark_deprecated_models(agent)
            
            logger.info("Completed refreshing available models from all providers")
            return True
        except Exception as e:
            logger.error(f"Failed to refresh models: {str(e)}")
            return False
            
    def _mark_deprecated_models(self, agent) -> None:
        """
        Mark models as deprecated if they're no longer available from providers
        
        Args:
            agent: Agent instance to access provider clients
        """
        from models import ModelPerformance
        
        # Get all active models from database
        active_models = ModelPerformance.query.filter_by(is_available=True).all()
        current_models = {}
        
        # Collect currently available models from each provider
        if hasattr(agent, 'venice_client') and agent.venice_client:
            try:
                venice_models = agent.venice_client.get_available_models()
                current_models["venice"] = [m["id"] for m in venice_models]
            except:
                current_models["venice"] = []
                
        if hasattr(agent, 'perplexity_client') and agent.perplexity_client:
            try:
                perplexity_models = agent.perplexity_client.get_available_models()
                current_models["perplexity"] = [m["id"] for m in perplexity_models]
            except:
                current_models["perplexity"] = []
                
        if hasattr(agent, 'anthropic_client') and agent.anthropic_client:
            try:
                anthropic_models = agent.anthropic_client.get_available_models()
                current_models["anthropic"] = [m["id"] for m in anthropic_models]
            except:
                current_models["anthropic"] = []
        
        # Check each active model to see if it's still available
        for model in active_models:
            provider = model.provider
            model_id = model.model_id
            
            if provider in current_models and model_id not in current_models[provider]:
                # Model is no longer available, mark as deprecated
                model.is_available = False
                logger.info(f"Marked model {provider}:{model_id} as deprecated")
        
        self.db.session.commit()


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
        stream: bool = False,
        session_id: Optional[str] = None
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
            session_id: Optional session identifier for tracking conversation continuity

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
            if stream:
                # For streaming, we need to get the direct stream from the provider
                if hasattr(self.agent, 'venice_client') and self.agent.venice_client:
                    try:
                        # If session_id is provided, get conversation history for continuity
                        conversation_history = []
                        if session_id and session_id in self.agent.active_conversations:
                            conversation_history = self.agent.active_conversations[session_id].get("messages", [])
                            logger.debug(f"Using conversation history from session {session_id} with {len(conversation_history)} messages")
                        
                        # Prepare context and messages similar to agent's process_query
                        relevant_memories = self.agent.memory_manager.get_relevant_memories(query, limit=5)
                        context = self.agent._create_context_from_memories(relevant_memories)
                        messages = self.agent._construct_prompt(query, system_prompt, context, conversation_history)
                        
                        # Get the model to use - simplified version of what the agent does
                        model_to_use = model_id if model_id else self.agent.current_model
                        
                        # Create streaming response
                        streaming_response = self.agent.venice_client.generate_stream(
                            messages=messages,
                            model=model_to_use
                        )
                        
                        # Store interaction once we're done with streaming
                        self.agent.memory_manager.store_interaction(
                            query=query,
                            response="[Streaming response]",  # Placeholder
                            system_prompt=system_prompt
                        )
                        
                        # Update conversation history if session_id is provided
                        if session_id:
                            if session_id not in self.agent.active_conversations:
                                self.agent.active_conversations[session_id] = {
                                    "messages": [],
                                    "last_updated": datetime.now(),
                                    "query_count": 1
                                }
                            else:
                                self.agent.active_conversations[session_id]["last_updated"] = datetime.now()
                                self.agent.active_conversations[session_id]["query_count"] += 1
                        
                        # Return with stream
                        return {
                            "status": "success",
                            "response_stream": streaming_response,
                            "model_used": model_to_use,
                            "query_type": query_type,
                            "query_id": query_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "agent_id": agent_id,
                            "session_id": session_id
                        }
                    except Exception as e:
                        logger.error(f"Error setting up stream: {str(e)}")
                        # Fall back to non-streaming
                
                # If streaming setup failed or isn't available, fall back to normal processing
                logger.warning("Streaming requested but not available, falling back to normal processing")
                        
            # Standard non-streaming processing
            response_text, model_used = self.agent.process_query(
                query, 
                system_prompt, 
                query_type,
                session_id
            )
            
            # Return structured response
            return {
                "status": "success",
                "response": response_text,
                "model_used": model_used,
                "query_type": query_type,
                "query_id": query_id,
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": agent_id,
                "session_id": session_id
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
    
    def update_system_prompt(self, system_prompt: str, parent_node_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Update the default system prompt for the agent
        This allows parent nodes to modify agent behavior

        Args:
            system_prompt: New system prompt
            parent_node_id: Optional ID of the parent node making the request

        Returns:
            Status dictionary
        """
        try:
            if not system_prompt:
                return {
                    "status": "error",
                    "error": "System prompt cannot be empty"
                }
                
            # Log the request from the parent node
            if parent_node_id:
                logger.info(f"System prompt update requested by parent node: {parent_node_id}")
            
            # Update the agent's system prompt
            if hasattr(self.agent, 'default_system_prompt'):
                # Store the previous prompt for potential rollback
                previous_prompt = self.agent.default_system_prompt
                
                # Update the prompt
                self.agent.default_system_prompt = system_prompt
                logger.info(f"Default system prompt updated to: {system_prompt[:50]}...")
                
                # Save the prompt to database for persistence
                self._save_system_prompt_to_db(system_prompt, parent_node_id)
                
                return {
                    "status": "success",
                    "message": "System prompt updated successfully",
                    "previous_prompt": previous_prompt
                }
            else:
                return {
                    "status": "error",
                    "error": "Agent does not support default system prompt updates"
                }
        except Exception as e:
            logger.error(f"Error updating system prompt: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _save_system_prompt_to_db(self, system_prompt: str, parent_node_id: Optional[str] = None) -> None:
        """
        Save the system prompt to the database for persistence
        
        Args:
            system_prompt: The system prompt to save
            parent_node_id: Optional ID of the parent node that requested the update
        """
        try:
            # In a production implementation, this would store the prompt in a
            # database table specifically for system prompts with versioning
            
            # For now, we'll just log it to show the functionality
            logger.info(f"System prompt saved to persistence layer. Parent: {parent_node_id or 'none'}")
            
            # You could implement a real database storage like:
            # from models import SystemPrompt
            # prompt = SystemPrompt(
            #     prompt_text=system_prompt,
            #     parent_node_id=parent_node_id,
            #     is_active=True
            # )
            # db.session.add(prompt)
            # db.session.commit()
            
        except Exception as e:
            logger.error(f"Error saving system prompt to database: {str(e)}")
            # Don't raise the exception, just log it
    
    def get_available_models(self) -> Dict[str, Any]:
        """
        Get information about available models across all providers
        
        Returns:
            Dictionary with model information grouped by provider
        """
        try:
            # First refresh models to ensure we have the latest information
            self.model_manager.refresh_models(self.agent)
            
            # Get models from the model manager
            models_by_provider = self.model_manager.get_available_models()
            
            # If no models are available from the model manager, fall back to direct queries
            if not models_by_provider:
                logger.info("No models found in model manager, querying providers directly")
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
            
            # Add model status information from database
            from models import ModelPerformance
            all_models = ModelPerformance.query.all()
            model_status = {}
            
            for model in all_models:
                key = f"{model.provider}:{model.model_id}"
                model_status[key] = {
                    "is_available": model.is_available,
                    "is_current": model.is_current,
                    "success_rate": model.success_rate() if hasattr(model, 'success_rate') else 0,
                    "average_latency": model.average_latency() if hasattr(model, 'average_latency') else 0,
                    "average_quality": model.average_quality() if hasattr(model, 'average_quality') else 0
                }
                
            return {
                "status": "success",
                "models": models_by_provider,
                "model_status": model_status
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
    def register_external_node(self, node_id: str, node_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register an external node or agent system that can communicate with this agent
        
        Args:
            node_id: Unique identifier for the external node
            node_info: Information about the node (capabilities, etc.)
            
        Returns:
            Status dictionary with access details
        """
        try:
            if not node_id:
                return {
                    "status": "error",
                    "error": "Node ID is required",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            # Generate an access token for this node
            # In a real implementation, this would use proper auth mechanisms
            access_token = str(uuid.uuid4())
            
            # Extract node capabilities
            capabilities = node_info.get('capabilities', [])
            specialties = node_info.get('specialties', [])
            
            # Here we would store the node information in a database
            # For now, we'll just log it
            logger.info(f"Registered external node: {node_id} with capabilities: {capabilities}")
            
            # Return success with API endpoints and agent information
            return {
                "status": "success",
                "message": f"Successfully registered node {node_id}",
                "access_token": access_token,
                "timestamp": datetime.utcnow().isoformat(),
                "allowed_endpoints": [
                    "/api/node/query",
                    "/api/node/update_system_prompt",
                    "/api/node/refresh_models"
                ],
                "agent_info": {
                    "agent_id": getattr(self.agent, 'agent_id', 'main_agent'),
                    "current_model": self.agent.current_model,
                    "capabilities": ["text", "code", "image"],
                    "default_system_prompt": getattr(self.agent, 'default_system_prompt', 
                                                    "You are a helpful AI assistant.")
                }
            }
        except Exception as e:
            logger.error(f"Error registering external node: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def handle_perplexity_query(self, 
                       messages: List[Dict[str, str]], 
                       model: str = "llama-3.1-sonar-small-128k-online",
                       max_tokens: Optional[int] = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Process a query specifically for the Perplexity API
        
        Args:
            messages: List of message objects (as required by Perplexity API)
            model: Model ID to use
            max_tokens: Maximum number of tokens to generate
            kwargs: Additional parameters for the Perplexity API
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Check if Perplexity client is available
            if not hasattr(self.agent, 'perplexity_client') or not self.agent.perplexity_client:
                logger.warning("Perplexity client not available, trying to initialize it")
                try:
                    # Try to initialize on-demand if we have the key in environment
                    from agent.perplexity import PerplexityClient
                    self.agent.perplexity_client = PerplexityClient()
                    
                    # Register this provider's models with our model manager
                    perplexity_models = self.agent.perplexity_client.get_available_models()
                    for model_info in perplexity_models:
                        self.model_manager.register_model(
                            provider="perplexity",
                            model_id=model_info["id"],
                            model_info=model_info
                        )
                except Exception as init_error:
                    return {
                        "status": "error",
                        "error": f"Perplexity client not available and could not be initialized: {str(init_error)}",
                        "timestamp": datetime.utcnow().isoformat()
                    }
            
            # Make sure we have a valid Perplexity API key
            if not self.agent.perplexity_client.api_key:
                return {
                    "status": "error",
                    "error": "Perplexity API key not configured",
                    "message": "Please make sure a valid PERPLEXITY_API_KEY environment variable is set",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            # Send request to Perplexity API
            result = self.agent.perplexity_client.generate(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Track this usage in our system (using a unique query ID)
            query_id = str(uuid.uuid4())
            logger.info(f"Processed Perplexity query {query_id} with model {model}")
            
            try:
                # Log usage to the database
                from models import UsageCost
                from main import db
                
                # Extract tokens from response if available
                request_tokens = 0
                response_tokens = 0
                if "usage" in result:
                    request_tokens = result["usage"].get("prompt_tokens", 0)
                    response_tokens = result["usage"].get("completion_tokens", 0)
                
                # Create usage cost record
                usage = UsageCost()
                usage.model_id = model
                usage.provider = "perplexity"
                usage.request_tokens = request_tokens
                usage.response_tokens = response_tokens
                usage.total_tokens = request_tokens + response_tokens
                usage.query_id = query_id
                usage.request_type = "chat"
                db.session.add(usage)
                db.session.commit()
            except Exception as log_error:
                logger.error(f"Error logging usage: {str(log_error)}")
            
            return {
                "status": "success",
                "response": result,
                "model_used": model,
                "timestamp": datetime.utcnow().isoformat(),
                "query_id": query_id
            }
            
        except Exception as e:
            logger.error(f"Error processing Perplexity query: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def handle_anthropic_query(self, 
                        messages: List[Dict[str, str]], 
                        model: str = "claude-3.5-sonnet-20241022",
                        max_tokens: int = 1024,
                        **kwargs) -> Dict[str, Any]:
        """
        Process a query specifically for the Anthropic API
        
        Args:
            messages: List of message objects (as required by Anthropic API)
            model: Model ID to use
            max_tokens: Maximum number of tokens to generate
            kwargs: Additional parameters for the Anthropic API
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Check if Anthropic client is available
            if not hasattr(self.agent, 'anthropic_client') or not self.agent.anthropic_client:
                logger.warning("Anthropic client not available, trying to initialize it")
                try:
                    # Try to initialize on-demand if we have the key in environment
                    from agent.anthropic_client import AnthropicClient
                    self.agent.anthropic_client = AnthropicClient()
                except Exception as init_error:
                    return {
                        "status": "error",
                        "error": f"Anthropic client not available and could not be initialized: {str(init_error)}",
                        "timestamp": datetime.utcnow().isoformat()
                    }
            
            # Make sure we have a valid Anthropic API key
            if not self.agent.anthropic_client.api_key:
                return {
                    "status": "error",
                    "error": "Anthropic API key not configured",
                    "message": "Please make sure a valid ANTHROPIC_API_KEY environment variable is set",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            # Send request to Anthropic API
            result = self.agent.anthropic_client.generate(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Track this usage in our system
            # In a real implementation, this would log to a database
            logger.info(f"Processed Anthropic query with model {model}")
            
            return {
                "status": "success",
                "response": result,
                "model_used": model,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing Anthropic query: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }