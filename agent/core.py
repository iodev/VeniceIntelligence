import logging
import random
import time
import re
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from sqlalchemy import func, desc
from agent.memory import MemoryManager
from agent.models import VeniceClient
from agent.perplexity import PerplexityClient
from agent.anthropic_client import AnthropicClient
from agent.evaluation import evaluate_model_response
from agent.registry import ModelRegistry
from models import ModelPerformance, UsageCost
import config
from flask import current_app

logger = logging.getLogger(__name__)

def init_default_models():
    """
    Initialize default models in the database if they don't already exist
    """
    from models import ModelPerformance
    from main import db
    
    # Define default Venice.AI models
    default_models = [
        "mistral-31-24b",  # High-quality generalist model
        "mistral-29-7b",   # Smaller but faster model
        "phi3-mini-4k",    # Compact model for simple tasks
        "llama-3-15b",     # Recent model with strong performance
    ]
    
    # Check if we already have models in the database
    existing_models = ModelPerformance.query.all()
    if existing_models:
        logger.info(f"Found {len(existing_models)} existing models in database")
        return
    
    # Add default models to database
    for model_id in default_models:
        model = ModelPerformance()
        model.model_id = model_id
        model.provider = "venice"  # Default provider
        model.total_calls = 0
        model.successful_calls = 0
        model.total_latency = 0.0
        model.quality_score = 0.0
        model.quality_evaluations = 0
        model.is_current = (model_id == "mistral-31-24b")  # Set the default model
        model.capabilities = "text"  # Default capability
        model.context_window = 8192  # Default context window
        model.display_name = model_id  # Default display name
        db.session.add(model)
    
    db.session.commit()
    logger.info(f"Initialized {len(default_models)} default models in database")

class Agent:
    """
    Self-learning agent that adapts to different AI models from multiple providers
    (Venice.ai, Anthropic, Perplexity) based on performance
    and maintains persistent memory with Qdrant.
    """
    
    def __init__(
        self,
        venice_client: VeniceClient,
        memory_manager: MemoryManager,
        available_models: List[str],
        default_model: str,
        default_system_prompt: str = "You are a helpful AI assistant."
    ):
        """
        Initialize the agent with the Venice API client and memory manager
        
        Args:
            venice_client: Initialized Venice API client
            memory_manager: Memory manager with Qdrant integration
            available_models: List of available Venice.ai models to use
            default_model: Default model to start with
            default_system_prompt: Default system prompt to use when none is provided
        """
        # Set the default system prompt
        self.default_system_prompt = default_system_prompt
        
        # Conversation tracking for continuity between user queries
        self.active_conversations = {}  # Map of session_id to conversation state
        
        # Initialize provider status tracking
        self._provider_status = {
            "venice": True,  # Venice is always available as our primary provider
            "anthropic": False,  # Set to False by default until tested
            "perplexity": False  # Set to False by default until tested
        }
        
        # Track multi-provider usage statistics
        # This allows us to learn when using multiple providers is beneficial
        self._multi_provider_stats = {
            "text": {
                "total_uses": 0,
                "positive_feedback": 0,
                "negative_feedback": 0
            },
            "code": {
                "total_uses": 0,
                "positive_feedback": 0,
                "negative_feedback": 0
            },
            "image": {
                "total_uses": 0,
                "positive_feedback": 0,
                "negative_feedback": 0
            }
        }
        
        # Temporarily store the last multi-provider query details
        self._last_multi_provider_query = {
            "used": False,
            "query_type": None,
            "timestamp": None,
            "providers_used": []
        }
        from models import ModelPerformance
        from main import db
        from agent.cost_control import CostMonitor
        
        # Set up API clients
        self.venice_client = venice_client
        self.memory_manager = memory_manager
        self.available_models = available_models
        self.cost_monitor = CostMonitor()
        
        # Create model registry 
        self.model_registry = ModelRegistry()
        
        # Try to initialize additional API clients if keys are available
        # Initialize Perplexity client first as Anthropic can use it for model discovery
        try:
            self.perplexity_client = PerplexityClient()
            if self.perplexity_client.api_key:
                logger.info("Perplexity API client initialized successfully")
                
                # Register this client with the model registry
                self.model_registry.register_client("perplexity", self.perplexity_client)
                
                # We'll register Perplexity models at the end of initialization
                self._register_provider_models_pending = True
            else:
                self.perplexity_client = None
                logger.warning("No Perplexity API key found, client not available")
        except Exception as e:
            logger.warning(f"Failed to initialize Perplexity client: {str(e)}")
            self.perplexity_client = None
            
        # Initialize Anthropic client with Perplexity for dynamic model lookup
        try:
            self.anthropic_client = AnthropicClient(perplexity_client=self.perplexity_client)
            if self.anthropic_client.api_key:
                logger.info("Anthropic API client initialized successfully")
                
                # Register with model registry
                self.model_registry.register_client("anthropic", self.anthropic_client)
                
                # Proactively update Anthropic models via Perplexity
                if self.perplexity_client and self.perplexity_client.api_key:
                    try:
                        # Attempt to fetch and register current Anthropic models
                        self._register_provider_models("anthropic")
                        logger.info("Successfully updated Anthropic models via Perplexity")
                    except Exception as model_err:
                        logger.warning(f"Failed to update Anthropic models: {str(model_err)}")
            else:
                self.anthropic_client = None
                logger.warning("No Anthropic API key found, client not available")
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic client: {str(e)}")
            self.anthropic_client = None
        
        # Load or initialize models in database
        init_default_models()
        
        # Process any pending model registrations
        if hasattr(self, '_register_provider_models_pending') and self._register_provider_models_pending:
            # Register Perplexity models
            if self.perplexity_client and self.perplexity_client.api_key:
                try:
                    self._register_provider_models('perplexity')
                    logger.info("Successfully registered Perplexity models")
                except Exception as e:
                    logger.warning(f"Failed to register Perplexity models: {str(e)}")
            
            # Clear the pending flag
            self._register_provider_models_pending = False
        
        # Get current model from database
        current_model_record = ModelPerformance.query.filter_by(is_current=True).first()
        if current_model_record:
            self.current_model = current_model_record.model_id
        else:
            # If no current model is set, use the provided default
            self.current_model = default_model
            # And mark it as current in the database
            model_record = ModelPerformance.query.filter_by(model_id=default_model).first()
            if model_record:
                model_record.is_current = True
                db.session.commit()
        
        # Interaction counter for model evaluation
        self.interaction_count = 0
        
        # Initialize model performance tracking
        self.model_performance = {}
        
        logger.info(f"Agent initialized with current model: {self.current_model}")
    
    def process_query(self, query: str, system_prompt: Optional[str] = None, query_type: str = "text", session_id: Optional[str] = None) -> Tuple[str, str]:
        """
        Process a user query and return the response using the most appropriate model.
        
        Args:
            query: The user's query
            system_prompt: Optional system prompt describing the agent's purpose.
                           If None, the default_system_prompt will be used.
            query_type: Type of query (text, code, image)
            session_id: Optional session identifier for tracking conversation continuity
            
        Returns:
            Tuple of (response text, model used)
        """
        # Use default system prompt if none is provided
        if system_prompt is None:
            system_prompt = self.default_system_prompt
            logger.debug(f"Using default system prompt: {system_prompt[:50]}...")
            
        self.interaction_count += 1
        
        # Track conversation by session_id if provided
        conversation_history = []
        if session_id:
            # Initialize conversation entry if this is the first query
            if session_id not in self.active_conversations:
                self.active_conversations[session_id] = {
                    "messages": [],
                    "last_updated": datetime.now(),
                    "query_count": 0
                }
            
            # Update the conversation
            self.active_conversations[session_id]["last_updated"] = datetime.now()
            self.active_conversations[session_id]["query_count"] += 1
            conversation_history = self.active_conversations[session_id]["messages"]
        
        # Get the most relevant memories for this query
        relevant_memories = self.memory_manager.get_relevant_memories(query, limit=5)
        
        # Create context from relevant memories
        context = self._create_context_from_memories(relevant_memories)
        
        # Include context from active conversation if available
        if conversation_history:
            logger.debug(f"Including {len(conversation_history)} messages from ongoing conversation")
            
        # Construct messages with context and conversation history
        messages = self._construct_prompt(query, system_prompt, context, conversation_history)
        
        # If it's time to evaluate models, try a different one
        if self.interaction_count % config.MODEL_EVALUATION_INTERVAL == 0:
            model_to_use = self._select_model_for_evaluation()
            logger.info(f"Evaluating model: {model_to_use}")
        else:
            # Convert query_type to task_type for cost monitoring
            task_type = "general"
            if query_type == "code":
                task_type = "code"
            elif query_type == "image":
                task_type = "image"
                
            # Try to get the best model from cost monitor based on query type
            try:
                model_id, provider = self.cost_monitor.select_model_for_task(
                    task_type=task_type, 
                    content_type=query_type
                )
                if model_id and provider == "venice":
                    model_to_use = model_id
                    logger.info(f"Selected model for {query_type}/{task_type}: {model_to_use}")
                else:
                    # Fall back to standard model selection if not a Venice model
                    model_to_use = self._get_best_model()
                    logger.info(f"Using best model: {model_to_use}")
            except Exception as e:
                logger.error(f"Error selecting model by task: {str(e)}")
                model_to_use = self._get_best_model()
                logger.info(f"Fallback to best model: {model_to_use}")
        
        # Add helper method to determine provider for model_id
        def get_provider_for_model(model_id):
            # Check for Anthropic models
            if model_id.startswith("claude-"):
                return "anthropic"
                
            # Check for Perplexity models
            if model_id.startswith("llama-3.1-sonar-"):
                return "perplexity"
                
            # Default to Venice for all other models
            return "venice"
            
        # Track provider availability
        self._update_provider_status()
        
        # Determine which provider to use based on the model
        provider = get_provider_for_model(model_to_use)
        
        # Get a list of fallback models ordered by performance
        fallback_models = self._get_fallback_models(provider, model_to_use, query_type)
        logger.debug(f"Fallback models prepared: {', '.join(fallback_models) if fallback_models else 'None'}")
        
        # Check if selected provider is available, otherwise prepare for fallback
        if (provider == "anthropic" and not self._provider_status.get("anthropic", False)) or \
           (provider == "perplexity" and not self._provider_status.get("perplexity", False)):
            logger.warning(f"Provider {provider} is unavailable, will try fallback models")
            
            # If we have fallback models available, use the first one
            if fallback_models:
                model_to_use = fallback_models[0]
                provider = get_provider_for_model(model_to_use)
                logger.info(f"Using fallback model {model_to_use} from provider {provider}")
            else:
                # If no fallback models, default to Venice
                logger.warning("No suitable fallback models found, defaulting to Venice")
                provider = "venice"
                model_to_use = self.current_model
        
        # Determine if this is a query that requires high accuracy
        high_accuracy_required = self._requires_high_accuracy(query, system_prompt, query_type)
        
        # Call the appropriate model based on provider
        start_time = time.time()
        # Initialize response in case all attempts fail
        response = "I apologize, but I'm unable to generate a response at this time."
        try:
            # For high accuracy queries, potentially use multiple providers in parallel
            if high_accuracy_required and self._has_multiple_available_providers():
                logger.info("High accuracy query detected - using multiple providers in parallel")
                
                # Collect responses from multiple providers in parallel
                response, model_to_use, provider = self._query_multiple_providers(messages, query_type)
                
                # Track that we used multi-provider mode
                self._last_multi_provider_query = {
                    "used": True,
                    "query_type": query_type,
                    "timestamp": time.time(),
                    "providers_used": ["venice", "perplexity"]  # Currently available providers
                }
                
                # Update statistics
                if query_type in self._multi_provider_stats:
                    self._multi_provider_stats[query_type]["total_uses"] += 1
                
                success = True
            else:
                # Initialize success flag and track remaining fallback models
                success = False
                remaining_fallbacks = fallback_models.copy() if fallback_models else []
                attempted_models = []
                
                # First attempt with the selected model
                while not success:
                    # Track this attempt
                    attempted_models.append(model_to_use)
                    current_provider = self._get_provider_for_model_id(model_to_use)
                    
                    try:
                        logger.info(f"Attempting to use model: {model_to_use} from provider: {current_provider}")
                        
                        if current_provider == "venice":
                            # Venice API
                            response_data = self.venice_client.generate(messages, model=model_to_use)
                            response = response_data
                            success = True
                            
                        elif current_provider == "anthropic" and hasattr(self, 'anthropic_client') and self.anthropic_client and self._provider_status.get("anthropic", False):
                            # Anthropic API
                            response_data = self.anthropic_client.generate(messages, model=model_to_use)
                            response = response_data.get('content', [])[0].get('text', "")
                            success = True
                            
                        elif current_provider == "perplexity" and hasattr(self, 'perplexity_client') and self.perplexity_client and self._provider_status.get("perplexity", False):
                            # Perplexity API
                            response_data = self.perplexity_client.generate(messages, model=model_to_use)
                            response = response_data.get('choices', [{}])[0].get('message', {}).get('content', "")
                            success = True
                            
                        else:
                            # Unsupported provider or unavailable
                            logger.warning(f"Provider {current_provider} for model {model_to_use} is not available")
                            raise ValueError(f"Provider {current_provider} is unavailable")
                            
                    except Exception as e:
                        # Log the failure
                        logger.error(f"Error using model {model_to_use} from {current_provider}: {str(e)}")
                        
                        # Mark provider as unavailable if needed
                        if current_provider in self._provider_status:
                            self._provider_status[current_provider] = False
                            logger.info(f"Marked provider {current_provider} as unavailable")
                        
                        # Check if we have any fallback models left
                        if remaining_fallbacks:
                            # Get next fallback model that hasn't been attempted yet
                            next_models = [m for m in remaining_fallbacks if m not in attempted_models]
                            
                            if next_models:
                                model_to_use = next_models[0]
                                remaining_fallbacks.remove(model_to_use)
                                provider = self._get_provider_for_model_id(model_to_use)
                                logger.info(f"Trying fallback model: {model_to_use} from provider: {provider}")
                            else:
                                # We've exhausted all fallbacks, use Venice as last resort
                                venice_default = "mistral-31-24b"  # Venice default model
                                logger.warning(f"All fallback models failed, using Venice default model {venice_default} as last resort")
                                model_to_use = venice_default
                                provider = "venice"
                                
                                try:
                                    # Last attempt with Venice default model
                                    response = self.venice_client.generate(messages, model=model_to_use)
                                    success = True
                                except Exception as final_error:
                                    # Even Venice failed - this is a critical error
                                    logger.critical(f"Critical: All models including Venice default model failed: {final_error}")
                                    response = "I apologize, but I'm currently experiencing technical difficulties. Please try again later."
                                    success = False
                                    break
                        else:
                            # No fallbacks configured, try Venice directly
                            venice_default = "mistral-31-24b"  # Venice default model
                            logger.warning(f"No fallback models available, using Venice default model {venice_default}")
                            model_to_use = venice_default
                            provider = "venice"
                            
                            try:
                                # Last attempt with Venice default model
                                response = self.venice_client.generate(messages, model=model_to_use)
                                success = True
                            except Exception as final_error:
                                # Even Venice failed - this is a critical error
                                logger.critical(f"Critical: Venice default model failed with no fallbacks: {final_error}")
                                response = "I apologize, but I'm currently experiencing technical difficulties. Please try again later."
                                success = False
                                break
        except Exception as e:
            logger.error(f"Error generating response with {provider} model {model_to_use}: {str(e)}")
            # Fallback to default model if available
            if model_to_use != self.current_model:
                try:
                    logger.info(f"Falling back to current model: {self.current_model}")
                    response = self.venice_client.generate(messages, model=self.current_model)
                    model_to_use = self.current_model
                    provider = "venice"
                    success = True
                except Exception as e2:
                    logger.error(f"Error with fallback model: {str(e2)}")
                    response = f"I'm sorry, I encountered an error: {str(e2)}"
                    success = False
            else:
                response = f"I'm sorry, I encountered an error: {str(e)}"
                success = False
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Ensure we have a response defined in all cases
        if 'response' not in locals() or response is None:
            response = "I apologize, but I'm having technical difficulties. Please try again later."
            success = False
            logger.critical("Critical: No response was generated through any model")
        
        # Update model performance metrics
        self._update_model_performance(model_to_use, success, latency)
        
        # Store interaction in memory
        self.memory_manager.store_interaction(query, response, system_prompt)
        
        # Update conversation history for session continuity if session_id provided
        if session_id and session_id in self.active_conversations:
            # Add query and response to the conversation history for future context
            self.active_conversations[session_id]["messages"].append({
                "role": "user",
                "content": query
            })
            self.active_conversations[session_id]["messages"].append({
                "role": "assistant",
                "content": response
            })
            logger.debug(f"Updated conversation history for session {session_id}, now has {len(self.active_conversations[session_id]['messages'])} messages")
        
        # Evaluate response quality
        if success:
            quality_score = evaluate_model_response(query, response)
            self._update_model_quality(model_to_use, quality_score)
        
        return response, model_to_use
    
    def _construct_prompt(self, query: str, system_prompt: str, context: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> list:
        """
        Construct the messages for the Venice.ai Chat API
        
        Args:
            query: The user's query
            system_prompt: System instructions for the agent
            context: Context from memory
            conversation_history: Previous messages in the current conversation session
            
        Returns:
            List of message objects for the Venice.ai Chat API
        """
        messages = []
        
        # System message
        system_content = system_prompt
        if context:
            system_content += f"\n\nContext from previous interactions:\n{context}"
        
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Add previous conversation messages if available
        if conversation_history and len(conversation_history) > 0:
            messages.extend(conversation_history)
        
        # User message
        messages.append({
            "role": "user",
            "content": query
        })
        
        logger.debug(f"Constructed messages: {len(messages)} total messages")
        return messages
    
    def _create_context_from_memories(self, memories: List[Dict[str, Any]]) -> str:
        """
        Create a context string from retrieved memories
        
        Args:
            memories: List of memory objects retrieved from vector store
            
        Returns:
            Formatted context string
        """
        if not memories:
            return ""
        
        context_parts = []
        for i, memory in enumerate(memories):
            query = memory.get("query", "")
            response = memory.get("response", "")
            if query and response:
                context_parts.append(f"Previous exchange {i+1}:\nUser: {query}\nAssistant: {response}")
        
        return "\n\n".join(context_parts)
    
    def randomly_select_model_for_evaluation(self, selection_mode: str = "balanced", 
                                     filter_criteria: dict = None, 
                                     exclude_current: bool = True) -> str:
        """
        Utility for randomly selecting models with various strategies for testing/evaluation
        
        Args:
            selection_mode: Strategy for selection
                - 'uniform': Completely random selection with equal probability
                - 'weighted': Probability weighted by inverse of evaluation count
                - 'balanced': Balanced approach considering capabilities & provider distribution
                - 'least_evaluated': Prioritize least evaluated models
                - 'provider_balanced': Ensure even distribution across providers
            filter_criteria: Optional dictionary of criteria to filter models
                - 'provider': List of providers to include
                - 'capabilities': List of required capabilities
                - 'min_success_rate': Minimum success rate threshold
            exclude_current: Whether to exclude the current model
            
        Returns:
            Selected model ID
        """
        from models import ModelPerformance
        import random
        
        # Get all models with their evaluation counts from the database
        models_data = {}
        
        # Fetch all model data from database
        db_models = ModelPerformance.query.all()
        
        for model in db_models:
            models_data[model.model_id] = {
                'provider': model.provider,
                'capabilities': model.capabilities.split(',') if model.capabilities else ['text'],
                'total_calls': model.total_calls,
                'successful_calls': model.successful_calls,
                'quality_evaluations': model.quality_evaluations,
                'success_rate': model.success_rate,
                'is_available': model.is_available
            }
        
        # Add any available models that aren't yet in the database
        for model_id in self.available_models:
            if model_id not in models_data:
                # Get provider from internal mapping or default to venice
                provider = self._get_provider_for_model_id(model_id)
                models_data[model_id] = {
                    'provider': provider,
                    'capabilities': ['text'],  # Default capability
                    'total_calls': 0,
                    'successful_calls': 0,
                    'quality_evaluations': 0,
                    'success_rate': 0.0,
                    'is_available': True
                }
        
        # Start with all available models
        candidates = []
        for model_id, data in models_data.items():
            # Skip current model if flagged
            if exclude_current and model_id == self.current_model:
                continue
                
            # Skip unavailable models
            if not data['is_available']:
                continue
                
            # Apply filter criteria if provided
            if filter_criteria:
                # Filter by provider
                if 'provider' in filter_criteria and data['provider'] not in filter_criteria['provider']:
                    continue
                    
                # Filter by capabilities
                if 'capabilities' in filter_criteria:
                    if not all(cap in data['capabilities'] for cap in filter_criteria['capabilities']):
                        continue
                        
                # Filter by success rate
                if 'min_success_rate' in filter_criteria and data['success_rate'] < filter_criteria['min_success_rate']:
                    continue
            
            # Add qualifying model to candidates
            candidates.append(model_id)
        
        # If no candidates, return current model
        if not candidates:
            logger.warning("No models meet the selection criteria, using current model")
            return self.current_model
            
        # Apply selection strategy
        if selection_mode == "uniform":
            # Completely random selection with equal probability
            return random.choice(candidates)
            
        elif selection_mode == "weighted":
            # Weight by inverse of evaluation count (less evaluated = higher probability)
            weights = []
            for model_id in candidates:
                evals = models_data[model_id]['quality_evaluations']
                # Add small constant to prevent division by zero
                weight = 1.0 / (evals + 0.1)
                weights.append(weight)
                
            # Normalize weights
            total = sum(weights)
            norm_weights = [w/total for w in weights]
            
            return random.choices(candidates, weights=norm_weights, k=1)[0]
            
        elif selection_mode == "least_evaluated":
            # Sort by evaluation count (ascending) and pick from the least evaluated 25%
            sorted_candidates = sorted(candidates, 
                                    key=lambda x: models_data[x]['quality_evaluations'])
            
            # Take the 25% least evaluated models
            selection_size = max(1, len(sorted_candidates) // 4)
            least_evaluated = sorted_candidates[:selection_size]
            
            return random.choice(least_evaluated)
            
        elif selection_mode == "provider_balanced":
            # Group by provider
            provider_groups = {}
            for model_id in candidates:
                provider = models_data[model_id]['provider']
                if provider not in provider_groups:
                    provider_groups[provider] = []
                provider_groups[provider].append(model_id)
                
            # Select a provider randomly
            providers = list(provider_groups.keys())
            if not providers:
                return random.choice(candidates)  # Fallback
                
            selected_provider = random.choice(providers)
            
            # Select a model from that provider
            return random.choice(provider_groups[selected_provider])
            
        else:  # "balanced" mode (default)
            # Balanced approach with some randomness
            # 1. First 30% chance: completely random
            # 2. 40% chance: weighted by inverse eval count
            # 3. 30% chance: provider balanced
            
            choice = random.random()
            
            if choice < 0.3:
                # 30% random
                return random.choice(candidates)
            elif choice < 0.7:
                # 40% weighted by inverse eval count
                weights = []
                for model_id in candidates:
                    evals = models_data[model_id]['quality_evaluations']
                    weight = 1.0 / (evals + 0.1)
                    weights.append(weight)
                    
                total = sum(weights)
                norm_weights = [w/total for w in weights]
                
                return random.choices(candidates, weights=norm_weights, k=1)[0]
            else:
                # 30% provider balanced
                provider_groups = {}
                for model_id in candidates:
                    provider = models_data[model_id]['provider']
                    if provider not in provider_groups:
                        provider_groups[provider] = []
                    provider_groups[provider].append(model_id)
                    
                providers = list(provider_groups.keys())
                if not providers:
                    return random.choice(candidates)  # Fallback
                    
                selected_provider = random.choice(providers)
                return random.choice(provider_groups[selected_provider])

    def _select_model_for_evaluation(self) -> str:
        """
        Select a model for evaluation using an adaptive approach:
        1. Round-robin for unevaluated models (priority)
        2. Gradually back off to criteria-based selection as models get evaluated
        
        This ensures all models get evaluated fairly at the beginning,
        but transitions to a smarter selection strategy over time.
        
        Returns:
            Model ID to evaluate
        """
        from models import ModelPerformance
        from app import db
        import random
        import math
        
        # Get all models with their evaluation counts from the database
        model_evaluations = {}
        
        # Check the database for model performance records
        models = ModelPerformance.query.all()
        for model in models:
            model_evaluations[model.model_id] = {
                'total_calls': model.total_calls,
                'quality_evaluations': model.quality_evaluations if hasattr(model, 'quality_evaluations') else 0
            }
        
        # Find unevaluated models (these get priority)
        unevaluated_models = []
        for model_name in self.available_models:
            if model_name not in model_evaluations:
                unevaluated_models.append(model_name)
            elif model_evaluations[model_name]['quality_evaluations'] == 0:
                unevaluated_models.append(model_name)
        
        # If there are unevaluated models, use round-robin on them
        if unevaluated_models:
            logger.info(f"Round-robin mode: selecting unevaluated model from {unevaluated_models}")
            candidates = [m for m in unevaluated_models if m != self.current_model]
            if candidates:
                return candidates[0]  # Return first unevaluated model
        
        # If enough models have been evaluated, use our new random selection utility
        EVAL_THRESHOLD = 5  # If average evals > 5, transition to new utility
        
        # Calculate average evaluation count 
        eval_counts = [
            model_evaluations.get(model_id, {}).get('quality_evaluations', 0) 
            for model_id in self.available_models
        ]
        avg_evals = sum(eval_counts) / len(eval_counts) if eval_counts else 0
        
        # If we've done enough evaluations, use the new utility with a balanced approach
        if avg_evals >= EVAL_THRESHOLD:
            logger.info("Using balanced random selection utility")
            return self.randomly_select_model_for_evaluation(
                selection_mode="balanced", 
                exclude_current=True
            )
        
        # All models have some evaluation - use a weighted probability approach
        # that gradually transitions from round-robin to criteria-based selection
        
        # Calculate the minimum evaluation count needed for full criteria-based selection
        # (We'll use 10 evaluations as a threshold for considering a model fully evaluated)
        FULL_EVALUATION_THRESHOLD = 10
        
        # If average evaluations are still low, bias towards round-robin
        if avg_evals < FULL_EVALUATION_THRESHOLD:
            # Sort models by evaluation count (ascending)
            sorted_models = sorted(
                [(m, model_evaluations.get(m, {}).get('quality_evaluations', 0)) 
                 for m in self.available_models if m != self.current_model],
                key=lambda x: x[1]
            )
            
            if sorted_models:
                # Calculate probability weights with strong bias for less evaluated models
                weights = [max(0.1, 1.0 - (eval_count / FULL_EVALUATION_THRESHOLD)) 
                          for _, eval_count in sorted_models]
                
                # Normalize weights
                total_weight = sum(weights)
                norm_weights = [w / total_weight for w in weights] if total_weight > 0 else None
                
                # Select model using weighted probability
                if norm_weights:
                    model_ids = [model_id for model_id, _ in sorted_models]
                    selected_model = random.choices(model_ids, weights=norm_weights, k=1)[0]
                    logger.info(f"Transition mode: selected {selected_model} using weighted probability")
                    return selected_model
        
        # If we reached here, either all models have good evaluation counts
        # or something went wrong - fall back to best model selection
        logger.info("Criteria-based mode: selecting best performing model")
        
        # Fall back to criteria-based selection (best performing model)
        best_model = self._get_best_performance_model()
        if best_model != self.current_model:
            return best_model
        
        # If best model is current model, randomly select a different one
        candidates = [m for m in self.available_models if m != self.current_model]
        if candidates:
            return random.choice(candidates)
        
        # If all else fails, return the current model
        logger.warning("No alternative models available for evaluation")
        return self.current_model
        
    def _update_provider_status(self) -> None:
        """
        Check the status of different providers and update availability
        
        This method tests each provider to determine if it's currently
        available or if it's hitting rate limits or returning errors.
        """
        # Venice is always our primary provider and considered available
        self._provider_status["venice"] = True
        
        # Check Anthropic availability if we have a client
        if hasattr(self, 'anthropic_client') and self.anthropic_client:
            try:
                # Test connection to see if Anthropic API is working
                if hasattr(self.anthropic_client, 'test_connection'):
                    self._provider_status["anthropic"] = self.anthropic_client.test_connection()
                else:
                    # If no test_connection method, use a simple test message
                    test_message = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello"}
                    ]
                    try:
                        self.anthropic_client.generate(test_message, max_tokens=10)
                        self._provider_status["anthropic"] = True
                    except Exception as e:
                        logger.warning(f"Anthropic API test failed: {str(e)}")
                        self._provider_status["anthropic"] = False
            except Exception as e:
                self._provider_status["anthropic"] = False
                logger.warning(f"Anthropic API unavailable: {str(e)}")
        else:
            self._provider_status["anthropic"] = False
            
        # Check Perplexity availability if we have a client
        if hasattr(self, 'perplexity_client') and self.perplexity_client:
            try:
                # Test connection to see if Perplexity API is working
                if hasattr(self.perplexity_client, 'test_connection'):
                    self._provider_status["perplexity"] = self.perplexity_client.test_connection()
                else:
                    # If no test_connection method, use a simple test message
                    test_message = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello"}
                    ]
                    try:
                        self.perplexity_client.generate(test_message, max_tokens=10)
                        self._provider_status["perplexity"] = True
                    except Exception as e:
                        logger.warning(f"Perplexity API test failed: {str(e)}")
                        self._provider_status["perplexity"] = False
            except Exception as e:
                self._provider_status["perplexity"] = False
                logger.warning(f"Perplexity API unavailable: {str(e)}")
        else:
            self._provider_status["perplexity"] = False
            
        logger.info(f"Provider status: {self._provider_status}")

    def _calculate_confidence_score(self, query: str, system_prompt: str, query_type: str) -> float:
        """
        Calculate a confidence score that determines if a query benefits from using
        multiple providers for higher accuracy
        
        This uses a sophisticated scoring system that evaluates query attributes
        and adapts to different query types.
        
        Args:
            query: The user's query
            system_prompt: System instructions for the agent
            query_type: Type of query (text, code, image)
            
        Returns:
            Confidence score between 0.0 and 1.0 where higher values suggest
            greater benefit from using multiple providers
        """
        import re
        
        # Base factors - common to all query types
        base_score = 0.0
        
        # 1. Keyword analysis - check for specific indicators of complex or important requests
        accuracy_keywords = {
            # High importance keywords
            "critical": 0.15, "important": 0.12, "accurate": 0.15, "precise": 0.15, "exact": 0.15,
            # Complexity keywords
            "complex": 0.12, "difficult": 0.10, "analyze": 0.08, "compare": 0.08, 
            # Domain-specific keywords
            "scientific": 0.12, "research": 0.10, "academic": 0.10,
            "evaluate": 0.08, "math": 0.10, "calculation": 0.10,
            "financial": 0.12, "medical": 0.15, "legal": 0.15, 
            "technical": 0.10, "engineering": 0.10
        }
        
        # Evaluate keyword matches with partial matching and context awareness
        keyword_score = 0.0
        for keyword, weight in accuracy_keywords.items():
            # Check for direct match or word boundary match (avoiding partial word matches)
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, query.lower()):
                keyword_score += weight
        
        # Cap keyword score at 0.3 maximum contribution
        keyword_score = min(0.3, keyword_score)
        
        # 2. Query complexity analysis - evaluate structural complexity
        # Length-based complexity (0.0-0.2)
        length_score = min(0.2, len(query) / 1000)
        
        # Structural complexity - sentences, questions, and request structure (0.0-0.2)
        sentence_count = len(re.split(r'[.!?;]', query))
        question_count = len(re.findall(r'\?', query))
        structural_score = min(0.2, (sentence_count * 0.05) + (question_count * 0.03))
        
        # 3. Domain-specific factors based on query type
        domain_score = 0.0
        if query_type == "code":
            # Code queries benefit more from multiple providers when they involve:
            code_keywords = ["debug", "optimize", "refactor", "architecture", "design pattern"]
            code_matches = sum(1 for kw in code_keywords if kw.lower() in query.lower())
            domain_score = min(0.2, code_matches * 0.05)
            
            # Longer code requests also benefit from multiple providers
            code_length_factor = min(0.1, len(query) / 800)
            domain_score += code_length_factor
            
        elif query_type == "image":
            # Image generation generally less suited for multiple providers
            domain_score = 0.05
            
        else:  # text queries
            # Text queries with specific attributes may benefit more:
            creative_pattern = r'\b(creative|story|write|narrative|poem|fiction)\b'
            factual_pattern = r'\b(fact|statistic|historical|science|explain|how does|why does)\b'
            
            if re.search(creative_pattern, query.lower()):
                # Creative queries benefit from diverse models
                domain_score += 0.15
            if re.search(factual_pattern, query.lower()):
                # Factual queries benefit from multiple providers for verification
                domain_score += 0.20
                
        # 4. System prompt analysis
        system_score = 0.0
        if system_prompt:
            # Complex system prompts suggest more specialized tasks
            system_score = min(0.15, len(system_prompt) / 1200)
            
            # Check for specific requirements in system prompt
            accuracy_pattern = r'\b(accurate|precise|exact|correct|verified|factual)\b'
            if re.search(accuracy_pattern, system_prompt.lower()):
                system_score += 0.1
        
        # 5. Historical performance-based scoring (basic implementation)
        history_score = 0.05  # Small baseline score that would be adjusted based on historical data
        
        # Combine all factors with appropriate weights
        base_score = (
            (keyword_score * 0.25) +       # 25% keyword relevance
            (length_score * 0.10) +        # 10% length complexity
            (structural_score * 0.15) +    # 15% structural complexity
            (domain_score * 0.25) +        # 25% domain-specific factors
            (system_score * 0.15) +        # 15% system prompt analysis 
            (history_score * 0.10)         # 10% historical performance
        )
        
        # Log detailed scoring for debugging and improvement
        logger.debug(f"Confidence scoring: keyword={keyword_score:.2f}, length={length_score:.2f}, "
                   f"structure={structural_score:.2f}, domain={domain_score:.2f}, "
                   f"system={system_score:.2f}, history={history_score:.2f}, total={base_score:.2f}")
        
        return base_score
    
    def _requires_high_accuracy(self, query: str, system_prompt: str, query_type: str) -> bool:
        """
        Determine if a query requires high accuracy and should use multiple providers
        
        The agent learns over time which types of queries benefit from multiple providers.
        This is a sophisticated evaluation based on confidence scoring, budget constraints,
        and provider availability.
        
        Args:
            query: The user's query
            system_prompt: System instructions for the agent
            query_type: Type of query (text, code, image)
            
        Returns:
            Whether high accuracy mode should be used (multiple providers)
        """
        # Avoid using high accuracy mode if we don't have multiple available providers
        if not self._has_multiple_available_providers():
            logger.info("Not enough available providers for high accuracy mode")
            return False
            
        # Only enable high accuracy mode if budget allows
        if hasattr(self, 'cost_monitor') and self.cost_monitor:
            try:
                if not self.cost_monitor.can_use_high_accuracy_mode():
                    logger.info("High accuracy mode not available due to budget constraints")
                    return False
            except Exception as e:
                logger.error(f"Error checking high accuracy budget: {str(e)}")
                return False
        
        # Calculate confidence score to determine if multiple providers would be beneficial
        confidence_score = self._calculate_confidence_score(query, system_prompt, query_type)
        
        # Adapt threshold based on query type, costs, and other factors
        base_threshold = 0.45  # Default threshold
        
        # Adjust threshold based on query type
        if query_type == "code":
            threshold = base_threshold + 0.05  # Higher threshold for code
        elif query_type == "image":
            threshold = base_threshold + 0.15  # Much higher threshold for image
        else:
            threshold = base_threshold
        
        # Log decision for monitoring
        use_high_accuracy = confidence_score >= threshold
        
        logger.info(f"High accuracy decision: score={confidence_score:.2f}, "
                   f"threshold={threshold:.2f}, result={'YES' if use_high_accuracy else 'NO'}")
        
        return use_high_accuracy
        
    def _has_multiple_available_providers(self) -> bool:
        """
        Check if multiple providers are available for query processing
        
        Returns:
            Whether multiple providers are available
        """
        available_count = 0
        
        # Venice is always considered available
        available_count += 1
        
        # Check Anthropic availability
        if hasattr(self, 'anthropic_client') and self.anthropic_client and self._provider_status.get("anthropic", False):
            available_count += 1
            
        # Check Perplexity availability
        if hasattr(self, 'perplexity_client') and self.perplexity_client and self._provider_status.get("perplexity", False):
            available_count += 1
            
        return available_count >= 2
        
    def _query_multiple_providers(self, messages: list, query_type: str) -> Tuple[str, str, str]:
        """
        Query multiple providers in parallel and combine the results for high accuracy
        
        Args:
            messages: List of message objects for providers
            query_type: Type of query (text, code, image)
            
        Returns:
            Tuple of (response, model_used, provider_used)
        """
        import concurrent.futures
        import time
        
        responses = []
        start_time = time.time()
        
        # Define provider query functions
        def query_venice():
            try:
                venice_response = self.venice_client.generate(messages, model=self.current_model)
                return {
                    "provider": "venice",
                    "model": self.current_model,
                    "response": venice_response,
                    "success": True,
                    "latency": time.time() - start_time
                }
            except Exception as e:
                logger.error(f"Error with Venice API: {str(e)}")
                return {
                    "provider": "venice",
                    "model": self.current_model,
                    "response": "",
                    "success": False,
                    "error": str(e),
                    "latency": time.time() - start_time
                }
        
        def query_perplexity():
            if not (hasattr(self, 'perplexity_client') and self.perplexity_client and self._provider_status.get("perplexity", False)):
                return None
                
            try:
                perplexity_model = "llama-3.1-sonar-small-128k-online"  # Use consistent model
                response_data = self.perplexity_client.generate(messages, model=perplexity_model)
                content = response_data.get('choices', [{}])[0].get('message', {}).get('content', "")
                return {
                    "provider": "perplexity",
                    "model": perplexity_model,
                    "response": content,
                    "success": True,
                    "latency": time.time() - start_time
                }
            except Exception as e:
                logger.error(f"Error with Perplexity API in multi-provider mode: {str(e)}")
                return None
        
        def query_anthropic():
            if not (hasattr(self, 'anthropic_client') and self.anthropic_client and self._provider_status.get("anthropic", False)):
                return None
                
            try:
                # Use the most recent model from our dynamic discovery
                from models import ModelPerformance
                anthropic_models = ModelPerformance.query.filter_by(provider="anthropic").all()
                anthropic_model = "claude-3-7-sonnet-20241022"  # Default to latest model
                
                # If we have discovered models, use the first available
                if anthropic_models:
                    anthropic_model = anthropic_models[0].model_id
                
                response_data = self.anthropic_client.generate(messages, model=anthropic_model)
                content = ""
                if 'content' in response_data and len(response_data['content']) > 0:
                    if isinstance(response_data['content'], list):
                        content = response_data['content'][0].get('text', "")
                    elif isinstance(response_data['content'], str):
                        content = response_data['content']
                        
                return {
                    "provider": "anthropic",
                    "model": anthropic_model, 
                    "response": content,
                    "success": True,
                    "latency": time.time() - start_time
                }
            except Exception as e:
                logger.error(f"Error with Anthropic API in multi-provider mode: {str(e)}")
                return None
        
        # Execute provider queries in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Start all provider queries in parallel
            future_to_provider = {
                executor.submit(query_venice): "venice",
                executor.submit(query_perplexity): "perplexity",
                executor.submit(query_anthropic): "anthropic"
            }
            
            # Set a timeout for all queries (30 seconds)
            timeout = 30
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_provider, timeout=timeout):
                provider_name = future_to_provider[future]
                try:
                    result = future.result()
                    if result:
                        responses.append(result)
                        logger.info(f"Received response from {provider_name} in {result.get('latency', 0):.2f} seconds")
                except Exception as exc:
                    logger.error(f"{provider_name} generated an exception: {exc}")
        
        logger.info(f"Parallel query completed in {time.time() - start_time:.2f} seconds with {len(responses)} responses")
        
        # If we have multiple responses, combine them or select the best
        if len(responses) > 1:
            logger.info(f"Received {len(responses)} responses in parallel multi-provider mode")
            
            # First, sort responses by success and then by latency
            successful_responses = [r for r in responses if r.get("success", False)]
            
            if successful_responses:
                # Sort successful responses by latency (faster responses first)
                sorted_responses = sorted(successful_responses, key=lambda x: x.get("latency", float('inf')))
                
                # Priority to Venice (our primary provider) if successful
                primary_response = next((r for r in sorted_responses if r["provider"] == "venice"), sorted_responses[0])
                
                # Track that we used multiple providers for this query
                self._track_multi_provider_usage(query_type, len(responses))
                
                # Return primary response
                return primary_response["response"], primary_response["model"], primary_response["provider"]
            
        # If only one response or no responses, use it or fall back to default
        if len(responses) == 1:
            return responses[0]["response"], responses[0]["model"], responses[0]["provider"]
        else:
            # Fallback in case all providers failed
            try:
                response = self.venice_client.generate(messages, model=self.current_model)
                return response, self.current_model, "venice"
            except Exception as e:
                logger.error(f"All providers failed, and fallback to Venice also failed: {e}")
                # Last resort fallback message
                return "I apologize, but I'm currently experiencing technical difficulties. Please try again in a moment.", self.current_model, "fallback"
            
    def _track_multi_provider_usage(self, query_type: str, provider_count: int) -> None:
        """
        Track usage of multiple providers for analytics and optimization
        
        Args:
            query_type: Type of query (text, code, image)
            provider_count: Number of providers used
        """
        logger.info(f"Multi-provider mode: used {provider_count} providers for {query_type} query")
        
        # In a more sophisticated implementation, we would store this in the database
        # to track which query types benefit most from multiple providers
        
    def _get_best_performance_model(self) -> str:
        """
        Get the best performing model based on quality and success rate
        
        This is used by the adaptive model evaluation system
        when we have enough evaluations.
        
        Returns:
            Best model ID based on performance
        """
        from models import ModelPerformance
        
        best_score = -1
        best_model = self.current_model
        
        # Query all models from database
        models = ModelPerformance.query.all()
        
        for model in models:
            # Skip models that aren't in our available list
            if model.model_id not in self.available_models:
                continue
                
            # Calculate weighted score based on quality score and success rate
            quality_weight = 0.7
            success_weight = 0.3
            
            quality_score = model.average_quality() if hasattr(model, 'average_quality') else 0
            success_rate = model.success_rate() if hasattr(model, 'success_rate') else 0
            
            weighted_score = (quality_weight * quality_score) + (success_weight * success_rate)
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_model = model.model_id
                
        return best_model
    
    def _get_best_model(self) -> str:
        """
        Get the best performing model based on success rate and latency
        
        Returns:
            Best model ID
        """
        from models import ModelPerformance
        
        best_score = -1
        best_model = self.current_model
        
        # Get model records from database
        model_records = ModelPerformance.query.all()
        
        # Convert to dictionary for easier access and calculations
        model_data = {}
        max_speed_value = 0
        
        for record in model_records:
            # Skip models that aren't in our available list
            if record.model_id not in self.available_models:
                continue
                
            if record.total_calls < 5:  # Need minimum calls for reliability
                continue
                
            success_rate = record.success_rate()
            avg_latency = record.average_latency()
            
            # Store for later use
            model_data[record.model_id] = {
                "success_rate": success_rate,
                "average_latency": avg_latency
            }
            
            # Track maximum speed for normalization
            if avg_latency > 0:
                speed = 1.0 / avg_latency
                max_speed_value = max(max_speed_value, speed)
        
        # Calculate scores for each model
        for model_id, data in model_data.items():
            success_rate = data["success_rate"] if data["success_rate"] > 0 else 0
            
            # Convert latency to speed score (lower latency is better)
            avg_latency = data["average_latency"] if data["average_latency"] > 0 else 1
            speed_score = 1.0 / avg_latency
            
            # Normalize speed score to 0-1 range
            if max_speed_value > 0:
                speed_score = speed_score / max_speed_value
            
            # Combined score
            score = (0.7 * success_rate) + (0.3 * speed_score)
            
            if score > best_score:
                best_score = score
                best_model = model_id
        
        return best_model
    
    def _update_model_performance(self, model: str, success: bool, latency: float) -> None:
        """
        Update the performance metrics for a specific model
        
        Args:
            model: Model ID
            success: Whether the model call succeeded
            latency: Response time in seconds
        """
        from models import ModelPerformance
        from main import db
        
        # Get model record from database
        model_record = ModelPerformance.query.filter_by(model_id=model).first()
        
        # If model doesn't exist, create it
        if not model_record:
            model_record = ModelPerformance()
            model_record.model_id = model
            model_record.provider = "venice"  # Default provider
            model_record.total_calls = 0
            model_record.successful_calls = 0
            model_record.total_latency = 0.0
            model_record.quality_score = 0.0
            model_record.quality_evaluations = 0
            model_record.is_current = (model == self.current_model)
            model_record.capabilities = "text"  # Default capability
            model_record.context_window = 8192  # Default context window
            model_record.display_name = model   # Default display name
            db.session.add(model_record)
        
        # Update metrics
        model_record.total_calls += 1
        
        if success:
            model_record.successful_calls += 1
        
        model_record.total_latency += latency
        model_record.updated_at = datetime.utcnow()
        
        # Save changes
        db.session.commit()
        
        # Check if this model is performing better than current model
        if success and model != self.current_model:
            current_model_record = ModelPerformance.query.filter_by(model_id=self.current_model).first()
            
            if (current_model_record and 
                model_record.success_rate > current_model_record.success_rate and
                model_record.total_calls >= 5):
                
                # Update current model flag
                logger.info(f"Switching default model from {self.current_model} to {model} based on performance")
                
                # Update database records
                current_model_record.is_current = False
                model_record.is_current = True
                db.session.commit()
                
                # Update local reference
                self.current_model = model
    
    def _register_provider_models(self, provider: str) -> None:
        """
        Register models from a specific provider dynamically
        
        Args:
            provider: The provider name (e.g., "anthropic", "perplexity")
        """
        from models import ModelPerformance
        from main import db
        
        logger.info(f"Registering models for provider: {provider}")
        
        if provider == "anthropic" and self.anthropic_client:
            # Get models from Anthropic client, which might use Perplexity for discovery
            new_models = self.anthropic_client.get_available_models()
            logger.info(f"Discovered {len(new_models)} models from Anthropic")
            
            # Register each model in the database if it doesn't exist
            for model_data in new_models:
                model_id = model_data.get("id")
                if not model_id:
                    continue
                    
                # Check if model already exists
                existing_model = ModelPerformance.query.filter_by(model_id=model_id).first()
                if existing_model:
                    logger.debug(f"Model {model_id} already exists in database")
                    continue
                    
                # Create new model record
                model = ModelPerformance()
                model.model_id = model_id
                model.provider = provider
                model.total_calls = 0
                model.successful_calls = 0
                model.total_latency = 0.0
                model.quality_score = 0.0
                model.quality_evaluations = 0
                model.is_current = False  # Don't set as current by default
                model.capabilities = "text"  # Default capability
                
                # Set context window if available
                context_window = model_data.get("context_window") or model_data.get("context_length")
                model.context_window = context_window or 8192  # Default
                
                # Set display name if available
                model.display_name = model_data.get("name") or model_id
                
                # Add to database
                db.session.add(model)
                logger.info(f"Registered new model: {model_id} from provider {provider}")
                
            # Commit all changes
            db.session.commit()
                
        elif provider == "perplexity" and self.perplexity_client:
            # Get models from Perplexity client
            new_models = self.perplexity_client.get_available_models()
            logger.info(f"Discovered {len(new_models)} models from Perplexity")
            
            # Register each model in the database if it doesn't exist
            for model_data in new_models:
                model_id = model_data.get("id")
                if not model_id:
                    continue
                    
                # Check if model already exists
                existing_model = ModelPerformance.query.filter_by(model_id=model_id).first()
                if existing_model:
                    logger.debug(f"Model {model_id} already exists in database")
                    continue
                    
                # Create new model record
                model = ModelPerformance()
                model.model_id = model_id
                model.provider = provider
                model.total_calls = 0
                model.successful_calls = 0
                model.total_latency = 0.0
                model.quality_score = 0.0
                model.quality_evaluations = 0
                model.is_current = False  # Don't set as current by default
                model.capabilities = "text"  # Default capability
                
                # Set context window if available
                context_window = model_data.get("context_length")
                model.context_window = context_window or 8192  # Default
                
                # Set display name if available
                model.display_name = model_data.get("name") or model_id
                
                # Add to database
                db.session.add(model)
                logger.info(f"Registered new model: {model_id} from provider {provider}")
                
            # Commit all changes
            db.session.commit()
        else:
            logger.warning(f"Unable to register models for provider {provider}: client not available or provider not supported")
    
    def _update_model_quality(self, model: str, quality_score: float) -> None:
        """
        Update the quality metrics for a specific model
        
        Args:
            model: Model ID
            quality_score: Evaluation score (0-1)
        """
        from models import ModelPerformance
        from main import db
        
        # Get model record from database
        model_record = ModelPerformance.query.filter_by(model_id=model).first()
        
        if not model_record:
            return
            
        # Update quality metrics
        model_record.quality_score += quality_score
        model_record.quality_evaluations += 1
        model_record.updated_at = datetime.utcnow()
        
        # Save changes
        db.session.commit()
        
        # Log quality score
        logger.info(f"Model {model} quality score: {quality_score}, avg: {model_record.average_quality}")
    
    def _get_fallback_models(self, current_provider: str, current_model: str, query_type: str) -> List[str]:
        """
        Get a list of fallback models in case the current model fails
        
        Args:
            current_provider: The provider of the current model
            current_model: The current model that might fail
            query_type: Type of query (text, code, image)
            
        Returns:
            List of model IDs to try as fallbacks, in order of preference
        """
        from models import ModelPerformance
        
        fallback_models = []
        
        # We don't want to use the current model as a fallback
        # First, get models from different providers with good performance for this query type
        try:
            # Get available models with good success rate (>80%) ordered by success rate
            fallback_candidates = ModelPerformance.query.filter(
                ModelPerformance.model_id != current_model,
                ModelPerformance.provider != current_provider,
                ModelPerformance.is_available == True,
                ModelPerformance.capabilities.like(f"%{query_type}%"),
                ModelPerformance.total_calls >= 5,  # Only consider models with some usage history
                ModelPerformance.successful_calls * 100 / ModelPerformance.total_calls >= 80
            ).order_by(ModelPerformance.successful_calls * 100 / ModelPerformance.total_calls.desc()).all()
            
            # Add these high-performing models from other providers to our fallback list
            for model in fallback_candidates:
                # Check if this provider is currently available
                provider = self._get_provider_for_model_id(model.model_id)
                if self._provider_status.get(provider, False):
                    fallback_models.append(model.model_id)
            
            # If we don't have enough fallbacks yet, add Venice models as they're most reliable
            if len(fallback_models) < 2:
                venice_models = ModelPerformance.query.filter(
                    ModelPerformance.model_id != current_model,
                    ModelPerformance.provider == "venice",
                    ModelPerformance.is_available == True,
                    ModelPerformance.capabilities.like(f"%{query_type}%")
                ).order_by(ModelPerformance.successful_calls * 100 / ModelPerformance.total_calls.desc()).all()
                
                for model in venice_models:
                    if model.model_id not in fallback_models:
                        fallback_models.append(model.model_id)
            
            # If we still don't have fallbacks, add any available model as last resort
            if len(fallback_models) == 0:
                last_resort_models = ModelPerformance.query.filter(
                    ModelPerformance.model_id != current_model,
                    ModelPerformance.is_available == True,
                    ModelPerformance.capabilities.like(f"%{query_type}%")
                ).all()
                
                for model in last_resort_models:
                    provider = self._get_provider_for_model_id(model.model_id)
                    if self._provider_status.get(provider, False):
                        fallback_models.append(model.model_id)
                        
            # Always ensure the original Venice default model is included as final fallback
            venice_default = "mistral-31-24b"  # Default Venice model if all else fails
            if venice_default not in fallback_models and venice_default != current_model:
                fallback_models.append(venice_default)
                
            return fallback_models
                
        except Exception as e:
            logger.error(f"Error getting fallback models: {str(e)}")
            
            # In case of database error, return a hardcoded fallback list
            # This ensures the system remains operational even if DB queries fail
            hardcoded_fallbacks = []
            
            # Add fallbacks from each provider that isn't the current one
            if current_provider != "venice":
                hardcoded_fallbacks.append("mistral-31-24b")  # Venice fallback
            if current_provider != "anthropic" and self._provider_status.get("anthropic", False):
                hardcoded_fallbacks.append("claude-3-5-haiku-20241022")  # Anthropic fallback
            if current_provider != "perplexity" and self._provider_status.get("perplexity", False):
                hardcoded_fallbacks.append("llama-3.1-sonar-small-128k-online")  # Perplexity fallback
                
            return hardcoded_fallbacks
    
    def _get_provider_for_model_id(self, model_id: str) -> str:
        """
        Get the provider for a model ID
        
        Args:
            model_id: The model ID
            
        Returns:
            Provider string (venice, anthropic, perplexity, etc.)
        """
        from models import ModelPerformance
        
        # First check the database
        model_record = ModelPerformance.query.filter_by(model_id=model_id).first()
        if model_record:
            return model_record.provider
            
        # If not in database, try to determine from model ID pattern
        if model_id.startswith("claude-"):
            return "anthropic"
        elif model_id.startswith("llama-"):
            return "perplexity"
        else:
            # Default to Venice
            return "venice"
    
    def get_models_performance(self) -> Dict[str, Dict]:
        """
        Get performance metrics for all models
        
        Returns:
            Dictionary of model performance metrics
        """
        from models import ModelPerformance
        
        # Get all models from database
        model_records = ModelPerformance.query.all()
        
        # Format for the API response
        result = {}
        for record in model_records:
            result[record.model_id] = {
                "success_rate": record.success_rate,
                "average_latency": record.average_latency,
                "total_calls": record.total_calls,
                "successes": record.successful_calls,
                "failures": record.total_calls - record.successful_calls,
                "total_latency": record.total_latency,
                "average_quality": record.average_quality,
                "is_current": record.is_current,
                "last_used": record.updated_at.timestamp() if record.updated_at else None
            }
        
        return result
