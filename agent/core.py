import logging
import random
import time
from typing import Dict, List, Tuple, Optional, Any
from agent.memory import MemoryManager
from agent.models import VeniceClient
from agent.perplexity import PerplexityClient
from agent.anthropic_client import AnthropicClient
from agent.evaluation import evaluate_model_response
import config
from datetime import datetime
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
        model = ModelPerformance(
            model_id=model_id,
            total_calls=0,
            successful_calls=0,
            total_latency=0.0,
            quality_score=0.0,
            quality_evaluations=0,
            is_current=(model_id == "mistral-31-24b")  # Set the default model
        )
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
        
        # Try to initialize additional API clients if keys are available
        # Initialize Perplexity client first as Anthropic can use it for model discovery
        try:
            self.perplexity_client = PerplexityClient()
            if self.perplexity_client.api_key:
                logger.info("Perplexity API client initialized successfully")
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
            else:
                self.anthropic_client = None
                logger.warning("No Anthropic API key found, client not available")
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic client: {str(e)}")
            self.anthropic_client = None
        
        # Load or initialize models in database
        init_default_models()
        
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
    
    def process_query(self, query: str, system_prompt: Optional[str] = None, query_type: str = "text") -> Tuple[str, str]:
        """
        Process a user query and return the response using the most appropriate model.
        
        Args:
            query: The user's query
            system_prompt: Optional system prompt describing the agent's purpose.
                           If None, the default_system_prompt will be used.
            query_type: Type of query (text, code, image)
            
        Returns:
            Tuple of (response text, model used)
        """
        # Use default system prompt if none is provided
        if system_prompt is None:
            system_prompt = self.default_system_prompt
            logger.debug(f"Using default system prompt: {system_prompt[:50]}...")
            
        self.interaction_count += 1
        
        # Get relevant memories based on the query
        relevant_memories = self.memory_manager.get_relevant_memories(query, limit=5)
        
        # Create context from relevant memories
        context = self._create_context_from_memories(relevant_memories)
        
        # Construct messages with context
        messages = self._construct_prompt(query, system_prompt, context)
        
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
        
        # Check if selected provider is available, otherwise use Venice
        if (provider == "anthropic" and not self._provider_status.get("anthropic", False)) or \
           (provider == "perplexity" and not self._provider_status.get("perplexity", False)):
            logger.warning(f"Provider {provider} is unavailable, using Venice instead")
            provider = "venice"
            model_to_use = self.current_model
        
        # Determine if this is a query that requires high accuracy
        high_accuracy_required = self._requires_high_accuracy(query, system_prompt, query_type)
        
        # Call the appropriate model based on provider
        start_time = time.time()
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
            elif provider == "venice":
                # Always prioritize Venice API as it's the most reliable
                response_data = self.venice_client.generate(messages, model=model_to_use)
                response = response_data
                success = True
            elif provider == "anthropic" and hasattr(self, 'anthropic_client') and self.anthropic_client and self._provider_status.get("anthropic", False):
                try:
                    response_data = self.anthropic_client.generate(messages, model=model_to_use)
                    response = response_data.get('content', [])[0].get('text', "")
                    success = True
                except Exception as e:
                    # Mark Anthropic as unavailable and fall back to Venice
                    logger.error(f"Error with Anthropic API, falling back to Venice: {str(e)}")
                    self._provider_status["anthropic"] = False
                    response_data = self.venice_client.generate(messages, model=self.current_model)
                    response = response_data
                    model_to_use = self.current_model
                    provider = "venice"
                    success = True
            elif provider == "perplexity" and hasattr(self, 'perplexity_client') and self.perplexity_client and self._provider_status.get("perplexity", False):
                try:
                    response_data = self.perplexity_client.generate(messages, model=model_to_use)
                    response = response_data.get('choices', [{}])[0].get('message', {}).get('content', "")
                    success = True
                except Exception as e:
                    # Mark Perplexity as unavailable and fall back to Venice
                    logger.error(f"Error with Perplexity API, falling back to Venice: {str(e)}")
                    self._provider_status["perplexity"] = False
                    response_data = self.venice_client.generate(messages, model=self.current_model)
                    response = response_data
                    model_to_use = self.current_model
                    provider = "venice"
                    success = True
            else:
                # Default to Venice client as the safest option
                response = self.venice_client.generate(messages, model=self.current_model)
                model_to_use = self.current_model
                provider = "venice"
                success = True
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
        
        # Update model performance metrics
        self._update_model_performance(model_to_use, success, latency)
        
        # Store interaction in memory
        self.memory_manager.store_interaction(query, response, system_prompt)
        
        # Evaluate response quality
        if success:
            quality_score = evaluate_model_response(query, response)
            self._update_model_quality(model_to_use, quality_score)
        
        return response, model_to_use
    
    def _construct_prompt(self, query: str, system_prompt: str, context: str) -> list:
        """
        Construct the messages for the Venice.ai Chat API
        
        Args:
            query: The user's query
            system_prompt: System instructions for the agent
            context: Context from memory
            
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
        
        # User message
        messages.append({
            "role": "user",
            "content": query
        })
        
        logger.debug(f"Constructed messages: {messages}")
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
        
        # All models have some evaluation - use a weighted probability approach
        # that gradually transitions from round-robin to criteria-based selection
        
        # Calculate the minimum evaluation count needed for full criteria-based selection
        # (We'll use 10 evaluations as a threshold for considering a model fully evaluated)
        FULL_EVALUATION_THRESHOLD = 10
        
        # Calculate average evaluation count 
        eval_counts = [
            model_evaluations.get(model_id, {}).get('quality_evaluations', 0) 
            for model_id in self.available_models
        ]
        avg_evals = sum(eval_counts) / len(eval_counts) if eval_counts else 0
        
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

    def _requires_high_accuracy(self, query: str, system_prompt: str, query_type: str) -> bool:
        """
        Determine if a query requires high accuracy and should use multiple providers
        
        The agent learns over time which types of queries benefit from multiple providers.
        This is a heuristic approach based on keywords, query complexity, and past performance.
        
        Args:
            query: The user's query
            system_prompt: System instructions for the agent
            query_type: Type of query (text, code, image)
            
        Returns:
            Whether high accuracy mode should be used (multiple providers)
        """
        # Avoid using high accuracy mode if we don't have multiple available providers
        if not self._has_multiple_available_providers():
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
        
        # Keywords that suggest high accuracy is needed
        high_accuracy_keywords = [
            "important", "critical", "crucial", "exact", "precise", 
            "accuracy", "accurate", "factual", "verify", "validate",
            "double-check", "ensure", "proof", "technical", "math",
            "medical", "legal", "financial", "analysis", "compare",
            "research", "investigate", "detailed", "complex"
        ]
        
        # Check for high accuracy keywords in query
        for keyword in high_accuracy_keywords:
            if keyword.lower() in query.lower():
                logger.info(f"High accuracy mode triggered by keyword: {keyword}")
                return True
                
        # Check for high accuracy markers in system prompt
        system_accuracy_indicators = [
            "accuracy is critical", "high accuracy", "precise",
            "factual correctness", "verification", "critical system"
        ]
        
        for indicator in system_accuracy_indicators:
            if indicator.lower() in system_prompt.lower():
                logger.info(f"High accuracy mode triggered by system prompt indicator: {indicator}")
                return True
                
        # Check query length - longer queries often benefit from multiple providers
        if len(query.split()) > 50:  # Arbitrary threshold for longer queries
            # For long queries, use high accuracy mode with some probability
            # This lets the agent explore the value of using multiple providers
            if random.random() < 0.3:  # 30% chance for long queries
                logger.info("High accuracy mode triggered for long query")
                return True
                
        # Code queries may benefit from multiple providers
        if query_type == "code" and len(query.split()) > 30:
            if random.random() < 0.4:  # 40% chance for code queries
                logger.info("High accuracy mode triggered for code query")
                return True
                
        # Default to not using high accuracy mode to reduce cost
        return False
        
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
        Query multiple providers and combine the results for high accuracy
        
        Args:
            messages: List of message objects for providers
            query_type: Type of query (text, code, image)
            
        Returns:
            Tuple of (response, model_used, provider_used)
        """
        responses = []
        
        # Always include Venice as primary provider
        try:
            venice_response = self.venice_client.generate(messages, model=self.current_model)
            responses.append({
                "provider": "venice",
                "model": self.current_model,
                "response": venice_response,
                "success": True
            })
        except Exception as e:
            logger.error(f"Error with Venice API: {str(e)}")
        
        # Try Perplexity if available
        if hasattr(self, 'perplexity_client') and self.perplexity_client and self._provider_status.get("perplexity", False):
            try:
                perplexity_model = "llama-3.1-sonar-small-128k-online"  # Use consistent model
                response_data = self.perplexity_client.generate(messages, model=perplexity_model)
                content = response_data.get('choices', [{}])[0].get('message', {}).get('content', "")
                responses.append({
                    "provider": "perplexity",
                    "model": perplexity_model,
                    "response": content,
                    "success": True
                })
            except Exception as e:
                logger.error(f"Error with Perplexity API in multi-provider mode: {str(e)}")
        
        # Try Anthropic if available
        if hasattr(self, 'anthropic_client') and self.anthropic_client and self._provider_status.get("anthropic", False):
            try:
                anthropic_model = "claude-3-sonnet-20240229"  # Use consistent model
                response_data = self.anthropic_client.generate(messages, model=anthropic_model)
                content = ""
                if 'content' in response_data and len(response_data['content']) > 0:
                    content = response_data['content'][0].get('text', "")
                responses.append({
                    "provider": "anthropic",
                    "model": anthropic_model, 
                    "response": content,
                    "success": True
                })
            except Exception as e:
                logger.error(f"Error with Anthropic API in multi-provider mode: {str(e)}")
                
        # If we have multiple responses, combine them or select the best
        if len(responses) > 1:
            logger.info(f"Received {len(responses)} responses in multi-provider mode")
            
            # For now, use Venice as primary and only use other providers to enhance it
            # In a more sophisticated implementation, we could compare and merge them
            
            primary_response = next((r for r in responses if r["provider"] == "venice"), responses[0])
            
            # Track that we used multiple providers for this query
            self._track_multi_provider_usage(query_type, len(responses))
            
            # Return primary response (typically Venice)
            return primary_response["response"], primary_response["model"], primary_response["provider"]
            
        # If only one response or no responses, use it or fall back to default
        if len(responses) == 1:
            return responses[0]["response"], responses[0]["model"], responses[0]["provider"]
        else:
            # Fallback in case all providers failed
            response = self.venice_client.generate(messages, model=self.current_model)
            return response, self.current_model, "venice"
            
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
        best_score = -1
        best_model = self.current_model
        
        for model, perf in self.model_performance.items():
            if perf["total_calls"] < 5:  # Need minimum calls for reliability
                continue
                
            # Calculate a combined score (70% success rate, 30% speed)
            success_rate = perf["success_rate"] if perf["success_rate"] > 0 else 0
            
            # Convert latency to speed score (lower is better)
            avg_latency = perf["average_latency"] if perf["average_latency"] > 0 else 1
            speed_score = 1.0 / avg_latency
            
            # Normalize speed score to 0-1 range
            max_speed = max([1.0 / p["average_latency"] if p["average_latency"] > 0 else 0 
                           for p in self.model_performance.values()])
            if max_speed > 0:
                speed_score = speed_score / max_speed
            
            # Combined score
            score = (0.7 * success_rate) + (0.3 * speed_score)
            
            if score > best_score:
                best_score = score
                best_model = model
        
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
            model_record = ModelPerformance(
                model_id=model,
                total_calls=0,
                successful_calls=0,
                total_latency=0.0,
                quality_score=0.0,
                quality_evaluations=0,
                is_current=(model == self.current_model)
            )
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
