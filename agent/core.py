import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from agent.memory import MemoryManager
from agent.models import VeniceClient
from agent.evaluation import evaluate_model_response
import config

logger = logging.getLogger(__name__)

class Agent:
    """
    Self-learning agent that adapts to different Venice.ai API models
    based on performance and maintains persistent memory with Qdrant.
    """
    
    def __init__(
        self,
        venice_client: VeniceClient,
        memory_manager: MemoryManager,
        available_models: List[str],
        default_model: str
    ):
        """
        Initialize the agent with the Venice API client and memory manager
        
        Args:
            venice_client: Initialized Venice API client
            memory_manager: Memory manager with Qdrant integration
            available_models: List of available Venice.ai models to use
            default_model: Default model to start with
        """
        self.venice_client = venice_client
        self.memory_manager = memory_manager
        self.available_models = available_models
        self.current_model = default_model
        
        # Model performance tracking
        self.model_performance = {model: {
            "success_rate": 0.0,
            "average_latency": 0.0,
            "total_calls": 0,
            "successes": 0,
            "failures": 0,
            "total_latency": 0.0,
            "last_used": None
        } for model in available_models}
        
        # Interaction counter for model evaluation
        self.interaction_count = 0
        logger.info(f"Agent initialized with default model: {default_model}")
    
    def process_query(self, query: str, system_prompt: str) -> Tuple[str, str]:
        """
        Process a user query and return the response using the most appropriate model.
        
        Args:
            query: The user's query
            system_prompt: System prompt describing the agent's purpose
            
        Returns:
            Tuple of (response text, model used)
        """
        self.interaction_count += 1
        
        # Get relevant memories based on the query
        relevant_memories = self.memory_manager.get_relevant_memories(query, limit=5)
        
        # Create context from relevant memories
        context = self._create_context_from_memories(relevant_memories)
        
        # Construct prompt with context
        prompt = self._construct_prompt(query, system_prompt, context)
        
        # If it's time to evaluate models, try a different one
        if self.interaction_count % config.MODEL_EVALUATION_INTERVAL == 0:
            model_to_use = self._select_model_for_evaluation()
            logger.info(f"Evaluating model: {model_to_use}")
        else:
            model_to_use = self._get_best_model()
            logger.info(f"Using best model: {model_to_use}")
        
        # Call the model
        start_time = time.time()
        try:
            response = self.venice_client.generate(prompt, model=model_to_use)
            success = True
        except Exception as e:
            logger.error(f"Error generating response with model {model_to_use}: {str(e)}")
            # Fallback to default model if available
            if model_to_use != self.current_model:
                try:
                    logger.info(f"Falling back to current model: {self.current_model}")
                    response = self.venice_client.generate(prompt, model=self.current_model)
                    model_to_use = self.current_model
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
        Select a model for evaluation based on usage patterns
        
        Returns:
            Model ID to evaluate
        """
        # Find least recently used model
        least_used = None
        min_calls = float('inf')
        
        for model, perf in self.model_performance.items():
            if perf["total_calls"] < min_calls:
                min_calls = perf["total_calls"]
                least_used = model
        
        # If there's a model with fewer calls, use it
        if least_used and least_used != self.current_model:
            return least_used
        
        # Otherwise, randomly select a model that's not the current one
        import random
        candidates = [m for m in self.available_models if m != self.current_model]
        if candidates:
            return random.choice(candidates)
        
        # If all else fails, return the current model
        return self.current_model
    
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
        if model not in self.model_performance:
            self.model_performance[model] = {
                "success_rate": 0.0,
                "average_latency": 0.0,
                "total_calls": 0,
                "successes": 0,
                "failures": 0,
                "total_latency": 0.0,
                "last_used": None
            }
        
        perf = self.model_performance[model]
        perf["total_calls"] += 1
        
        if success:
            perf["successes"] += 1
        else:
            perf["failures"] += 1
        
        perf["success_rate"] = perf["successes"] / perf["total_calls"]
        perf["total_latency"] += latency
        perf["average_latency"] = perf["total_latency"] / perf["total_calls"]
        perf["last_used"] = time.time()
        
        # If this model is performing better, update current model
        if (success and model != self.current_model and 
            perf["success_rate"] > self.model_performance[self.current_model]["success_rate"] and
            perf["total_calls"] >= 5):
            logger.info(f"Switching default model from {self.current_model} to {model} based on performance")
            self.current_model = model
    
    def _update_model_quality(self, model: str, quality_score: float) -> None:
        """
        Update the quality metrics for a specific model
        
        Args:
            model: Model ID
            quality_score: Evaluation score (0-1)
        """
        if model not in self.model_performance:
            return
            
        # We could enhance this with more sophisticated quality tracking
        # For now, we just log it
        logger.info(f"Model {model} quality score: {quality_score}")
    
    def get_models_performance(self) -> Dict[str, Dict]:
        """
        Get performance metrics for all models
        
        Returns:
            Dictionary of model performance metrics
        """
        # Add current_model flag to the model data
        result = {}
        for model, perf in self.model_performance.items():
            model_data = perf.copy()
            model_data["is_current"] = (model == self.current_model)
            result[model] = model_data
        
        return result
