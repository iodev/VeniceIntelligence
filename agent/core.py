import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from agent.memory import MemoryManager
from agent.models import VeniceClient
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
        from models import ModelPerformance
        from main import db
        from agent.cost_control import CostMonitor
        
        self.venice_client = venice_client
        self.memory_manager = memory_manager
        self.available_models = available_models
        self.cost_monitor = CostMonitor()
        
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
        logger.info(f"Agent initialized with current model: {self.current_model}")
    
    def process_query(self, query: str, system_prompt: str, query_type: str = "text") -> Tuple[str, str]:
        """
        Process a user query and return the response using the most appropriate model.
        
        Args:
            query: The user's query
            system_prompt: System prompt describing the agent's purpose
            query_type: Type of query (text, code, image)
            
        Returns:
            Tuple of (response text, model used)
        """
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
        
        # Call the model
        start_time = time.time()
        try:
            response = self.venice_client.generate(messages, model=model_to_use)
            success = True
        except Exception as e:
            logger.error(f"Error generating response with model {model_to_use}: {str(e)}")
            # Fallback to default model if available
            if model_to_use != self.current_model:
                try:
                    logger.info(f"Falling back to current model: {self.current_model}")
                    response = self.venice_client.generate(messages, model=self.current_model)
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
