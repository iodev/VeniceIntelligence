"""
Model Registry System for tracking all available models across providers

This module provides a centralized registry for managing model metadata,
capabilities, and performance across different AI providers.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
from sqlalchemy import desc, asc, func, or_, and_

from models import db, ModelPerformance
from agent.models import VeniceClient
from agent.anthropic_client import AnthropicClient
from agent.perplexity import PerplexityClient
from agent.huggingface_client import HuggingFaceClient

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Centralized registry for tracking AI models across providers and managing their metadata
    
    This class handles:
    - Model registration and discovery
    - Capability tracking and filtering
    - Status management (available, deprecated, etc.)
    - Provider-specific metadata storage
    - Centralized query interface for model selection
    """
    
    def __init__(self):
        """Initialize the model registry system"""
        self.providers = {
            "venice": True,
            "anthropic": False,
            "perplexity": False,
            "huggingface": False
        }
        self.clients = {}
        self.last_refresh = {}
        self.discovery_in_progress = False
        self._discovery_errors = {}
    
    def register_client(self, provider: str, client: Any) -> bool:
        """
        Register a provider's client with the registry
        
        Args:
            provider: Provider name (venice, anthropic, perplexity, huggingface)
            client: Provider's client instance
            
        Returns:
            Success status
        """
        if provider not in self.providers:
            logger.warning(f"Attempted to register unknown provider: {provider}")
            return False
            
        self.clients[provider] = client
        self.providers[provider] = True
        logger.info(f"Registered client for provider: {provider}")
        return True
    
    def get_model_capabilities(self, model_id: str) -> Set[str]:
        """
        Get the capabilities of a specific model
        
        Args:
            model_id: The model identifier
            
        Returns:
            Set of capability strings (text, code, image, etc.)
        """
        model = ModelPerformance.query.filter_by(model_id=model_id).first()
        if not model:
            return set()
        
        capabilities = set()
        if model.capabilities:
            capabilities = set(model.capabilities.split(','))
        
        return capabilities
    
    def get_models_by_capability(self, capability: str, min_success_rate: float = 0.0) -> List[Dict[str, Any]]:
        """
        Get all models that support a specific capability
        
        Args:
            capability: Required capability (text, code, image, etc.)
            min_success_rate: Minimum success rate threshold
            
        Returns:
            List of model info dictionaries
        """
        models = ModelPerformance.query.filter(
            ModelPerformance.capabilities.like(f"%{capability}%"),
            (ModelPerformance.successful_calls / func.nullif(ModelPerformance.total_calls, 0)) >= min_success_rate,
            ModelPerformance.is_deprecated == False
        ).all()
        
        result = []
        for model in models:
            # Calculate success rate with null safety
            success_rate = 0
            if model.total_calls > 0:
                success_rate = model.successful_calls / model.total_calls
                
            result.append({
                "model_id": model.model_id,
                "provider": model.provider,
                "success_rate": success_rate,
                "quality_score": model.quality_score,
                "average_latency": (model.total_latency / model.total_calls) if model.total_calls > 0 else 0,
                "capabilities": model.capabilities.split(',') if model.capabilities else [],
                "context_window": model.context_window,
                "display_name": model.display_name
            })
            
        return result
    
    def discover_models(self, provider: str = None, force: bool = False) -> Dict[str, Any]:
        """
        Discover models available from a provider
        
        Args:
            provider: Provider to query (or None for all providers)
            force: Whether to force refresh regardless of cache status
            
        Returns:
            Dictionary with discovery results
        """
        if self.discovery_in_progress:
            logger.warning("Model discovery already in progress, skipping")
            return {"status": "in_progress", "message": "Discovery already in progress"}
            
        self.discovery_in_progress = True
        results = {"discovered": 0, "updated": 0, "errors": 0, "details": {}}
        
        try:
            providers_to_check = [provider] if provider else list(self.providers.keys())
            
            for provider_name in providers_to_check:
                # Skip providers that aren't registered or available
                if provider_name not in self.clients or not self.providers.get(provider_name, False):
                    logger.warning(f"Provider {provider_name} not available for discovery")
                    results["details"][provider_name] = "Provider not available"
                    continue
                    
                # Check if we need to refresh based on time
                last_check = self.last_refresh.get(provider_name, datetime.min)
                if not force and datetime.now() - last_check < timedelta(hours=24):
                    logger.info(f"Skipping {provider_name} discovery (cache valid)")
                    results["details"][provider_name] = "Using cached data"
                    continue
                
                # Get the appropriate discovery method for this provider
                result = self._discover_provider_models(provider_name)
                
                # Update results
                results["discovered"] += result.get("discovered", 0)
                results["updated"] += result.get("updated", 0)
                results["errors"] += result.get("errors", 0)
                results["details"][provider_name] = result.get("message", "Completed")
                
                # Update last refresh time on success
                if result.get("success", False):
                    self.last_refresh[provider_name] = datetime.now()
            
            results["status"] = "success"
            return results
            
        except Exception as e:
            logger.error(f"Error during model discovery: {str(e)}")
            results["status"] = "error"
            results["message"] = str(e)
            return results
        finally:
            self.discovery_in_progress = False
    
    def _discover_provider_models(self, provider: str) -> Dict[str, Any]:
        """
        Discover models for a specific provider
        
        Args:
            provider: Provider name
            
        Returns:
            Discovery result details
        """
        result = {"discovered": 0, "updated": 0, "errors": 0, "success": False}
        
        if provider == "venice":
            # Venice uses direct API methods for discovery
            result = self._discover_venice_models()
            
        elif provider == "anthropic":
            # Anthropic doesn't have a models endpoint, use Perplexity to get info
            result = self._discover_anthropic_models()
            
        elif provider == "perplexity":
            # Perplexity has standard models we can register
            result = self._discover_perplexity_models()
            
        elif provider == "huggingface":
            # Hugging Face has too many models, we register only what we use
            result = self._discover_huggingface_models()
            
        else:
            result["message"] = f"Unknown provider: {provider}"
            
        return result
    
    def _discover_venice_models(self) -> Dict[str, Any]:
        """Discover models from Venice.ai"""
        result = {"discovered": 0, "updated": 0, "errors": 0, "success": False}
        
        try:
            client = self.clients.get("venice")
            if not client:
                result["message"] = "Venice client not registered"
                return result
                
            # Get text models
            text_models = client.list_models(type="text")
            self._register_venice_models(text_models, "text")
            result["discovered"] += len(text_models)
            
            # Get image models
            image_models = client.list_models(type="image")
            self._register_venice_models(image_models, "image")
            result["discovered"] += len(image_models)
            
            result["message"] = f"Successfully discovered {result['discovered']} Venice models"
            result["success"] = True
            
        except Exception as e:
            logger.error(f"Error discovering Venice models: {str(e)}")
            result["errors"] += 1
            result["message"] = f"Error: {str(e)}"
            self._discovery_errors["venice"] = str(e)
            
        return result
    
    def _register_venice_models(self, models: List[Dict[str, Any]], capability: str) -> None:
        """
        Register Venice models in the database
        
        Args:
            models: List of model info dictionaries from Venice API
            capability: Model capability (text, image, etc.)
        """
        for model_info in models:
            model_id = model_info.get("id")
            if not model_id:
                continue
                
            # Check if model already exists
            existing = ModelPerformance.query.filter_by(
                model_id=model_id, 
                provider="venice"
            ).first()
            
            if existing:
                # Update existing model info
                if capability not in existing.capabilities:
                    existing.capabilities = f"{existing.capabilities},{capability}"
                # Update other fields as needed
                db.session.add(existing)
            else:
                # Create new model entry
                model = ModelPerformance()
                model.model_id = model_id
                model.provider = "venice"
                model.capabilities = capability
                model.display_name = model_info.get("display_name", model_id)
                # Add other fields from API response
                model.context_window = model_info.get("context_window", 4096)
                model.total_calls = 0
                model.successful_calls = 0
                model.total_latency = 0
                model.quality_score = 0
                model.quality_evaluations = 0
                model.is_deprecated = False
                db.session.add(model)
                
        db.session.commit()
    
    def _discover_anthropic_models(self) -> Dict[str, Any]:
        """Discover models from Anthropic"""
        result = {"discovered": 0, "updated": 0, "errors": 0, "success": False}
        
        try:
            # Use direct API connection if available
            client = self.clients.get("anthropic")
            if not client:
                result["message"] = "Anthropic client not registered"
                return result
            
            # Try using Perplexity to obtain Anthropic model info
            perplexity_client = self.clients.get("perplexity")
            if perplexity_client:
                model_info = perplexity_client.get_anthropic_models()
                discovered_models = model_info.get("models", [])
                
                for model in discovered_models:
                    # Register model
                    existing = ModelPerformance.query.filter_by(
                        model_id=model, 
                        provider="anthropic"
                    ).first()
                    
                    if existing:
                        # Update existing model
                        result["updated"] += 1
                    else:
                        # Create new model
                        model_obj = ModelPerformance()
                        model_obj.model_id = model
                        model_obj.provider = "anthropic"
                        model_obj.capabilities = "text"
                        model_obj.display_name = model
                        model_obj.context_window = 100000  # Estimated
                        model_obj.total_calls = 0
                        model_obj.successful_calls = 0
                        model_obj.total_latency = 0
                        model_obj.quality_score = 0
                        model_obj.quality_evaluations = 0
                        model_obj.is_deprecated = False
                        db.session.add(model_obj)
                        result["discovered"] += 1
                
                db.session.commit()
                result["message"] = f"Discovered {result['discovered']} Anthropic models"
                result["success"] = True
            else:
                # Register known models
                known_models = [
                    "claude-3-5-sonnet-20241022",
                    "claude-3-7-sonnet-20241022", 
                    "claude-3-haiku-20240307",
                    "claude-3-opus-20240229"
                ]
                
                for model in known_models:
                    existing = ModelPerformance.query.filter_by(
                        model_id=model, 
                        provider="anthropic"
                    ).first()
                    
                    if not existing:
                        model_obj = ModelPerformance()
                        model_obj.model_id = model
                        model_obj.provider = "anthropic"
                        model_obj.capabilities = "text"
                        model_obj.display_name = model
                        model_obj.context_window = 100000  # Claude models typically have large context
                        model_obj.total_calls = 0
                        model_obj.successful_calls = 0
                        model_obj.total_latency = 0
                        model_obj.quality_score = 0
                        model_obj.quality_evaluations = 0
                        model_obj.is_deprecated = False
                        db.session.add(model_obj)
                        result["discovered"] += 1
                
                db.session.commit()
                result["message"] = f"Registered {result['discovered']} known Anthropic models"
                result["success"] = True
                
        except Exception as e:
            logger.error(f"Error discovering Anthropic models: {str(e)}")
            result["errors"] += 1
            result["message"] = f"Error: {str(e)}"
            self._discovery_errors["anthropic"] = str(e)
            
        return result
    
    def _discover_perplexity_models(self) -> Dict[str, Any]:
        """Discover models from Perplexity"""
        result = {"discovered": 0, "updated": 0, "errors": 0, "success": False}
        
        try:
            # Standard Perplexity models
            models = [
                "llama-3.1-sonar-small-128k-online",
                "llama-3.1-sonar-large-128k-online",
                "llama-3.1-sonar-huge-128k-online"
            ]
            
            for model_id in models:
                existing = ModelPerformance.query.filter_by(
                    model_id=model_id, 
                    provider="perplexity"
                ).first()
                
                if existing:
                    # Update existing model if needed
                    result["updated"] += 1
                else:
                    # Create new model
                    model = ModelPerformance()
                    model.model_id = model_id
                    model.provider = "perplexity"
                    model.capabilities = "text,search"
                    model.display_name = model_id
                    
                    # Set context window based on model size
                    if "small" in model_id:
                        model.context_window = 65536
                    elif "large" in model_id:
                        model.context_window = 100000
                    elif "huge" in model_id:
                        model.context_window = 128000
                    else:
                        model.context_window = 32768
                        
                    model.total_calls = 0
                    model.successful_calls = 0
                    model.total_latency = 0
                    model.quality_score = 0
                    model.quality_evaluations = 0
                    model.is_deprecated = False
                    db.session.add(model)
                    result["discovered"] += 1
            
            db.session.commit()
            result["message"] = f"Registered {result['discovered']} Perplexity models"
            result["success"] = True
            
        except Exception as e:
            logger.error(f"Error discovering Perplexity models: {str(e)}")
            result["errors"] += 1
            result["message"] = f"Error: {str(e)}"
            self._discovery_errors["perplexity"] = str(e)
            
        return result
    
    def _discover_huggingface_models(self) -> Dict[str, Any]:
        """Register known HuggingFace models"""
        result = {"discovered": 0, "updated": 0, "errors": 0, "success": False}
        
        try:
            # Register a few key models
            models = [
                {"id": "huggingface/mistralai/Mistral-7B-v0.1", "display": "Mistral 7B", "context": 8192},
                {"id": "huggingface/meta-llama/Llama-2-7b-chat-hf", "display": "Llama-2 7B Chat", "context": 4096},
                {"id": "huggingface/meta-llama/Llama-2-13b-chat-hf", "display": "Llama-2 13B Chat", "context": 4096},
                {"id": "huggingface/microsoft/phi-2", "display": "Phi-2", "context": 2048}
            ]
            
            for model in models:
                existing = ModelPerformance.query.filter_by(
                    model_id=model["id"], 
                    provider="huggingface"
                ).first()
                
                if existing:
                    # Update existing model
                    result["updated"] += 1
                else:
                    # Create new model
                    model_obj = ModelPerformance()
                    model_obj.model_id = model["id"]
                    model_obj.provider = "huggingface"
                    model_obj.capabilities = "text"
                    model_obj.display_name = model["display"]
                    model_obj.context_window = model["context"]
                    model_obj.total_calls = 0
                    model_obj.successful_calls = 0
                    model_obj.total_latency = 0
                    model_obj.quality_score = 0
                    model_obj.quality_evaluations = 0
                    model_obj.is_deprecated = False
                    db.session.add(model_obj)
                    result["discovered"] += 1
            
            db.session.commit()
            result["message"] = f"Registered {result['discovered']} HuggingFace models"
            result["success"] = True
            
        except Exception as e:
            logger.error(f"Error registering HuggingFace models: {str(e)}")
            result["errors"] += 1
            result["message"] = f"Error: {str(e)}"
            self._discovery_errors["huggingface"] = str(e)
            
        return result
    
    def get_best_models(self, 
                      capability: str = "text", 
                      provider: str = None,
                      criteria: str = "balanced",
                      limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get the best performing models based on various criteria
        
        Args:
            capability: Required capability (text, code, image, etc.)
            provider: Optional provider filter
            criteria: Selection criteria:
                - "accuracy": Best quality scores
                - "speed": Lowest average latency
                - "success": Highest success rate
                - "balanced": Balanced scoring of all factors
            limit: Maximum number of models to return
            
        Returns:
            List of best model info dictionaries
        """
        query = ModelPerformance.query.filter(
            ModelPerformance.capabilities.like(f"%{capability}%"),
            ModelPerformance.is_deprecated == False,
            ModelPerformance.total_calls > 0
        )
        
        if provider:
            query = query.filter(ModelPerformance.provider == provider)
        
        # Apply different sorting based on criteria
        if criteria == "accuracy":
            query = query.order_by(desc(ModelPerformance.quality_score))
        elif criteria == "speed":
            # Use average latency for sorting
            query = query.order_by(
                (ModelPerformance.total_latency / ModelPerformance.total_calls)
            )
        elif criteria == "success":
            # Use success rate for sorting
            query = query.order_by(
                desc(ModelPerformance.successful_calls / func.nullif(ModelPerformance.total_calls, 0))
            )
        else:  # balanced - combine multiple factors
            # This is more complex in raw SQL, we'll fetch and sort in memory
            models = query.all()
            
            # Create scored list
            scored_models = []
            for model in models:
                if model.total_calls == 0:
                    continue
                    
                # Calculate normalized scores (0-1)
                success_rate = model.successful_calls / model.total_calls
                avg_latency = model.total_latency / model.total_calls
                
                # Invert latency so that lower is better (1 = fast, 0 = slow)
                speed_score = 1.0 - min(1.0, avg_latency / 5.0)  # Normalize to 0-1 (5s is worst)
                
                # Combined score with weights
                combined_score = (
                    (model.quality_score * 0.4) +  # 40% quality weight
                    (success_rate * 0.3) +         # 30% success rate weight
                    (speed_score * 0.3)            # 30% speed weight
                )
                
                scored_models.append({
                    "model_id": model.model_id,
                    "provider": model.provider,
                    "success_rate": success_rate,
                    "quality_score": model.quality_score,
                    "average_latency": avg_latency,
                    "capabilities": model.capabilities.split(',') if model.capabilities else [],
                    "context_window": model.context_window,
                    "display_name": model.display_name,
                    "combined_score": combined_score
                })
            
            # Sort by combined score
            sorted_models = sorted(scored_models, key=lambda x: x["combined_score"], reverse=True)
            return sorted_models[:limit]
        
        # For simple criteria, use the query directly
        models = query.limit(limit).all()
        
        result = []
        for model in models:
            # Calculate metrics
            success_rate = 0
            avg_latency = 0
            if model.total_calls > 0:
                success_rate = model.successful_calls / model.total_calls
                avg_latency = model.total_latency / model.total_calls
                
            result.append({
                "model_id": model.model_id,
                "provider": model.provider,
                "success_rate": success_rate,
                "quality_score": model.quality_score,
                "average_latency": avg_latency,
                "capabilities": model.capabilities.split(',') if model.capabilities else [],
                "context_window": model.context_window,
                "display_name": model.display_name
            })
            
        return result
    
    def deprecate_model(self, model_id: str, reason: str = None) -> bool:
        """
        Mark a model as deprecated
        
        Args:
            model_id: The model to deprecate
            reason: Optional reason for deprecation
            
        Returns:
            Success status
        """
        model = ModelPerformance.query.filter_by(model_id=model_id).first()
        if not model:
            logger.warning(f"Attempted to deprecate unknown model: {model_id}")
            return False
            
        model.is_deprecated = True
        model.deprecation_reason = reason
        db.session.add(model)
        db.session.commit()
        
        logger.info(f"Model {model_id} marked as deprecated. Reason: {reason}")
        return True
    
    def restore_model(self, model_id: str) -> bool:
        """
        Restore a previously deprecated model
        
        Args:
            model_id: The model to restore
            
        Returns:
            Success status
        """
        model = ModelPerformance.query.filter_by(model_id=model_id).first()
        if not model:
            logger.warning(f"Attempted to restore unknown model: {model_id}")
            return False
            
        model.is_deprecated = False
        model.deprecation_reason = None
        db.session.add(model)
        db.session.commit()
        
        logger.info(f"Model {model_id} restored from deprecated status")
        return True
    
    def get_all_models(self, include_deprecated: bool = False) -> List[Dict[str, Any]]:
        """
        Get information about all registered models
        
        Args:
            include_deprecated: Whether to include deprecated models
            
        Returns:
            List of all model info dictionaries
        """
        query = ModelPerformance.query
        
        if not include_deprecated:
            query = query.filter(ModelPerformance.is_deprecated == False)
            
        models = query.all()
        
        result = []
        for model in models:
            # Calculate success rate with null safety
            success_rate = 0
            avg_latency = 0
            if model.total_calls > 0:
                success_rate = model.successful_calls / model.total_calls
                avg_latency = model.total_latency / model.total_calls
                
            result.append({
                "model_id": model.model_id,
                "provider": model.provider,
                "success_rate": success_rate,
                "quality_score": model.quality_score,
                "average_latency": avg_latency,
                "total_calls": model.total_calls,
                "successful_calls": model.successful_calls,
                "capabilities": model.capabilities.split(',') if model.capabilities else [],
                "context_window": model.context_window,
                "display_name": model.display_name,
                "is_deprecated": model.is_deprecated,
                "deprecation_reason": model.deprecation_reason
            })
            
        return result
    
    def search_models(self, 
                    search_term: str = None,
                    provider: str = None, 
                    capability: str = None, 
                    min_success_rate: float = 0.0,
                    min_quality_score: float = 0.0,
                    include_deprecated: bool = False) -> List[Dict[str, Any]]:
        """
        Search for models matching specified criteria
        
        Args:
            search_term: Optional text to search in model ID and display name
            provider: Optional provider filter
            capability: Optional capability filter
            min_success_rate: Minimum success rate threshold
            min_quality_score: Minimum quality score threshold
            include_deprecated: Whether to include deprecated models
            
        Returns:
            List of matching model info dictionaries
        """
        query = ModelPerformance.query
        
        if not include_deprecated:
            query = query.filter(ModelPerformance.is_deprecated == False)
            
        if provider:
            query = query.filter(ModelPerformance.provider == provider)
            
        if capability:
            query = query.filter(ModelPerformance.capabilities.like(f"%{capability}%"))
            
        if min_success_rate > 0:
            query = query.filter(
                (ModelPerformance.successful_calls / func.nullif(ModelPerformance.total_calls, 0)) >= min_success_rate
            )
            
        if min_quality_score > 0:
            query = query.filter(ModelPerformance.quality_score >= min_quality_score)
            
        if search_term:
            query = query.filter(
                or_(
                    ModelPerformance.model_id.like(f"%{search_term}%"),
                    ModelPerformance.display_name.like(f"%{search_term}%")
                )
            )
            
        models = query.all()
        
        result = []
        for model in models:
            # Calculate metrics
            success_rate = 0
            avg_latency = 0
            if model.total_calls > 0:
                success_rate = model.successful_calls / model.total_calls
                avg_latency = model.total_latency / model.total_calls
                
            result.append({
                "model_id": model.model_id,
                "provider": model.provider,
                "success_rate": success_rate,
                "quality_score": model.quality_score,
                "average_latency": avg_latency,
                "capabilities": model.capabilities.split(',') if model.capabilities else [],
                "context_window": model.context_window,
                "display_name": model.display_name,
                "is_deprecated": model.is_deprecated,
                "deprecation_reason": model.deprecation_reason
            })
            
        return result
    
    def get_provider_status(self) -> Dict[str, bool]:
        """
        Get the current status of all providers
        
        Returns:
            Dictionary of provider status (provider_name: is_active)
        """
        return dict(self.providers)
        
    def set_provider_status(self, provider: str, status: bool) -> bool:
        """
        Set the status of a provider
        
        Args:
            provider: Provider name
            status: Active status
            
        Returns:
            Success status
        """
        if provider not in self.providers:
            logger.warning(f"Attempted to set status for unknown provider: {provider}")
            return False
            
        self.providers[provider] = status
        logger.info(f"Provider {provider} status set to: {status}")
        return True
        
    def update_model_stats(self, model_id: str, success: bool, latency: float) -> bool:
        """
        Update usage statistics for a model
        
        Args:
            model_id: The model identifier
            success: Whether the call was successful
            latency: Response time in seconds
            
        Returns:
            Success status
        """
        model = ModelPerformance.query.filter_by(model_id=model_id).first()
        if not model:
            logger.warning(f"Attempted to update stats for unknown model: {model_id}")
            return False
            
        # Update statistics
        model.total_calls += 1
        model.total_latency += latency
        
        if success:
            model.successful_calls += 1
            
        db.session.add(model)
        db.session.commit()
        
        return True
        
    def update_model_quality(self, model_id: str, quality_score: float) -> bool:
        """
        Update the quality score for a model
        
        Args:
            model_id: The model identifier
            quality_score: Evaluation score (0-1)
            
        Returns:
            Success status
        """
        model = ModelPerformance.query.filter_by(model_id=model_id).first()
        if not model:
            logger.warning(f"Attempted to update quality for unknown model: {model_id}")
            return False
            
        # Get current total quality and update with new score
        current_total = model.quality_score * model.quality_evaluations
        model.quality_evaluations += 1
        model.quality_score = (current_total + quality_score) / model.quality_evaluations
        
        db.session.add(model)
        db.session.commit()
        
        return True
    
    def get_discovery_status(self) -> Dict[str, Any]:
        """
        Get the status of model discovery
        
        Returns:
            Discovery status information
        """
        return {
            "in_progress": self.discovery_in_progress,
            "last_refresh": self.last_refresh,
            "errors": self._discovery_errors
        }