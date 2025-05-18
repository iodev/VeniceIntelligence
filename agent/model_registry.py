"""
Model Registry System for tracking all available models across providers

This module provides a centralized registry for managing model metadata,
capabilities, and performance across different AI providers without requiring
database schema changes.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
import os
import threading
import re

from sqlalchemy import func, or_, text
from models import ModelPerformance, db
from agent.models import VeniceClient

logger = logging.getLogger(__name__)

# In-memory registry data
_deprecated_models = {}  # model_id -> reason
_model_metadata = {}  # model_id -> extra metadata
_provider_clients = {}  # provider -> client
_provider_status = {
    "venice": True,
    "anthropic": False,
    "perplexity": False,
    "huggingface": False
}
_last_refresh = {}  # provider -> timestamp
_registry_lock = threading.RLock()

class ModelRegistry:
    """
    Centralized registry for tracking AI models across providers 
    and managing their metadata
    
    This class maintains both database and in-memory model information
    to provide a complete registry without modifying the database schema.
    """
    
    def __init__(self):
        """Initialize the model registry system"""
        # We're using module-level variables for thread safety
        pass
    
    def register_client(self, provider: str, client: Any) -> bool:
        """
        Register a provider's client with the registry
        
        Args:
            provider: Provider name (venice, anthropic, perplexity, huggingface)
            client: Provider's client instance
            
        Returns:
            Success status
        """
        with _registry_lock:
            if provider not in _provider_status:
                logger.warning(f"Attempted to register unknown provider: {provider}")
                return False
                
            _provider_clients[provider] = client
            _provider_status[provider] = True
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
        # Query models with the required capability - using LIKE operator for string contains
        capability_filter = f"%{capability}%"
        models = ModelPerformance.query.filter(
            ModelPerformance.capabilities.like(capability_filter)
        ).all()
        
        result = []
        for model in models:
            # Skip deprecated models
            if model.model_id in _deprecated_models:
                continue
                
            # Calculate success rate
            success_rate = 0
            if model.total_calls > 0:
                success_rate = model.successful_calls / model.total_calls
                
            # Apply success rate filter
            if success_rate < min_success_rate:
                continue
                
            # Add model to results
            result.append({
                "model_id": model.model_id,
                "provider": model.provider,
                "success_rate": success_rate,
                "quality_score": model.quality_score,
                "average_latency": (model.total_latency / model.total_calls) if model.total_calls > 0 else 0,
                "capabilities": model.capabilities.split(',') if model.capabilities else [],
                "context_window": model.context_window,
                "display_name": model.display_name,
                "is_deprecated": model.model_id in _deprecated_models,
                "metadata": _model_metadata.get(model.model_id, {})
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
        with _registry_lock:
            if provider is not None and provider not in _provider_status:
                return {"status": "error", "message": f"Unknown provider: {provider}"}
                
            providers_to_check = [provider] if provider else list(_provider_status.keys())
            results = {"discovered": 0, "updated": 0, "errors": 0, "details": {}}
            
            for provider_name in providers_to_check:
                # Skip providers that aren't registered or available
                if provider_name not in _provider_clients or not _provider_status.get(provider_name, False):
                    logger.warning(f"Provider {provider_name} not available for discovery")
                    results["details"][provider_name] = "Provider not available"
                    continue
                    
                # Check if we need to refresh based on time
                last_check = _last_refresh.get(provider_name, datetime.min)
                if not force and datetime.now() - last_check < timedelta(hours=24):
                    logger.info(f"Skipping {provider_name} discovery (cache valid)")
                    results["details"][provider_name] = "Using cached data"
                    continue
                
                # Call the appropriate discovery method
                if provider_name == "venice":
                    provider_result = self._discover_venice_models()
                elif provider_name == "anthropic":
                    provider_result = self._discover_anthropic_models()
                elif provider_name == "perplexity":
                    provider_result = self._discover_perplexity_models()
                elif provider_name == "huggingface":
                    provider_result = self._discover_huggingface_models()
                else:
                    provider_result = {"status": "error", "message": f"Unknown provider: {provider_name}"}
                
                # Update results
                results["discovered"] += provider_result.get("discovered", 0)
                results["updated"] += provider_result.get("updated", 0)
                results["errors"] += provider_result.get("errors", 0)
                results["details"][provider_name] = provider_result.get("message", "Completed")
                
                # Update last refresh time on success
                if provider_result.get("status") == "success":
                    _last_refresh[provider_name] = datetime.now()
            
            results["status"] = "success"
            return results
    
    def _discover_venice_models(self) -> Dict[str, Any]:
        """Discover models from Venice.ai"""
        result = {"status": "success", "discovered": 0, "updated": 0, "errors": 0}
        
        try:
            client = _provider_clients.get("venice")
            if not client:
                return {"status": "error", "message": "Venice client not registered"}
                
            # Get text models
            text_models = client.list_models(type="text")
            self._register_venice_models(text_models, "text")
            result["discovered"] += len(text_models)
            
            # Get image models
            image_models = client.list_models(type="image")
            self._register_venice_models(image_models, "image")
            result["discovered"] += len(image_models)
            
            result["message"] = f"Successfully discovered {result['discovered']} Venice models"
            
        except Exception as e:
            logger.error(f"Error discovering Venice models: {str(e)}")
            result["errors"] += 1
            result["message"] = f"Error: {str(e)}"
            result["status"] = "error"
            
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
                db.session.add(model)
                
            # Store additional metadata
            _model_metadata[model_id] = {
                "display_name": model_info.get("display_name", model_id),
                "last_updated": datetime.now().isoformat()
            }
                
        db.session.commit()
    
    def _discover_anthropic_models(self) -> Dict[str, Any]:
        """Discover models from Anthropic"""
        result = {"status": "success", "discovered": 0, "updated": 0, "errors": 0}
        
        try:
            # Use direct API connection if available
            client = _provider_clients.get("anthropic")
            if not client:
                return {"status": "error", "message": "Anthropic client not registered"}
            
            # Try using Perplexity to obtain Anthropic model info
            perplexity_client = _provider_clients.get("perplexity")
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
                        db.session.add(model_obj)
                        result["discovered"] += 1
                
                db.session.commit()
                result["message"] = f"Discovered {result['discovered']} Anthropic models"
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
                        db.session.add(model_obj)
                        result["discovered"] += 1
                
                db.session.commit()
                result["message"] = f"Registered {result['discovered']} known Anthropic models"
                
        except Exception as e:
            logger.error(f"Error discovering Anthropic models: {str(e)}")
            result["errors"] += 1
            result["message"] = f"Error: {str(e)}"
            result["status"] = "error"
            
        return result
    
    def _discover_perplexity_models(self) -> Dict[str, Any]:
        """Discover models from Perplexity"""
        result = {"status": "success", "discovered": 0, "updated": 0, "errors": 0}
        
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
                    db.session.add(model)
                    result["discovered"] += 1
            
            db.session.commit()
            result["message"] = f"Registered {result['discovered']} Perplexity models"
            
        except Exception as e:
            logger.error(f"Error discovering Perplexity models: {str(e)}")
            result["errors"] += 1
            result["message"] = f"Error: {str(e)}"
            result["status"] = "error"
            
        return result
    
    def _discover_huggingface_models(self) -> Dict[str, Any]:
        """Register known HuggingFace models"""
        result = {"status": "success", "discovered": 0, "updated": 0, "errors": 0}
        
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
                    db.session.add(model_obj)
                    result["discovered"] += 1
            
            db.session.commit()
            result["message"] = f"Registered {result['discovered']} HuggingFace models"
            
        except Exception as e:
            logger.error(f"Error registering HuggingFace models: {str(e)}")
            result["errors"] += 1
            result["message"] = f"Error: {str(e)}"
            result["status"] = "error"
            
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
        # Base query with capability filter using LIKE
        capability_filter = f"%{capability}%"
        query = ModelPerformance.query.filter(
            ModelPerformance.capabilities.like(capability_filter),
            ModelPerformance.total_calls > 0
        )
        
        # Apply provider filter if specified
        if provider:
            query = query.filter(ModelPerformance.provider == provider)
        
        # Get all matching models
        models = query.all()
        
        # Filter out deprecated models
        models = [m for m in models if m.model_id not in _deprecated_models]
        
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
            
            # Combined score based on specified criteria
            if criteria == "accuracy":
                combined_score = model.quality_score
            elif criteria == "speed":
                combined_score = speed_score
            elif criteria == "success":
                combined_score = success_rate
            else:  # balanced
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
                "combined_score": combined_score,
                "metadata": _model_metadata.get(model.model_id, {})
            })
        
        # Sort by combined score
        sorted_models = sorted(scored_models, key=lambda x: x["combined_score"], reverse=True)
        return sorted_models[:limit]
    
    def deprecate_model(self, model_id: str, reason: str = None) -> bool:
        """
        Mark a model as deprecated
        
        Args:
            model_id: The model to deprecate
            reason: Optional reason for deprecation
            
        Returns:
            Success status
        """
        with _registry_lock:
            model = ModelPerformance.query.filter_by(model_id=model_id).first()
            if not model:
                logger.warning(f"Attempted to deprecate unknown model: {model_id}")
                return False
                
            # Mark as deprecated in our registry
            _deprecated_models[model_id] = reason or "Manually deprecated"
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
        with _registry_lock:
            model = ModelPerformance.query.filter_by(model_id=model_id).first()
            if not model:
                logger.warning(f"Attempted to restore unknown model: {model_id}")
                return False
                
            # Remove from deprecated list
            if model_id in _deprecated_models:
                del _deprecated_models[model_id]
                logger.info(f"Model {model_id} restored from deprecated status")
                return True
            else:
                logger.info(f"Model {model_id} was not deprecated")
                return False
    
    def get_all_models(self, include_deprecated: bool = False) -> List[Dict[str, Any]]:
        """
        Get information about all registered models
        
        Args:
            include_deprecated: Whether to include deprecated models
            
        Returns:
            List of all model info dictionaries
        """
        # Get all models from database
        models = ModelPerformance.query.all()
        
        result = []
        for model in models:
            # Skip deprecated models if not requested
            if not include_deprecated and model.model_id in _deprecated_models:
                continue
                
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
                "is_deprecated": model.model_id in _deprecated_models,
                "deprecation_reason": _deprecated_models.get(model.model_id),
                "metadata": _model_metadata.get(model.model_id, {})
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
        # Build query
        query = ModelPerformance.query
        
        # Apply filters
        if provider:
            query = query.filter(ModelPerformance.provider == provider)
            
        if capability:
            query = query.filter(ModelPerformance.capabilities.like(f"%{capability}%"))
            
        if min_quality_score > 0:
            query = query.filter(ModelPerformance.quality_score >= min_quality_score)
            
        if search_term:
            query = query.filter(
                or_(
                    ModelPerformance.model_id.like(f"%{search_term}%"),
                    ModelPerformance.display_name.like(f"%{search_term}%")
                )
            )
            
        # Get all models
        models = query.all()
        
        result = []
        for model in models:
            # Skip deprecated models if not requested
            if not include_deprecated and model.model_id in _deprecated_models:
                continue
                
            # Calculate metrics
            success_rate = 0
            avg_latency = 0
            if model.total_calls > 0:
                success_rate = model.successful_calls / model.total_calls
                avg_latency = model.total_latency / model.total_calls
                
            # Skip if below success rate threshold
            if success_rate < min_success_rate:
                continue
                
            result.append({
                "model_id": model.model_id,
                "provider": model.provider,
                "success_rate": success_rate,
                "quality_score": model.quality_score,
                "average_latency": avg_latency,
                "capabilities": model.capabilities.split(',') if model.capabilities else [],
                "context_window": model.context_window,
                "display_name": model.display_name,
                "is_deprecated": model.model_id in _deprecated_models,
                "deprecation_reason": _deprecated_models.get(model.model_id),
                "metadata": _model_metadata.get(model.model_id, {})
            })
            
        return result
    
    def get_provider_status(self) -> Dict[str, bool]:
        """
        Get the current status of all providers
        
        Returns:
            Dictionary of provider status (provider_name: is_active)
        """
        return dict(_provider_status)
        
    def set_provider_status(self, provider: str, status: bool) -> bool:
        """
        Set the status of a provider
        
        Args:
            provider: Provider name
            status: Active status
            
        Returns:
            Success status
        """
        with _registry_lock:
            if provider not in _provider_status:
                logger.warning(f"Attempted to set status for unknown provider: {provider}")
                return False
                
            _provider_status[provider] = status
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
            "last_refresh": {k: v.isoformat() if isinstance(v, datetime) else str(v) 
                           for k, v in _last_refresh.items()},
            "deprecated_models": len(_deprecated_models),
            "model_metadata": len(_model_metadata),
            "provider_status": dict(_provider_status)
        }
    
    def get_model_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the model registry
        
        Returns:
            Statistics about the registry
        """
        models_by_provider = {}
        total_models = 0
        
        for provider in _provider_status.keys():
            count = ModelPerformance.query.filter_by(provider=provider).count()
            models_by_provider[provider] = count
            total_models += count
            
        return {
            "total_models": total_models,
            "models_by_provider": models_by_provider,
            "deprecated_models": len(_deprecated_models),
            "active_providers": sum(1 for v in _provider_status.values() if v)
        }


# Create a global registry instance
registry = ModelRegistry()