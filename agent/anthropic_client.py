import logging
import json
import os
import re
import time
from typing import List, Dict, Any, Optional, Union
import requests

logger = logging.getLogger(__name__)

class AnthropicClient:
    """
    Client for interacting with the Anthropic API
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: str = "https://api.anthropic.com/v1",
        perplexity_client=None
    ):
        """
        Initialize the Anthropic API client
        
        Args:
            api_key: Anthropic API key
            base_url: Base URL for Anthropic API
            perplexity_client: Optional Perplexity client for dynamic model updates
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = base_url
        self.perplexity_client = perplexity_client
        
        # Model registry keeps track of available models
        self.model_registry = []
        self.model_registry_last_updated = 0
        
        # Default model to use if not specified
        self.default_model = "claude-3-5-sonnet-20241022"
        
        # Basic validation
        if not self.api_key:
            logger.warning("No Anthropic API key provided")
            self.is_available = False
            return
            
        # Set up main session for completions
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        })
        
        self.is_available = True
        
        # Test connection if key is available
        if self.is_available:
            self.is_available = self.test_connection()
    
    def test_connection(self) -> bool:
        """
        Test the connection to the Anthropic API
        
        Returns:
            Whether the connection was successful
        """
        if not self.api_key:
            logger.warning("Cannot test Anthropic API connection without API key")
            return False
            
        try:
            # Make a minimal direct API call instead of using generate()
            # This avoids recursive calls and better captures specific error types
            url = f"{self.base_url}/messages"
            test_messages = [
                {"role": "user", "content": "Hello, this is a test."}
            ]
            
            data = {
                "model": self.default_model,
                "messages": test_messages,
                "max_tokens": 5,
                "temperature": 0.1
            }
            
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                url, 
                json=data, 
                headers=headers,
                timeout=5  # Short timeout for testing
            )
            
            if response.status_code == 200:
                logger.info("Successfully connected to Anthropic API")
                return True
            else:
                error_text = response.text
                logger.warning(f"Anthropic API test failed: {response.status_code}, {error_text}")
                
                # Check for specific error types
                if "credit balance is too low" in error_text:
                    logger.error("Anthropic API unavailable: Insufficient credit balance")
                    return False
                elif "rate limit" in error_text.lower():
                    logger.error("Anthropic API unavailable: Rate limit exceeded")
                    return False
                elif "unauthorized" in error_text.lower() or response.status_code == 401:
                    logger.error("Anthropic API unavailable: Authentication error")
                    return False
                elif "not_found_error" in error_text.lower() and "model" in error_text.lower():
                    # Handle model not found error specifically
                    logger.warning(f"Model {self.default_model} not found, attempting to find current models")
                    
                    # Try to extract suggested model from error message
                    suggested_model = self._extract_suggested_model_from_error(error_text)
                    if suggested_model:
                        logger.info(f"API suggested model: {suggested_model}")
                        self.default_model = suggested_model
                        # Try connection again with suggested model
                        return self.test_connection()
                    
                    # If we have Perplexity client, try to update models
                    if self.perplexity_client:
                        success = self._update_models_via_perplexity()
                        if success:
                            # Try the connection again with updated models
                            return self.test_connection()
                
                return False
        except Exception as e:
            logger.error(f"Error connecting to Anthropic API: {str(e)}")
            return False
            
    def _extract_suggested_model_from_error(self, error_text: str) -> Optional[str]:
        """
        Extract suggested model name from API error response
        
        Args:
            error_text: Error response from API
            
        Returns:
            Suggested model name if found, None otherwise
        """
        try:
            # Look for patterns like "Did you mean claude-3-5-sonnet-20241022"
            pattern = r"Did you mean ([a-zA-Z0-9\-]+)"
            match = re.search(pattern, error_text)
            if match:
                return match.group(1)
                
            return None
        except Exception as e:
            logger.error(f"Error extracting suggested model: {str(e)}")
            return None
            
    def _update_models_via_perplexity(self) -> bool:
        """
        Use Perplexity to fetch current Anthropic models
        
        Returns:
            Whether models were successfully updated
        """
        if not self.perplexity_client:
            logger.warning("No Perplexity client available for model update")
            return False
            
        try:
            logger.info("Attempting to update Anthropic models via Perplexity")
            
            # Fetch current models from Perplexity
            current_models = self.perplexity_client.fetch_current_anthropic_models()
            
            if not current_models or len(current_models) == 0:
                logger.warning("No models returned from Perplexity")
                return False
                
            # Update our model registry
            self.model_registry = current_models
            self.model_registry_last_updated = time.time()
            
            # Update default model to the first returned model
            if len(current_models) > 0:
                # Find the model with "sonnet" in the name for default
                sonnet_models = [m for m in current_models if "sonnet" in m["id"].lower()]
                if sonnet_models:
                    self.default_model = sonnet_models[0]["id"]
                else:
                    self.default_model = current_models[0]["id"]
                    
            logger.info(f"Updated model registry with {len(current_models)} models")
            logger.info(f"New default model: {self.default_model}")
            
            return True
        except Exception as e:
            logger.error(f"Error updating models via Perplexity: {str(e)}")
            return False
    
    def generate(
        self, 
        messages: list, 
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text using Anthropic API
        
        Args:
            messages: List of message objects (user, assistant)
            model: Model ID to use (if None, default model is used)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Top-p sampling parameter
            stream: Whether to stream the response
            
        Returns:
            Generated text with metadata
        """
        if not self.is_available:
            raise Exception("Anthropic API client is not available. Please check API key.")
            
        if not messages:
            raise ValueError("Messages cannot be empty")
            
        # Use default model if none specified
        if not model:
            model = self.default_model
            
        logger.info(f"Generating with Anthropic model: {model}")
        
        # Convert messages to Anthropic format if needed
        anthropic_messages = messages.copy()
        
        # Anthropic API payload
        payload = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
            
        try:
            response = self.session.post(
                f"{self.base_url}/messages",
                json=payload,
                timeout=60  # Longer timeout for generation
            )
            
            if response.status_code != 200:
                error_text = response.text if hasattr(response, 'text') else ""
                
                # Check for model not found error
                if "not_found_error" in error_text.lower() and "model" in error_text.lower():
                    logger.warning(f"Model {model} not found")
                    
                    # Try to extract suggested model from error message
                    suggested_model = self._extract_suggested_model_from_error(error_text)
                    if suggested_model:
                        logger.info(f"API suggested model: {suggested_model}")
                        # Retry with suggested model
                        if model != suggested_model:
                            return self.generate(
                                messages=messages,
                                model=suggested_model,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                stream=stream
                            )
                    
                    # If we have Perplexity client, try to update models
                    if self.perplexity_client and time.time() - self.model_registry_last_updated > 3600:  # Update once per hour max
                        success = self._update_models_via_perplexity()
                        if success and self.default_model != model:
                            logger.info(f"Retrying with updated model: {self.default_model}")
                            return self.generate(
                                messages=messages,
                                model=self.default_model,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                stream=stream
                            )
                
                # If we get here, we couldn't recover from the error
                error_msg = f"Anthropic API error: {response.status_code}"
                if error_text:
                    error_msg += f" - {error_text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            result = response.json()
            logger.debug(f"Response JSON: {result}")
            
            return result
            
        except requests.RequestException as e:
            logger.error(f"Request error with Anthropic API: {str(e)}")
            raise Exception(f"Failed to communicate with Anthropic API: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {str(e)}")
            raise Exception(f"Invalid response format from Anthropic API")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
            
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models for Anthropic
        
        Returns:
            List of model information
        """
        # Check if we have dynamic model information from registry
        if self.model_registry and len(self.model_registry) > 0:
            logger.info(f"Using {len(self.model_registry)} models from dynamic registry")
            return self.model_registry
            
        # If no models in registry, or if registry is stale (over 24 hours)
        if time.time() - self.model_registry_last_updated > 86400 and self.perplexity_client:
            try:
                # Try to fetch updated models
                current_models = self.perplexity_client.fetch_current_anthropic_models()
                if current_models and len(current_models) > 0:
                    self.model_registry = current_models
                    self.model_registry_last_updated = time.time()
                    return self.model_registry
            except Exception as e:
                logger.error(f"Error updating model registry: {str(e)}")
                # Fall back to static models if update fails
        
        # Fallback: Use static list of known models
        # The newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
        static_models = [
            {
                "id": "claude-3-5-sonnet-20241022",
                "name": "Claude 3.5 Sonnet",
                "context_window": 200000,
                "provider": "anthropic"
            },
            {
                "id": "claude-3-opus-20240229",
                "name": "Claude 3 Opus",
                "context_window": 200000,
                "provider": "anthropic"
            },
            {
                "id": "claude-3-sonnet-20240229",
                "name": "Claude 3 Sonnet",
                "context_window": 200000,
                "provider": "anthropic"
            },
            {
                "id": "claude-3-haiku-20240307",
                "name": "Claude 3 Haiku",
                "context_window": 200000,
                "provider": "anthropic"
            }
        ]
        
        return static_models