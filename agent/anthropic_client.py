import logging
import json
import os
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
        base_url: str = "https://api.anthropic.com/v1"
    ):
        """
        Initialize the Anthropic API client
        
        Args:
            api_key: Anthropic API key
            base_url: Base URL for Anthropic API
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = base_url
        
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
            # Simple query to test the connection with minimal tokens
            test_messages = [
                {"role": "user", "content": "Hello, this is a test."}
            ]
            
            response = self.generate(
                messages=test_messages, 
                model="claude-3.5-sonnet-20241022",
                max_tokens=5
            )
            logger.info("Successfully connected to Anthropic API")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Anthropic API: {str(e)}")
            return False
    
    def generate(
        self, 
        messages: list, 
        model: str = "claude-3.5-sonnet-20241022",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text using Anthropic API
        
        Args:
            messages: List of message objects (user, assistant)
            model: Model ID to use
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
                error_msg = f"Anthropic API error: {response.status_code}"
                if hasattr(response, 'text'):
                    error_msg += f" - {response.text}"
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
        # Use static list of known models
        # The newest Anthropic model is "claude-3.5-sonnet-20241022" which was released October a22, 2024
        return [
            {
                "id": "claude-3.5-sonnet-20241022",
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