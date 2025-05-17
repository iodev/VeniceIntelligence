import logging
import json
import os
from typing import List, Dict, Any, Optional, Union
import requests

logger = logging.getLogger(__name__)

class PerplexityClient:
    """
    Client for interacting with the Perplexity API
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: str = "https://api.perplexity.ai"
    ):
        """
        Initialize the Perplexity API client
        
        Args:
            api_key: Perplexity API key
            base_url: Base URL for Perplexity API
        """
        self.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY", "")
        self.base_url = base_url
        
        # Basic validation
        if not self.api_key:
            logger.warning("No Perplexity API key provided")
        
        # Set up main session for completions
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
        # Test connection
        self.test_connection()
    
    def test_connection(self) -> bool:
        """
        Test the connection to the Perplexity API
        
        Returns:
            Whether the connection was successful
        """
        try:
            # Simple query to test the connection
            test_messages = [
                {"role": "system", "content": "Be precise and concise."},
                {"role": "user", "content": "Hello, this is a test."}
            ]
            
            response = self.generate(messages=test_messages, max_tokens=5)
            logger.info("Successfully connected to Perplexity API")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Perplexity API: {str(e)}")
            return False
    
    def generate(
        self, 
        messages: list, 
        model: str = "llama-3.1-sonar-small-128k-online",
        max_tokens: Optional[int] = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text using Perplexity API
        
        Args:
            messages: List of message objects (system, user, etc.)
            model: Model ID to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Top-p sampling parameter
            stream: Whether to stream the response
            
        Returns:
            Generated text with metadata
        """
        if not messages:
            raise ValueError("Messages cannot be empty")
            
        logger.info(f"Generating with Perplexity model: {model}")
        
        # Perplexity API payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "return_images": False,
            "return_related_questions": False,
            "frequency_penalty": 1
        }
        
        # Add max_tokens if provided
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
            
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=60  # Longer timeout for generation
            )
            
            if response.status_code != 200:
                error_msg = f"Perplexity API error: {response.status_code}"
                if hasattr(response, 'text'):
                    error_msg += f" - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            result = response.json()
            logger.debug(f"Response JSON: {result}")
            
            return result
            
        except requests.RequestException as e:
            logger.error(f"Request error with Perplexity API: {str(e)}")
            raise Exception(f"Failed to communicate with Perplexity API: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {str(e)}")
            raise Exception(f"Invalid response format from Perplexity API")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
            
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models for Perplexity
        
        Returns:
            List of model information
        """
        # Perplexity doesn't have a dedicated models endpoint
        # Return static list of known models
        return [
            {
                "id": "llama-3.1-sonar-small-128k-online",
                "name": "Llama 3.1 Sonar Small (128k)",
                "context_window": 128000,
                "provider": "perplexity"
            },
            {
                "id": "llama-3.1-sonar-large-128k-online",
                "name": "Llama 3.1 Sonar Large (128k)",
                "context_window": 128000,
                "provider": "perplexity"
            },
            {
                "id": "llama-3.1-sonar-huge-128k-online", 
                "name": "Llama 3.1 Sonar Huge (128k)",
                "context_window": 128000,
                "provider": "perplexity"
            }
        ]