import logging
import json
from typing import List, Dict, Any, Optional
import requests
import time

logger = logging.getLogger(__name__)

class VeniceClient:
    """
    Client for interacting with the Venice.ai API
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.venice.ai/v1/openai"):
        """
        Initialize the Venice API client
        
        Args:
            api_key: Venice API key
            base_url: Base URL for Venice API (OpenAI-compatible endpoint)
        """
        self.api_key = api_key
        self.base_url = base_url
        
        # Basic validation
        if not api_key:
            logger.warning("No Venice API key provided")
        
        # Set up session
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
        
        # Test connection
        self.test_connection()
    
    def test_connection(self) -> bool:
        """
        Test the connection to the Venice API
        
        Returns:
            Whether the connection was successful
        """
        try:
            # Try to get available models as a test using OpenAI-compatible endpoint
            response = self.session.get(f"{self.base_url}/models")
            
            if response.status_code == 200:
                logger.info("Successfully connected to Venice API")
                return True
            else:
                logger.error(f"Failed to connect to Venice API: {response.status_code} {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Venice API: {str(e)}")
            return False
    
    def generate(
        self, 
        prompt: str, 
        model: str = "venice-large-beta",
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate text using Venice LLM via OpenAI-compatible API
        
        Args:
            prompt: The prompt to generate from
            model: Model ID to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Top-p sampling parameter
            stop: Optional list of stop sequences
            
        Returns:
            Generated text
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        
        # OpenAI-compatible format for chat completions
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        if stop:
            payload["stop"] = stop
        
        try:
            # Use chat completions endpoint for OpenAI compatibility
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=60  # Longer timeout for generation
            )
            
            if response.status_code != 200:
                error_msg = f"Venice API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            result = response.json()
            
            # Extract the generated text from the OpenAI-compatible response format
            generated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not generated_text:
                logger.warning("Empty response from Venice API")
            
            return generated_text
        
        except requests.RequestException as e:
            logger.error(f"Request error with Venice API: {str(e)}")
            raise Exception(f"Failed to communicate with Venice API: {str(e)}")
        except json.JSONDecodeError as e:
            if 'response' in locals():
                logger.error(f"Invalid JSON response: {response.text}")
            else:
                logger.error(f"JSON decode error: {e}")
            raise Exception("Invalid response format from Venice API")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    def get_embedding(
        self, 
        text: str, 
        model: str = "venice-embedding"
    ) -> List[float]:
        """
        Get embedding vector for text via OpenAI-compatible API
        
        Args:
            text: Text to get embedding for
            model: Embedding model to use
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text:
            raise ValueError("Text cannot be empty")
        
        # OpenAI-compatible format
        payload = {
            "model": model,
            "input": text
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/embeddings",
                json=payload
            )
            
            if response.status_code != 200:
                error_msg = f"Venice API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            result = response.json()
            
            # Extract the embedding from the OpenAI-compatible response
            embedding = result.get("data", [{}])[0].get("embedding", [])
            
            if not embedding:
                logger.warning("Empty embedding from Venice API")
                # Return zeros as fallback
                return [0.0] * 1536  # Default embedding size
            
            return embedding
        
        except requests.RequestException as e:
            logger.error(f"Request error with Venice API: {str(e)}")
            raise Exception(f"Failed to get embedding: {str(e)}")
        except json.JSONDecodeError as e:
            if 'response' in locals():
                logger.error(f"Invalid JSON response: {response.text}")
            else:
                logger.error(f"JSON decode error: {e}")
            raise Exception("Invalid response format from Venice API")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models via OpenAI-compatible API
        
        Returns:
            List of model information
        """
        try:
            response = self.session.get(f"{self.base_url}/models")
            
            if response.status_code != 200:
                error_msg = f"Venice API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            result = response.json()
            
            # Extract models list from OpenAI-compatible response
            models = result.get("data", [])
            
            return models
        
        except requests.RequestException as e:
            logger.error(f"Request error with Venice API: {str(e)}")
            raise Exception(f"Failed to get models: {str(e)}")
        except json.JSONDecodeError as e:
            if 'response' in locals():
                logger.error(f"Invalid JSON response: {response.text}")
            else:
                logger.error(f"JSON decode error: {e}")
            raise Exception("Invalid response format from Venice API")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
