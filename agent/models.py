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
    
    def __init__(self, api_key: str, base_url: str = "https://api.venice.ai/api/v1"):
        """
        Initialize the Venice API client
        
        Args:
            api_key: Venice API key
            base_url: Base URL for Venice API (native endpoint)
        """
        # Import json at the module level
        import json
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
            # Print the base URL and headers for debugging
            logger.info(f"Testing connection to Venice API at: {self.base_url}")
            logger.info(f"Request headers: {self.session.headers}")
            
            # Try to get available models as a test
            response = self.session.get(f"{self.base_url}/models")
            
            logger.info(f"Venice API response status: {response.status_code}")
            
            if response.status_code == 200:
                logger.info("Successfully connected to Venice API")
                return True
            else:
                logger.error(f"Failed to connect to Venice API: {response.status_code} {response.text}")
                logger.error(f"Full URL: {self.base_url}/models")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Venice API: {str(e)}")
            return False
    
    def generate(
        self, 
        messages: list, 
        model: str = "mistral-31-24b",
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate text using Venice.ai Chat API
        
        Args:
            messages: List of message objects (system, user, etc.)
            model: Model ID to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Top-p sampling parameter
            stop: Optional list of stop sequences
            
        Returns:
            Generated text
        """
        if not messages:
            raise ValueError("Messages cannot be empty")
            
        logger.info(f"Generating with model: {model}")
        logger.debug(f"Messages: {messages}")
            
        # Venice.ai Chat API payload
        payload = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        if stop:
            payload["stop"] = stop
        
        try:
            # Use chat endpoint for Venice.ai API
            logger.info(f"Calling Venice API at: {self.base_url}/chat/completions")
            # Store response variable at a higher scope
            response = None
            try:
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
                logger.debug(f"Response JSON: {result}")
                
                # Extract the generated text from the Venice.ai Chat API response
                generated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            except json.JSONDecodeError as e:
                error_text = "No response text available"
                if response is not None:
                    try:
                        error_text = response.text
                    except:
                        pass
                logger.error(f"Invalid JSON response: {error_text}")
                logger.error(f"JSON decode error: {e}")
                raise Exception(f"Invalid response format from Venice API: {error_text}")
            
            if not generated_text:
                logger.warning("Empty response from Venice API")
            
            return generated_text
        
        except requests.RequestException as e:
            logger.error(f"Request error with Venice API: {str(e)}")
            raise Exception(f"Failed to communicate with Venice API: {str(e)}")
        except json.JSONDecodeError as e:
            error_text = ""
            try:
                if 'response' in locals():
                    error_text = response.text
                    logger.error(f"Invalid JSON response: {error_text}")
            except:
                pass
            logger.error(f"JSON decode error: {e}")
            raise Exception(f"Invalid response format from Venice API: {error_text}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    def get_embedding(
        self, 
        text: str, 
        model: str = "llama-3.2-3b"
    ) -> List[float]:
        """
        Get embedding vector for text via native Venice API
        
        Args:
            text: Text to get embedding for
            model: Embedding model to use
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text:
            raise ValueError("Text cannot be empty")
        
        # Native Venice API format for embeddings
        payload = {
            "model": model,
            "input": text
        }
        
        try:
            # Store response variable at a higher scope
            response = None
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
                
                # Extract the embedding from the Venice API response
                embedding = result.get("data", [{}])[0].get("embedding", [])
                
                if not embedding:
                    logger.warning("Empty embedding from Venice API")
                    # Return zeros as fallback
                    return [0.0] * 1536  # Default embedding size
                
                return embedding
            except json.JSONDecodeError as e:
                error_text = "No response text available"
                if response is not None:
                    try:
                        error_text = response.text
                    except:
                        pass
                logger.error(f"Invalid JSON response: {error_text}")
                logger.error(f"JSON decode error: {e}")
                raise Exception(f"Invalid response format from Venice API: {error_text}")
        
        except requests.RequestException as e:
            logger.error(f"Request error with Venice API: {str(e)}")
            raise Exception(f"Failed to get embedding: {str(e)}")
        except json.JSONDecodeError as e:
            error_text = ""
            try:
                if 'response' in locals():
                    error_text = response.text
                    logger.error(f"Invalid JSON response: {error_text}")
            except:
                pass
            logger.error(f"JSON decode error: {e}")
            raise Exception(f"Invalid response format from Venice API: {error_text}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models via native Venice API
        
        Returns:
            List of model information
        """
        try:
            # Store response at a higher scope
            response = None
            try:
                response = self.session.get(f"{self.base_url}/models")
                
                if response.status_code != 200:
                    error_msg = f"Venice API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                
                result = response.json()
                
                # Extract models list from Venice API response
                models = result.get("data", [])
                
                return models
            except json.JSONDecodeError as e:
                error_text = "No response text available"
                if response is not None:
                    try:
                        error_text = response.text
                    except:
                        pass
                logger.error(f"Invalid JSON response: {error_text}")
                logger.error(f"JSON decode error: {e}")
                raise Exception(f"Invalid response format from Venice API: {error_text}")
        
        except requests.RequestException as e:
            logger.error(f"Request error with Venice API: {str(e)}")
            raise Exception(f"Failed to get models: {str(e)}")
        except json.JSONDecodeError as e:
            error_text = ""
            try:
                if 'response' in locals():
                    error_text = response.text
                    logger.error(f"Invalid JSON response: {error_text}")
            except:
                pass
            logger.error(f"JSON decode error: {e}")
            raise Exception(f"Invalid response format from Venice API: {error_text}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
