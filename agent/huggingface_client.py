import logging
import json
import os
from typing import List, Dict, Any, Optional, Union
import requests

logger = logging.getLogger(__name__)

class HuggingFaceClient:
    """
    Client for interacting with the Hugging Face API
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: str = "https://api-inference.huggingface.co/models"
    ):
        """
        Initialize the Hugging Face API client
        
        Args:
            api_key: Hugging Face API key
            base_url: Base URL for Hugging Face Inference API
        """
        self.api_key = api_key or os.environ.get("HUGGING_FACE_API_KEY_IO", "")
        self.base_url = base_url
        
        # Basic validation
        if not self.api_key:
            logger.warning("No Hugging Face API key provided")
            self.is_available = False
            return
            
        # Set up main session for inference requests
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
        self.is_available = True
        
        # Test connection if key is available
        if self.is_available:
            self.is_available = self.test_connection()
    
    def test_connection(self) -> bool:
        """
        Test the connection to the Hugging Face API
        
        Returns:
            Whether the connection was successful
        """
        if not self.api_key:
            logger.warning("Cannot test Hugging Face API connection without API key")
            return False
            
        try:
            # Use a simple model for testing
            test_model = "gpt2"
            test_input = {"inputs": "Hello, I'm testing the connection"}
            
            response = self.query(model=test_model, payload=test_input)
            
            if response and isinstance(response, list) and len(response) > 0:
                logger.info("Successfully connected to Hugging Face API")
                return True
            else:
                logger.error(f"Failed to connect to Hugging Face API: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to Hugging Face API: {str(e)}")
            return False
    
    def query(
        self, 
        model: str,
        payload: Dict[str, Any]
    ) -> Any:
        """
        Query a Hugging Face model
        
        Args:
            model: Model ID to use (e.g., "gpt2", "mistralai/Mistral-7B-v0.1")
            payload: Input payload for the model
            
        Returns:
            Model response
        """
        if not self.is_available:
            raise Exception("Hugging Face API client is not available. Please check API key.")
            
        logger.info(f"Querying Hugging Face model: {model}")
        logger.debug(f"Payload: {payload}")
        
        try:
            response = self.session.post(
                f"{self.base_url}/{model}",
                json=payload,
                timeout=60  # Longer timeout for inference
            )
            
            if response.status_code != 200:
                error_msg = f"Hugging Face API error: {response.status_code}"
                if hasattr(response, 'text'):
                    error_msg += f" - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            result = response.json()
            logger.debug(f"Response JSON: {result}")
            
            return result
            
        except requests.RequestException as e:
            logger.error(f"Request error with Hugging Face API: {str(e)}")
            raise Exception(f"Failed to communicate with Hugging Face API: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {str(e)}")
            raise Exception(f"Invalid response format from Hugging Face API")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of recommended Hugging Face models
        
        Returns:
            List of model information
        """
        # Hugging Face has thousands of models, so we return a curated list
        # of popular models in different categories
        return [
            {
                "id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "name": "Mixtral 8x7B Instruct",
                "context_window": 32768,
                "provider": "huggingface",
                "category": "text"
            },
            {
                "id": "meta-llama/Llama-2-70b-chat-hf",
                "name": "Llama 2 70B Chat",
                "context_window": 4096,
                "provider": "huggingface",
                "category": "text"
            },
            {
                "id": "mistralai/Mistral-7B-Instruct-v0.2",
                "name": "Mistral 7B Instruct v0.2",
                "context_window": 8192,
                "provider": "huggingface",
                "category": "text"
            },
            {
                "id": "stabilityai/stable-diffusion-xl-base-1.0",
                "name": "Stable Diffusion XL Base 1.0",
                "context_window": 0,
                "provider": "huggingface",
                "category": "image"
            },
            {
                "id": "microsoft/phi-2",
                "name": "Phi-2",
                "context_window": 2048,
                "provider": "huggingface",
                "category": "text"
            },
            {
                "id": "bigcode/starcoder2-15b",
                "name": "StarCoder2 15B",
                "context_window": 16384,
                "provider": "huggingface",
                "category": "code"
            },
            {
                "id": "codellama/CodeLlama-34b-Instruct-hf",
                "name": "CodeLlama 34B Instruct",
                "context_window": 16384,
                "provider": "huggingface",
                "category": "code"
            }
        ]