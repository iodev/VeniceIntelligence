"""
Image generation capabilities for the agent using Venice.ai API.
"""
import logging
import json
from typing import List, Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

class VeniceImageClient:
    """
    Client for accessing Venice.ai image generation capabilities.
    """
    
    def __init__(
        self, 
        api_key: str,
        base_url: str = "https://api.venice.ai/api/v1"
    ):
        """
        Initialize the Venice Image API client
        
        Args:
            api_key: Venice API key
            base_url: Base URL for Venice API
        """
        self.api_key = api_key
        self.base_url = base_url
        
        # Set up session with auth headers
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
        logger.info("Venice Image client initialized")
        
    def generate_image(
        self, 
        prompt: str, 
        model: str = "stable-diffusion-xl-1024-v1-0",
        size: str = "1024x1024",
        num_images: int = 1
    ) -> List[Dict[str, str]]:
        """
        Generate images using Venice.ai's image generation models
        
        Args:
            prompt: Text prompt describing the image to generate
            model: Image generation model to use
            size: Size of the generated image (e.g., "1024x1024")
            num_images: Number of images to generate
            
        Returns:
            List of dicts with image URLs and metadata
        """
        logger.info(f"Generating image with model: {model}")
        
        # Image generation API payload
        payload = {
            "model": model,
            "prompt": prompt,
            "n": num_images,
            "size": size
        }
        
        try:
            # Use images endpoint for Venice.ai API
            logger.info(f"Calling Venice API at: {self.base_url}/images/generations")
            
            response = self.session.post(
                f"{self.base_url}/images/generations",
                json=payload,
                timeout=60  # Longer timeout for image generation
            )
            
            if response.status_code != 200:
                error_msg = f"Venice API image generation error: {response.status_code}"
                if hasattr(response, 'text'):
                    error_msg += f" - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            result = response.json()
            logger.debug(f"Response JSON: {result}")
            
            # Extract image URLs from the response
            if "data" in result and len(result["data"]) > 0:
                return result["data"]
            
            # If we can't extract the URLs, return an error
            logger.error(f"Unexpected response format: {result}")
            raise Exception("Failed to extract image URLs from response")
        
        except requests.RequestException as e:
            logger.error(f"Request error with Venice API image generation: {str(e)}")
            raise Exception(f"Failed to communicate with Venice API: {str(e)}")
        except json.JSONDecodeError as e:
            error_text = "No response text available"
            if response is not None and hasattr(response, 'text'):
                error_text = response.text
            logger.error(f"Invalid JSON response: {error_text}")
            logger.error(f"JSON decode error: {e}")
            raise Exception(f"Invalid response format from Venice API: {error_text}")
        except Exception as e:
            logger.error(f"Unexpected error in image generation: {str(e)}")
            raise
    
    def get_available_image_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available image models via Venice API
        
        Returns:
            List of image model information
        """
        try:
            response = self.session.get(
                f"{self.base_url}/models",
                params={"type": "image"},
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error(f"Error fetching image models: {response.status_code}")
                return []
                
            result = response.json()
            
            # Filter for image models only
            models = [model for model in result.get("data", []) 
                     if model.get("capabilities", {}).get("image_generation", False)]
                
            return models
            
        except Exception as e:
            logger.error(f"Error fetching image models: {e}")
            return []