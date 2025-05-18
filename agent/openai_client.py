"""
OpenAI API client for the Agent system
"""
import os
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union, Generator

from openai import OpenAI

logger = logging.getLogger(__name__)

class OpenAIClient:
    """
    Client for interacting with the OpenAI API
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None
    ):
        """
        Initialize the OpenAI API client
        
        Args:
            api_key: OpenAI API key (optional, will use environment variable if not provided)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key provided, functionality will be limited")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
            
        # The newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.default_model = "gpt-4o"
        
    def test_connection(self) -> bool:
        """
        Test the connection to the OpenAI API
        
        Returns:
            Whether the connection was successful
        """
        if not self.client:
            logger.error("Cannot test connection without API key")
            return False
            
        try:
            # Use a simple query to test the connection
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "user", "content": "Hello, are you working?"}
                ],
                max_tokens=5  # Just enough to test
            )
            
            if response:
                logger.info("Successfully connected to OpenAI API")
                return True
                
        except Exception as e:
            logger.error(f"Error testing OpenAI API connection: {str(e)}")
            
        return False

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stream: bool = False,
        json_response: bool = False
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """
        Generate text using OpenAI API
        
        Args:
            messages: List of message objects (user, assistant, system)
            model: Model ID (defaults to gpt-4o if None)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Top-p sampling parameter
            frequency_penalty: Frequency penalty parameter
            presence_penalty: Presence penalty parameter
            stream: Whether to stream the response
            json_response: Whether to force JSON response format
            
        Returns:
            Generated text with metadata or a generator if streaming
        """
        if not self.client:
            raise ValueError("Cannot generate text without API key")
        
        # Use default model if none specified
        if not model:
            model = self.default_model
            
        # Build request parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": stream
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
            
        if json_response:
            params["response_format"] = {"type": "json_object"}
            
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(**params)
            
            if stream:
                # Return generator that yields chunks
                def generate_stream_chunks():
                    # Initialize variables for monitoring streaming
                    start_time = time.time()
                    timeout_seconds = 120  # 2-minute timeout for streaming
                    accumulated_text = ""
                    
                    try:
                        for chunk in response:
                            # Check for timeout
                            if time.time() - start_time > timeout_seconds:
                                logger.warning(f"Streaming response timeout after {timeout_seconds} seconds")
                                yield "I apologize, but the response stream timed out. Please try again."
                                yield "[DONE]"
                                break
                                
                            content = chunk.choices[0].delta.content
                            if content is not None:
                                accumulated_text += content
                                yield content
                    except Exception as e:
                        logger.error(f"Error in OpenAI streaming response: {str(e)}")
                        # If we have some text already, don't raise an error
                        if not accumulated_text:
                            yield "I apologize, but there was an error in the response stream."
                    
                    # Check if we received any content
                    if not accumulated_text:
                        logger.warning("Empty accumulated response from OpenAI API streaming")
                        # Try to yield a fallback message to prevent UI from getting stuck
                        yield "I apologize, but the response stream ended unexpectedly. Please try again."
                    
                    # Send completion signal
                    yield "[DONE]"
                
                return {"stream": generate_stream_chunks()}
            else:
                # Return the full response data
                content = response.choices[0].message.content
                
                # Calculate latency
                latency = time.time() - start_time
                
                return {
                    "choices": [{"message": {"content": content, "role": "assistant"}}],
                    "model": model,
                    "_meta": {
                        "latency": latency
                    }
                }
                
        except Exception as e:
            logger.error(f"Error while generating text with OpenAI API: {e}")
            raise ValueError(f"OpenAI API error: {e}")
            
    def generate_image(
        self,
        prompt: str,
        model: str = "dall-e-3",
        size: str = "1024x1024", 
        quality: str = "standard",
        n: int = 1
    ) -> Dict[str, Any]:
        """
        Generate an image using DALL-E
        
        Args:
            prompt: Text description of the desired image
            model: Model to use (dall-e-3 or dall-e-2)
            size: Image size (1024x1024, 1024x1792, 1792x1024)
            quality: Image quality (standard or hd for dall-e-3)
            n: Number of images to generate
            
        Returns:
            Generated image URLs and metadata
        """
        if not self.client:
            raise ValueError("Cannot generate image without API key")
            
        try:
            start_time = time.time()
            
            response = self.client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                n=n
            )
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Format response similar to text generation
            return {
                "data": [{"url": item.url, "revised_prompt": getattr(item, "revised_prompt", None)} 
                         for item in response.data],
                "model": model,
                "_meta": {
                    "latency": latency
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating image with DALL-E: {str(e)}")
            raise ValueError(f"OpenAI Image API error: {e}")
            
    def analyze_image(
        self,
        image_url: str,
        prompt: str,
        model: str = "gpt-4o",
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze an image using GPT-4 Vision capabilities
        
        Args:
            image_url: URL of the image to analyze (can be external URL or data URL)
            prompt: Text instruction for image analysis
            model: Model to use (must support vision)
            max_tokens: Maximum tokens in the response
            
        Returns:
            Analysis of the image with metadata
        """
        if not self.client:
            raise ValueError("Cannot analyze image without API key")
            
        if model != "gpt-4o" and not model.startswith("gpt-4-vision"):
            logger.warning(f"Model {model} may not support vision capabilities, using gpt-4o instead")
            model = "gpt-4o"
            
        try:
            start_time = time.time()
            
            # Construct the message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                }
            ]
            
            # Set up parameters
            params = {
                "model": model,
                "messages": messages
            }
            
            if max_tokens:
                params["max_tokens"] = max_tokens
                
            response = self.client.chat.completions.create(**params)
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Format response similar to text generation
            return {
                "choices": [{"message": {"content": response.choices[0].message.content, "role": "assistant"}}],
                "model": model,
                "_meta": {
                    "latency": latency
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image with OpenAI Vision: {str(e)}")
            raise ValueError(f"OpenAI Vision API error: {e}")
    
    def transcribe_audio(
        self,
        audio_file_path: str,
        model: str = "whisper-1",
        prompt: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio file
        
        Args:
            audio_file_path: Path to audio file
            model: Whisper model to use
            prompt: Optional prompt to guide transcription
            language: Optional language code
            
        Returns:
            Transcription with metadata
        """
        if not self.client:
            raise ValueError("Cannot transcribe audio without API key")
            
        try:
            start_time = time.time()
            
            with open(audio_file_path, "rb") as audio_file:
                params = {
                    "model": model,
                    "file": audio_file
                }
                
                if prompt:
                    params["prompt"] = prompt
                    
                if language:
                    params["language"] = language
                    
                response = self.client.audio.transcriptions.create(**params)
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Format response
            return {
                "text": response.text,
                "model": model,
                "_meta": {
                    "latency": latency
                }
            }
            
        except Exception as e:
            logger.error(f"Error transcribing audio with OpenAI: {str(e)}")
            raise ValueError(f"OpenAI Audio API error: {e}")
            
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models
        
        Returns:
            List of model information
        """
        if not self.client:
            logger.warning("Cannot get models without API key")
            return []
            
        # Return hard-coded available models
        # The actual API endpoint for listing models doesn't provide detailed capability info
        return [
            {
                "id": "gpt-4o",
                "provider": "openai",
                "name": "GPT-4o",
                "context_length": 128000,
                "capabilities": ["text", "code", "vision"],
                "cost_per_1k_tokens": 0.01,  # Input tokens
                "cost_per_1k_tokens_output": 0.03  # Output tokens
            },
            {
                "id": "gpt-4-turbo",
                "provider": "openai",
                "name": "GPT-4 Turbo",
                "context_length": 128000,
                "capabilities": ["text", "code"],
                "cost_per_1k_tokens": 0.01,
                "cost_per_1k_tokens_output": 0.03
            },
            {
                "id": "gpt-3.5-turbo",
                "provider": "openai",
                "name": "GPT-3.5 Turbo",
                "context_length": 16000,
                "capabilities": ["text", "code"],
                "cost_per_1k_tokens": 0.0005,
                "cost_per_1k_tokens_output": 0.0015
            },
            {
                "id": "dall-e-3",
                "provider": "openai",
                "name": "DALL-E 3",
                "context_length": 0,  # Not applicable for image models
                "capabilities": ["image"],
                "cost_per_image": 0.02  # Standard quality 1024x1024
            }
        ]