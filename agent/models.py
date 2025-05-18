import logging
import json
from typing import List, Dict, Any, Optional, Union, Iterator
import requests
import time
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

class VeniceClient:
    """
    Client for interacting with the Venice.ai API with OpenAI for embeddings
    """
    
    def __init__(
        self, 
        api_key: str, 
        openai_api_key: Optional[str] = None,
        base_url: str = "https://api.venice.ai/api/v1"
    ):
        """
        Initialize the Venice API client with OpenAI for embeddings
        
        Args:
            api_key: Venice API key for chat/completion endpoints
            openai_api_key: Optional OpenAI API key for embeddings
            base_url: Base URL for Venice API (native endpoint)
        """
        self.api_key = api_key
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        
        # Basic validation
        if not api_key:
            logger.warning("No Venice API key provided")
        
        if not self.openai_api_key:
            logger.warning("No OpenAI API key provided for embeddings")
        
        # Set up main session for chat completions
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
        
        # Initialize OpenAI client for embeddings
        try:
            if self.openai_api_key:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI client initialized for embeddings")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            self.openai_client = None
        
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
        stop: Optional[List[str]] = None,
        stream: bool = False
    ) -> Any:
        """
        Generate text using Venice.ai Chat API
        
        Args:
            messages: List of message objects (system, user, etc.)
            model: Model ID to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Top-p sampling parameter
            stop: Optional list of stop sequences
            stream: Whether to stream the response
            
        Returns:
            Generated text or stream iterator if stream=True
        """
        if not messages:
            raise ValueError("Messages cannot be empty")
            
        logger.info(f"Generating with model: {model} (stream={stream})")
        
        # Use streaming endpoint if requested
        if stream:
            return self.generate_stream(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop
            )
            
        # For non-streaming requests:
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
        
        response = None
        try:
            # Use chat endpoint for Venice.ai API
            logger.info(f"Calling Venice API at: {self.base_url}/chat/completions")
            
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=60  # Longer timeout for generation
            )
            
            if response.status_code != 200:
                error_msg = f"Venice API error: {response.status_code}"
                if hasattr(response, 'text'):
                    error_msg += f" - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            result = response.json()
            logger.debug(f"Response JSON: {result}")
            
            # Extract the generated text from the Venice.ai Chat API response
            generated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not generated_text:
                logger.warning("Empty response from Venice API")
            
            return generated_text
        
        except requests.Timeout as e:
            logger.error(f"Timeout error with Venice API: {str(e)}")
            raise Exception(f"Venice API request timed out: {str(e)}")
        except requests.RequestException as e:
            logger.error(f"Request error with Venice API: {str(e)}")
            raise Exception(f"Failed to communicate with Venice API: {str(e)}")
        except json.JSONDecodeError as e:
            error_text = "No response text available"
            if response is not None and hasattr(response, 'text'):
                error_text = response.text
            logger.error(f"Invalid JSON response: {error_text}")
            logger.error(f"JSON decode error: {e}")
            raise Exception(f"Invalid response format from Venice API: {error_text}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    def _generate_streaming(
        self,
        messages: list,
        model: str = "mistral-31-24b",
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ):
        """
        Generate text using Venice.ai Chat API with streaming support
        
        Args:
            messages: List of message objects (system, user, etc.)
            model: Model ID to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Top-p sampling parameter
            stop: Optional list of stop sequences
            
        Yields:
            Chunks of generated text as they become available
        """
        if not messages:
            raise ValueError("Messages cannot be empty")
            
        logger.info(f"Generating with streaming using model: {model}")
        
        # Venice.ai Chat API payload
        payload = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True
        }
        
        if stop:
            payload["stop"] = stop
        
        try:
            # Use chat endpoint for Venice.ai API with streaming
            logger.info(f"Calling Venice API streaming at: {self.base_url}/chat/completions")
            
            with self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=60,  # Longer timeout for generation
                stream=True
            ) as response:
                if response.status_code != 200:
                    error_msg = f"Venice API streaming error: {response.status_code}"
                    if hasattr(response, 'text'):
                        error_msg += f" - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                
                # Process the streamed response with timeout handling
                accumulated_text = ""
                start_time = time.time()
                timeout_seconds = 120  # 2-minute timeout for streaming
                
                for line in response.iter_lines():
                    # Check for timeout
                    if time.time() - start_time > timeout_seconds:
                        logger.warning(f"Streaming response timeout after {timeout_seconds} seconds")
                        break
                        
                    if line:
                        try:
                            # Remove 'data: ' prefix if present (SSE format)
                            if line.startswith(b'data: '):
                                line = line[6:]
                            
                            # Skip empty lines or [DONE] markers
                            if not line or line == b'[DONE]':
                                continue
                                
                            # Safely decode and parse JSON
                            try:
                                line_str = line.decode('utf-8') if isinstance(line, bytes) else line
                                chunk_data = json.loads(line_str)
                            except UnicodeDecodeError as e:
                                logger.warning(f"Unicode decode error in streaming response: {e}")
                                continue
                            except json.JSONDecodeError as e:
                                logger.warning(f"JSON decode error in streaming response: {e}, line: {line}")
                                continue
                            
                            # Safely extract the delta text with appropriate error handling
                            try:
                                if not chunk_data or "choices" not in chunk_data or not chunk_data["choices"]:
                                    logger.warning(f"Missing choices in streaming response chunk: {chunk_data}")
                                    continue

                                choice = chunk_data["choices"][0]
                                if not choice or "delta" not in choice:
                                    logger.warning(f"Missing delta in streaming response choice: {choice}")
                                    continue
                                    
                                delta = choice.get("delta", {}).get("content", "")
                                if delta:
                                    accumulated_text += delta
                                    yield delta
                            except (KeyError, IndexError, TypeError) as e:
                                logger.warning(f"Error extracting delta from chunk: {e}, chunk_data: {chunk_data}")
                                continue
                        except json.JSONDecodeError as e:
                            logger.warning(f"Could not parse streaming response line: {line}, error: {e}")
                            continue
                        except Exception as e:
                            logger.warning(f"Error processing streaming response: {e}, line: {line}")
                            continue
                
                # Check if we received any content
                if not accumulated_text:
                    logger.warning("Empty accumulated response from Venice API streaming")
                    # Try to yield a fallback message to prevent UI from getting stuck
                    yield "I apologize, but the response stream ended unexpectedly. Please try again."
                
                # Success signal for complete streams
                yield "[DONE]"
        
        except (requests.ConnectionError, requests.Timeout) as e:
            error_type = "connection" if isinstance(e, requests.ConnectionError) else "timeout"
            logger.error(f"{error_type.capitalize()} error with Venice API streaming: {str(e)}")
            
            # Yield appropriate user-friendly error message
            if isinstance(e, requests.ConnectionError):
                yield "I apologize, but there was a problem connecting to the AI service. Please try again in a moment."
            else:  # Timeout
                yield "I apologize, but the request to the AI service timed out. Please try again with a simpler query."
                
            yield "[DONE]"
            raise Exception(f"Venice API streaming {error_type} error: {str(e)}")
        except requests.RequestException as e:
            logger.error(f"Request error with Venice API streaming: {str(e)}")
            # Yield a user-friendly error message before raising the exception
            yield "I apologize, but there was a problem with the request to the AI service. Please try again."
            yield "[DONE]"
            raise Exception(f"Failed to communicate with Venice API streaming: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in streaming: {str(e)}")
            # Yield a user-friendly unexpected error message
            yield "I apologize, but an unexpected error occurred. Please try again."
            yield "[DONE]"
            raise
    
    def generate_stream(
        self, 
        messages: list, 
        model: str = "mistral-31-24b",
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> Iterator[str]:
        """
        Generate text using Venice.ai Chat API with streaming support
        
        Args:
            messages: List of message objects (system, user, etc.)
            model: Model ID to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Top-p sampling parameter
            stop: Optional list of stop sequences
            
        Returns:
            Iterator yielding chunks of generated text
        """
        # Use the internal streaming method
        return self._generate_streaming(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop
        )
        
    def get_embedding(
        self, 
        text: str, 
        model: str = "text-embedding-3-large"
    ) -> List[float]:
        """
        Get embedding vector for text via OpenAI embedding API
        
        Args:
            text: Text to get embedding for
            model: Embedding model to use (OpenAI model name)
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text:
            raise ValueError("Text cannot be empty")
        
        if not self.openai_client:
            logger.error("OpenAI client not initialized, cannot generate embeddings")
            # Return zeros as fallback (not ideal, but prevents crashing)
            return [0.0] * 3072  # text-embedding-3-large size
        
        try:
            # the newest OpenAI model is "text-embedding-3-large" which was released in 2024
            # do not change this unless explicitly requested by the user
            response = self.openai_client.embeddings.create(
                model=model,
                input=text,
                encoding_format="float"
            )
            
            # Extract the embedding from the OpenAI response
            embedding = response.data[0].embedding
            
            if not embedding:
                logger.warning("Empty embedding from OpenAI API")
                # Return zeros as fallback
                return [0.0] * 3072  # text-embedding-3-large size
            
            logger.info(f"Successfully generated embedding with OpenAI model {model}, dimension: {len(embedding)}")
            return embedding
        
        except Exception as e:
            logger.error(f"Error generating embedding with OpenAI: {str(e)}")
            # Return zeros as fallback
            return [0.0] * 3072  # text-embedding-3-large size
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models via native Venice API
        
        Returns:
            List of model information
        """
        response = None
        try:
            response = self.session.get(f"{self.base_url}/models")
            
            if response.status_code != 200:
                error_msg = f"Venice API error: {response.status_code}"
                if hasattr(response, 'text'):
                    error_msg += f" - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            result = response.json()
            
            # Extract models list from Venice API response
            models = result.get("data", [])
            
            return models
        
        except requests.RequestException as e:
            logger.error(f"Request error with Venice API: {str(e)}")
            raise Exception(f"Failed to get models: {str(e)}")
        except json.JSONDecodeError as e:
            error_text = "No response text available"
            if response is not None and hasattr(response, 'text'):
                error_text = response.text
            logger.error(f"Invalid JSON response: {error_text}")
            logger.error(f"JSON decode error: {e}")
            raise Exception(f"Invalid response format from Venice API: {error_text}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise