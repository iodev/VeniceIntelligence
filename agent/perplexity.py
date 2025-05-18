"""
Perplexity API client for the Agent system.
"""
import os
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union

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
        self.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        if not self.api_key:
            logger.warning("No Perplexity API key provided, functionality will be limited")
            
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
    def test_connection(self) -> bool:
        """
        Test the connection to the Perplexity API
        
        Returns:
            Whether the connection was successful
        """
        if not self.api_key:
            logger.error("Cannot test connection without API key")
            return False
            
        try:
            # Use a simple query to test the connection
            test_messages = [
                {"role": "system", "content": "Be precise and concise."},
                {"role": "user", "content": "Hi"}
            ]
            
            # Make a lightweight request to test connection
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": test_messages,
                "max_tokens": 5,  # Minimal tokens to reduce cost
                "temperature": 0.1,
                "stream": False
            }
            
            response = requests.post(
                url, 
                headers=self.headers,
                json=payload,
                timeout=5  # Short timeout for quick failure
            )
            
            if response.status_code == 200:
                logger.info("Successfully connected to Perplexity API")
                return True
            else:
                logger.warning(f"Perplexity API test failed: {response.status_code}, {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error testing Perplexity API connection: {str(e)}")
            return False
            
        try:
            # Simple query to test if the API key is valid
            response = self.generate(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, are you working?"}
                ],
                model="llama-3.1-sonar-small-128k-online",
                max_tokens=10
            )
            return "id" in response
        except Exception as e:
            logger.error(f"Failed to connect to Perplexity API: {e}")
            return False
    
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "llama-3.1-sonar-small-128k-online",
        max_tokens: Optional[int] = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 0,
        presence_penalty: float = 0,
        frequency_penalty: float = 1,
        stream: bool = False,
        search_domain_filter: Optional[List[str]] = None,
        search_recency_filter: Optional[str] = "month",
        return_images: bool = False,
        return_related_questions: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text using Perplexity API
        
        Args:
            messages: List of message objects (user, assistant, system)
            model: Model ID to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            presence_penalty: Presence penalty parameter
            frequency_penalty: Frequency penalty parameter
            stream: Whether to stream the response
            search_domain_filter: List of domains to search
            search_recency_filter: Time range for search
            return_images: Whether to return images
            return_related_questions: Whether to return related questions
            
        Returns:
            Generated text with metadata
        """
        if not self.api_key:
            raise ValueError("Cannot generate text without API key")
            
        # Ensure valid model choice
        valid_models = ["llama-3.1-sonar-small-128k-online", 
                        "llama-3.1-sonar-large-128k-online", 
                        "llama-3.1-sonar-huge-128k-online"]
        
        if model not in valid_models:
            logger.warning(f"Model {model} not in valid models: {valid_models}. Using llama-3.1-sonar-small-128k-online")
            model = "llama-3.1-sonar-small-128k-online"
            
        # Build request
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "stream": stream,
            "return_images": return_images,
            "return_related_questions": return_related_questions
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
            
        if search_domain_filter:
            payload["search_domain_filter"] = search_domain_filter
            
        if search_recency_filter:
            payload["search_recency_filter"] = search_recency_filter
            
        start_time = time.time()
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            if stream:
                # Return generator that yields chunks
                def generate_stream_chunks():
                    # Initialize variables for monitoring streaming
                    start_time = time.time()
                    timeout_seconds = 120  # 2-minute timeout for streaming
                    accumulated_text = ""
                    
                    with response:
                        for line in response.iter_lines():
                            # Check for timeout
                            if time.time() - start_time > timeout_seconds:
                                logger.warning(f"Streaming response timeout after {timeout_seconds} seconds")
                                yield "I apologize, but the response stream timed out. Please try again."
                                yield "[DONE]"
                                break
                                
                            if not line:
                                continue
                            
                            # Skip the "data: " prefix in SSE responses
                            if line.startswith(b'data: '):
                                line = line[6:]
                                
                            # Skip empty lines or [DONE] markers
                            if line == b'[DONE]':
                                continue
                                
                            try:
                                # Safely decode and parse JSON
                                line_str = line.decode('utf-8') if isinstance(line, bytes) else line
                                chunk_data = json.loads(line_str)
                                
                                # Extract delta content carefully
                                delta = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if delta:
                                    accumulated_text += delta
                                    yield delta
                            except UnicodeDecodeError as e:
                                logger.warning(f"Unicode decode error in streaming response: {e}")
                                continue
                            except json.JSONDecodeError:
                                logger.warning(f"Could not parse streaming response line: {line}")
                                continue
                            except Exception as e:
                                logger.warning(f"Error processing streaming response: {e}")
                                continue
                    
                    # Check if we received any content
                    if not accumulated_text:
                        logger.warning("Empty accumulated response from Perplexity API streaming")
                        # Try to yield a fallback message to prevent UI from getting stuck
                        yield "I apologize, but the response stream ended unexpectedly. Please try again."
                    
                    # Send completion signal
                    yield "[DONE]"
                
                return {"stream": generate_stream_chunks()}
            else:
                # Return the full response data
                result = response.json()
                
                # Calculate latency
                latency = time.time() - start_time
                result["_meta"] = {
                    "latency": latency
                }
                
                return result
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error while generating text with Perplexity API: {e}")
            
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
                
            raise ValueError(f"Perplexity API error: {e}")
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models for Perplexity
        
        Returns:
            List of model information
        """
        return [
            {
                "id": "llama-3.1-sonar-small-128k-online",
                "provider": "perplexity",
                "name": "Llama 3.1 Sonar Small",
                "context_length": 128000,
                "capabilities": ["text", "search"],
                "cost_per_1k_tokens": 0.00025,  # Input tokens
                "cost_per_1k_tokens_output": 0.00125  # Output tokens
            },
            {
                "id": "llama-3.1-sonar-large-128k-online",
                "provider": "perplexity",
                "name": "Llama 3.1 Sonar Large",
                "context_length": 128000,
                "capabilities": ["text", "search"],
                "cost_per_1k_tokens": 0.0008,  # Input tokens
                "cost_per_1k_tokens_output": 0.0024  # Output tokens
            },
            {
                "id": "llama-3.1-sonar-huge-128k-online",
                "provider": "perplexity",
                "name": "Llama 3.1 Sonar Huge",
                "context_length": 128000,
                "capabilities": ["text", "search"],
                "cost_per_1k_tokens": 0.0016,  # Input tokens
                "cost_per_1k_tokens_output": 0.0048  # Output tokens
            }
        ]
    
    def count_tokens(self, text: str) -> int:
        """
        Approximate token count for Perplexity models
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Very simple approximation - in production would use a proper tokenizer
        words = text.split()
        return len(words) * 4 // 3  # ~1.33 tokens per word on average
        
    def fetch_current_anthropic_models(self) -> List[Dict[str, Any]]:
        """
        Use Perplexity's web access to find current Anthropic models
        
        This is used when we encounter model name errors with Anthropic,
        allowing our system to self-heal and update model lists dynamically.
        
        Returns:
            List of current Anthropic models with their details
        """
        logger.info("Using Perplexity to fetch current Anthropic model information")
        
        try:
            # Craft a specific query to get current Claude model information
            query = """
            Please provide the complete and accurate list of all currently available Claude AI models from Anthropic that can be used with their API (as of today).
            For each model, include:
            1. The exact API model identifier string used in API calls (e.g., "claude-3-5-sonnet-20241022")
            2. The human-readable name (e.g., "Claude 3.5 Sonnet")
            3. The context window size in tokens
            
            Format your response as a clean JSON array of objects with the following keys exactly:
            - "id": the API model identifier
            - "name": the human-readable name
            - "context_window": the context window size (as an integer)
            
            Only include Claude models that are actually available for API use today. Ensure all model names are correctly formatted with hyphens and dates exactly as used in the API.
            """
            
            # Build system message that emphasizes factual correctness
            system_message = "You are an AI assistant tasked with providing only factual, accurate, and up-to-date information about Anthropic's Claude AI models. Always format model IDs exactly as they appear in Anthropic's API documentation. Your response must be in valid JSON format only."
            
            response = self.generate(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query}
                ],
                model="llama-3.1-sonar-small-128k-online",  # Using Sonar for web search capabilities
                temperature=0.1,  # Low temperature for factual response
                search_domain_filter=["anthropic.com", "docs.anthropic.com"],  # Focus on Anthropic docs
                search_recency_filter="day"  # Ensure we get recent information
            )
            
            # Extract the model information from the response
            content = response.get('choices', [{}])[0].get('message', {}).get('content', "")
            
            # Parse JSON from the content - need to extract JSON from potential text
            import re
            json_match = re.search(r'(\[\s*\{.*\}\s*\])', content, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                models_data = json.loads(json_str)
                
                # Add provider information
                for model in models_data:
                    model["provider"] = "anthropic"
                    
                logger.info(f"Successfully retrieved {len(models_data)} Anthropic models from Perplexity")
                return models_data
            else:
                # Try to parse the whole response as JSON if no match found
                try:
                    models_data = json.loads(content)
                    if isinstance(models_data, list):
                        # Add provider information
                        for model in models_data:
                            model["provider"] = "anthropic"
                        
                        logger.info(f"Successfully retrieved {len(models_data)} Anthropic models from Perplexity")
                        return models_data
                except:
                    logger.error("Failed to parse JSON from Perplexity response")
                    logger.debug(f"Raw content: {content}")
                    
            # Return empty list as a fallback to prevent None return
            return []
                    
        except Exception as e:
            logger.error(f"Error fetching Anthropic models from Perplexity: {str(e)}")
            return []
            
    def get_anthropic_models(self) -> Dict[str, Any]:
        """
        Get current Anthropic models using Perplexity's web access
        
        Returns:
            Dictionary with model information
        """
        models = self.fetch_current_anthropic_models()
        
        # Convert to dictionary format for registry
        if not models:
            # Fallback to default models if Perplexity can't fetch them
            models = [
                {
                    "id": "claude-3-5-sonnet-20241022",
                    "name": "Claude 3.5 Sonnet",
                    "context_window": 200000,
                    "provider": "anthropic"
                },
                {
                    "id": "claude-3-7-sonnet-20241022",
                    "name": "Claude 3.7 Sonnet",
                    "context_window": 200000,
                    "provider": "anthropic"
                }
            ]
        
        return {
            "models": [model["id"] for model in models],
            "details": models
        }