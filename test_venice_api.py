#!/usr/bin/env python3
"""
Test script for validating Venice.ai API integration and agent functionality.
This script tests the core functionality independently from the implementation.
"""

import os
import sys
import logging
import json
import requests
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test constants
VENICE_API_KEY = os.environ.get("VENICE_API_KEY")
VENICE_EMBEDDINGS_API_KEY = os.environ.get("VENICE_EMBEDDINGS_API_KEY")
VENICE_BASE_URL = "https://api.venice.ai/api/v1"

def test_api_connection():
    """Test basic connection to Venice API"""
    logger.info("Testing connection to Venice API...")
    
    if not VENICE_API_KEY:
        logger.error("VENICE_API_KEY environment variable not set")
        return False
    
    headers = {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(
            f"{VENICE_BASE_URL}/models",
            headers=headers
        )
        
        if response.status_code == 200:
            logger.info("✅ Successfully connected to Venice API")
            models = response.json().get("data", [])
            logger.info(f"Available models: {', '.join([m.get('id', 'unknown') for m in models])}")
            return True
        else:
            logger.error(f"❌ Failed to connect to Venice API: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Error connecting to Venice API: {str(e)}")
        return False

def test_chat_completion(model: str = "mistral-31-24b"):
    """Test chat completion endpoint"""
    logger.info(f"Testing chat completion with model: {model}...")
    
    if not VENICE_API_KEY:
        logger.error("VENICE_API_KEY environment variable not set")
        return False
    
    headers = {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Simple test message
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hello, can you tell me about Venice.ai?"
        }
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": 250,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    try:
        logger.info(f"Calling Venice API at: {VENICE_BASE_URL}/chat/completions")
        logger.info(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            f"{VENICE_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if generated_text:
                logger.info("✅ Successfully generated text from Venice API")
                logger.info(f"Response: {generated_text[:100]}...")
                return True
            else:
                logger.error("❌ Empty response from Venice API")
                return False
        else:
            logger.error(f"❌ Venice API error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Error generating response: {str(e)}")
        return False

def test_embeddings(model: str = "text-embedding-bge-m3"):
    """Test embeddings endpoint"""
    logger.info(f"Testing embeddings with model: {model}...")
    
    # Let's try using the main API key instead of a separate embeddings key
    # Venice.ai might use the same key for all endpoints
    logger.info("Using main API key for embeddings test...")
    
    if not VENICE_API_KEY:
        logger.error("VENICE_API_KEY environment variable not set")
        return False
    
    headers = {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Test text for embedding
    text = "This is a test text for generating embeddings from Venice.ai API."
    
    payload = {
        "model": model,
        "input": text,
        "encoding_format": "float"
    }
    
    try:
        response = requests.post(
            f"{VENICE_BASE_URL}/embeddings",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            embedding = result.get("data", [{}])[0].get("embedding", [])
            
            if embedding and len(embedding) > 0:
                logger.info("✅ Successfully generated embedding from Venice API")
                logger.info(f"Embedding dimension: {len(embedding)}")
                return True
            else:
                logger.error("❌ Empty embedding from Venice API")
                return False
        else:
            logger.error(f"❌ Venice API error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Error generating embedding: {str(e)}")
        return False

def run_all_tests():
    """Run all Venice API integration tests"""
    logger.info("Starting Venice API integration tests...")
    
    # Test 1: Basic connection
    connection_success = test_api_connection()
    if not connection_success:
        logger.error("⚠️  Connection test failed, stopping further tests")
        return False
    
    # Test 2: Chat completion
    chat_success = test_chat_completion()
    
    # Test 3: Embeddings
    embedding_success = test_embeddings()
    
    # Overall result
    if connection_success and chat_success and embedding_success:
        logger.info("🎉 All tests passed successfully")
        return True
    else:
        failed = []
        if not connection_success:
            failed.append("API connection")
        if not chat_success:
            failed.append("Chat completion")
        if not embedding_success:
            failed.append("Embeddings")
        
        logger.error(f"⚠️  Some tests failed: {', '.join(failed)}")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)