#!/usr/bin/env python3
"""
Test script for validating OpenAI embeddings integration.
"""

import os
import sys
import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_openai_embeddings():
    """Test OpenAI embeddings API"""
    logger.info("Testing OpenAI embeddings API...")
    
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return False
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)
        
        # Test text for embedding
        text = "This is a test text for generating embeddings from OpenAI API."
        
        # Get embedding
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        
        # Extract the embedding vector
        embedding = response.data[0].embedding
        
        if embedding and len(embedding) > 0:
            logger.info("✅ Successfully generated embedding from OpenAI API")
            logger.info(f"Embedding dimension: {len(embedding)}")
            logger.info(f"First 5 values: {embedding[:5]}")
            return True
        else:
            logger.error("❌ Empty embedding from OpenAI API")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error generating embedding: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_openai_embeddings()
    sys.exit(0 if success else 1)