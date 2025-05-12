#!/usr/bin/env python3
"""
Test script for validating the full Agent functionality independent of the web interface.
"""

import os
import sys
import logging
import json
import time
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import agent modules - we need to make sure they're properly initialized
try:
    from agent.models import VeniceClient
    from agent.memory import MemoryManager
    from agent.core import Agent
except ImportError as e:
    logger.error(f"Failed to import agent modules: {e}")
    logger.error("Make sure you're running this from the root directory")
    sys.exit(1)

def create_test_venice_client():
    """Create and test a Venice API client"""
    logger.info("Creating test Venice client...")
    
    venice_api_key = os.environ.get("VENICE_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not venice_api_key:
        logger.error("VENICE_API_KEY environment variable not set")
        return None
    
    # Use the updated base URL
    base_url = "https://api.venice.ai/api/v1"
    
    try:
        client = VeniceClient(
            api_key=venice_api_key, 
            openai_api_key=openai_api_key,
            base_url=base_url)
        
        # Test the connection
        if client.test_connection():
            logger.info("✅ Successfully created and connected Venice client")
            return client
        else:
            logger.error("❌ Failed to connect to Venice API")
            return None
    except Exception as e:
        logger.error(f"❌ Error creating Venice client: {str(e)}")
        return None

def create_test_memory_manager(venice_client):
    """Create and test a Memory Manager with Qdrant integration"""
    logger.info("Creating test Memory Manager...")
    
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    
    if not qdrant_url or not qdrant_api_key:
        logger.error("QDRANT_URL or QDRANT_API_KEY environment variables not set")
        return None
    
    try:
        # Create memory manager with test collection
        memory_manager = MemoryManager(
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            collection_name="test_agent_memory",
            vector_size=1536,  # Default size for most embedding models
            embedding_model="text-embedding-3-large",
            venice_client=venice_client
        )
        
        # Test storing an interaction
        store_success = memory_manager.store_interaction(
            query="This is a test query",
            response="This is a test response",
            system_prompt="You are a test agent",
            metadata={"test": True}
        )
        
        if store_success:
            logger.info("✅ Successfully created and tested Memory Manager")
            
            # Test retrieving memories
            memories = memory_manager.get_relevant_memories("test query")
            logger.info(f"Retrieved {len(memories)} relevant memories")
            
            return memory_manager
        else:
            logger.error("❌ Failed to store test interaction in Memory Manager")
            return None
    except Exception as e:
        logger.error(f"❌ Error creating Memory Manager: {str(e)}")
        return None

def create_test_agent(venice_client, memory_manager):
    """Create and test the main Agent"""
    logger.info("Creating test Agent...")
    
    try:
        # Get available models
        available_models = []
        try:
            model_data = venice_client.get_available_models()
            available_models = [model.get("id") for model in model_data if model.get("id")]
        except Exception as e:
            logger.warning(f"Failed to get available models: {e}")
            # Use fallback models
            available_models = ["venice-large-beta", "venice-medium"]
        
        logger.info(f"Using models: {available_models}")
        
        # Create agent
        agent = Agent(
            venice_client=venice_client,
            memory_manager=memory_manager,
            available_models=available_models,
            default_model=available_models[0] if available_models else "venice-large-beta"
        )
        
        logger.info("✅ Successfully created Agent")
        return agent
    except Exception as e:
        logger.error(f"❌ Error creating Agent: {str(e)}")
        return None

def test_agent_query(agent):
    """Test the agent with a query"""
    logger.info("Testing agent with a query...")
    
    system_prompt = "You are a helpful AI assistant. You provide concise and accurate answers."
    query = "What is the capital of France?"
    
    try:
        logger.info(f"Sending query to agent: '{query}'")
        start_time = time.time()
        
        response, model = agent.process_query(query, system_prompt)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Got response in {elapsed_time:.2f} seconds using model: {model}")
        logger.info(f"Response: {response}")
        
        if response and len(response) > 0:
            logger.info("✅ Successfully tested agent query")
            return True
        else:
            logger.error("❌ Empty response from agent")
            return False
    except Exception as e:
        logger.error(f"❌ Error testing agent query: {str(e)}")
        return False

def run_all_agent_tests():
    """Run all agent functionality tests"""
    logger.info("Starting agent functionality tests...")
    
    # Step 1: Create and test Venice API client
    venice_client = create_test_venice_client()
    if not venice_client:
        logger.error("⚠️  Venice client creation failed, stopping further tests")
        return False
    
    # Step 2: Create and test Memory Manager
    memory_manager = create_test_memory_manager(venice_client)
    if not memory_manager:
        logger.error("⚠️  Memory Manager creation failed, stopping further tests")
        return False
    
    # Step 3: Create and test Agent
    agent = create_test_agent(venice_client, memory_manager)
    if not agent:
        logger.error("⚠️  Agent creation failed, stopping further tests")
        return False
    
    # Step 4: Test agent query processing
    query_success = test_agent_query(agent)
    
    # Overall result
    if query_success:
        logger.info("🎉 All agent tests passed successfully")
        return True
    else:
        logger.error("⚠️  Some agent tests failed")
        return False

if __name__ == "__main__":
    success = run_all_agent_tests()
    sys.exit(0 if success else 1)