import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Venice.ai API Configuration
VENICE_API_KEY = os.getenv("VENICE_API_KEY", "")
# Venice.ai native API endpoint (not using OpenAI compatibility layer)
VENICE_API_BASE_URL = "https://api.venice.ai/v1"

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "agent_memory")
QDRANT_VECTOR_SIZE = 1536  # Default for most embedding models

# Agent Configuration
DEFAULT_MODEL = "venice-large-beta"
AVAILABLE_MODELS = [
    "venice-large-beta",
    "venice-medium-beta",
    "venice-small-beta"
]
EMBEDDING_MODEL = "venice-embedding"
MODEL_EVALUATION_INTERVAL = 10  # Number of interactions before re-evaluating models
MEMORY_RETENTION_LIMIT = 1000   # Maximum number of memories to retain
