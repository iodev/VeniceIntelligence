import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Venice.ai API Configuration
VENICE_API_KEY = os.getenv("VENICE_API_KEY", "")
# Venice.ai native API endpoint (not using OpenAI compatibility layer)
VENICE_API_BASE_URL = "https://api.venice.ai/api/v1"

# OpenAI Configuration for Embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Perplexity API Configuration
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")
PERPLEXITY_API_BASE_URL = "https://api.perplexity.ai"

# Anthropic API Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_API_BASE_URL = "https://api.anthropic.com/v1"

# Hugging Face API Configuration  
HUGGINGFACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY_IO", "")
HUGGINGFACE_API_BASE_URL = "https://api-inference.huggingface.co/models"

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "agent_memory")
QDRANT_VECTOR_SIZE = 3072  # Size for OpenAI text-embedding-3-large model

# Agent Configuration
DEFAULT_MODEL = "mistral-31-24b"
AVAILABLE_MODELS = [
    "mistral-31-24b",
    "llama-3.2-3b",
    "llama-3.3-70b"
]
EMBEDDING_MODEL = "text-embedding-3-large"  # Using OpenAI's embedding model
MODEL_EVALUATION_INTERVAL = 10  # Number of interactions before re-evaluating models
MEMORY_RETENTION_LIMIT = 1000   # Maximum number of memories to retain
