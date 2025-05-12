import logging
import time
import json
from typing import Dict, List, Any, Optional
import numpy as np

from agent.models import VeniceClient

# Check if Qdrant is available, otherwise use a fallback
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    from qdrant_client.http.exceptions import UnexpectedResponse
    QDRANT_AVAILABLE = True
except ImportError:
    logging.warning("Qdrant client not available, using local fallback storage")
    QDRANT_AVAILABLE = False

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages persistent memory for the agent using Qdrant vector database
    or a local fallback if Qdrant is not available.
    """
    
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        collection_name: str,
        vector_size: int,
        embedding_model: str,
        venice_client: VeniceClient
    ):
        """
        Initialize the memory manager
        
        Args:
            qdrant_url: URL of the Qdrant service
            qdrant_api_key: API key for Qdrant
            collection_name: Name of the collection to use
            vector_size: Size of embedding vectors
            embedding_model: Name of the embedding model to use
            venice_client: Venice client for generating embeddings
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.embedding_model = embedding_model
        self.venice_client = venice_client
        
        # Initialize Qdrant client if available
        if QDRANT_AVAILABLE and qdrant_url:
            try:
                self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                # Check if collection exists, create if not
                collections = self.client.get_collections().collections
                collection_names = [c.name for c in collections]
                
                if collection_name not in collection_names:
                    logger.info(f"Creating collection {collection_name}")
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=qdrant_models.VectorParams(
                            size=vector_size,
                            distance=qdrant_models.Distance.COSINE
                        )
                    )
                self.using_qdrant = True
                logger.info(f"Using Qdrant for memory persistence at {qdrant_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {str(e)}")
                self.client = None
                self.using_qdrant = False
                self._init_local_storage()
        else:
            self.client = None
            self.using_qdrant = False
            self._init_local_storage()
    
    def _init_local_storage(self):
        """Initialize local storage fallback"""
        logger.warning("Using local in-memory storage - memory will not persist")
        self.local_storage = []
        self.local_id_counter = 0
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text using OpenAI embedding model
        
        Args:
            text: Text to embed
            
        Returns:
            List of float values representing the embedding
        """
        try:
            embedding = self.venice_client.get_embedding(text, model=self.embedding_model)
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            # Return a zero vector as a fallback
            return [0.0] * self.vector_size
    
    def store_interaction(
        self, 
        query: str, 
        response: str, 
        system_prompt: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store an interaction in the memory
        
        Args:
            query: User query
            response: Agent response
            system_prompt: System prompt used
            metadata: Additional metadata for the interaction
            
        Returns:
            Success status
        """
        if not query or not response:
            return False
        
        # Combine query and response for embedding
        combined_text = f"Query: {query}\nResponse: {response}"
        embedding = self._get_embedding(combined_text)
        
        if not metadata:
            metadata = {}
            
        # Create payload
        payload = {
            "query": query,
            "response": response,
            "system_prompt": system_prompt,
            "timestamp": time.time(),
            "metadata": metadata
        }
        
        try:
            if self.using_qdrant:
                # Store in Qdrant
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        qdrant_models.PointStruct(
                            id=int(time.time() * 1000),  # Use timestamp as ID
                            vector=embedding,
                            payload=payload
                        )
                    ]
                )
            else:
                # Store in local storage
                self.local_id_counter += 1
                self.local_storage.append({
                    "id": self.local_id_counter,
                    "vector": embedding,
                    "payload": payload
                })
                # Limit local storage size
                if len(self.local_storage) > 1000:
                    self.local_storage = self.local_storage[-1000:]
            
            return True
        except Exception as e:
            logger.error(f"Error storing interaction: {str(e)}")
            return False
    
    def get_relevant_memories(
        self, 
        query: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get memories relevant to a query
        
        Args:
            query: Query to find relevant memories for
            limit: Maximum number of memories to return
            
        Returns:
            List of memory payloads
        """
        query_embedding = self._get_embedding(query)
        
        try:
            if self.using_qdrant:
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=limit
                )
                return [point.payload for point in search_result]
            else:
                # Local vector search
                if not self.local_storage:
                    return []
                
                # Calculate cosine similarities
                similarities = []
                for item in self.local_storage:
                    vec1 = np.array(query_embedding)
                    vec2 = np.array(item["vector"])
                    
                    # Compute cosine similarity
                    dot_product = np.dot(vec1, vec2)
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    
                    if norm1 == 0 or norm2 == 0:
                        similarity = 0
                    else:
                        similarity = dot_product / (norm1 * norm2)
                    
                    similarities.append((item, similarity))
                
                # Sort by similarity (descending)
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Return top matches
                return [item[0]["payload"] for item in similarities[:limit]]
        
        except Exception as e:
            logger.error(f"Error getting relevant memories: {str(e)}")
            return []
    
    def get_recent_interactions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get the most recent interactions
        
        Args:
            limit: Maximum number of interactions to return
            
        Returns:
            List of memory payloads
        """
        try:
            if self.using_qdrant:
                # Qdrant doesn't directly support sorting by payload fields
                # Get a larger batch and sort locally
                search_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    with_vectors=False
                )
                
                points = search_result[0]
                memories = [point.payload for point in points]
                
                # Sort by timestamp (descending)
                memories.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
                return memories[:limit]
            else:
                # Sort local storage by timestamp
                sorted_memories = sorted(
                    self.local_storage,
                    key=lambda x: x["payload"].get("timestamp", 0),
                    reverse=True
                )
                return [item["payload"] for item in sorted_memories[:limit]]
                
        except Exception as e:
            logger.error(f"Error getting recent interactions: {str(e)}")
            return []
    
    def clear_memory(self) -> bool:
        """
        Clear all stored memories
        
        Returns:
            Success status
        """
        try:
            if self.using_qdrant:
                # Delete and recreate collection
                self.client.delete_collection(collection_name=self.collection_name)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.vector_size,
                        distance=qdrant_models.Distance.COSINE
                    )
                )
            else:
                # Clear local storage
                self.local_storage = []
                self.local_id_counter = 0
            
            return True
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
            return False
