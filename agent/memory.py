import logging
import time
import json
from typing import Dict, List, Any, Optional, Sequence
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
    # Create stub classes for type checking when Qdrant is not available
    class MockQdrantModels:
        class Distance:
            COSINE = "cosine"
        
        class VectorParams:
            def __init__(self, size, distance):
                self.size = size
                self.distance = distance
                
        class PointStruct:
            def __init__(self, id, vector, payload):
                self.id = id
                self.vector = vector
                self.payload = payload
    
    # Create mock models for type checking
    qdrant_models = MockQdrantModels()

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
        self.using_qdrant = False
        self.client = None
        
        # Set up local storage as fallback
        self.local_storage = []
        
        # Try to use Qdrant if available
        if QDRANT_AVAILABLE and qdrant_url:
            try:
                # Initialize with proper error handling
                from qdrant_client import QdrantClient
                from qdrant_client.http import models as qdrant_models
                
                self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                
                # Check if collection exists with the correct vector size, recreate if necessary
                collections_response = self.client.get_collections()
                if not hasattr(collections_response, 'collections'):
                    logger.warning("Unexpected response format from Qdrant get_collections()")
                    raise AttributeError("Response object has no 'collections' attribute")
                    
                collections = collections_response.collections
                collection_names = [c.name for c in collections if hasattr(c, 'name')]
                create_new_collection = False
                self.using_qdrant = True
                
                if collection_name not in collection_names:
                    logger.info(f"Collection {collection_name} not found, will create it")
                    create_new_collection = True
                else:
                    # Check if vector size matches
                    try:
                        collection_info = self.client.get_collection(collection_name=collection_name)
                        # Safely check vector size with proper type handling
                        if hasattr(collection_info, 'config') and hasattr(collection_info.config, 'params'):
                            # Default to None, will be set if we can determine size
                            current_vector_size = None
                            
                            # Safely extract vector size
                            try:
                                if hasattr(collection_info, 'config') and hasattr(collection_info.config, 'params'):
                                    # Extract config data as a dictionary to handle various structures
                                    config_dict = {}
                                    
                                    # Convert config to dictionary if it has __dict__ attribute
                                    if hasattr(collection_info, 'config'):
                                        if hasattr(collection_info.config, '__dict__'):
                                            config_dict = collection_info.config.__dict__
                                        elif isinstance(collection_info.config, dict):
                                            config_dict = collection_info.config
                                    
                                    # Try to find vector size by accessing properties safely
                                    try:
                                        # Try dict approach first
                                        if 'params' in config_dict:
                                            params = config_dict['params']
                                            
                                            # If params is a dict
                                            if isinstance(params, dict) and 'vectors' in params:
                                                vectors_data = params['vectors']
                                                
                                                # Case 1: Direct size in dict
                                                if isinstance(vectors_data, dict) and 'size' in vectors_data:
                                                    current_vector_size = int(vectors_data['size'])
                                                
                                                # Case 2: List of dicts with size
                                                elif isinstance(vectors_data, list) and len(vectors_data) > 0:
                                                    for item in vectors_data:
                                                        if isinstance(item, dict) and 'size' in item:
                                                            current_vector_size = int(item['size'])
                                                            break
                                    except (TypeError, KeyError, AttributeError, ValueError) as e:
                                        logger.debug(f"Error in dictionary approach to find vector size: {e}")
                                    
                                    # Attempt object attribute approach if needed
                                    if current_vector_size is None:
                                        try:
                                            if hasattr(collection_info.config.params, 'vectors'):
                                                # Extract vectors safely using a type-agnostic approach
                                                try:
                                                    vectors = getattr(collection_info.config.params, 'vectors')
                                                    # First, try to extract directly from a Qdrant Vector object (if provided)
                                                    if vectors:
                                                        # Method 1: Access vector size from the first vector config
                                                        # This handles the case where vectors is a dictionary of vector configs
                                                        if isinstance(vectors, dict) and len(vectors) > 0:
                                                            # Get the first vector config
                                                            vector_name = next(iter(vectors))
                                                            vector_config = vectors[vector_name]
                                                            
                                                            # Try to extract size as a number
                                                            size_value = None
                                                            
                                                            # Try dictionary-style access
                                                            if isinstance(vector_config, dict) and 'size' in vector_config:
                                                                size_value = vector_config['size']
                                                            # Try attribute access
                                                            elif hasattr(vector_config, 'size'):
                                                                size_value = getattr(vector_config, 'size')
                                                                
                                                            # Convert to int if possible
                                                            if size_value is not None:
                                                                if isinstance(size_value, (int, float)):
                                                                    current_vector_size = int(size_value)
                                                                elif isinstance(size_value, str) and size_value.isdigit():
                                                                    current_vector_size = int(size_value)
                                                except Exception as extract_error:
                                                    logger.debug(f"Error extracting vector size from vectors object: {extract_error}")
                                        except (AttributeError, ValueError, TypeError) as e:
                                            logger.debug(f"Error in attribute approach to find vector size: {e}")
                                    
                                    # Log what we found for debugging
                                    logger.debug(f"Extracted vector size: {current_vector_size}")
                            except Exception as e:
                                logger.warning(f"Error accessing vector size: {e}")
                                
                            if current_vector_size is None:
                                logger.warning("Could not determine vector size from collection info")
                                create_new_collection = True
                            
                            # Only check if we were able to determine the size
                            if current_vector_size is not None and current_vector_size != vector_size:
                                logger.warning(f"Collection {collection_name} has vector size {current_vector_size} but {vector_size} is required")
                                logger.info(f"Recreating collection {collection_name} with correct vector size")
                                
                                # Only try to delete if client exists
                                if self.client is not None:
                                    self.client.delete_collection(collection_name=collection_name)
                                create_new_collection = True
                        else:
                            logger.warning("Collection info doesn't have expected structure")
                            create_new_collection = True
                    except Exception as e:
                        logger.error(f"Error checking collection vector size: {e}")
                        create_new_collection = True
                
                if create_new_collection:
                    logger.info(f"Creating collection {collection_name} with vector size {vector_size}")
                    try:
                        if QDRANT_AVAILABLE:
                            # Use VectorParams from the correct module
                            from qdrant_client.models import VectorParams, Distance
                            self.client.create_collection(
                                collection_name=collection_name,
                                vectors_config=VectorParams(
                                    size=vector_size,
                                    distance=Distance.COSINE
                                )
                            )
                        else:
                            # Using mock classes (should not reach this in normal flow)
                            logger.warning("Using mock Qdrant classes - this should not happen")
                            self.using_qdrant = False
                            self._init_local_storage()
                    except Exception as e:
                        logger.error(f"Failed to create collection: {e}")
                        self.using_qdrant = False
                        self._init_local_storage()
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
                if self.client is not None:
                    # Create point with the appropriate model class
                    if QDRANT_AVAILABLE:
                        from qdrant_client.http import models as qdrant_models
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
                        # In case we're using MockQdrantModels (shouldn't happen normally)
                        logger.warning("Using mock Qdrant client which cannot store points")
                else:
                    raise ValueError("Qdrant client is not initialized but using_qdrant is True")
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
            if self.using_qdrant and self.client is not None:
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=limit
                )
                
                # Ensure we properly handle the response
                if search_result is not None and len(search_result) > 0:
                    # Handle different point types by safely extracting payload
                    result_list: List[Dict[str, Any]] = []
                    for point in search_result:
                        if hasattr(point, 'payload') and isinstance(point.payload, dict):
                            result_list.append(point.payload)
                        elif isinstance(point, dict) and 'payload' in point and isinstance(point['payload'], dict):
                            result_list.append(point['payload'])
                        else:
                            # Skip invalid points
                            logger.warning(f"Skipping invalid search result point: {point}")
                    return result_list
                return []
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
            if self.using_qdrant and self.client is not None:
                # Qdrant doesn't directly support sorting by payload fields
                # Get a larger batch and sort locally
                search_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    with_vectors=False
                )
                
                if search_result and len(search_result) > 0:
                    points = search_result[0]
                    # Safely extract payloads
                    memories: List[Dict[str, Any]] = []
                    for point in points:
                        if hasattr(point, 'payload') and isinstance(point.payload, dict):
                            memories.append(point.payload)
                    
                    # Sort by timestamp (descending)
                    memories.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
                    return memories[:limit]
                return []
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
    
    def clear_memories(self) -> bool:
        """
        Clear all stored memories
        
        Returns:
            Success status
        """
        try:
            if self.using_qdrant and self.client is not None:
                # Delete and recreate collection
                try:
                    self.client.delete_collection(collection_name=self.collection_name)
                except Exception as e:
                    logger.warning(f"Error deleting collection (might not exist): {str(e)}")
                
                # Create a new collection with proper error handling
                try:
                    # Ensure we're using the proper Qdrant models
                    if QDRANT_AVAILABLE:
                        from qdrant_client.models import VectorParams, Distance
                        self.client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=VectorParams(
                                size=self.vector_size,
                                distance=Distance.COSINE
                            )
                        )
                    else:
                        logger.warning("Cannot create collection: Qdrant not available")
                        self.using_qdrant = False
                        self._init_local_storage()
                        return False
                except Exception as e:
                    logger.error(f"Failed to create collection: {str(e)}")
                    self.using_qdrant = False
                    self._init_local_storage()
                    return False
            else:
                # Clear local storage
                self.local_storage = []
                self.local_id_counter = 0
            
            return True
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
            return False
