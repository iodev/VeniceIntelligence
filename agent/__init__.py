"""
Self-learning agent package using Venice.ai API and Qdrant for persistent memory.
"""

from agent.core import Agent
from agent.memory import MemoryManager
from agent.models import VeniceClient

__all__ = ['Agent', 'MemoryManager', 'VeniceClient']
