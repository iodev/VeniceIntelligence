import logging
import json
import time
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def format_time_ago(timestamp: float) -> str:
    """
    Format a timestamp as a human-readable "time ago" string
    
    Args:
        timestamp: Unix timestamp
        
    Returns:
        Human-readable string like "2 hours ago"
    """
    now = time.time()
    diff = now - timestamp
    
    if diff < 60:
        return "just now"
    elif diff < 3600:
        minutes = int(diff / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif diff < 86400:
        hours = int(diff / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff < 604800:
        days = int(diff / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    else:
        weeks = int(diff / 604800)
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length, adding ellipsis if truncated
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - 3] + "..."

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely load JSON string
    
    Args:
        json_str: JSON string
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else {}

def get_model_display_name(model_id: str) -> str:
    """
    Get a user-friendly display name for a model ID
    
    Args:
        model_id: Raw model ID
        
    Returns:
        User-friendly display name
    """
    # Map of model IDs to display names
    model_names = {
        "venice-large-beta": "Venice Large",
        "venice-medium-beta": "Venice Medium",
        "venice-small-beta": "Venice Small",
        "venice-embedding": "Venice Embeddings"
    }
    
    return model_names.get(model_id, model_id)
