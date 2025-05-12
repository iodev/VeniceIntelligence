import logging
import re
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)

def evaluate_model_response(query: str, response: str) -> float:
    """
    Evaluate the quality of a model's response to a query
    
    This is a simple heuristic-based evaluation. In a production system,
    this would be more sophisticated, possibly using a separate LLM to evaluate.
    
    Args:
        query: The user query
        response: The model's response
        
    Returns:
        Quality score between 0 and 1
    """
    # Skip empty responses
    if not response:
        return 0.0
    
    score = 0.0
    
    # Length-based scoring (penalize very short responses)
    response_length = len(response.split())
    if response_length < 10:
        length_score = 0.3  # Very short response
    elif response_length < 25:
        length_score = 0.6  # Short response
    elif response_length < 300:
        length_score = 1.0  # Good length
    else:
        length_score = 0.8  # Very long response - slightly penalize
    
    score += length_score * 0.3  # 30% of score is length-based
    
    # Relevance-based scoring
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    query_words = {w for w in query_words if len(w) > 3}  # Filter short words
    
    response_words = set(re.findall(r'\b\w+\b', response.lower()))
    
    # Calculate word overlap
    if query_words:
        overlap = len(query_words.intersection(response_words)) / len(query_words)
        relevance_score = min(1.0, overlap * 1.5)  # Scale up, but cap at 1.0
    else:
        relevance_score = 0.5  # Neutral score for empty query
    
    score += relevance_score * 0.4  # 40% of score is relevance-based
    
    # Structure-based scoring
    structure_score = 0.7  # Default
    
    # Check for patterns of good responses
    if response.count('\n') > 2:
        structure_score += 0.1  # Formatted with paragraphs
    
    if re.search(r'\b(however|but|on the other hand)\b', response, re.IGNORECASE):
        structure_score += 0.1  # Shows nuanced thinking
    
    if re.search(r'^\s*I\'ll|I can|Let me', response, re.IGNORECASE):
        structure_score += 0.1  # Helpful opening
    
    # Cap at 1.0
    structure_score = min(1.0, structure_score)
    
    score += structure_score * 0.3  # 30% of score is structure-based
    
    logger.debug(f"Response evaluation: length={length_score}, relevance={relevance_score}, structure={structure_score}, total={score}")
    
    return score
