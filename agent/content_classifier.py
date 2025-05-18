"""
Content classifier for determining optimal model selection based on query type
"""
import logging
import re
from typing import Dict, List, Any, Tuple, Set, Optional

logger = logging.getLogger(__name__)

class ContentClassifier:
    """
    Analyzes queries to determine optimal model selection for different content types
    
    This classifier specializes in identifying when to use specific AI capabilities
    for different types of content creation tasks (text, code, image generation).
    """
    
    # Code-specific patterns
    CODE_PATTERNS = [
        r"(create|write|generate|implement|coding|code|function|class|method)",
        r"(html|css|javascript|python|java|typescript|c\+\+|php|ruby|go|rust|swift)",
        r"(algorithm|api|app|application|database|website|webpage|component|module)",
        r"(function|class|method|constructor|variable|const|let|var|def|import|export)",
        r"(syntax|compiler|interpreter|runtime|framework|library|package|dependency)",
        r"(snippet|gist|repository|github|gitlab|bitbucket)",
        r"(programming|development|software|engineering|implementation)"
    ]
    
    # Image generation patterns
    IMAGE_PATTERNS = [
        r"(image|picture|photo|illustration|graphic|artwork|drawing|painting|sketch)",
        r"(generate|create|draw|design|render|visualize|imagine|make|produce)",
        r"(visual|visually|aesthetic|artistical|creative|imaginative)",
        r"(style|styled|artistic|photorealistic|anime|cartoon|realistic|abstract)",
        r"(scene|background|foreground|lighting|shadow|perspective|angle|view)",
        r"(portrait|landscape|cityscape|seascape|character|figure|face|person)",
        r"(color|vibrant|monochrome|black and white|pastel|neon|dark|bright)",
        r"(dall-e|midjourney|stable diffusion|generative art|AI art)"
    ]
    
    # Mathematical and technical patterns
    MATH_PATTERNS = [
        r"(equation|formula|calculation|solve|math|mathematics|algebra|calculus)",
        r"(numerical|compute|statistical|algorithm|optimization|analysis)",
        r"(proof|theorem|lemma|corollary|hypothesis|axiom|postulate)",
        r"(linear|nonlinear|differential|integral|vector|matrix|tensor)",
        r"(probability|statistics|distribution|variance|standard deviation|mean)",
        r"(geometry|geometric|topological|differential geometry|manifold)",
        r"(trigonometry|sine|cosine|tangent|radian|degree)"
    ]
    
    def __init__(self):
        """Initialize the content classifier"""
        pass
        
    def classify_query(self, query: str, system_prompt: Optional[str] = None) -> Dict[str, float]:
        """
        Classify a query to determine confidence scores for different content types
        
        Args:
            query: The user's query text
            system_prompt: Optional system prompt for additional context
            
        Returns:
            Dictionary with confidence scores for each content type 
            (text, code, image, math) with values between 0.0 and 1.0
        """
        # Normalize the query for analysis
        normalized_query = query.lower()
        
        # Initialize default scores
        scores = {
            "text": 0.6,  # Default content type
            "code": 0.0,
            "image": 0.0,
            "math": 0.0
        }
        
        # Check for code-related patterns
        code_matches = self._count_pattern_matches(normalized_query, self.CODE_PATTERNS)
        code_score = min(1.0, code_matches * 0.2)  # Scale up to max 1.0
        
        # Check for image-related patterns
        image_matches = self._count_pattern_matches(normalized_query, self.IMAGE_PATTERNS)
        image_score = min(1.0, image_matches * 0.2)  # Scale up to max 1.0
        
        # Check for math-related patterns
        math_matches = self._count_pattern_matches(normalized_query, self.MATH_PATTERNS)
        math_score = min(1.0, math_matches * 0.2)  # Scale up to max 1.0
        
        # Assign scores
        scores["code"] = code_score
        scores["image"] = image_score
        scores["math"] = math_score
        
        # If both system prompt and query have strong code indicators, reinforce code score
        if system_prompt and code_score > 0.3:
            system_prompt_code_matches = self._count_pattern_matches(
                system_prompt.lower(), self.CODE_PATTERNS
            )
            if system_prompt_code_matches > 1:
                scores["code"] = min(1.0, scores["code"] + 0.2)
                
        # Check for explicit requests in the query text
        if re.search(r'\b(generate|create|make|produce) (an |a )?(image|picture|illustration|drawing|visual)', 
                    normalized_query, re.IGNORECASE):
            scores["image"] = max(scores["image"], 0.9)  # Strong indicator for image creation
            
        if re.search(r'\b(write|generate|create|implement)( some| a)? (code|function|class|program|script)',
                    normalized_query, re.IGNORECASE):
            scores["code"] = max(scores["code"], 0.9)  # Strong indicator for code creation
            
        # Length-based adjustments
        if len(normalized_query) > 300:  # Long query
            scores["text"] += 0.1  # Favor text for long, detailed queries
            
        # Normalize text score based on other scores
        max_specialized_score = max(scores["code"], scores["image"], scores["math"])
        if max_specialized_score > 0.5:
            scores["text"] = max(0.3, 1.0 - max_specialized_score)
            
        # Ensure all scores are within bounds 0.0-1.0
        for key in scores:
            scores[key] = max(0.0, min(1.0, scores[key]))
            
        logger.debug(f"Content type scores: {scores}")
        return scores
    
    def get_optimal_provider(self, content_scores: Dict[str, float], 
                           available_providers: List[str]) -> Tuple[str, str]:
        """
        Determine the optimal provider based on content classification
        
        Args:
            content_scores: Scores for different content types
            available_providers: List of available provider names
            
        Returns:
            Tuple of (provider_name, content_type) with the best match
        """
        # Default providers by content type (in order of preference)
        content_provider_map = {
            "code": ["openai", "anthropic", "perplexity", "venice", "huggingface"],
            "image": ["openai", "venice"],
            "math": ["anthropic", "openai", "perplexity", "venice", "huggingface"],
            "text": ["venice", "anthropic", "perplexity", "openai", "huggingface"]
        }
        
        # Find the dominant content type
        max_score = 0.0
        dominant_type = "text"  # Default
        
        for content_type, score in content_scores.items():
            if score > max_score:
                max_score = score
                dominant_type = content_type
                
        # Get preferred providers for the dominant content type
        preferred_providers = content_provider_map[dominant_type]
        
        # Find the best available provider
        for provider in preferred_providers:
            if provider in available_providers:
                return (provider, dominant_type)
                
        # Fallback to any available provider
        if available_providers:
            return (available_providers[0], dominant_type)
            
        # Ultimate fallback
        return ("venice", dominant_type)
    
    def get_optimal_model(self, provider: str, content_type: str, 
                         available_models: List[Dict[str, Any]]) -> Optional[str]:
        """
        Get the optimal model for a specific provider and content type
        
        Args:
            provider: Provider name
            content_type: Content type (text, code, image, math)
            available_models: List of available model information dictionaries
            
        Returns:
            Model ID of the best model, or None if no suitable model is found
        """
        # Filter models by provider
        provider_models = [m for m in available_models if m.get("provider") == provider]
        if not provider_models:
            return None
            
        # For image content type, select models with image capability
        if content_type == "image":
            image_models = [m for m in provider_models 
                           if "image" in (m.get("capabilities") or [])]
            if image_models:
                # Prefer more advanced image models
                return image_models[0].get("id")
                
        # For code content type, prefer models good at code
        if content_type == "code":
            code_models = [m for m in provider_models 
                          if "code" in (m.get("capabilities") or [])]
            if code_models:
                # Sort by context window size (larger is better for code)
                sorted_models = sorted(code_models, 
                                      key=lambda m: m.get("context_window", 0), 
                                      reverse=True)
                return sorted_models[0].get("id")
                
        # For math content type, prefer models with larger context windows
        if content_type == "math":
            # Sort by context window size (larger is better for math)
            sorted_models = sorted(provider_models, 
                                 key=lambda m: m.get("context_window", 0), 
                                 reverse=True)
            return sorted_models[0].get("id")
            
        # For general text, default to the first available model for the provider
        return provider_models[0].get("id")
    
    def _count_pattern_matches(self, text: str, patterns: List[str]) -> int:
        """
        Count the number of pattern matches in text
        
        Args:
            text: Text to check
            patterns: List of regex patterns
            
        Returns:
            Number of matches found
        """
        match_count = 0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            match_count += len(matches)
            
        return match_count