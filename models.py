from main import db
from datetime import datetime
from sqlalchemy import Integer, String, Float, DateTime, Boolean


class ModelPerformance(db.Model):
    __tablename__ = 'model_performance'
    
    id = db.Column(Integer, primary_key=True)
    model_id = db.Column(String(100), nullable=False, index=True)
    provider = db.Column(String(50), nullable=False, default="venice", index=True)
    total_calls = db.Column(Integer, default=0)
    successful_calls = db.Column(Integer, default=0)
    total_latency = db.Column(Float, default=0.0)
    quality_score = db.Column(Float, default=0.0)
    quality_evaluations = db.Column(Integer, default=0)
    is_current = db.Column(Boolean, default=False)
    capabilities = db.Column(String(200), default="text", nullable=False)  # text, code, image, etc.
    context_window = db.Column(Integer, default=8192)  # Token context window
    cost_per_1k_tokens = db.Column(Float, default=0.0)  # Cost per 1k tokens in USD
    display_name = db.Column(String(100))  # Human-readable name
    created_at = db.Column(DateTime, default=datetime.utcnow)
    updated_at = db.Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    @property
    def success_rate(self):
        """Calculate success rate based on successful calls vs total calls"""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls
    
    @property
    def average_latency(self):
        """Calculate average latency"""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency / self.total_calls
    
    @property
    def average_quality(self):
        """Calculate average quality score"""
        if self.quality_evaluations == 0:
            return 0.0
        return self.quality_score / self.quality_evaluations


class ImageGeneration(db.Model):
    __tablename__ = 'image_generation'
    
    id = db.Column(Integer, primary_key=True)
    model_id = db.Column(String(100), nullable=False, index=True)
    prompt = db.Column(String(500), nullable=False)
    image_url = db.Column(String(500), nullable=False)
    size = db.Column(String(50), nullable=False)
    created_at = db.Column(DateTime, default=datetime.utcnow)


class UsageCost(db.Model):
    __tablename__ = 'usage_cost'
    
    id = db.Column(Integer, primary_key=True)
    model_id = db.Column(String(100), nullable=False, index=True)
    provider = db.Column(String(50), nullable=False, default="venice", index=True)
    timestamp = db.Column(DateTime, default=datetime.utcnow)
    request_tokens = db.Column(Integer, default=0)  # Input tokens
    response_tokens = db.Column(Integer, default=0)  # Output tokens
    total_tokens = db.Column(Integer, default=0)    # Total tokens processed
    cost = db.Column(Float, default=0.0)  # Cost in USD
    request_type = db.Column(String(20), default="chat")  # chat, embedding, image, etc.
    query_id = db.Column(String(100), nullable=True)  # To group related requests
    
    @classmethod
    def calculate_cost(cls, model_id, provider, request_tokens, response_tokens):
        """
        Calculate cost based on token counts and model pricing
        
        Returns:
            Cost in USD
        """
        costs = {
            "venice": {
                "mistral-31-24b": {"input": 0.0002, "output": 0.0006},
                "llama-3.2-3b": {"input": 0.0001, "output": 0.0003},
                "llama-3.3-70b": {"input": 0.0003, "output": 0.0009},
                "default": {"input": 0.0002, "output": 0.0006}
            },
            "perplexity": {
                "llama-3.1-sonar-small-128k-online": {"input": 0.0001, "output": 0.0005},
                "llama-3.1-sonar-large-128k-online": {"input": 0.0003, "output": 0.0015},
                "llama-3.1-sonar-huge-128k-online": {"input": 0.0006, "output": 0.0025},
                "default": {"input": 0.0003, "output": 0.0015}
            },
            "anthropic": {
                "claude-3.5-sonnet-20241022": {"input": 0.0003, "output": 0.0015},
                "claude-3-opus-20240229": {"input": 0.0015, "output": 0.0075},
                "claude-3-sonnet-20240229": {"input": 0.0003, "output": 0.0015},
                "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
                "default": {"input": 0.0005, "output": 0.0025}
            },
            "huggingface": {
                # Most Hugging Face models are self-hosted, so costs are approximate
                "default": {"input": 0.0001, "output": 0.0002}
            },
            "default": {"input": 0.0002, "output": 0.0006}
        }
        
        # Get provider pricing or default
        provider_costs = costs.get(provider, costs["default"])
        
        # Get model pricing or default for that provider
        model_costs = provider_costs.get(model_id, provider_costs["default"])
        
        # Calculate cost based on token counts
        input_cost = (request_tokens / 1000) * model_costs["input"]
        output_cost = (response_tokens / 1000) * model_costs["output"]
        
        return input_cost + output_cost


class ModelEfficiency(db.Model):
    __tablename__ = 'model_efficiency'
    
    id = db.Column(Integer, primary_key=True)
    model_id = db.Column(String(100), nullable=False, index=True)
    provider = db.Column(String(50), nullable=False, default="venice", index=True)
    task_type = db.Column(String(50), nullable=False, default="general")  # general, code, creative, math, etc.
    avg_time_to_first_token = db.Column(Float, default=0.0)  # Seconds
    avg_tokens_per_second = db.Column(Float, default=0.0)
    avg_cost_per_100_tokens = db.Column(Float, default=0.0)  # Cost per 100 tokens in USD
    accuracy_score = db.Column(Float, default=0.0)  # 0-1 scale
    user_satisfaction = db.Column(Float, default=0.0)  # 0-1 scale
    samples_count = db.Column(Integer, default=0)
    updated_at = db.Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    @property
    def efficiency_score(self):
        """Calculate overall efficiency score based on multiple factors"""
        # Higher score is better (1.0 is best)
        time_score = min(1.0, max(0.0, 1.0 - (self.avg_time_to_first_token / 5.0)))  # 0-5 seconds
        token_score = min(1.0, max(0.0, self.avg_tokens_per_second / 50.0))  # 0-50 tokens/sec
        
        # Lower cost is better (inverse relationship)
        max_cost_per_100 = 0.01  # $0.01 per 100 tokens as baseline
        cost_score = min(1.0, max(0.0, 1.0 - (self.avg_cost_per_100_tokens / max_cost_per_100)))
        
        # Weighted average
        return (time_score * 0.2) + (token_score * 0.3) + (cost_score * 0.2) + (self.accuracy_score * 0.3)


class CostControlStrategy(db.Model):
    __tablename__ = 'cost_control_strategy'
    
    id = db.Column(Integer, primary_key=True)
    active = db.Column(Boolean, default=True)
    name = db.Column(String(100), nullable=False)
    description = db.Column(String(500))
    daily_budget = db.Column(Float, default=1.0)  # Daily budget in USD
    current_spending = db.Column(Float, default=0.0)  # Current spending in USD
    budget_reset_at = db.Column(DateTime)  # When the budget was last reset
    
    # Model selection strategy
    prioritize_cost = db.Column(Float, default=0.3)  # 0-1 weight
    prioritize_speed = db.Column(Float, default=0.3)  # 0-1 weight
    prioritize_accuracy = db.Column(Float, default=0.4)  # 0-1 weight
    
    # Thresholds to switch models
    cost_threshold = db.Column(Float, default=0.8)  # % of budget to trigger cost saving
    
    # Task-specific model mapping (JSON stored as string)
    text_task_mapping = db.Column(String(1000), default='{}')  # JSON mapping task types to models
    code_task_mapping = db.Column(String(1000), default='{}')  # JSON mapping task types to models
    image_task_mapping = db.Column(String(1000), default='{}')  # JSON mapping task types to models
    
    fallback_model = db.Column(String(100), default="llama-3.2-3b")  # Low-cost fallback model
    created_at = db.Column(DateTime, default=datetime.utcnow)
    updated_at = db.Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)