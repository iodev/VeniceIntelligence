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
    is_available = db.Column(Boolean, default=True, index=True)  # Whether the model is still available from provider
    capabilities = db.Column(String(200), default="text", nullable=False)  # text, code, image, etc.
    context_window = db.Column(Integer, default=8192)  # Token context window
    cost_per_1k_tokens = db.Column(Float, default=0.0)  # Cost per 1k tokens in USD
    display_name = db.Column(String(100))  # Human-readable name
    created_at = db.Column(DateTime, default=datetime.utcnow)
    updated_at = db.Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __init__(self, model_id=None, provider="venice", capabilities="text", 
                 context_window=8192, cost_per_1k_tokens=0.0, display_name=None):
        """
        Initialize a model performance record
        
        Args:
            model_id: Unique identifier for the model
            provider: Provider of the model (venice, anthropic, perplexity, etc.)
            capabilities: Comma-separated list of capabilities (text, code, image)
            context_window: Token context window size
            cost_per_1k_tokens: Cost per 1000 tokens in USD
            display_name: Human-readable model name for UI display
        """
        self.model_id = model_id
        self.provider = provider
        self.capabilities = capabilities
        self.context_window = context_window
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.display_name = display_name
        
        # Initialize counters and metrics
        self.total_calls = 0
        self.successful_calls = 0
        self.total_latency = 0.0
        self.quality_score = 0.0
        self.quality_evaluations = 0
        self.is_current = False
        self.is_available = True
    
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
    
    def __init__(self, model_id, prompt, image_url, size="1024x1024"):
        """
        Initialize an image generation record
        
        Args:
            model_id: ID of the model used to generate the image
            prompt: Text prompt used to generate the image
            image_url: URL where the generated image is stored
            size: Image dimensions in format 'WIDTHxHEIGHT'
        """
        self.model_id = model_id
        self.prompt = prompt
        self.image_url = image_url
        self.size = size


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
    agent_id = db.Column(String(100), nullable=True, index=True)  # Which agent used this model

    # Token efficiency metrics
    tokens_per_dollar_input = db.Column(Float, default=0.0)  # Input tokens per dollar
    tokens_per_dollar_output = db.Column(Float, default=0.0)  # Output tokens per dollar
    tokens_per_dollar_total = db.Column(Float, default=0.0)  # Total tokens per dollar
    
    def __init__(self, model_id=None, provider="venice", request_tokens=0, response_tokens=0,
                 cost=0.0, request_type="chat", query_id=None, agent_id=None):
        """
        Initialize a usage cost record
        
        Args:
            model_id: ID of the model used
            provider: Provider of the model (venice, anthropic, perplexity, etc.)
            request_tokens: Number of tokens in the request
            response_tokens: Number of tokens in the response
            cost: Cost in USD
            request_type: Type of request (chat, embedding, image)
            query_id: Optional ID to group related requests
            agent_id: Optional ID to identify which agent used this model
        """
        self.model_id = model_id
        self.provider = provider
        self.request_tokens = request_tokens
        self.response_tokens = response_tokens
        self.total_tokens = request_tokens + response_tokens
        self.cost = cost
        self.request_type = request_type
        self.query_id = query_id
        self.agent_id = agent_id
        
        # Calculate token efficiency metrics if cost > 0
        if cost > 0:
            self.tokens_per_dollar_input = request_tokens / cost if request_tokens > 0 else 0
            self.tokens_per_dollar_output = response_tokens / cost if response_tokens > 0 else 0
            self.tokens_per_dollar_total = self.total_tokens / cost if self.total_tokens > 0 else 0
        else:
            self.tokens_per_dollar_input = 0
            self.tokens_per_dollar_output = 0
            self.tokens_per_dollar_total = 0
    
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
    
    def __init__(self, model_id=None, provider="venice", task_type="general",
                 avg_time_to_first_token=0.0, avg_tokens_per_second=0.0, 
                 avg_cost_per_100_tokens=0.0, accuracy_score=0.0,
                 user_satisfaction=0.0, samples_count=0):
        """
        Initialize a model efficiency record
        
        Args:
            model_id: Unique identifier for the model
            provider: Provider of the model (venice, anthropic, perplexity, etc.)
            task_type: Type of task this efficiency data is for (general, code, creative, etc.)
            avg_time_to_first_token: Average time to first token in seconds
            avg_tokens_per_second: Average tokens generated per second
            avg_cost_per_100_tokens: Average cost per 100 tokens in USD
            accuracy_score: Accuracy score (0-1 scale)
            user_satisfaction: User satisfaction score (0-1 scale)
            samples_count: Number of samples used to calculate averages
        """
        self.model_id = model_id
        self.provider = provider
        self.task_type = task_type
        self.avg_time_to_first_token = avg_time_to_first_token
        self.avg_tokens_per_second = avg_tokens_per_second
        self.avg_cost_per_100_tokens = avg_cost_per_100_tokens
        self.accuracy_score = accuracy_score
        self.user_satisfaction = user_satisfaction
        self.samples_count = samples_count
    
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
    
    def __init__(self, name="Default Strategy", description=None, daily_budget=1.0,
                 prioritize_cost=0.3, prioritize_speed=0.3, prioritize_accuracy=0.4,
                 cost_threshold=0.8, text_task_mapping=None, code_task_mapping=None,
                 image_task_mapping=None, fallback_model="llama-3.2-3b", active=True):
        """
        Initialize a cost control strategy
        
        Args:
            name: Name of the strategy
            description: Description of the strategy
            daily_budget: Daily budget in USD
            prioritize_cost: Weight for cost optimization (0-1)
            prioritize_speed: Weight for speed optimization (0-1)
            prioritize_accuracy: Weight for accuracy optimization (0-1)
            cost_threshold: Threshold (% of budget) to trigger cost saving mode
            text_task_mapping: JSON mapping of text task types to models
            code_task_mapping: JSON mapping of code task types to models
            image_task_mapping: JSON mapping of image task types to models
            fallback_model: Low-cost fallback model
            active: Whether this strategy is active
        """
        self.name = name
        self.description = description
        self.daily_budget = daily_budget
        self.current_spending = 0.0
        self.budget_reset_at = datetime.utcnow()
        
        self.prioritize_cost = prioritize_cost
        self.prioritize_speed = prioritize_speed
        self.prioritize_accuracy = prioritize_accuracy
        self.cost_threshold = cost_threshold
        
        self.text_task_mapping = text_task_mapping or '{}'
        self.code_task_mapping = code_task_mapping or '{}'
        self.image_task_mapping = image_task_mapping or '{}'
        
        self.fallback_model = fallback_model
        self.active = active