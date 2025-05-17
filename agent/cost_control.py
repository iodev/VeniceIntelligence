import logging
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

from sqlalchemy import func, desc
from models import ModelPerformance, UsageCost, ModelEfficiency, CostControlStrategy
from main import db

logger = logging.getLogger(__name__)

class CostMonitor:
    """
    Monitor and control costs across different AI model providers
    """
    
    def __init__(self):
        """Initialize the cost monitor with default strategies"""
        # Check for existing strategy or create default
        self._initialize_default_strategy()
        
        # Cache for quick access
        self._current_strategy = None
        self._model_efficiencies = {}
        self._daily_usage = {}
        
        # Load current strategy from database
        self.load_strategy()
    
    def _initialize_default_strategy(self):
        """Initialize default cost control strategy if none exists"""
        strategy = CostControlStrategy.query.filter_by(active=True).first()
        if not strategy:
            default_strategy = CostControlStrategy(
                name="Default Balanced Strategy",
                description="Balances cost, speed, and accuracy with slight preference for accuracy",
                daily_budget=5.0,  # $5 per day
                budget_reset_at=datetime.utcnow(),
                text_task_mapping=json.dumps({
                    "general": "llama-3.1-sonar-small-128k-online",
                    "creative": "claude-3-sonnet-20240229",
                    "analytical": "mistral-31-24b"
                }),
                code_task_mapping=json.dumps({
                    "general": "mistral-31-24b",
                    "python": "llama-3.2-3b", 
                    "javascript": "llama-3.2-3b"
                }),
                image_task_mapping=json.dumps({
                    "general": "stable-diffusion-xl-1024-v1-0"
                })
            )
            
            db.session.add(default_strategy)
            db.session.commit()
            logger.info("Created default cost control strategy")
    
    def load_strategy(self):
        """Load active cost control strategy from database"""
        self._current_strategy = CostControlStrategy.query.filter_by(active=True).first()
        if not self._current_strategy:
            logger.error("No active cost control strategy found")
            return
            
        # Check if budget should be reset (daily)
        self._check_budget_reset()
        
        # Load model efficiencies
        self._load_model_efficiencies()
        
        # Load today's usage
        self._load_daily_usage()
        
        logger.info(f"Loaded cost control strategy: {self._current_strategy.name}")
    
    def _check_budget_reset(self):
        """Check if budget needs to be reset (once per day)"""
        if not self._current_strategy:
            return
            
        now = datetime.utcnow()
        last_reset = self._current_strategy.budget_reset_at
        
        # If reset was more than 24 hours ago, reset the budget
        if last_reset and (now - last_reset) > timedelta(days=1):
            self._current_strategy.current_spending = 0.0
            self._current_strategy.budget_reset_at = now
            db.session.commit()
            logger.info(f"Reset daily budget to {self._current_strategy.daily_budget}")
    
    def _load_model_efficiencies(self):
        """Load model efficiency data for all models"""
        efficiencies = ModelEfficiency.query.all()
        for eff in efficiencies:
            key = f"{eff.provider}:{eff.model_id}:{eff.task_type}"
            self._model_efficiencies[key] = eff
    
    def _load_daily_usage(self):
        """Load today's usage data for cost tracking"""
        today = datetime.utcnow().date()
        daily_usage = UsageCost.query.filter(
            func.date(UsageCost.timestamp) == today
        ).all()
        
        # Aggregate by provider and model
        for usage in daily_usage:
            key = f"{usage.provider}:{usage.model_id}"
            if key not in self._daily_usage:
                self._daily_usage[key] = {
                    "cost": 0.0,
                    "request_tokens": 0,
                    "response_tokens": 0,
                    "requests": 0
                }
                
            self._daily_usage[key]["cost"] += usage.cost
            self._daily_usage[key]["request_tokens"] += usage.request_tokens
            self._daily_usage[key]["response_tokens"] += usage.response_tokens
            self._daily_usage[key]["requests"] += 1
    
    def record_usage(self, model_id: str, provider: str, 
                     request_tokens: int, response_tokens: int,
                     request_type: str = "chat", latency: float = 0.0,
                     query_id: Optional[str] = None, agent_id: Optional[str] = None) -> float:
        """
        Record usage and cost for a model call
        
        Args:
            model_id: Model identifier
            provider: Provider name (venice, perplexity, etc.)
            request_tokens: Number of tokens in the request
            response_tokens: Number of tokens in the response
            request_type: Type of request (chat, embedding, image)
            latency: Response latency in seconds
            query_id: Optional ID to group related requests
            agent_id: Optional ID to identify which agent used this model
            
        Returns:
            Calculated cost in USD
        """
        # Generate query ID if not provided
        if not query_id:
            query_id = str(uuid.uuid4())
            
        # Calculate cost
        cost = UsageCost.calculate_cost(model_id, provider, request_tokens, response_tokens)
        
        # Calculate token efficiency metrics
        tokens_per_dollar_input = request_tokens / cost if cost > 0 else 0
        tokens_per_dollar_output = response_tokens / cost if cost > 0 else 0
        tokens_per_dollar_total = (request_tokens + response_tokens) / cost if cost > 0 else 0
        
        # Create usage record
        usage = UsageCost(
            model_id=model_id,
            provider=provider,
            request_tokens=request_tokens,
            response_tokens=response_tokens,
            total_tokens=request_tokens + response_tokens,
            cost=cost,
            request_type=request_type,
            query_id=query_id,
            agent_id=agent_id,
            tokens_per_dollar_input=tokens_per_dollar_input,
            tokens_per_dollar_output=tokens_per_dollar_output,
            tokens_per_dollar_total=tokens_per_dollar_total
        )
        
        # Update strategy current spending
        if self._current_strategy:
            self._current_strategy.current_spending += cost
            
        # Save to database
        db.session.add(usage)
        db.session.commit()
        
        # Update daily usage cache
        key = f"{provider}:{model_id}"
        if key not in self._daily_usage:
            self._daily_usage[key] = {
                "cost": 0.0,
                "request_tokens": 0,
                "response_tokens": 0,
                "requests": 0
            }
            
        self._daily_usage[key]["cost"] += cost
        self._daily_usage[key]["request_tokens"] += request_tokens
        self._daily_usage[key]["response_tokens"] += response_tokens
        self._daily_usage[key]["requests"] += 1
        
        # Update efficiency metrics if latency is provided
        if latency > 0 and response_tokens > 0:
            self._update_efficiency_metrics(
                model_id, provider, "general", 
                latency, response_tokens, cost
            )
        
        logger.info(f"Recorded {request_tokens + response_tokens} tokens costing ${cost:.4f} for {provider}:{model_id}")
        return cost
    
    def _update_efficiency_metrics(self, model_id: str, provider: str, 
                                 task_type: str, latency: float,
                                 response_tokens: int, cost: float):
        """Update efficiency metrics for a model"""
        key = f"{provider}:{model_id}:{task_type}"
        
        # Find existing record or create new one
        efficiency = ModelEfficiency.query.filter_by(
            model_id=model_id, 
            provider=provider,
            task_type=task_type
        ).first()
        
        if not efficiency:
            efficiency = ModelEfficiency(
                model_id=model_id,
                provider=provider,
                task_type=task_type,
                samples_count=0
            )
            db.session.add(efficiency)
        
        # Calculate tokens per second
        tokens_per_second = response_tokens / latency if latency > 0 else 0
        cost_per_100_tokens = (cost / (response_tokens / 100)) if response_tokens > 0 else 0
        
        # Update rolling averages
        n = efficiency.samples_count
        
        if n == 0:
            # First sample
            efficiency.avg_time_to_first_token = latency
            efficiency.avg_tokens_per_second = tokens_per_second
            efficiency.avg_cost_per_100_tokens = cost_per_100_tokens
        else:
            # Update rolling averages
            efficiency.avg_time_to_first_token = (efficiency.avg_time_to_first_token * n + latency) / (n + 1)
            efficiency.avg_tokens_per_second = (efficiency.avg_tokens_per_second * n + tokens_per_second) / (n + 1)
            efficiency.avg_cost_per_100_tokens = (efficiency.avg_cost_per_100_tokens * n + cost_per_100_tokens) / (n + 1)
        
        efficiency.samples_count += 1
        db.session.commit()
        
        # Update cache
        self._model_efficiencies[key] = efficiency
    
    def select_model_for_task(self, task_type: str, content_type: str = "text") -> Tuple[str, str]:
        """
        Select the optimal model for a given task based on current strategy
        
        Args:
            task_type: Type of task (general, code, creative, analytical, etc.)
            content_type: Content type (text, code, image)
            
        Returns:
            Tuple of (model_id, provider)
        """
        if not self._current_strategy:
            # No strategy, use default model
            return "mistral-31-24b", "venice"
        
        # Check if we're approaching budget limit
        budget_ratio = self._current_strategy.current_spending / self._current_strategy.daily_budget
        cost_saving_mode = budget_ratio >= self._current_strategy.cost_threshold
        
        # Get model mapping based on content type
        if content_type == "code":
            mapping_str = self._current_strategy.code_task_mapping
        elif content_type == "image":
            mapping_str = self._current_strategy.image_task_mapping
        else:
            mapping_str = self._current_strategy.text_task_mapping
        
        # Parse mapping
        try:
            mapping = json.loads(mapping_str)
        except json.JSONDecodeError:
            mapping = {}
        
        # Check if we have a specific model for this task
        if task_type in mapping:
            model_id = mapping[task_type]
            
            # For Venice models, use Venice provider
            if model_id in ["mistral-31-24b", "llama-3.2-3b", "llama-3.3-70b"]:
                provider = "venice"
            # For Perplexity models
            elif "sonar" in model_id:
                provider = "perplexity"
            # For Anthropic models
            elif "claude" in model_id:
                provider = "anthropic"
            # For image models
            elif "stable-diffusion" in model_id:
                provider = "venice"
            # Hugging Face models usually contain a slash
            elif "/" in model_id:
                provider = "huggingface"
            else:
                # Default to Venice for unknown models
                provider = "venice"
                
            # If in cost saving mode, use fallback model
            if cost_saving_mode and content_type != "image":
                logger.info(f"Cost saving mode active ({budget_ratio:.1%} of budget used), using fallback model")
                return self._current_strategy.fallback_model, "venice"
                
            return model_id, provider
        
        # No specific mapping, use smart selection based on efficiency metrics
        return self._select_optimal_model(task_type, content_type, cost_saving_mode)
    
    def _select_optimal_model(self, task_type: str, content_type: str, cost_saving_mode: bool) -> Tuple[str, str]:
        """Select optimal model based on efficiency metrics and current strategy"""
        # Filter efficiencies by content type and task
        candidates = []
        
        for key, efficiency in self._model_efficiencies.items():
            provider, model_id, model_task = key.split(":")
            
            # Skip if task type doesn't match (unless it's "general")
            if model_task != "general" and model_task != task_type:
                continue
                
            # Skip image models for text tasks and vice versa
            if content_type == "image" and not ("diffusion" in model_id or "stable" in model_id):
                continue
            elif content_type != "image" and ("diffusion" in model_id or "stable" in model_id):
                continue
                
            # Calculate weighted score based on strategy priorities
            cost_weight = self._current_strategy.prioritize_cost
            speed_weight = self._current_strategy.prioritize_speed
            accuracy_weight = self._current_strategy.prioritize_accuracy
            
            # In cost saving mode, increase cost weight
            if cost_saving_mode:
                cost_weight *= 2
                # Normalize weights
                total = cost_weight + speed_weight + accuracy_weight
                cost_weight /= total
                speed_weight /= total
                accuracy_weight /= total
            
            # Cost score (1.0 is lowest cost)
            max_cost = 0.01  # $0.01 per 100 tokens reference
            cost_score = 1.0 - min(1.0, efficiency.avg_cost_per_100_tokens / max_cost)
            
            # Speed score (1.0 is fastest)
            speed_score = min(1.0, efficiency.avg_tokens_per_second / 50.0)  # 50 tokens/sec reference
            
            # Use accuracy score directly
            accuracy_score = efficiency.accuracy_score
            
            # Combined weighted score
            score = (cost_score * cost_weight + 
                    speed_score * speed_weight + 
                    accuracy_score * accuracy_weight)
                    
            candidates.append({
                "model_id": model_id,
                "provider": provider,
                "score": score,
                "efficiency": efficiency
            })
        
        # If no candidates with efficiency data, use default models
        if not candidates:
            if content_type == "image":
                return "stable-diffusion-xl-1024-v1-0", "venice"
            elif content_type == "code":
                return "mistral-31-24b", "venice"
            else:
                return "llama-3.1-sonar-small-128k-online", "perplexity"
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Return best model
        best = candidates[0]
        return best["model_id"], best["provider"]
    
    def update_model_accuracy(self, model_id: str, provider: str, 
                             task_type: str, accuracy_score: float):
        """
        Update accuracy score for a model
        
        Args:
            model_id: Model identifier
            provider: Provider name
            task_type: Type of task
            accuracy_score: Score from 0 to 1
        """
        key = f"{provider}:{model_id}:{task_type}"
        
        # Find existing record or create new one
        efficiency = ModelEfficiency.query.filter_by(
            model_id=model_id, 
            provider=provider,
            task_type=task_type
        ).first()
        
        if not efficiency:
            efficiency = ModelEfficiency(
                model_id=model_id,
                provider=provider,
                task_type=task_type,
                samples_count=1,
                accuracy_score=accuracy_score
            )
            db.session.add(efficiency)
        else:
            # Update rolling average
            n = efficiency.samples_count
            efficiency.accuracy_score = (efficiency.accuracy_score * n + accuracy_score) / (n + 1)
            efficiency.samples_count += 1
        
        db.session.commit()
        
        # Update cache
        self._model_efficiencies[key] = efficiency
        
        logger.info(f"Updated accuracy score for {provider}:{model_id} ({task_type}): {accuracy_score:.2f}")
    
    def update_user_satisfaction(self, model_id: str, provider: str,
                               task_type: str, satisfaction_score: float):
        """
        Update user satisfaction score for a model
        
        Args:
            model_id: Model identifier
            provider: Provider name
            task_type: Type of task
            satisfaction_score: Score from 0 to 1
        """
        key = f"{provider}:{model_id}:{task_type}"
        
        # Find existing record or create new one
        efficiency = ModelEfficiency.query.filter_by(
            model_id=model_id, 
            provider=provider,
            task_type=task_type
        ).first()
        
        if not efficiency:
            efficiency = ModelEfficiency(
                model_id=model_id,
                provider=provider,
                task_type=task_type,
                samples_count=1,
                user_satisfaction=satisfaction_score
            )
            db.session.add(efficiency)
        else:
            # Update rolling average
            n = efficiency.samples_count
            efficiency.user_satisfaction = (efficiency.user_satisfaction * n + satisfaction_score) / (n + 1)
            efficiency.samples_count += 1
        
        db.session.commit()
        
        # Update cache
        self._model_efficiencies[key] = efficiency
        
        logger.info(f"Updated user satisfaction for {provider}:{model_id} ({task_type}): {satisfaction_score:.2f}")
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get summary of costs across all providers,
        focusing on agent, model, and token efficiency metrics
        
        Returns:
            Dictionary with detailed cost metrics by agent and model
        """
        summary = {
            "total_spend": 0.0,
            "request_count": 0,
            "agent_costs": {},
            "model_costs": {},
            "provider_costs": {},
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "token_efficiency": {}
        }
            
        # Load all usage records
        all_usage = UsageCost.query.all()
        summary["request_count"] = len(all_usage)
        
        # Process usage data
        for usage in all_usage:
            # Add to total spend
            summary["total_spend"] += usage.cost
            
            # Add to token counts
            summary["total_tokens"] += usage.total_tokens
            summary["input_tokens"] += usage.request_tokens
            summary["output_tokens"] += usage.response_tokens
            
            # Track by agent
            agent_id = usage.agent_id or "unassigned"
            if agent_id not in summary["agent_costs"]:
                summary["agent_costs"][agent_id] = {
                    "cost": 0.0,
                    "request_tokens": 0,
                    "response_tokens": 0,
                    "tokens_per_dollar": 0.0
                }
            summary["agent_costs"][agent_id]["cost"] += usage.cost
            summary["agent_costs"][agent_id]["request_tokens"] += usage.request_tokens
            summary["agent_costs"][agent_id]["response_tokens"] += usage.response_tokens
            
            # Track by model
            model_key = f"{usage.provider}:{usage.model_id}"
            if model_key not in summary["model_costs"]:
                summary["model_costs"][model_key] = {
                    "cost": 0.0,
                    "request_tokens": 0,
                    "response_tokens": 0,
                    "tokens_per_dollar_input": 0.0,
                    "tokens_per_dollar_output": 0.0,
                    "tokens_per_dollar_total": 0.0
                }
            summary["model_costs"][model_key]["cost"] += usage.cost
            summary["model_costs"][model_key]["request_tokens"] += usage.request_tokens
            summary["model_costs"][model_key]["response_tokens"] += usage.response_tokens
            
            # Add to provider costs
            if usage.provider not in summary["provider_costs"]:
                summary["provider_costs"][usage.provider] = 0.0
            summary["provider_costs"][usage.provider] += usage.cost
        
        # Calculate token efficiency for each agent
        for agent_id, data in summary["agent_costs"].items():
            if data["cost"] > 0:
                data["tokens_per_dollar"] = (data["request_tokens"] + data["response_tokens"]) / data["cost"]
        
        # Calculate token efficiency for each model
        for model_key, data in summary["model_costs"].items():
            if data["cost"] > 0:
                data["tokens_per_dollar_input"] = data["request_tokens"] / data["cost"]
                data["tokens_per_dollar_output"] = data["response_tokens"] / data["cost"] 
                data["tokens_per_dollar_total"] = (data["request_tokens"] + data["response_tokens"]) / data["cost"]
        
        # Overall token efficiency
        if summary["total_spend"] > 0:
            summary["token_efficiency"] = {
                "tokens_per_dollar_input": summary["input_tokens"] / summary["total_spend"],
                "tokens_per_dollar_output": summary["output_tokens"] / summary["total_spend"],
                "tokens_per_dollar_total": summary["total_tokens"] / summary["total_spend"]
            }
            
        return summary
    
    def get_efficiency_metrics(self) -> List[Dict[str, Any]]:
        """
        Get efficiency metrics for all models
        
        Returns:
            List of model efficiency metrics
        """
        result = []
        
        # Convert cache to list of dicts
        for key, efficiency in self._model_efficiencies.items():
            provider, model_id, task_type = key.split(":")
            
            result.append({
                "model_id": model_id,
                "provider": provider,
                "task_type": task_type,
                "avg_time_to_first_token": efficiency.avg_time_to_first_token,
                "avg_tokens_per_second": efficiency.avg_tokens_per_second,
                "avg_cost_per_100_tokens": efficiency.avg_cost_per_100_tokens,
                "accuracy_score": efficiency.accuracy_score,
                "user_satisfaction": efficiency.user_satisfaction,
                "efficiency_score": efficiency.efficiency_score,
                "samples_count": efficiency.samples_count
            })
        
        # Sort by efficiency score (highest first)
        result.sort(key=lambda x: x["efficiency_score"], reverse=True)
        
        return result
    
    def get_current_strategy(self) -> Dict[str, Any]:
        """
        Get current cost control strategy
        
        Returns:
            Dictionary with strategy details
        """
        if not self._current_strategy:
            return {}
            
        try:
            text_mapping = json.loads(self._current_strategy.text_task_mapping)
        except json.JSONDecodeError:
            text_mapping = {}
            
        try:
            code_mapping = json.loads(self._current_strategy.code_task_mapping)
        except json.JSONDecodeError:
            code_mapping = {}
            
        try:
            image_mapping = json.loads(self._current_strategy.image_task_mapping)
        except json.JSONDecodeError:
            image_mapping = {}
        
        return {
            "id": self._current_strategy.id,
            "name": self._current_strategy.name,
            "description": self._current_strategy.description,
            "daily_budget": self._current_strategy.daily_budget,
            "current_spending": self._current_strategy.current_spending,
            "prioritize_cost": self._current_strategy.prioritize_cost,
            "prioritize_speed": self._current_strategy.prioritize_speed,
            "prioritize_accuracy": self._current_strategy.prioritize_accuracy,
            "cost_threshold": self._current_strategy.cost_threshold,
            "text_task_mapping": text_mapping,
            "code_task_mapping": code_mapping,
            "image_task_mapping": image_mapping,
            "fallback_model": self._current_strategy.fallback_model
        }
    
    def update_strategy(self, strategy_data: Dict[str, Any]) -> bool:
        """
        Update cost control strategy
        
        Args:
            strategy_data: Dictionary with strategy parameters
            
        Returns:
            Success status
        """
        if not self._current_strategy:
            return False
            
        try:
            # Update fields
            if "name" in strategy_data:
                self._current_strategy.name = strategy_data["name"]
                
            if "description" in strategy_data:
                self._current_strategy.description = strategy_data["description"]
                
            if "daily_budget" in strategy_data:
                self._current_strategy.daily_budget = float(strategy_data["daily_budget"])
                
            if "prioritize_cost" in strategy_data:
                self._current_strategy.prioritize_cost = float(strategy_data["prioritize_cost"])
                
            if "prioritize_speed" in strategy_data:
                self._current_strategy.prioritize_speed = float(strategy_data["prioritize_speed"])
                
            if "prioritize_accuracy" in strategy_data:
                self._current_strategy.prioritize_accuracy = float(strategy_data["prioritize_accuracy"])
                
            if "cost_threshold" in strategy_data:
                self._current_strategy.cost_threshold = float(strategy_data["cost_threshold"])
                
            if "text_task_mapping" in strategy_data:
                self._current_strategy.text_task_mapping = json.dumps(strategy_data["text_task_mapping"])
                
            if "code_task_mapping" in strategy_data:
                self._current_strategy.code_task_mapping = json.dumps(strategy_data["code_task_mapping"])
                
            if "image_task_mapping" in strategy_data:
                self._current_strategy.image_task_mapping = json.dumps(strategy_data["image_task_mapping"])
                
            if "fallback_model" in strategy_data:
                self._current_strategy.fallback_model = strategy_data["fallback_model"]
            
            db.session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error updating strategy: {str(e)}")
            db.session.rollback()
            return False