"""
Billing and subscription management for the AI Agent System
Handles usage tracking, rate limiting, and premium feature access
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class SubscriptionTier(Enum):
    """Subscription tiers with different feature access"""
    FREE = "free"
    STARTER = "starter"  # $29/month
    PROFESSIONAL = "professional"  # $99/month
    ENTERPRISE = "enterprise"  # $299/month

class FeatureFlag(Enum):
    """Premium features that can be gated"""
    MULTI_PROVIDER_ROUTING = "multi_provider_routing"
    ADVANCED_CONTENT_CLASSIFICATION = "advanced_content_classification"
    COST_OPTIMIZATION = "cost_optimization"
    PRIORITY_SUPPORT = "priority_support"
    CUSTOM_MODEL_TRAINING = "custom_model_training"
    ANALYTICS_DASHBOARD = "analytics_dashboard"
    API_RATE_UNLIMITED = "api_rate_unlimited"
    ENTERPRISE_MODELS = "enterprise_models"

@dataclass
class UsageMetrics:
    """Track usage metrics for billing and analytics"""
    api_calls_today: int = 0
    api_calls_month: int = 0
    tokens_used_today: int = 0
    tokens_used_month: int = 0
    cost_saved_today: float = 0.0
    cost_saved_month: float = 0.0
    models_used: int = 0
    providers_used: int = 0
    
class BillingManager:
    """
    Manages subscription tiers, usage tracking, and feature access
    """
    
    # Rate limits by tier (requests per hour)
    RATE_LIMITS = {
        SubscriptionTier.FREE: 100,
        SubscriptionTier.STARTER: 1000,
        SubscriptionTier.PROFESSIONAL: 10000,
        SubscriptionTier.ENTERPRISE: 100000
    }
    
    # Feature access by tier
    TIER_FEATURES = {
        SubscriptionTier.FREE: [
            # Core features only
        ],
        SubscriptionTier.STARTER: [
            FeatureFlag.MULTI_PROVIDER_ROUTING,
            FeatureFlag.ANALYTICS_DASHBOARD
        ],
        SubscriptionTier.PROFESSIONAL: [
            FeatureFlag.MULTI_PROVIDER_ROUTING,
            FeatureFlag.ADVANCED_CONTENT_CLASSIFICATION,
            FeatureFlag.COST_OPTIMIZATION,
            FeatureFlag.ANALYTICS_DASHBOARD,
            FeatureFlag.API_RATE_UNLIMITED
        ],
        SubscriptionTier.ENTERPRISE: [
            FeatureFlag.MULTI_PROVIDER_ROUTING,
            FeatureFlag.ADVANCED_CONTENT_CLASSIFICATION,
            FeatureFlag.COST_OPTIMIZATION,
            FeatureFlag.PRIORITY_SUPPORT,
            FeatureFlag.CUSTOM_MODEL_TRAINING,
            FeatureFlag.ANALYTICS_DASHBOARD,
            FeatureFlag.API_RATE_UNLIMITED,
            FeatureFlag.ENTERPRISE_MODELS
        ]
    }
    
    def __init__(self):
        """Initialize billing manager"""
        self.usage_data = {}  # user_id -> UsageMetrics
        self.user_tiers = {}  # user_id -> SubscriptionTier
        self.rate_limit_data = {}  # user_id -> {"count": int, "window_start": datetime}
        
    def get_user_tier(self, user_id: str) -> SubscriptionTier:
        """Get subscription tier for user"""
        return self.user_tiers.get(user_id, SubscriptionTier.FREE)
    
    def set_user_tier(self, user_id: str, tier: SubscriptionTier) -> None:
        """Set subscription tier for user"""
        self.user_tiers[user_id] = tier
        logger.info(f"Updated user {user_id} to tier {tier.value}")
    
    def has_feature_access(self, user_id: str, feature: FeatureFlag) -> bool:
        """Check if user has access to a specific feature"""
        user_tier = self.get_user_tier(user_id)
        return feature in self.TIER_FEATURES.get(user_tier, [])
    
    def check_rate_limit(self, user_id: str) -> Tuple[bool, int]:
        """
        Check if user is within rate limits
        
        Returns:
            Tuple of (allowed, remaining_requests)
        """
        user_tier = self.get_user_tier(user_id)
        
        # Enterprise and Professional have unlimited API calls
        if user_tier in [SubscriptionTier.ENTERPRISE, SubscriptionTier.PROFESSIONAL]:
            return True, 999999
            
        limit = self.RATE_LIMITS[user_tier]
        now = datetime.now()
        
        # Initialize or reset rate limit window
        if user_id not in self.rate_limit_data:
            self.rate_limit_data[user_id] = {"count": 0, "window_start": now}
        
        user_data = self.rate_limit_data[user_id]
        
        # Reset if hour has passed
        if now - user_data["window_start"] >= timedelta(hours=1):
            user_data["count"] = 0
            user_data["window_start"] = now
        
        # Check limit
        if user_data["count"] >= limit:
            return False, 0
        
        # Increment counter
        user_data["count"] += 1
        remaining = limit - user_data["count"]
        
        return True, remaining
    
    def track_usage(self, user_id: str, tokens_used: int, cost_saved: float = 0.0, 
                   model_used: str = None, provider_used: str = None) -> None:
        """Track usage metrics for analytics and billing"""
        if user_id not in self.usage_data:
            self.usage_data[user_id] = UsageMetrics()
        
        metrics = self.usage_data[user_id]
        
        # Update counters
        metrics.api_calls_today += 1
        metrics.api_calls_month += 1
        metrics.tokens_used_today += tokens_used
        metrics.tokens_used_month += tokens_used
        metrics.cost_saved_today += cost_saved
        metrics.cost_saved_month += cost_saved
        
        # Track unique models and providers
        if model_used:
            metrics.models_used += 1
        if provider_used:
            metrics.providers_used += 1
    
    def get_usage_metrics(self, user_id: str) -> UsageMetrics:
        """Get usage metrics for a user"""
        return self.usage_data.get(user_id, UsageMetrics())
    
    def get_upgrade_recommendation(self, user_id: str) -> Optional[Dict]:
        """
        Analyze usage and recommend subscription upgrade
        
        Returns:
            Dictionary with upgrade recommendation or None
        """
        metrics = self.get_usage_metrics(user_id)
        current_tier = self.get_user_tier(user_id)
        
        # Skip if already enterprise
        if current_tier == SubscriptionTier.ENTERPRISE:
            return None
        
        recommendations = []
        
        # High usage patterns
        if metrics.api_calls_month > 500 and current_tier == SubscriptionTier.FREE:
            recommendations.append({
                "tier": SubscriptionTier.STARTER,
                "reason": f"You've made {metrics.api_calls_month} API calls this month. Starter plan offers 10x higher limits.",
                "value_prop": "Unlock multi-provider routing for better results"
            })
        
        if metrics.api_calls_month > 5000 and current_tier == SubscriptionTier.STARTER:
            recommendations.append({
                "tier": SubscriptionTier.PROFESSIONAL,
                "reason": f"Heavy usage detected ({metrics.api_calls_month} calls). Professional offers unlimited API calls.",
                "value_prop": f"You've saved ${metrics.cost_saved_month:.2f} in AI costs - Professional plan includes advanced optimization"
            })
        
        # Cost savings opportunity
        if metrics.cost_saved_month > 50 and current_tier in [SubscriptionTier.FREE, SubscriptionTier.STARTER]:
            recommendations.append({
                "tier": SubscriptionTier.PROFESSIONAL,
                "reason": f"You've already saved ${metrics.cost_saved_month:.2f} this month with basic routing.",
                "value_prop": "Professional tier could save you 3-5x more with advanced optimization"
            })
        
        return recommendations[0] if recommendations else None
    
    def generate_usage_report(self, user_id: str) -> Dict:
        """Generate detailed usage report for dashboard"""
        metrics = self.get_usage_metrics(user_id)
        tier = self.get_user_tier(user_id)
        _, remaining_requests = self.check_rate_limit(user_id)
        
        return {
            "subscription_tier": tier.value,
            "usage": {
                "api_calls_today": metrics.api_calls_today,
                "api_calls_month": metrics.api_calls_month,
                "tokens_used_today": metrics.tokens_used_today,
                "tokens_used_month": metrics.tokens_used_month,
                "cost_saved_today": round(metrics.cost_saved_today, 2),
                "cost_saved_month": round(metrics.cost_saved_month, 2),
                "models_used": metrics.models_used,
                "providers_used": metrics.providers_used
            },
            "limits": {
                "remaining_requests_hour": remaining_requests,
                "rate_limit": self.RATE_LIMITS[tier]
            },
            "features_available": [f.value for f in self.TIER_FEATURES.get(tier, [])],
            "upgrade_recommendation": self.get_upgrade_recommendation(user_id)
        }

# Global billing manager instance
billing_manager = BillingManager()

def require_feature(feature: FeatureFlag):
    """Decorator to check feature access before executing function"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract user_id from function arguments or session
            user_id = kwargs.get('user_id', 'anonymous')
            
            if not billing_manager.has_feature_access(user_id, feature):
                tier_needed = None
                for tier, features in billing_manager.TIER_FEATURES.items():
                    if feature in features:
                        tier_needed = tier
                        break
                
                raise PermissionError(
                    f"Feature '{feature.value}' requires {tier_needed.value if tier_needed else 'premium'} subscription. "
                    f"Upgrade at /upgrade to unlock this feature."
                )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def check_rate_limit(user_id: str = 'anonymous'):
    """Decorator to check rate limits before executing function"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            allowed, remaining = billing_manager.check_rate_limit(user_id)
            
            if not allowed:
                tier = billing_manager.get_user_tier(user_id)
                raise PermissionError(
                    f"Rate limit exceeded for {tier.value} tier. "
                    f"Upgrade at /upgrade for higher limits."
                )
            
            # Track usage
            billing_manager.track_usage(user_id, tokens_used=1)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator