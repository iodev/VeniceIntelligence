"""
Analytics and metrics tracking for the AI Agent System
Provides insights for both users and business intelligence
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)

@dataclass
class QueryAnalytics:
    """Analytics data for individual queries"""
    timestamp: datetime
    user_id: str
    query_type: str  # text, code, image, math
    model_used: str
    provider_used: str
    latency_ms: float
    tokens_used: int
    cost_estimated: float
    cost_saved: float
    success: bool
    quality_score: Optional[float] = None
    user_rating: Optional[int] = None  # 1-5 stars

@dataclass
class UserAnalytics:
    """Aggregated analytics for a user"""
    user_id: str
    total_queries: int = 0
    total_tokens: int = 0
    total_cost_saved: float = 0.0
    average_latency: float = 0.0
    favorite_model: str = ""
    most_used_provider: str = ""
    query_types: Dict[str, int] = None
    success_rate: float = 0.0
    average_quality: float = 0.0
    
    def __post_init__(self):
        if self.query_types is None:
            self.query_types = {"text": 0, "code": 0, "image": 0, "math": 0}

@dataclass
class SystemAnalytics:
    """System-wide analytics and performance metrics"""
    total_users: int = 0
    total_queries_today: int = 0
    total_queries_month: int = 0
    average_cost_savings: float = 0.0
    top_models: List[Dict[str, Any]] = None
    provider_distribution: Dict[str, int] = None
    conversion_rate: float = 0.0  # free to paid
    churn_rate: float = 0.0
    
    def __post_init__(self):
        if self.top_models is None:
            self.top_models = []
        if self.provider_distribution is None:
            self.provider_distribution = {}

class AnalyticsManager:
    """
    Manages analytics collection and reporting for business intelligence
    """
    
    def __init__(self):
        """Initialize analytics manager"""
        self.query_history = []  # List of QueryAnalytics
        self.user_cache = {}  # user_id -> UserAnalytics (cached)
        self.system_cache = None  # SystemAnalytics (cached)
        self.cache_updated = datetime.now()
        
    def track_query(self, user_id: str, query_type: str, model_used: str, 
                   provider_used: str, latency_ms: float, tokens_used: int,
                   cost_estimated: float, cost_saved: float, success: bool,
                   quality_score: Optional[float] = None) -> None:
        """Track a query for analytics"""
        
        analytics = QueryAnalytics(
            timestamp=datetime.now(),
            user_id=user_id,
            query_type=query_type,
            model_used=model_used,
            provider_used=provider_used,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            cost_estimated=cost_estimated,
            cost_saved=cost_saved,
            success=success,
            quality_score=quality_score
        )
        
        self.query_history.append(analytics)
        
        # Clear cache to force recalculation
        self._invalidate_cache()
        
        logger.debug(f"Tracked query analytics for user {user_id}: {query_type} via {provider_used}")
    
    def track_user_rating(self, user_id: str, rating: int) -> None:
        """Track user rating for the last query"""
        # Find the most recent query for this user
        for query in reversed(self.query_history):
            if query.user_id == user_id:
                query.user_rating = rating
                logger.info(f"User {user_id} rated their last query: {rating}/5 stars")
                break
    
    def get_user_analytics(self, user_id: str, force_refresh: bool = False) -> UserAnalytics:
        """Get analytics for a specific user"""
        
        if not force_refresh and user_id in self.user_cache:
            return self.user_cache[user_id]
        
        # Calculate user analytics from query history
        user_queries = [q for q in self.query_history if q.user_id == user_id]
        
        if not user_queries:
            return UserAnalytics(user_id=user_id)
        
        # Aggregate metrics
        total_queries = len(user_queries)
        total_tokens = sum(q.tokens_used for q in user_queries)
        total_cost_saved = sum(q.cost_saved for q in user_queries)
        average_latency = sum(q.latency_ms for q in user_queries) / total_queries
        
        # Success rate
        successful_queries = sum(1 for q in user_queries if q.success)
        success_rate = successful_queries / total_queries if total_queries > 0 else 0
        
        # Average quality
        quality_scores = [q.quality_score for q in user_queries if q.quality_score is not None]
        average_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Most used model and provider
        model_counts = {}
        provider_counts = {}
        query_type_counts = {"text": 0, "code": 0, "image": 0, "math": 0}
        
        for query in user_queries:
            # Count models
            model_counts[query.model_used] = model_counts.get(query.model_used, 0) + 1
            
            # Count providers
            provider_counts[query.provider_used] = provider_counts.get(query.provider_used, 0) + 1
            
            # Count query types
            if query.query_type in query_type_counts:
                query_type_counts[query.query_type] += 1
        
        favorite_model = max(model_counts, key=model_counts.get) if model_counts else ""
        most_used_provider = max(provider_counts, key=provider_counts.get) if provider_counts else ""
        
        analytics = UserAnalytics(
            user_id=user_id,
            total_queries=total_queries,
            total_tokens=total_tokens,
            total_cost_saved=total_cost_saved,
            average_latency=average_latency,
            favorite_model=favorite_model,
            most_used_provider=most_used_provider,
            query_types=query_type_counts,
            success_rate=success_rate,
            average_quality=average_quality
        )
        
        # Cache the result
        self.user_cache[user_id] = analytics
        
        return analytics
    
    def get_system_analytics(self, force_refresh: bool = False) -> SystemAnalytics:
        """Get system-wide analytics"""
        
        if not force_refresh and self.system_cache and \
           datetime.now() - self.cache_updated < timedelta(minutes=5):
            return self.system_cache
        
        # Calculate system analytics
        total_users = len(set(q.user_id for q in self.query_history))
        
        today = datetime.now().date()
        month_start = datetime.now().replace(day=1).date()
        
        queries_today = sum(1 for q in self.query_history if q.timestamp.date() == today)
        queries_month = sum(1 for q in self.query_history if q.timestamp.date() >= month_start)
        
        # Average cost savings
        cost_savings = [q.cost_saved for q in self.query_history if q.cost_saved > 0]
        average_cost_savings = sum(cost_savings) / len(cost_savings) if cost_savings else 0
        
        # Top models by usage
        model_counts = {}
        provider_counts = {}
        
        for query in self.query_history:
            model_counts[query.model_used] = model_counts.get(query.model_used, 0) + 1
            provider_counts[query.provider_used] = provider_counts.get(query.provider_used, 0) + 1
        
        # Sort models by usage
        top_models = [
            {"model": model, "usage_count": count, "percentage": (count / len(self.query_history)) * 100}
            for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True)
        ][:10]  # Top 10 models
        
        analytics = SystemAnalytics(
            total_users=total_users,
            total_queries_today=queries_today,
            total_queries_month=queries_month,
            average_cost_savings=average_cost_savings,
            top_models=top_models,
            provider_distribution=provider_counts,
            conversion_rate=0.0,  # TODO: Calculate from billing data
            churn_rate=0.0  # TODO: Calculate from user activity
        )
        
        # Cache the result
        self.system_cache = analytics
        self.cache_updated = datetime.now()
        
        return analytics
    
    def get_dashboard_data(self, user_id: str) -> Dict[str, Any]:
        """Get formatted data for user dashboard"""
        user_analytics = self.get_user_analytics(user_id)
        
        return {
            "summary": {
                "total_queries": user_analytics.total_queries,
                "cost_saved": f"${user_analytics.total_cost_saved:.2f}",
                "favorite_model": user_analytics.favorite_model,
                "success_rate": f"{user_analytics.success_rate * 100:.1f}%"
            },
            "usage_breakdown": user_analytics.query_types,
            "performance": {
                "average_latency": f"{user_analytics.average_latency:.0f}ms",
                "quality_score": f"{user_analytics.average_quality:.1f}/5.0",
                "tokens_used": user_analytics.total_tokens
            },
            "insights": self._generate_user_insights(user_analytics)
        }
    
    def get_business_intelligence(self) -> Dict[str, Any]:
        """Get business intelligence data for admin dashboard"""
        system_analytics = self.get_system_analytics()
        
        return {
            "growth": {
                "total_users": system_analytics.total_users,
                "queries_today": system_analytics.total_queries_today,
                "queries_month": system_analytics.total_queries_month,
                "conversion_rate": f"{system_analytics.conversion_rate:.1f}%"
            },
            "performance": {
                "average_savings": f"${system_analytics.average_cost_savings:.2f}",
                "top_models": system_analytics.top_models[:5],
                "provider_split": system_analytics.provider_distribution
            },
            "opportunities": self._generate_business_insights(system_analytics)
        }
    
    def _generate_user_insights(self, analytics: UserAnalytics) -> List[str]:
        """Generate actionable insights for users"""
        insights = []
        
        if analytics.total_cost_saved > 100:
            insights.append(f"You've saved ${analytics.total_cost_saved:.0f} in AI costs!")
        
        if analytics.success_rate < 0.8:
            insights.append("Consider upgrading for better model routing and higher success rates")
        
        if analytics.query_types.get("code", 0) > 10:
            insights.append("You frequently use code generation - Professional tier offers specialized code models")
        
        if analytics.average_latency > 3000:
            insights.append("Upgrade to Professional for priority processing and faster responses")
        
        return insights
    
    def _generate_business_insights(self, analytics: SystemAnalytics) -> List[str]:
        """Generate business insights for admin dashboard"""
        insights = []
        
        if analytics.total_queries_today > analytics.total_queries_month / 30 * 1.5:
            insights.append("Above-average daily usage detected - consider scaling infrastructure")
        
        if analytics.average_cost_savings > 50:
            insights.append("High value delivery - highlight cost savings in marketing")
        
        # Identify popular models for partnership opportunities
        if analytics.top_models:
            top_model = analytics.top_models[0]
            insights.append(f"Consider enterprise partnership with {top_model['model']} provider")
        
        return insights
    
    def _invalidate_cache(self):
        """Clear analytics cache to force recalculation"""
        self.user_cache.clear()
        self.system_cache = None
    
    def export_analytics(self, user_id: Optional[str] = None, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Export analytics data for reporting"""
        
        # Filter queries by date range
        filtered_queries = self.query_history
        
        if start_date:
            filtered_queries = [q for q in filtered_queries if q.timestamp >= start_date]
        if end_date:
            filtered_queries = [q for q in filtered_queries if q.timestamp <= end_date]
        if user_id:
            filtered_queries = [q for q in filtered_queries if q.user_id == user_id]
        
        # Convert to serializable format
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "filters": {
                "user_id": user_id,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None
            },
            "query_count": len(filtered_queries),
            "queries": [asdict(q) for q in filtered_queries]
        }
        
        return export_data

# Global analytics manager instance
analytics_manager = AnalyticsManager()