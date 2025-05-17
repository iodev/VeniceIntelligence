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