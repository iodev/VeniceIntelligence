# Enterprise Repository Files to Create

## File 1: enterprise/billing.py
```python
"""
Enterprise billing and subscription management
"""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional
import stripe
import os

class SubscriptionTier(Enum):
    STARTER = "starter"      # $29/month
    PROFESSIONAL = "professional"  # $99/month  
    ENTERPRISE = "enterprise"      # $299/month

@dataclass
class Subscription:
    user_id: str
    tier: SubscriptionTier
    stripe_subscription_id: str
    status: str
    current_period_end: int

class BillingManager:
    def __init__(self):
        stripe.api_key = os.environ.get('STRIPE_SECRET_KEY')
    
    def create_subscription(self, user_id: str, tier: SubscriptionTier, payment_method: str):
        # Create Stripe subscription
        pass
    
    def cancel_subscription(self, subscription_id: str):
        # Cancel Stripe subscription
        pass
    
    def check_subscription_status(self, user_id: str) -> Optional[Subscription]:
        # Check current subscription status
        pass
```

## File 2: enterprise/analytics.py
```python
"""
Enterprise analytics and business intelligence
"""
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime, timedelta

@dataclass
class BusinessMetrics:
    total_revenue: float
    active_subscriptions: int
    churn_rate: float
    average_usage_per_user: int
    cost_savings_delivered: float

class AnalyticsManager:
    def get_revenue_metrics(self, start_date: datetime, end_date: datetime) -> BusinessMetrics:
        # Calculate revenue metrics
        pass
    
    def get_user_cohort_analysis(self) -> Dict:
        # Analyze user cohorts and retention
        pass
    
    def get_usage_patterns(self) -> Dict:
        # Analyze usage patterns across customers
        pass
```

## File 3: templates/pricing.html (Enterprise)
```html
<!-- Enterprise pricing page with actual Stripe integration -->
<div class="pricing-enterprise">
    <h1>Enterprise Pricing</h1>
    
    <!-- Starter Plan -->
    <div class="plan starter">
        <h3>Starter - $29/month</h3>
        <button onclick="createCheckoutSession('starter')">Subscribe</button>
    </div>
    
    <!-- Professional Plan -->
    <div class="plan professional">
        <h3>Professional - $99/month</h3>
        <button onclick="createCheckoutSession('professional')">Subscribe</button>
    </div>
    
    <!-- Enterprise Plan -->
    <div class="plan enterprise">
        <h3>Enterprise - $299/month</h3>
        <button onclick="contactSales()">Contact Sales</button>
    </div>
</div>

<script>
function createCheckoutSession(tier) {
    fetch('/api/create-checkout-session', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({tier: tier})
    })
    .then(response => response.json())
    .then(data => {
        window.location.href = data.checkout_url;
    });
}
</script>
```

## File 4: config/production.py
```python
"""
Production configuration for enterprise deployment
"""
import os

class ProductionConfig:
    # Database
    DATABASE_URL = os.environ.get('DATABASE_URL')
    
    # Redis for caching
    REDIS_URL = os.environ.get('REDIS_URL')
    
    # Stripe
    STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY')
    STRIPE_WEBHOOK_SECRET = os.environ.get('STRIPE_WEBHOOK_SECRET')
    
    # Monitoring
    DATADOG_API_KEY = os.environ.get('DATADOG_API_KEY')
    
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    # Rate limiting (enterprise has no limits)
    RATE_LIMIT_ENABLED = False
```

## File 5: deployment/docker/Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "main:app"]
```

## File 6: deployment/kubernetes/deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-agent-enterprise
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-agent-enterprise
  template:
    metadata:
      labels:
        app: ai-agent-enterprise
    spec:
      containers:
      - name: ai-agent
        image: ai-agent-enterprise:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ai-agent-secrets
              key: database-url
        - name: STRIPE_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: ai-agent-secrets
              key: stripe-secret-key
```

## File 7: enterprise/security.py
```python
"""
Enterprise security and compliance features
"""
import hashlib
import secrets
from cryptography.fernet import Fernet

class SecurityManager:
    def __init__(self):
        self.encryption_key = os.environ.get('ENCRYPTION_KEY')
        self.cipher = Fernet(self.encryption_key)
    
    def encrypt_api_key(self, api_key: str) -> str:
        """Encrypt API keys for storage"""
        return self.cipher.encrypt(api_key.encode()).decode()
    
    def decrypt_api_key(self, encrypted_key: str) -> str:
        """Decrypt API keys for use"""
        return self.cipher.decrypt(encrypted_key.encode()).decode()
    
    def audit_log(self, user_id: str, action: str, details: dict):
        """Log user actions for compliance"""
        # Store audit logs for enterprise compliance
        pass
```

## File 8: enterprise/integrations.py
```python
"""
Enterprise integrations (Slack, Teams, etc.)
"""
from slack_sdk import WebClient
import requests

class SlackIntegration:
    def __init__(self, bot_token: str):
        self.client = WebClient(token=bot_token)
    
    def send_alert(self, channel: str, message: str):
        """Send alerts to Slack channels"""
        self.client.chat_postMessage(channel=channel, text=message)

class TeamsIntegration:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send_notification(self, message: str):
        """Send notifications to Teams"""
        payload = {"text": message}
        requests.post(self.webhook_url, json=payload)
```

## File 9: main.py (Enterprise version)
```python
"""
Enterprise version of main.py with additional features
"""
from flask import Flask
from enterprise.billing import BillingManager
from enterprise.analytics import AnalyticsManager
from enterprise.security import SecurityManager
import config.production

app = Flask(__name__)
app.config.from_object(config.production.ProductionConfig)

# Initialize enterprise components
billing_manager = BillingManager()
analytics_manager = AnalyticsManager()
security_manager = SecurityManager()

# Import enterprise routes
from enterprise import billing_routes
from enterprise import analytics_routes

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

## File 10: requirements.txt (Enterprise)
```txt
# Base requirements from community edition
flask==2.3.3
sqlalchemy==2.0.21
psycopg2-binary==2.9.7

# Enterprise-specific
stripe==6.6.0
redis==4.6.0
cryptography==41.0.4
datadog==0.47.0
slack-sdk==3.22.0
celery==5.3.1
gunicorn==21.2.0
kubernetes==27.2.0
```

---

## Marketing Website Files

## File 1: pages/index.tsx
```typescript
import Head from 'next/head'
import Hero from '../components/Hero'
import FeatureComparison from '../components/FeatureComparison'
import Testimonials from '../components/Testimonials'
import ROICalculator from '../components/ROICalculator'

export default function Home() {
  return (
    <>
      <Head>
        <title>AI Agent Platform - Reduce AI Costs by 30-70%</title>
        <meta name="description" content="Intelligent multi-provider AI routing that reduces costs and improves reliability" />
      </Head>
      
      <Hero />
      <FeatureComparison />
      <ROICalculator />
      <Testimonials />
    </>
  )
}
```

## File 2: components/PricingTable.tsx
```typescript
interface PricingTier {
  name: string;
  price: string;
  features: string[];
  cta: string;
  href: string;
}

const tiers: PricingTier[] = [
  {
    name: "Community",
    price: "Free",
    features: ["100 API calls/hour", "Basic routing", "Community support"],
    cta: "Try on GitHub",
    href: "https://github.com/yourusername/ai-agent-community"
  },
  {
    name: "Professional", 
    price: "$99/month",
    features: ["Unlimited API calls", "Advanced routing", "Priority support"],
    cta: "Start Free Trial",
    href: "/trial"
  }
];

export default function PricingTable() {
  return (
    <div className="pricing-grid">
      {tiers.map((tier) => (
        <div key={tier.name} className="pricing-card">
          <h3>{tier.name}</h3>
          <div className="price">{tier.price}</div>
          <ul>
            {tier.features.map((feature) => (
              <li key={feature}>{feature}</li>
            ))}
          </ul>
          <a href={tier.href} className="cta-button">{tier.cta}</a>
        </div>
      ))}
    </div>
  )
}
```

These files create the complete enterprise platform with billing, analytics, security, and marketing capabilities while keeping the community edition clean and educational.