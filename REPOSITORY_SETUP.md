# Repository Setup Guide

## Repository 1: Community Edition (Current → Public)

**Current Status**: ✅ Ready for public release
**Action Required**: Make this repository public on GitHub

### Files Ready:
- ✅ Cleaned README.md for community focus
- ✅ Removed all commercial strategy files
- ✅ Updated TODO.md for community contributions
- ✅ Clean code demonstrating AI agent patterns

### Next Steps:
1. Push current state to GitHub
2. Make repository public
3. Add topics: `ai`, `machine-learning`, `multi-provider`, `agent-system`
4. Create initial GitHub issues for community contributions

---

## Repository 2: Enterprise Platform (New Private Repo)

**Name**: `ai-agent-enterprise`
**Status**: 📋 Setup required
**Visibility**: Private

### Files to Create:
```
ai-agent-enterprise/
├── enterprise/
│   ├── billing.py          # Subscription management
│   ├── analytics.py        # Business intelligence
│   ├── security.py         # Enterprise security layer
│   └── integrations.py     # Enterprise connectors
├── config/
│   ├── production.py       # Production settings
│   └── enterprise.py       # Enterprise configuration
├── deployment/
│   ├── docker/            # Container setup
│   ├── kubernetes/        # K8s manifests
│   └── terraform/         # Infrastructure as code
└── monitoring/
    ├── metrics.py         # Enterprise monitoring
    └── alerts.py          # Alert management
```

### Key Enterprise Features to Implement:
- Unlimited API rate limits
- Advanced cost optimization algorithms
- Multi-tenant architecture
- Custom model training capabilities
- Enterprise security and compliance
- Advanced analytics and reporting

---

## Repository 3: Marketing Website (New Public Repo)

**Name**: `ai-agent-website`
**Status**: 📋 Setup required
**Visibility**: Public
**Technology**: Next.js or React

### Files to Create:
```
ai-agent-website/
├── pages/
│   ├── index.tsx          # Landing page
│   ├── pricing.tsx        # Pricing tiers
│   ├── enterprise.tsx     # Enterprise features
│   ├── demo.tsx           # Interactive demo
│   └── case-studies/      # Customer stories
├── components/
│   ├── Hero.tsx           # Landing hero section
│   ├── PricingTable.tsx   # Pricing comparison
│   ├── Testimonials.tsx   # Customer testimonials
│   └── ROICalculator.tsx  # Cost savings calculator
├── public/
│   ├── images/            # Marketing images
│   └── videos/            # Demo videos
└── styles/
    └── globals.css        # Marketing styles
```

### Key Marketing Pages:
- High-converting landing page
- ROI calculator showing cost savings
- Enterprise feature comparison
- Customer testimonials and case studies
- Free trial signup flow

---

## Manual Setup Instructions

### Step 1: Create GitHub Repositories

1. **Make current repo public**:
   ```bash
   # In current repo settings
   # Go to Settings → General → Danger Zone
   # Change repository visibility to Public
   ```

2. **Create enterprise repo**:
   ```bash
   gh repo create ai-agent-enterprise --private
   git clone https://github.com/yourusername/ai-agent-enterprise.git
   ```

3. **Create marketing website**:
   ```bash
   gh repo create ai-agent-website --public
   npx create-next-app@latest ai-agent-website
   ```

### Step 2: Copy Enterprise Features

From current repo, move these files to enterprise repo:
- `agent/billing.py` (if exists)
- `agent/analytics.py` (if exists)
- `templates/pricing.html` (if exists)
- `templates/dashboard.html` (if exists)

### Step 3: Set Up Deployment

**Enterprise Platform**:
- Use Railway, Vercel, or AWS for hosting
- Set up CI/CD with GitHub Actions
- Configure environment variables for production

**Marketing Website**:
- Deploy to Vercel or Netlify
- Set up analytics (Google Analytics, Mixpanel)
- Configure lead capture forms

### Step 4: Cross-Repository Links

Update each repository to reference the others:
- Community → "Enterprise features available"
- Enterprise → "Built on open source foundation"  
- Marketing → Links to both community and enterprise

---

## Environment Variables by Repository

### Community Edition:
```bash
# Basic API keys for demonstration
VENICE_API_KEY=demo_key
ANTHROPIC_API_KEY=demo_key
DATABASE_URL=sqlite:///demo.db
```

### Enterprise Platform:
```bash
# Production API keys
VENICE_API_KEY=prod_venice_key
ANTHROPIC_API_KEY=prod_anthropic_key
OPENAI_API_KEY=prod_openai_key
DATABASE_URL=postgresql://prod_db_url
STRIPE_SECRET_KEY=sk_live_...
QDRANT_URL=https://prod-qdrant.com
```

### Marketing Website:
```bash
# Analytics and tracking
GOOGLE_ANALYTICS_ID=GA-XXXXXXXXX
MIXPANEL_TOKEN=your_mixpanel_token
STRIPE_PUBLISHABLE_KEY=pk_live_...
```

---

## Repository Naming Convention

- `ai-agent-community` - Open source showcase
- `ai-agent-enterprise` - Commercial platform  
- `ai-agent-website` - Marketing and sales
- `ai-agent-docs` - Documentation (optional)

This naming makes the relationship between repositories clear while maintaining professional branding.