# Repository Setup Instructions

## CURRENT REPOSITORY (Community Edition)
**Status**: Ready for public release
**Action**: Make public on GitHub

### Files in Current Repo (Keep These):
```
Current Repository (ai-agent-community)
├── agent/                    # Core AI agent system
│   ├── api.py               # External API interface
│   ├── core.py              # Main agent logic
│   ├── content_classifier.py # Query type detection
│   ├── model_registry.py    # Model management
│   ├── memory.py            # Vector storage
│   ├── anthropic_client.py  # Anthropic integration
│   ├── perplexity.py        # Perplexity integration
│   ├── models.py            # Venice client
│   └── openai_client.py     # OpenAI integration
├── templates/               # Web interface
│   ├── base.html
│   ├── index.html
│   ├── history.html
│   └── admin.html
├── static/                  # CSS and assets
├── models.py               # Database models
├── main.py                 # Application entry
├── app.py                  # Flask routes
├── config.py               # Configuration
├── README.md               # Community-focused documentation
├── CONTRIBUTING.md         # Contribution guidelines
├── TODO.md                 # Community task list
└── requirements.txt        # Dependencies
```

### Make Repository Public:
1. Go to repository Settings
2. Scroll to "Danger Zone"
3. Click "Change repository visibility"
4. Select "Make public"
5. Add repository topics: `ai`, `machine-learning`, `multi-provider`, `agent-system`

---

## NEW REPOSITORY 1: Enterprise Platform
**Name**: `ai-agent-enterprise`
**Visibility**: PRIVATE
**Purpose**: Commercial features and billing

### Create New Private Repository:
```bash
gh repo create ai-agent-enterprise --private
git clone https://github.com/yourusername/ai-agent-enterprise.git
cd ai-agent-enterprise
```

### Files to Create in Enterprise Repo:
```
ai-agent-enterprise/
├── enterprise/
│   ├── __init__.py
│   ├── billing.py           # Subscription management
│   ├── analytics.py         # Business intelligence
│   ├── security.py          # Enterprise security
│   ├── integrations.py      # Slack, Teams, etc.
│   └── monitoring.py        # Advanced monitoring
├── templates/
│   ├── pricing.html         # Pricing tiers
│   ├── dashboard.html       # Analytics dashboard
│   ├── billing.html         # Billing interface
│   └── enterprise.html      # Enterprise features
├── config/
│   ├── production.py        # Production settings
│   └── enterprise.py        # Enterprise configuration
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   ├── kubernetes/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   └── terraform/
│       ├── main.tf
│       └── variables.tf
├── tests/
│   ├── test_billing.py
│   └── test_analytics.py
├── main.py                  # Enterprise app entry
├── requirements.txt         # Enterprise dependencies
├── README.md               # Enterprise documentation
└── .env.example            # Enterprise environment template
```

### Copy These Specific Files from Current Repo:
- Copy `agent/` folder as base
- Add enterprise-specific modules on top
- Copy `models.py` and extend with billing tables
- Copy `main.py` and modify for enterprise features

---

## NEW REPOSITORY 2: Marketing Website
**Name**: `ai-agent-website`
**Visibility**: PUBLIC
**Purpose**: Marketing, sales, lead generation

### Create Marketing Website:
```bash
npx create-next-app@latest ai-agent-website --typescript --tailwind --eslint
cd ai-agent-website
```

### Files to Create in Marketing Repo:
```
ai-agent-website/
├── pages/
│   ├── index.tsx            # Landing page
│   ├── pricing.tsx          # Pricing comparison
│   ├── enterprise.tsx       # Enterprise features
│   ├── demo.tsx             # Interactive demo
│   ├── case-studies/
│   │   ├── index.tsx        # Case studies list
│   │   └── [slug].tsx       # Individual case study
│   └── api/
│       ├── contact.ts       # Contact form handler
│       └── trial.ts         # Trial signup handler
├── components/
│   ├── Hero.tsx             # Landing hero section
│   ├── PricingTable.tsx     # Pricing tiers
│   ├── Testimonials.tsx     # Customer testimonials
│   ├── ROICalculator.tsx    # Cost savings calculator
│   ├── FeatureComparison.tsx # Community vs Enterprise
│   └── ContactForm.tsx      # Lead capture form
├── public/
│   ├── images/
│   │   ├── logo.svg
│   │   ├── hero-diagram.svg
│   │   └── screenshots/
│   └── videos/
│       └── demo.mp4
├── styles/
│   ├── globals.css
│   └── components.css
├── lib/
│   ├── analytics.ts         # Google Analytics
│   └── api.ts              # API helpers
└── content/
    ├── case-studies.json
    └── testimonials.json
```

---

## SPECIFIC CONTENT FOR EACH REPOSITORY

### Community Repository Content:
**Message**: "Learn advanced AI orchestration patterns"
**Features**: 
- Basic multi-provider routing
- Simple content classification  
- 100 requests/hour limit
- Educational examples
- Community support

### Enterprise Repository Content:
**Message**: "Production-ready AI infrastructure"
**Features**:
- Unlimited API calls
- Advanced analytics
- Custom model training
- Enterprise security
- SLA guarantees
- Priority support

### Marketing Website Content:
**Message**: "Reduce AI costs by 30-70%"
**Features**:
- Landing pages
- ROI calculator
- Pricing comparison
- Customer testimonials
- Free trial signup
- Enterprise contact forms

---

## ENVIRONMENT VARIABLES BY REPOSITORY

### Community Edition (.env):
```bash
# Demo/basic keys
VENICE_API_KEY=demo_key
ANTHROPIC_API_KEY=demo_key
PERPLEXITY_API_KEY=demo_key
DATABASE_URL=sqlite:///community.db
SESSION_SECRET=community_secret_key
```

### Enterprise Platform (.env):
```bash
# Production keys
VENICE_API_KEY=prod_venice_key
ANTHROPIC_API_KEY=prod_anthropic_key
OPENAI_API_KEY=prod_openai_key
PERPLEXITY_API_KEY=prod_perplexity_key
DATABASE_URL=postgresql://prod_db_url
STRIPE_SECRET_KEY=sk_live_...
QDRANT_URL=https://prod-qdrant.com
QDRANT_API_KEY=prod_qdrant_key
SESSION_SECRET=enterprise_secret_key
ADMIN_KEY=secure_admin_key
```

### Marketing Website (.env):
```bash
# Analytics and tracking
NEXT_PUBLIC_GA_ID=GA-XXXXXXXXX
MIXPANEL_TOKEN=mixpanel_token
STRIPE_PUBLISHABLE_KEY=pk_live_...
HUBSPOT_API_KEY=hubspot_key
SENDGRID_API_KEY=sendgrid_key
```

---

## DEPLOYMENT INSTRUCTIONS

### Community Edition:
- Deploy to Replit (current)
- Or deploy to Railway/Vercel for public demo
- Use basic database (SQLite/PostgreSQL)

### Enterprise Platform:
- Deploy to AWS/GCP/Azure
- Use production database (PostgreSQL)
- Set up Redis for caching
- Configure monitoring (DataDog/New Relic)
- Set up CI/CD pipeline

### Marketing Website:
- Deploy to Vercel/Netlify
- Configure custom domain
- Set up analytics tracking
- Configure form handlers
- Set up A/B testing

---

## CROSS-REPOSITORY LINKS

### In Community README:
"Enterprise features with unlimited API calls, advanced analytics, and dedicated support available separately."

### In Enterprise README:
"Built on our open source foundation. See the community edition for core architecture."

### In Marketing Website:
- Link to community repo: "Try the open source version"
- Link to enterprise: "Start enterprise trial"
- Link between pricing tiers

This setup creates a complete ecosystem where each repository serves a specific purpose in the customer journey: Discovery → Evaluation → Purchase → Enterprise Scaling.