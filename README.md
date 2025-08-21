# 🤖 AI Agent System - Community Edition

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contributors Welcome](https://img.shields.io/badge/contributors-welcome-brightgreen.svg)](CONTRIBUTING.md)

> An intelligent multi-provider AI agent system with dynamic model selection and content-type classification

## ✨ What Makes This Special

This AI agent system demonstrates **intelligent routing** across multiple AI providers, automatically selecting the best model for each query type. It's designed as a modular, self-contained component that showcases advanced AI orchestration patterns.

### 🎯 Key Features

- **🔀 Multi-Provider Routing**: Seamlessly switches between Venice.ai, Anthropic, Perplexity, and other providers
- **🧠 Content Classification**: Automatically detects query types (text, code, image, math) for optimal model selection  
- **📊 Performance Tracking**: Built-in analytics to monitor model performance and costs
- **🔄 Adaptive Learning**: Vector storage with Qdrant for persistent memory and learning
- **🛡️ Intelligent Fallbacks**: Automatic failover when providers are unavailable
- **🎨 Clean Architecture**: Completely modular design with clear separation of concerns

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL database
- API keys for at least one provider (Venice.ai, Anthropic, or Perplexity)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-agent-system.git
   cd ai-agent-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and database URL
   ```

4. **Initialize the database**
   ```bash
   python -c "from main import db; db.create_all()"
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

Visit `http://localhost:5000` to start using the agent!

## 🏗️ Architecture

The system is built with clean separation between components:

```
agent/
├── core.py              # Main agent logic with multi-provider support
├── content_classifier.py # Query type classification
├── model_registry.py    # Dynamic model management
├── api.py              # Clean API interface for external systems
├── memory.py           # Qdrant vector storage integration
├── providers/          # Individual provider implementations
│   ├── venice_client.py
│   ├── anthropic_client.py
│   └── perplexity.py
└── utils.py            # Shared utilities
```

## 💡 Usage Examples

### Basic Query Processing

```python
from agent.api import AgentAPI
from agent.core import Agent

# Initialize the agent
agent_api = AgentAPI(agent)

# Process different types of queries
text_result = agent_api.process_query(
    query="Explain quantum computing",
    query_type="text"
)

code_result = agent_api.process_query(
    query="Write a Python function to sort a list",
    query_type="code"
)

# The system automatically selects the best model for each query type
```

### Content Type Classification

The system automatically classifies queries into optimal categories:

- **Text**: General questions, explanations, creative writing
- **Code**: Programming tasks, debugging, code review
- **Image**: Image generation prompts, visual descriptions
- **Math**: Mathematical problems, calculations, formulas

### Multi-Provider Integration

```python
# The agent automatically handles provider fallbacks
# Venice.ai → Anthropic → Perplexity → OpenAI (if available)

result = agent_api.process_query(
    query="Your query here",
    provider=None  # Auto-select best provider
)
```

## 🔧 Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/aiagent

# AI Provider APIs (at least one required)
VENICE_API_KEY=your_venice_key
ANTHROPIC_API_KEY=your_anthropic_key
PERPLEXITY_API_KEY=your_perplexity_key
OPENAI_API_KEY=your_openai_key

# Vector Storage (optional)
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_key
```

### Model Configuration

The system automatically discovers available models from each provider. You can customize model preferences in `config.py`:

```python
AVAILABLE_MODELS = {
    'venice': ['mistral-31-24b', 'llama-3.2-90b'],
    'anthropic': ['claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307'],
    'perplexity': ['llama-3.1-sonar-large-128k-online']
}
```

## 📈 Performance Monitoring

The system includes built-in performance tracking:

- **Response times** for each provider and model
- **Success rates** and error patterns  
- **Cost estimation** and optimization opportunities
- **Model effectiveness** by query type

Access the monitoring dashboard at `/admin?key=your_admin_key`

## 🤝 Contributing

We welcome contributions! This is a community-driven project showcasing advanced AI agent patterns.

### Areas for Contribution

- **New Provider Integrations**: Add support for additional AI providers
- **Enhanced Classification**: Improve content type detection algorithms
- **Performance Optimization**: Optimize routing and caching strategies
- **Documentation**: Improve examples and tutorials
- **Testing**: Add comprehensive test coverage

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Ensure tests pass: `python -m pytest`
5. Submit a pull request

## 🏢 Enterprise Features

This community edition demonstrates the core capabilities. Enterprise features include:

- **Unlimited API calls** (vs 100/hour community limit)
- **Advanced analytics** and business intelligence
- **Custom model training** and fine-tuning
- **Enterprise security** and compliance
- **Priority support** and SLA guarantees
- **On-premise deployment** options

[Contact us for enterprise licensing →](mailto:enterprise@yourcompany.com)

## 📚 Documentation

- [API Reference](docs/api.md)
- [Architecture Guide](docs/architecture.md)
- [Provider Integration](docs/providers.md)
- [Deployment Guide](docs/deployment.md)

## 🐛 Troubleshooting

### Common Issues

**Agent initialization failed**
- Check your API keys are valid
- Ensure database is accessible
- Verify Python version (3.11+ required)

**No models available**
- At least one provider API key must be configured
- Check provider API status and quotas
- Review logs for specific error messages

**Memory errors with Qdrant**
- Qdrant configuration is optional for basic usage
- Set `QDRANT_URL=` to disable vector storage
- Check Qdrant service is running if enabled

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/ai-agent-system&type=Date)](https://star-history.com/#yourusername/ai-agent-system&Date)

## 💬 Community

- **GitHub Discussions**: [Join the conversation](https://github.com/yourusername/ai-agent-system/discussions)
- **Issues**: [Report bugs or request features](https://github.com/yourusername/ai-agent-system/issues)
- **Discord**: [Join our developer community](https://discord.gg/your-invite)

---

**Built with ❤️ by the AI agent community**

*Showcasing the future of intelligent AI orchestration*