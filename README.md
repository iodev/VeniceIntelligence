# Venice Intelligence - AI Agent Framework

A comprehensive multi-provider AI agent framework showcasing the Venice.ai API alongside OpenAI, Anthropic, Perplexity, and Hugging Face integrations. This project demonstrates best practices for building intelligent AI agents with cost monitoring, dynamic model selection, and persistent memory.

## 🌟 Features

### Multi-Provider AI Integration
- **Venice.ai** - Primary AI provider with OpenAI compatibility
- **OpenAI** - Text embeddings and fallback completions  
- **Anthropic** - Claude models for advanced reasoning
- **Perplexity** - Real-time web search and current information
- **Hugging Face** - Open-source model integration

### Intelligent Agent Capabilities
- **Dynamic Model Selection** - Automatically chooses optimal models based on task requirements
- **Cost Optimization** - Real-time cost tracking and budget management
- **Persistent Memory** - Vector-based memory storage using Qdrant
- **Image Generation** - Venice.ai native image generation with advanced parameters
- **Task-Aware Processing** - Specialized handling for text, code, and image tasks

### Web Interface
- Interactive chat interface
- Cost monitoring dashboard
- Model performance analytics
- Usage history and statistics
- Admin panel for configuration

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Venice.ai API key ([Get one here](https://venice.ai))
- Optional: OpenAI API key for embeddings
- Optional: Qdrant instance for persistent memory

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/iodev/VeniceIntelligence.git
   cd VeniceIntelligence
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # Or using uv (recommended)
   uv sync
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the application**
   ```bash
   # Using Python
   python main.py
   
   # Using uv
   uv run python main.py
   
   # Using Gunicorn (production)
   gunicorn --bind 0.0.0.0:5000 main:app
   ```

5. **Access the application**
   Open your browser to `http://localhost:5000`

## ⚙️ Configuration

### Required Environment Variables

```env
# Venice.ai API (Required)
VENICE_API_KEY=your-venice-api-key-here

# Optional: Additional providers
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
PERPLEXITY_API_KEY=your-perplexity-api-key-here
HUGGING_FACE_API_KEY_IO=your-hf-api-key-here

# Optional: Vector database for persistent memory
QDRANT_URL=your-qdrant-url-here
QDRANT_API_KEY=your-qdrant-api-key-here
```

### Venice.ai Setup

1. Visit [Venice.ai](https://venice.ai) and create an account
2. Navigate to API settings and generate an API key
3. Add the key to your `.env` file as `VENICE_API_KEY`

The application will work with just the Venice.ai API key, with other providers being optional enhancements.

## 🏗️ Architecture

### Core Components

```
├── agent/                  # Core AI agent implementation
│   ├── core.py            # Main agent logic and orchestration
│   ├── models.py          # Venice.ai client and model management
│   ├── memory.py          # Persistent memory with Qdrant
│   ├── cost_control.py    # Cost monitoring and optimization
│   ├── image.py           # Image generation capabilities
│   ├── anthropic_client.py # Anthropic integration
│   ├── perplexity.py      # Perplexity web search
│   └── huggingface_client.py # HuggingFace integration
├── templates/             # Web interface templates
├── static/               # CSS, JS, and other static assets
├── models.py            # Database models for tracking
├── app.py              # Flask web application
├── main.py            # Application entry point
└── config.py         # Configuration management
```

### Agent Workflow

1. **Request Processing** - Analyze user input and determine task type
2. **Model Selection** - Choose optimal model based on task requirements and cost
3. **Provider Routing** - Route to Venice.ai or fallback providers
4. **Memory Integration** - Store and retrieve relevant context
5. **Cost Tracking** - Monitor usage and optimize for efficiency
6. **Response Generation** - Generate and format final response

## 📊 Venice.ai Integration

### Supported Models

The agent automatically detects and utilizes available Venice.ai models:

- **mistral-31-24b** - High-performance reasoning
- **llama-3.2-3b** - Fast, efficient processing  
- **llama-3.3-70b** - Advanced language understanding
- **And more** - Automatically discovered via Venice.ai API

### Venice.ai Features Demonstrated

- **OpenAI Compatibility** - Seamless integration using OpenAI client libraries
- **Native API Access** - Direct Venice.ai API calls for advanced features
- **Image Generation** - Venice.ai's native image generation with custom parameters
- **Model Discovery** - Automatic detection of available models
- **Cost Optimization** - Real-time usage tracking and cost management

## 🔧 Development

### Project Structure

```python
# Core agent initialization
agent = Agent(
    venice_client=VeniceClient(api_key="your-key"),
    memory_manager=MemoryManager(),
    cost_monitor=CostMonitor()
)

# Process requests
response = await agent.process_message(
    message="Your question here",
    task_type="text"  # or "code", "image"
)
```

### Adding New Providers

1. Create client class in `agent/` directory
2. Implement standard interface methods
3. Add to provider registry in `agent/core.py`
4. Update configuration in `config.py`

### Running Tests

```bash
# Test Venice.ai API connectivity
python test_venice_api.py

# Test agent functionality  
python test_agent.py

# Test OpenAI embeddings (if configured)
python test_openai_embeddings.py
```

## 📈 Monitoring & Analytics

### Cost Tracking
- Real-time cost monitoring per provider
- Token usage analytics
- Cost optimization recommendations
- Budget alerts and controls

### Performance Metrics
- Response time tracking
- Model accuracy scoring
- Provider reliability statistics
- Usage pattern analysis

### Web Dashboard
- Interactive cost monitoring
- Model performance comparisons
- Usage history and trends
- Configuration management

## 🐳 Docker Deployment

```dockerfile
# Build image
docker build -t venice-intelligence .

# Run container
docker run -p 5000:5000 \
  -e VENICE_API_KEY=your-key \
  venice-intelligence
```

## 🤝 Contributing

We welcome contributions! Please see our guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/VeniceIntelligence.git

# Install development dependencies
uv sync --dev

# Run tests
python -m pytest

# Run with auto-reload
uv run python main.py
```

## 🔒 Security

- **No Hardcoded Secrets** - All API keys use environment variables
- **Input Sanitization** - User inputs are properly validated
- **Rate Limiting** - Configurable rate limits for cost control
- **Session Management** - Secure session handling for web interface

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Venice.ai](https://venice.ai) - Primary AI provider with excellent OpenAI compatibility
- [OpenAI](https://openai.com) - Embeddings and API compatibility standards
- [Anthropic](https://anthropic.com) - Claude models for advanced reasoning
- [Perplexity](https://perplexity.ai) - Real-time web search capabilities
- [Hugging Face](https://huggingface.co) - Open-source model ecosystem

## 🔗 Links

- [Venice.ai Documentation](https://docs.venice.ai)
- [Venice.ai API Reference](https://api.venice.ai/docs)
- [Project Issues](https://github.com/iodev/VeniceIntelligence/issues)
- [Project Wiki](https://github.com/iodev/VeniceIntelligence/wiki)

## 📞 Support

- **GitHub Issues** - For bugs and feature requests
- **Discussions** - For questions and community support
- **Venice.ai Discord** - For Venice.ai specific questions

---

**Start building intelligent AI agents with Venice.ai today!** 🚀