# Contributing to AI Agent System - Community Edition

Thank you for your interest in contributing to the AI Agent System! This community edition showcases advanced AI orchestration patterns and welcomes contributions from developers, researchers, and AI enthusiasts.

## 🎯 Project Goals

This repository demonstrates:
- **Multi-provider AI routing** with intelligent failover
- **Content classification** for optimal model selection
- **Adaptive learning** with vector memory storage
- **Clean architecture** patterns for AI systems

## 🚀 Quick Start for Contributors

### Prerequisites
- Python 3.11+
- PostgreSQL (local or cloud)
- At least one AI provider API key

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-agent-community.git
   cd ai-agent-community
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Initialize database**
   ```bash
   python -c "from main import db; db.create_all()"
   ```

6. **Run the application**
   ```bash
   python main.py
   ```

## 🤝 How to Contribute

### Types of Contributions We Welcome

#### 🔧 Code Contributions
- **New provider integrations** (Mistral, Cohere, etc.)
- **Enhanced content classification** algorithms
- **Performance optimizations** for routing logic
- **Better error handling** and user feedback
- **Testing improvements** and coverage

#### 📚 Documentation
- **API documentation** and examples
- **Tutorial notebooks** for common use cases
- **Architecture guides** explaining design patterns
- **Deployment guides** for different platforms

#### 🐛 Bug Reports
- Clear reproduction steps
- Environment details (Python version, OS, etc.)
- Expected vs actual behavior
- Relevant logs or error messages

#### 💡 Feature Requests
- Use case description
- Proposed implementation approach
- Benefit to the community
- Willingness to contribute implementation

### Contribution Process

1. **Check existing issues** for similar work
2. **Create an issue** to discuss major changes
3. **Fork the repository** and create a feature branch
4. **Make your changes** with tests
5. **Submit a pull request** with clear description

### Branch Naming Convention
- `feature/provider-integration-mistral`
- `fix/content-classifier-edge-case`
- `docs/api-reference-examples`
- `test/coverage-improvement`

## 🏗️ Architecture Overview

### Core Components

```
agent/
├── core.py              # Main agent logic
├── content_classifier.py # Query type detection
├── model_registry.py    # Model management
├── api.py              # External interface
├── memory.py           # Vector storage
└── providers/          # AI provider clients
    ├── venice_client.py
    ├── anthropic_client.py
    └── perplexity.py
```

### Key Design Principles

1. **Separation of Concerns**: Each component has a single responsibility
2. **Provider Abstraction**: Easy to add new AI providers
3. **Configurable Routing**: Customizable model selection logic
4. **Resilient Architecture**: Graceful handling of provider failures

## 📝 Coding Standards

### Python Style
- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Document classes and methods with docstrings
- Maximum line length: 100 characters

### Code Example
```python
from typing import Dict, List, Optional

class ProviderClient:
    """Base class for AI provider clients."""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """Initialize the provider client.
        
        Args:
            api_key: API key for the provider
            base_url: Optional base URL override
        """
        self.api_key = api_key
        self.base_url = base_url
    
    def generate_text(self, prompt: str, model: str) -> Dict[str, Any]:
        """Generate text using the provider's API.
        
        Args:
            prompt: Input text prompt
            model: Model identifier
            
        Returns:
            Dictionary containing response and metadata
        """
        # Implementation here
        pass
```

### Testing Requirements
- Add tests for new functionality
- Maintain or improve test coverage
- Use descriptive test names
- Mock external API calls

### Documentation
- Update docstrings for code changes
- Add examples for new features
- Update README if needed
- Include inline comments for complex logic

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=agent --cov-report=html

# Run specific test file
python -m pytest tests/test_content_classifier.py
```

### Test Structure
```
tests/
├── unit/
│   ├── test_core.py
│   ├── test_classifier.py
│   └── test_providers/
├── integration/
│   ├── test_api.py
│   └── test_memory.py
└── fixtures/
    ├── sample_queries.py
    └── mock_responses.py
```

## 📋 Issue Labels

- `good-first-issue` - Great for newcomers
- `enhancement` - New feature requests
- `bug` - Something isn't working
- `documentation` - Documentation improvements
- `provider-integration` - Adding new AI providers
- `performance` - Performance optimization
- `testing` - Test-related changes

## 🔍 Review Process

### Pull Request Requirements
- Clear description of changes
- Link to related issue
- Tests pass
- Documentation updated
- No merge conflicts

### Review Criteria
- Code quality and style
- Test coverage
- Performance impact
- Documentation completeness
- Backward compatibility

## 🌟 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Invited to maintainer discussions
- Eligible for contributor badges

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Discord**: Join our community chat
- **Email**: maintainers@ai-agent-community.org

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🎉 Thank You

Every contribution, no matter how small, helps make this project better for the entire AI development community. We appreciate your time and effort!

---

*This is the community edition showcasing AI agent architecture patterns. Enterprise features are available separately for production use.*