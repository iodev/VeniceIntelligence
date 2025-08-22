# AI Agent System - Project Documentation

## Overview
A dynamic self-learning Python AI agent powered by Venice.ai as the primary provider, with intelligent routing to additional providers for optimal performance and cost efficiency. The system functions as a modular, self-contained component that demonstrates advanced AI orchestration patterns.

## Key Components
- **Venice.ai Primary Integration** - Mistral 3.1 24B, Llama 3.2 3B, Llama 3.3 70B as primary models
- **Smart Multi-Provider Routing** - Intelligent fallback to Anthropic, Perplexity, OpenAI for specialized tasks
- **Qdrant vector storage** for persistent learning mechanisms
- **Intelligent cost monitoring** and performance tracking with Venice.ai prioritization
- **Adaptive learning framework** with Venice.ai-first model selection
- **Content classifier** for optimal routing between Venice.ai and specialist providers
- **Model registry system** with Venice.ai models prominently featured

## Recent Changes
- **2025-08-22**: Added comprehensive conversation history storage and retrieval system
- **2025-08-22**: Implemented chat session management with export/sharing functionality
- **2025-08-22**: Established Venice.ai as primary provider with prominent model selection UI
- **2025-08-22**: Fixed template errors and SQLAlchemy session binding issues
- **2025-08-21**: Prepared clean community edition for open source release
- **2025-05-18**: Implemented OpenAI client integration with support for text, code, and image generation
- **2025-05-18**: Created content classifier that intelligently routes queries to specialized models
- **2025-05-18**: Enhanced model selection system to consider query content type (text, code, image, math)
- **2025-05-18**: Completed model registry system with dynamic model discovery across providers
- **2025-05-18**: Fixed SQLAlchemy compatibility issues and added in-memory tracking

## Project Architecture
The system is designed with complete separation between UI and agent core:

### Core Agent System (`agent/`)
- **agent/core.py**: Main agent logic with multi-provider support
- **agent/api.py**: Clean API interface for external system integration
- **agent/openai_client.py**: OpenAI API integration (text, image, vision)
- **agent/content_classifier.py**: Query type classification for optimal routing
- **agent/model_registry.py**: Centralized model management across providers
- **agent/anthropic_client.py**: Anthropic API integration
- **agent/perplexity.py**: Perplexity API integration
- **agent/models.py**: Venice API client
- **agent/memory.py**: Qdrant vector storage for persistent learning
- **agent/cost_control.py**: Cost monitoring and efficiency tracking
- **agent/evaluation.py**: Model performance evaluation

### Web Interface (`templates/`, `static/`)
- **Flask-based UI** for demonstration and admin dashboard
- **Completely separated** from core agent functionality
- Can be replaced with any other interface without affecting agent logic

## User Preferences
- Focus on clean, educational code that demonstrates AI agent patterns
- Prioritize Venice API as primary provider with intelligent fallbacks
- System must be modular and well-documented for community contributions
- Demonstrate best practices in AI orchestration and multi-provider routing
- Showcase advanced patterns like content classification and adaptive learning

## Technical Design Decisions
- **Query types** (text, code, image, math) used for model selection
- **Content-based routing** to appropriate specialized models
- **Session-based tracking** for conversation context
- **Multi-threading** for parallel provider queries in high-accuracy mode
- **Confidence scoring** system for optimal provider selection
- **Dynamic model registration** during initialization to handle API changes

## Current Status
✅ **Complete separation** between UI and agent system
✅ **Multi-provider integration** with intelligent fallbacks
✅ **Content type classification** for specialized model routing
✅ **Model registry system** with dynamic discovery
✅ **Cost monitoring** and performance tracking
✅ **Persistent memory** with Qdrant integration
✅ **Parallel execution** for high-accuracy mode
✅ **Conversation history** storage, retrieval, and export/sharing
✅ **Community edition** ready for open source release

## Future Considerations
The system demonstrates advanced AI agent architecture patterns and is designed for community contributions and learning.