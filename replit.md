# AI Agent System - Project Documentation

## Overview
A dynamic self-learning Python AI agent that intelligently adapts model selection using advanced API integrations and vector storage technologies. The system functions as a modular, self-contained component that can be queried by and integrated with other agent systems in a larger network.

## Key Components
- **Venice.ai API integration** for model evaluation and selection
- **Multi-provider support** (Venice, Anthropic, Perplexity, OpenAI)
- **Qdrant vector storage** for persistent learning mechanisms
- **Intelligent cost monitoring** and performance tracking
- **Adaptive learning framework** with multi-model support
- **Content classifier** for optimal model selection based on query type
- **Model registry system** with dynamic discovery capabilities

## Recent Changes
- **2025-08-21**: Prepared repository for open source community release
- **2025-08-21**: Cleaned up commercial features for community edition focus
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

## Future Considerations
The system is positioned for potential commercialization or community contribution, with robust architecture supporting both paths.