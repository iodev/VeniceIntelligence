# System Patterns

## Architecture Overview
The system implements a multi-layer architecture with:
1. Core orchestration layer (supervisor_agent.py, orchestrator.py)
2. Agent communication layer (MCP servers with HTTP/SSE/TCP transports)
3. Infrastructure layer (Docker, deployment guides, and config files)
4. Security layer (OAuth proxy, documentation, and compliance checks)
5. User interface layer (standalone-web, static assets)

## Key Technical Decisions
1. Docker-based deployment for all components
2. Semantic versioning for component management
3. Unified configuration through JSON files
4. Dual transport support (HTTP/SSE + TCP) for flexibility
5. Security-first approach with OAuth integration

## Component Relationships
1. Orchestrator manages agent coordination
2. MCP servers provide tool interfaces to agents
3. API layer exposes endpoints for system control
4. Database layer stores workflow state
5. UI layer provides monitoring capabilities

## Critical Implementation Paths
1. Agent registration and discovery
2. Task distribution and execution tracking
3. Real-time status updates through SSE
4. TCP transport reliability and error handling
5. Security integration with OAuth proxy

## Design Patterns Used
1. Observer pattern for status updates
2. Factory pattern for agent creation
3. Strategy pattern for transport selection
4. Facade pattern for API abstraction
5. Composite pattern for workflow structure
