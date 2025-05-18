# Agent System TODO List

## Priority 1: Critical Fixes
- [✓] Fix naming inconsistency in memory management: app.py calls `clear_memory()` but MemoryManager implements `clear_memories()` (FIXED: the method was already using clear_memories correctly)
- [✓] Fix undeclared/undefined `cost_monitor` reference in app.py's update_strategy route (line 522) (FIXED: replaced with agent.cost_monitor references)
- [✓] Fix `'Agent' object has no attribute 'model_performance'` error in the agent core module (FIXED: updated _get_best_model to use database directly instead of the local model_performance dictionary)
- [✓] Implement missing can_use_high_accuracy_mode method in CostMonitor class (FIXED: implemented the method with budget and complexity-based logic to determine when to use high accuracy mode)

## Priority 2: Method Implementation Issues (By Folder)
### agent/api.py
- [✓] Fix LSP error for ModelPerformance constructor in register_model method (lines 70-77) (FIXED: already correctly implemented with object initialization first)
- [✓] Fix incorrect UsageCost constructor call (lines 699-707) (FIXED: already correctly implemented with object initialization first)

### agent/core.py
- [✓] Fix LSP error for ModelPerformance constructor in init_default_models (lines 39-47) (FIXED: added missing required properties like provider, capabilities, context_window, and display_name)
- [✓] Fix LSP error for ModelPerformance constructor in _update_model_performance method (FIXED: added missing required properties like provider, capabilities, context_window, and display_name)
- [✓] Implement streaming response handler for non-Venice providers (FIXED: implemented streaming response handler for Anthropic client)

### agent/cost_control.py
- [✓] Fix constructor calls for CostControlStrategy, UsageCost, and ModelEfficiency (FIXED: updated all constructors to initialize objects first then set attributes)
- [✓] Fix reference errors to prioritize_cost, prioritize_speed, and prioritize_accuracy attributes (lines 335-337) (FIXED: added null checks before accessing attributes)

### agent/memory.py
- Fix QdrantClient dependency issues (possibly unbound error)
- Resolve type errors in search response handling for Qdrant results (line 215)
- Fix Payload type incompatibility with Dict[str, Any] return type

### agent/perplexity.py
- [✓] Standardize stream response handling with other clients (FIXED: streaming implementation already matches the standardized approach)

### agent/models.py
- Ensure streaming implementation is working correctly with error handling (warning seen in logs)

## Priority 3: Enhancements to Complete
- Complete the integration of dynamic model discovery between Perplexity and Anthropic
- Add utility to randomly select models for testing/evaluation
- Fix constructor issues in data models (CostControlStrategy, UsageCost, ModelEfficiency, ModelPerformance)
- Implement proper fallback mechanisms when primary model fails
- Standardize streaming response format across all providers
- Fix format_timestamp filter missing in history.html template

## Priority 4: Suggested Enhancements
- Add parallel query execution for high-accuracy mode
- Implement confidence scoring system to determine when multiple providers should be used
- Create a model registry system that tracks all available models across providers
- Add performance metrics dashboard for comparing model efficiency
- Implement automatic deprecation of underperforming models

## Folders/Files Reviewed
- ✓ agent/api.py
- ✓ agent/core.py
- ✓ agent/cost_control.py
- ✓ agent/evaluation.py
- ✓ agent/huggingface_client.py
- ✓ agent/memory.py
- ✓ agent/models.py
- ✓ agent/anthropic_client.py
- ✓ agent/perplexity.py
- ✓ agent/image.py
- ✓ app.py
- ✓ main.py
- ✓ models.py