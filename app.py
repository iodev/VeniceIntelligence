import os
import logging
import json
from flask import render_template, request, jsonify, session, redirect, url_for, flash, Response, stream_with_context, current_app
from main import app, db  # Import Flask app from main.py
from agent.core import Agent
from agent.memory import MemoryManager
from agent.models import VeniceClient
from agent.image import VeniceImageClient
from agent.perplexity import PerplexityClient
from agent.anthropic_client import AnthropicClient
from agent.huggingface_client import HuggingFaceClient
from agent.cost_control import CostMonitor
from agent.api import AgentAPI
from models import ModelPerformance, UsageCost, ModelEfficiency, CostControlStrategy
import config

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create references for the agent components
venice_client = None
venice_image_client = None
memory_manager = None
agent = None
agent_api = None  # API layer for external system integration

# Function to initialize the agent (to be called within app context)
def init_agent():
    global venice_client, venice_image_client, memory_manager, agent, agent_api
    
    try:
        # Initialize Venice API clients
        venice_client = VeniceClient(
            api_key=config.VENICE_API_KEY,
            openai_api_key=config.OPENAI_API_KEY
        )
        
        # Initialize the image generation client
        venice_image_client = VeniceImageClient(
            api_key=config.VENICE_API_KEY
        )
        
        # Initialize memory manager
        memory_manager = MemoryManager(
            qdrant_url=config.QDRANT_URL,
            qdrant_api_key=config.QDRANT_API_KEY,
            collection_name=config.QDRANT_COLLECTION_NAME,
            vector_size=config.QDRANT_VECTOR_SIZE,
            embedding_model=config.EMBEDDING_MODEL,
            venice_client=venice_client
        )
        
        # Initialize agent with the components
        agent = Agent(
            venice_client=venice_client,
            memory_manager=memory_manager,
            available_models=config.AVAILABLE_MODELS,
            default_model=config.DEFAULT_MODEL
        )
        
        # Initialize API layer for external system integration
        # This serves as the primary interface for other systems/nodes to interact with the agent
        agent_api = AgentAPI(agent)
        
        logger.info("Agent initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize agent: {str(e)}")
        return False

# Initialize the agent within app context
with app.app_context():
    # Initialize the agent
    init_success = init_agent()
    if not init_success:
        logger.error("Failed to initialize agent properly")

@app.route('/')
def index():
    """Home page with agent interface"""
    # Check if agent is properly initialized
    if agent is None:
        flash("Agent initialization failed. Please check your API keys and Qdrant configuration.", "danger")
    return render_template('index.html')

@app.route('/chat', methods=['POST', 'GET'])
def chat():
    """Handle chat interactions with the agent"""
    if agent_api is None:
        return jsonify({"error": "Agent API is not initialized"}), 500
    
    # Support both POST (JSON) and GET (for EventSource)
    stream = False
    
    if request.method == 'POST':
        data = request.json
        query = data.get('query', '')
        system_prompt = data.get('system_prompt', 'You are a helpful AI assistant.')
        query_type = data.get('query_type', 'text')  # Default to text if not specified
        stream = data.get('stream', False)
    else:  # GET method for EventSource
        query = request.args.get('query', '')
        system_prompt = request.args.get('system_prompt', 'You are a helpful AI assistant.')
        query_type = request.args.get('query_type', 'text')  # Default to text if not specified
        stream = request.args.get('stream') == 'true'
    
    if not query:
        return jsonify({"error": "Empty query"}), 400
    
    try:
        if stream:
            return stream_chat(query, system_prompt, query_type)
        else:
            # Get response from the agent API
            # The agent API layer handles all business logic
            result = agent_api.process_query(
                query=query, 
                system_prompt=system_prompt, 
                query_type=query_type
            )
            
            if result.get('status') == 'success':
                # Return the response with metadata
                return jsonify({
                    "response": result.get('response'),
                    "model_used": result.get('model_used'),
                    "query_type": query_type,
                    "success": True
                })
            else:
                # Return error from API
                return jsonify({"error": result.get('error')}), 500
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500

def stream_chat(query, system_prompt, query_type="text"):
    """Stream chat responses to the client"""
    def generate():
        try:
            if agent_api is None:
                yield 'data: {"error": "Agent API is not initialized", "type": "error"}\n\n'
                return
                
            # Request streaming response from the API layer
            # This abstracts away the details of memory, model selection, etc.
            result = agent_api.process_query(
                query=query,
                system_prompt=system_prompt,
                query_type=query_type,
                stream=True
            )
            
            if result.get('status') != 'success':
                yield f'data: {{"error": "{result.get("error", "Unknown error")}", "type": "error"}}\n\n'
                return
                
            # Get the response stream from the result
            response_stream = result.get('response_stream')
            
            if not response_stream:
                yield f'data: {{"error": "No response stream available", "type": "error"}}\n\n'
                return
                
            # Stream the response chunks to the client
            for chunk in response_stream:
                if not chunk:
                    continue
                yield f'data: {{"chunk": "{chunk}", "type": "token"}}\n\n'
                
            # Signal that we're done
            yield 'data: {"done": true, "type": "done"}\n\n'
        except Exception as e:
            logger.error(f"Error streaming chat: {str(e)}")
            yield f'data: {{"error": "{str(e)}", "type": "error"}}\n\n'
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/api/models')
def get_models():
    """Return available models and their performance metrics"""
    if agent is None:
        return jsonify({"error": "Agent is not initialized"}), 500
    
    models_data = agent.get_models_performance()
    return jsonify(models_data)

@app.route('/api/set_system_prompt', methods=['POST'])
def set_system_prompt():
    """Set a new system prompt for the agent (for UI use)"""
    if agent is None:
        return jsonify({"error": "Agent is not initialized"}), 500
    
    data = request.json
    system_prompt = data.get('system_prompt', '')
    
    if not system_prompt:
        return jsonify({"error": "Empty system prompt"}), 400
    
    session['system_prompt'] = system_prompt
    return jsonify({"success": True})
    
@app.route('/api/node/update_system_prompt', methods=['POST'])
def node_update_system_prompt():
    """
    Endpoint for parent nodes to update the system prompt
    This allows external systems to modify the agent's behavior
    """
    if agent_api is None:
        return jsonify({"error": "Agent API is not initialized"}), 500
    
    try:
        data = request.json
        system_prompt = data.get('system_prompt', '')
        parent_node_id = data.get('parent_node_id')
        
        if not system_prompt:
            return jsonify({"error": "Empty system prompt"}), 400
            
        # Use the agent API to update the system prompt
        result = agent_api.update_system_prompt(system_prompt, parent_node_id)
        
        if result.get('status') == 'success':
            return jsonify({
                "success": True,
                "message": "System prompt updated successfully",
                "previous_prompt": result.get('previous_prompt')
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get('error', 'Unknown error')
            }), 400
            
    except Exception as e:
        logger.error(f"Error updating system prompt from node: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
        
@app.route('/api/node/register', methods=['POST'])
def register_external_node():
    """
    Endpoint for external nodes to register with this agent
    This allows parent nodes in a larger system to connect to this agent
    """
    if agent_api is None:
        return jsonify({"error": "Agent API is not initialized"}), 500
    
    try:
        data = request.json
        node_id = data.get('node_id')
        node_info = data.get('node_info', {})
        
        if not node_id:
            return jsonify({"error": "Node ID is required"}), 400
            
        # Register the external node
        result = agent_api.register_external_node(node_id, node_info)
        
        if result.get('status') == 'success':
            return jsonify(result)
        else:
            return jsonify({
                "success": False,
                "error": result.get('error', 'Unknown error')
            }), 400
            
    except Exception as e:
        logger.error(f"Error registering external node: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/history')
def history():
    """View interaction history and model performance"""
    if agent is None:
        return render_template('history.html', error="Agent is not initialized")
    
    try:
        recent_interactions = agent.memory_manager.get_recent_interactions(limit=50)
        model_performance = agent.get_models_performance()
        return render_template('history.html', 
                              interactions=recent_interactions,
                              model_performance=model_performance)
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        return render_template('history.html', error=str(e))

@app.route('/api/reset_memory', methods=['POST'])
def reset_memory():
    """Reset the agent's memory"""
    if agent is None:
        return jsonify({"error": "Agent is not initialized"}), 500
    
    try:
        agent.memory_manager.clear_memories()
        return jsonify({"success": True, "message": "Memory reset successfully"})
    except Exception as e:
        logger.error(f"Error resetting memory: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/api/manage_models', methods=['POST'])
def manage_models():
    """Endpoint to manage models (refresh, deprecate, restore)"""
    if agent is None or agent_api is None:
        return jsonify({"error": "Agent is not initialized"}), 500
    
    try:
        action = request.json.get('action')
        
        if action == 'refresh':
            # Refresh available models from all providers
            result = agent_api.get_available_models()
            return jsonify({
                "success": True,
                "message": "Models refreshed successfully",
                "data": result
            })
            
        elif action == 'deprecate':
            # Mark a specific model as deprecated
            provider = request.json.get('provider')
            model_id = request.json.get('model_id')
            
            if not provider or not model_id:
                return jsonify({"error": "Provider and model ID are required"}), 400
            
            # Update model availability in database
            model = ModelPerformance.query.filter_by(
                provider=provider,
                model_id=model_id
            ).first()
            
            if model:
                model.is_available = False
                db.session.commit()
                return jsonify({
                    "success": True,
                    "message": f"Model {provider}:{model_id} marked as deprecated"
                })
            else:
                return jsonify({"error": f"Model {provider}:{model_id} not found"}), 404
                
        elif action == 'restore':
            # Restore a deprecated model
            provider = request.json.get('provider')
            model_id = request.json.get('model_id')
            
            if not provider or not model_id:
                return jsonify({"error": "Provider and model ID are required"}), 400
            
            # Update model availability in database
            model = ModelPerformance.query.filter_by(
                provider=provider,
                model_id=model_id
            ).first()
            
            if model:
                model.is_available = True
                db.session.commit()
                return jsonify({
                    "success": True,
                    "message": f"Model {provider}:{model_id} restored"
                })
            else:
                return jsonify({"error": f"Model {provider}:{model_id} not found"}), 404
        else:
            return jsonify({"error": f"Unknown action: {action}"}), 400
            
    except Exception as e:
        logger.error(f"Error managing models: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500
    
@app.route('/api/generate-image', methods=['POST'])
def generate_image():
    """Generate an image using Venice.ai's image generation API"""
    if venice_image_client is None:
        return jsonify({"error": "Image generation client is not initialized"}), 500
    
    try:
        data = request.json
        prompt = data.get('prompt', '')
        model = data.get('model', 'stable-diffusion-xl-1024-v1-0')
        size = data.get('size', '1024x1024')
        num_images = min(data.get('num_images', 1), 4)  # Limit to max 4 images
        
        if not prompt:
            return jsonify({"error": "Empty prompt"}), 400
        
        # Generate the image
        results = venice_image_client.generate_image(
            prompt=prompt,
            model=model,
            size=size,
            num_images=num_images
        )
        
        # Return image URLs and metadata
        return jsonify({
            "success": True,
            "images": results,
            "model_used": model
        })
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/api/image-models')
def get_image_models():
    """Return available image generation models"""
    if venice_image_client is None:
        return jsonify({"error": "Image generation client is not initialized"}), 500
    
    try:
        models = venice_image_client.get_available_image_models()
        return jsonify({"success": True, "models": models})
    except Exception as e:
        logger.error(f"Error fetching image models: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/cost-monitor')
def cost_monitor_page():
    """Cost monitoring and optimization dashboard"""
    if cost_monitor is None:
        flash("Cost monitoring is not initialized", "danger")
        return redirect(url_for('index'))
    
    try:
        # Get cost summary and efficiency metrics
        cost_summary = cost_monitor.get_cost_summary()
        efficiency_metrics = cost_monitor.get_efficiency_metrics()
        strategy = cost_monitor.get_current_strategy()
        
        # Ensure cost_summary has all required fields with default values
        if cost_summary is None:
            cost_summary = {
                'daily_spend': 0.0,
                'daily_budget': 1.0,
                'total_spend': 0.0,
                'request_count': 0,
                'provider_costs': {},
                'request_type_costs': {}
            }
        else:
            # Ensure all required fields exist
            if 'daily_spend' not in cost_summary:
                cost_summary['daily_spend'] = 0.0
            if 'daily_budget' not in cost_summary:
                cost_summary['daily_budget'] = 1.0
            if 'total_spend' not in cost_summary:
                cost_summary['total_spend'] = 0.0
            if 'request_count' not in cost_summary:
                cost_summary['request_count'] = 0
            if 'provider_costs' not in cost_summary:
                cost_summary['provider_costs'] = {}
            if 'request_type_costs' not in cost_summary:
                cost_summary['request_type_costs'] = {}
        
        # Ensure strategy has all required fields with default values
        if strategy is None:
            strategy = {
                'name': 'Default Strategy',
                'description': 'Balance between cost, speed, and accuracy',
                'daily_budget': 1.0,
                'prioritize_cost': 0.3,
                'prioritize_speed': 0.3,
                'prioritize_accuracy': 0.4,
                'cost_threshold': 0.8,
                'fallback_model': 'llama-3.2-3b'
            }
        
        return render_template('cost_monitor.html',
                              cost_summary=cost_summary,
                              efficiency_metrics=efficiency_metrics if efficiency_metrics else [],
                              strategy=strategy)
    except Exception as e:
        logger.error(f"Error rendering cost monitor page: {str(e)}")
        flash(f"Error loading cost monitoring data: {str(e)}", "danger")
        return redirect(url_for('index'))

@app.route('/admin/update-budget', methods=['POST'])
def update_budget():
    """Update the daily budget for cost control"""
    if cost_monitor is None:
        return jsonify({"error": "Cost monitoring is not initialized"}), 500
    
    try:
        daily_budget = float(request.form.get('daily_budget', 1.0))
        
        # Update strategy with new budget
        strategy_data = {"daily_budget": daily_budget}
        success = cost_monitor.update_strategy(strategy_data)
        
        if success:
            flash(f"Daily budget updated to ${daily_budget:.2f}", "success")
        else:
            flash("Failed to update budget", "danger")
            
        return redirect(url_for('cost_monitor_page'))
    except Exception as e:
        logger.error(f"Error updating budget: {str(e)}")
        flash(f"Error: {str(e)}", "danger")
        return redirect(url_for('cost_monitor_page'))

@app.route('/admin/update-strategy', methods=['POST'])
def update_strategy():
    """Update the model selection strategy"""
    if cost_monitor is None:
        return jsonify({"error": "Cost monitoring is not initialized"}), 500
    
    try:
        # Get basic strategy parameters
        strategy_data = {
            "name": request.form.get('name', 'Default Strategy'),
            "description": request.form.get('description', ''),
            "prioritize_cost": float(request.form.get('prioritize_cost', 0.3)),
            "prioritize_speed": float(request.form.get('prioritize_speed', 0.3)),
            "prioritize_accuracy": float(request.form.get('prioritize_accuracy', 0.4)),
            "cost_threshold": float(request.form.get('cost_threshold', 0.8)),
            "fallback_model": request.form.get('fallback_model', 'llama-3.2-3b')
        }
        
        # Build task mappings
        text_mapping = {
            "general": request.form.get('text_mapping_general', 'llama-3.1-sonar-small-128k-online'),
            "creative": request.form.get('text_mapping_creative', 'claude-3.5-sonnet-20241022'),
            "analytical": request.form.get('text_mapping_analytical', 'mistral-31-24b')
        }
        
        code_mapping = {
            "general": request.form.get('code_mapping_general', 'mistral-31-24b'),
            "python": request.form.get('code_mapping_python', 'mistral-31-24b'),
            "javascript": request.form.get('code_mapping_javascript', 'llama-3.2-3b')
        }
        
        image_mapping = {
            "general": request.form.get('image_mapping_general', 'stable-diffusion-xl-1024-v1-0')
        }
        
        # Add mappings to strategy data
        strategy_data["text_task_mapping"] = text_mapping
        strategy_data["code_task_mapping"] = code_mapping
        strategy_data["image_task_mapping"] = image_mapping
        
        # Update strategy
        success = cost_monitor.update_strategy(strategy_data)
        
        if success:
            flash("Strategy updated successfully", "success")
        else:
            flash("Failed to update strategy", "danger")
            
        return redirect(url_for('cost_monitor_page'))
    except Exception as e:
        logger.error(f"Error updating strategy: {str(e)}")
        flash(f"Error: {str(e)}", "danger")
        return redirect(url_for('cost_monitor_page'))

@app.route('/admin')
def admin():
    """Admin page with model information across platforms"""
    # Check API availabilities
    api_status = {
        "venice": False,
        "perplexity": False,
        "anthropic": False,
        "huggingface": False,
        "qdrant": False
    }
    
    api_keys = {
        "venice": bool(config.VENICE_API_KEY),
        "perplexity": bool(config.PERPLEXITY_API_KEY),
        "anthropic": bool(config.ANTHROPIC_API_KEY),
        "huggingface": bool(config.HUGGINGFACE_API_KEY),
        "qdrant": bool(config.QDRANT_URL and config.QDRANT_API_KEY)
    }
    
    # Initialize clients as needed
    perplexity_client = None
    anthropic_client = None
    huggingface_client = None
    
    # Check Venice API status
    if venice_client:
        try:
            # Simple test call
            api_status["venice"] = venice_client.test_connection()
        except Exception as e:
            logger.error(f"Error testing Venice API: {str(e)}")
    
    # Check Perplexity API status if key is available
    if api_keys["perplexity"]:
        try:
            perplexity_client = PerplexityClient()
            api_status["perplexity"] = perplexity_client.test_connection()
        except Exception as e:
            logger.error(f"Error testing Perplexity API: {str(e)}")
    
    # Check Anthropic API status if key is available
    if api_keys["anthropic"]:
        try:
            anthropic_client = AnthropicClient()
            api_status["anthropic"] = anthropic_client.test_connection()
        except Exception as e:
            logger.error(f"Error testing Anthropic API: {str(e)}")
    
    # Check Hugging Face API status if key is available
    if api_keys["huggingface"]:
        try:
            huggingface_client = HuggingFaceClient()
            api_status["huggingface"] = huggingface_client.test_connection()
        except Exception as e:
            logger.error(f"Error testing Hugging Face API: {str(e)}")
    
    # Check Qdrant status
    if agent and agent.memory_manager:
        api_status["qdrant"] = True
    
    # Fetch model information from each platform
    venice_models = []
    perplexity_models = []
    anthropic_models = []
    huggingface_models = []
    model_performance = []
    
    # Fetch Venice.ai models
    if api_status["venice"]:
        try:
            venice_models = venice_client.get_available_models()
        except Exception as e:
            logger.error(f"Error fetching Venice models: {str(e)}")
    
    # Fetch Perplexity models
    if api_status["perplexity"] and perplexity_client:
        try:
            perplexity_models = perplexity_client.get_available_models()
        except Exception as e:
            logger.error(f"Error fetching Perplexity models: {str(e)}")
    
    # Fetch Anthropic models
    if api_status["anthropic"] and anthropic_client:
        try:
            anthropic_models = anthropic_client.get_available_models()
        except Exception as e:
            logger.error(f"Error fetching Anthropic models: {str(e)}")
    
    # Fetch Hugging Face models
    if api_status["huggingface"] and huggingface_client:
        try:
            huggingface_models = huggingface_client.get_available_models()
        except Exception as e:
            logger.error(f"Error fetching Hugging Face models: {str(e)}")
    
    # Get model performance metrics from database
    try:
        model_performance = ModelPerformance.query.all()
    except Exception as e:
        logger.error(f"Error fetching model performance: {str(e)}")
    
    # Get dynamic routing strategy settings
    # For now, use default values
    strategy = {
        "code_model": "mistral-31-24b",
        "text_model": "llama-3.1-sonar-small-128k-online",
        "image_model": "stable-diffusion-xl-1024-v1-0",
        "use_dynamic_routing": True,
        "use_collaboration": False
    }
    
    return render_template('admin.html',
                           api_status=api_status,
                           api_keys=api_keys,
                           venice_models=venice_models,
                           perplexity_models=perplexity_models,
                           anthropic_models=anthropic_models,
                           huggingface_models=huggingface_models,
                           model_performance=model_performance,
                           strategy=strategy)
