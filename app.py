import os
import logging
import json
import datetime
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
# Removed commercial features for open source release
from models import ModelPerformance, UsageCost, ModelEfficiency, CostControlStrategy, ChatSession, ChatMessage
import config

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add Jinja2 custom filters
@app.template_filter('format_timestamp')
def format_timestamp(timestamp):
    """Format a Unix timestamp to a readable date/time format"""
    try:
        dt = datetime.datetime.fromtimestamp(timestamp)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError):
        return 'Invalid timestamp'

@app.template_filter('format_model_name')
def format_model_name(model_id):
    """Format model names for display"""
    if not model_id:
        return 'Unknown Model'
    
    # Clean up model names for better display
    model_name = str(model_id)
    
    # Replace common patterns for readability
    replacements = {
        'claude-3-5-sonnet-20241022': 'Claude 3.5 Sonnet',
        'claude-3-opus-20240229': 'Claude 3 Opus',
        'claude-3-sonnet-20240229': 'Claude 3 Sonnet',
        'claude-3-haiku-20240307': 'Claude 3 Haiku',
        'llama-3.1-sonar-small-128k-online': 'Llama 3.1 Sonar Small',
        'llama-3.1-sonar-large-128k-online': 'Llama 3.1 Sonar Large',
        'llama-3.1-sonar-huge-128k-online': 'Llama 3.1 Sonar Huge',
        'mistral-31-24b': 'Mistral 3.1 24B',
        'gpt-4o': 'GPT-4o',
        'gpt-4': 'GPT-4',
        'gpt-3.5-turbo': 'GPT-3.5 Turbo'
    }
    
    return replacements.get(model_name, model_name.replace('-', ' ').title())

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

# Commercial features moved to enterprise repository



@app.route('/chat', methods=['POST', 'GET'])
def chat():
    """Handle chat interactions with the agent"""
    if agent_api is None:
        return jsonify({"error": "Agent API is not initialized"}), 500
    
# Simple rate limiting for community edition
    user_id = session.get('user_id', 'anonymous')
    
    # Basic rate limiting - 100 requests per hour for community version
    rate_limit_key = f"rate_limit_{user_id}"
    current_hour = datetime.datetime.now().hour
    session_key = f"{rate_limit_key}_{current_hour}"
    
    requests_this_hour = session.get(session_key, 0)
    if requests_this_hour >= 100:  # Community limit
        return jsonify({
            "error": "Rate limit exceeded (100 requests/hour). This is the community edition.",
            "note": "Enterprise version available with unlimited requests",
            "contact": "Visit our website for enterprise options"
        }), 429
    
    session[session_key] = requests_this_hour + 1
    
    # Support both POST (JSON) and GET (for EventSource)
    stream = False
    
    if request.method == 'POST':
        data = request.json if request.json else {}
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
    
    # Get or create session ID for conversation tracking
    session_id = session.get('conversation_id')
    if not session_id:
        session_id = str(datetime.datetime.now().timestamp())
        session['conversation_id'] = session_id
        logger.debug(f"Created new conversation session: {session_id}")
    
    try:
        if stream:
            return stream_chat(query, system_prompt, query_type, session_id)
        else:
            # Get response from the agent API
            # The agent API layer handles all business logic
            result = agent_api.process_query(
                query=query, 
                system_prompt=system_prompt, 
                query_type=query_type,
                agent_id=None,
                provider=None,
                model_id=None,
                stream=False,
                session_id=session_id  # Pass session ID for conversation continuity
            )
            
            if result.get('status') == 'success':
                # Return the response with basic metadata (community edition)
                return jsonify({
                    "response": result.get('response'),
                    "model_used": result.get('model_used', 'unknown'),
                    "provider_used": result.get('provider_used', 'unknown'),
                    "query_type": query_type,
                    "conversation_id": session_id,
                    "success": True,
                    "note": "Community Edition - Enterprise features available"
                })
            else:
                # Return error from API
                return jsonify({"error": result.get('error')}), 500
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500

def stream_chat(query, system_prompt, query_type="text", session_id=None):
    """Stream chat responses to the client"""
    def generate():
        try:
            if agent_api is None:
                error_json = json.dumps({"error": "Agent API is not initialized", "type": "error"})
                yield f'data: {error_json}\n\n'
                return
                
            # Request streaming response from the API layer
            # This abstracts away the details of memory, model selection, etc.
            result = agent_api.process_query(
                query=query,
                system_prompt=system_prompt,
                query_type=query_type,
                stream=True,
                session_id=session_id  # Pass session ID for conversation continuity
            )
            
            if result.get('status') != 'success':
                error_json = json.dumps({"error": result.get("error", "Unknown error"), "type": "error"})
                yield f'data: {error_json}\n\n'
                return
                
            # Get the response stream from the result
            response_stream = result.get('response_stream')
            
            if not response_stream:
                error_json = json.dumps({"error": "No response stream available", "type": "error"})
                yield f'data: {error_json}\n\n'
                return
                
            # Stream the response chunks to the client
            for chunk in response_stream:
                if not chunk:
                    continue
                chunk_json = json.dumps({"chunk": chunk, "type": "chunk"})
                yield f'data: {chunk_json}\n\n'
                
            # Signal that we're done
            done_json = json.dumps({"done": True, "type": "end"})
            yield f'data: {done_json}\n\n'
        except Exception as e:
            logger.error(f"Error streaming chat: {str(e)}")
            error_json = json.dumps({"error": str(e), "type": "error"})
            yield f'data: {error_json}\n\n'
    
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
    
    data = request.json if request.json else {}
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
        data = request.json if request.json else {}
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
        data = request.json if request.json else {}
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
        data = request.json if request.json else {}
        action = data.get('action')
        
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
            provider = data.get('provider')
            model_id = data.get('model_id')
            
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
            provider = data.get('provider')
            model_id = data.get('model_id')
            
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
        data = request.json if request.json else {}
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
    if agent is None or not hasattr(agent, 'cost_monitor') or agent.cost_monitor is None:
        flash("Cost monitoring is not initialized", "danger")
        return redirect(url_for('index'))
    
    try:
        # Get cost summary and efficiency metrics
        cost_summary = agent.cost_monitor.get_cost_summary()
        efficiency_metrics = agent.cost_monitor.get_efficiency_metrics()
        strategy = agent.cost_monitor.get_current_strategy()
        
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
    if agent is None or not hasattr(agent, 'cost_monitor') or agent.cost_monitor is None:
        return jsonify({"error": "Cost monitoring is not initialized"}), 500
    
    try:
        daily_budget = float(request.form.get('daily_budget', 1.0))
        
        # Update strategy with new budget
        strategy_data = {"daily_budget": daily_budget}
        success = agent.cost_monitor.update_strategy(strategy_data)
        
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
    if agent is None or not hasattr(agent, 'cost_monitor') or agent.cost_monitor is None:
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
        success = agent.cost_monitor.update_strategy(strategy_data)
        
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
    if api_status["venice"] and venice_client:
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


# ==================== CONVERSATION MANAGEMENT ROUTES ====================

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get all chat sessions"""
    try:
        sessions = ChatSession.query.filter_by(is_active=True).order_by(ChatSession.updated_at.desc()).all()
        return jsonify({
            'sessions': [session.to_dict() for session in sessions],
            'count': len(sessions)
        })
    except Exception as e:
        logger.error(f"Error retrieving sessions: {str(e)}")
        return jsonify({'error': 'Failed to retrieve sessions'}), 500


@app.route('/api/sessions', methods=['POST'])
def create_session():
    """Create a new chat session"""
    try:
        import uuid
        session_id = str(uuid.uuid4())
        title = request.json.get('title', f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")
        
        session = ChatSession(session_id=session_id, title=title)
        db.session.add(session)
        db.session.commit()
        
        return jsonify(session.to_dict()), 201
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        return jsonify({'error': 'Failed to create session'}), 500


@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get a specific session with its messages"""
    try:
        session = ChatSession.query.get_or_404(session_id)
        messages = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.timestamp).all()
        
        return jsonify({
            'session': session.to_dict(),
            'messages': [msg.to_dict() for msg in messages]
        })
    except Exception as e:
        logger.error(f"Error retrieving session {session_id}: {str(e)}")
        return jsonify({'error': 'Failed to retrieve session'}), 500


@app.route('/api/sessions/<session_id>/messages', methods=['POST'])
def add_message(session_id):
    """Add a message to a session"""
    try:
        data = request.json
        message = ChatMessage(
            session_id=session_id,
            message_type=data.get('message_type', 'user'),
            content=data.get('content', ''),
            query_type=data.get('query_type'),
            model_used=data.get('model_used'),
            provider_used=data.get('provider_used'),
            extra_data=json.dumps(data.get('metadata', {})) if data.get('metadata') else None
        )
        
        db.session.add(message)
        
        # Update session updated_at timestamp
        session = ChatSession.query.get(session_id)
        if session:
            session.updated_at = datetime.utcnow()
            
        db.session.commit()
        
        return jsonify(message.to_dict()), 201
    except Exception as e:
        logger.error(f"Error adding message to session {session_id}: {str(e)}")
        return jsonify({'error': 'Failed to add message'}), 500


@app.route('/api/sessions/<session_id>/share', methods=['POST'])
def share_session(session_id):
    """Generate a share token for a session"""
    try:
        session = ChatSession.query.get_or_404(session_id)
        share_token = session.generate_share_token()
        db.session.commit()
        
        return jsonify({
            'share_token': share_token,
            'share_url': f"{request.url_root}shared/{share_token}"
        })
    except Exception as e:
        logger.error(f"Error sharing session {session_id}: {str(e)}")
        return jsonify({'error': 'Failed to share session'}), 500


@app.route('/api/sessions/<session_id>/export')
def export_session(session_id):
    """Export session as JSON or markdown"""
    try:
        session = ChatSession.query.get_or_404(session_id)
        messages = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.timestamp).all()
        
        format_type = request.args.get('format', 'json')
        
        if format_type == 'markdown':
            md_content = f"# {session.title}\n\n"
            md_content += f"**Created:** {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            for msg in messages:
                if msg.message_type == 'user':
                    md_content += f"**User:** {msg.content}\n\n"
                elif msg.message_type == 'assistant':
                    model_info = f" (via {msg.model_used})" if msg.model_used else ""
                    md_content += f"**Assistant{model_info}:** {msg.content}\n\n"
            
            return md_content, 200, {'Content-Type': 'text/markdown'}
        
        else:  # JSON format
            export_data = {
                'session': session.to_dict(),
                'messages': [msg.to_dict() for msg in messages],
                'export_timestamp': datetime.utcnow().isoformat()
            }
            return jsonify(export_data)
            
    except Exception as e:
        logger.error(f"Error exporting session {session_id}: {str(e)}")
        return jsonify({'error': 'Failed to export session'}), 500


@app.route('/shared/<share_token>')
def view_shared_session(share_token):
    """View a shared session"""
    try:
        session = ChatSession.query.filter_by(share_token=share_token).first_or_404()
        messages = ChatMessage.query.filter_by(session_id=session.id).order_by(ChatMessage.timestamp).all()
        
        return render_template('shared_chat.html', 
                             session=session, 
                             messages=messages)
    except Exception as e:
        logger.error(f"Error viewing shared session {share_token}: {str(e)}")
        return render_template('500.html'), 404


@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a chat session"""
    try:
        session = ChatSession.query.get_or_404(session_id)
        session.is_active = False  # Soft delete
        db.session.commit()
        
        return jsonify({'message': 'Session deleted successfully'}), 200
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {str(e)}")
        return jsonify({'error': 'Failed to delete session'}), 500


@app.route('/history')
def chat_history():
    """Chat history page"""
    return render_template('history.html')
