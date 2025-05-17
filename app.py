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
from models import ModelPerformance
import config

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create references for the agent components
venice_client = None
venice_image_client = None
memory_manager = None
agent = None

# Function to initialize the agent (to be called within app context)
def init_agent():
    global venice_client, venice_image_client, memory_manager, agent
    
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
    if agent is None:
        return jsonify({"error": "Agent is not initialized"}), 500
    
    # Support both POST (JSON) and GET (for EventSource)
    stream = False
    
    if request.method == 'POST':
        data = request.json
        query = data.get('query', '')
        system_prompt = data.get('system_prompt', 'You are a helpful AI assistant.')
        stream = data.get('stream', False)
    else:  # GET method for EventSource
        query = request.args.get('query', '')
        system_prompt = request.args.get('system_prompt', 'You are a helpful AI assistant.')
        stream = request.args.get('stream') == 'true'
    
    if not query:
        return jsonify({"error": "Empty query"}), 400
    
    try:
        if stream:
            return stream_chat(query, system_prompt)
        else:
            # Get agent response (non-streaming)
            response, model_used = agent.process_query(query, system_prompt)
            
            # Return the response with metadata
            return jsonify({
                "response": response,
                "model_used": model_used,
                "success": True
            })
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500

def stream_chat(query, system_prompt):
    """Stream chat responses to the client"""
    def generate():
        try:
            # First, get the relevant context from memory
            memories = agent.memory_manager.get_relevant_memories(query)
            context = agent._create_context_from_memories(memories)
            
            # Construct the prompt with context
            messages = agent._construct_prompt(query, system_prompt, context)
            
            # Select a model (use the default for now)
            model = agent._get_best_model()
            
            # Start the generation with streaming
            stream_iterator = agent.venice_client.generate(
                messages=messages,
                model=model,
                max_tokens=500,
                temperature=0.7,
                stream=True
            )
            
            # First, yield the starting JSON
            yield 'data: {"model_used": "%s", "success": true, "type": "start"}\n\n' % model
            
            # Stream each chunk
            for chunk in stream_iterator:
                yield f'data: {{"chunk": {json.dumps(chunk)}, "type": "chunk"}}\n\n'
                
            # Final message
            # Store the interaction in memory asynchronously after sending response
            response_text = "Response saved to memory"  # Placeholder for stored response
            agent.memory_manager.store_interaction(query, response_text, system_prompt)
            
            # Update model performance asynchronously
            # For now we'll assume it was successful
            agent._update_model_performance(model, True, 1.0)  # Success, 1.0 second latency (placeholder)
            
            # End stream
            yield 'data: {"type": "end"}\n\n'
            
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            error_msg = str(e).replace('"', '\\"')
            yield f'data: {{"error": "{error_msg}", "type": "error"}}\n\n'
    
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
    """Set a new system prompt for the agent"""
    if agent is None:
        return jsonify({"error": "Agent is not initialized"}), 500
    
    data = request.json
    system_prompt = data.get('system_prompt', '')
    
    if not system_prompt:
        return jsonify({"error": "Empty system prompt"}), 400
    
    session['system_prompt'] = system_prompt
    return jsonify({"success": True})

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
        agent.memory_manager.clear_memory()
        return jsonify({"success": True, "message": "Memory reset successfully"})
    except Exception as e:
        logger.error(f"Error resetting memory: {str(e)}")
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

@app.route('/admin')
def admin():
    """Admin page with model information across platforms"""
    # Check API availabilities
    api_status = {
        "venice": False,
        "perplexity": False,
        "anthropic": False,
        "qdrant": False
    }
    
    api_keys = {
        "venice": bool(config.VENICE_API_KEY),
        "perplexity": bool(config.PERPLEXITY_API_KEY),
        "anthropic": bool(config.ANTHROPIC_API_KEY),
        "qdrant": bool(config.QDRANT_URL and config.QDRANT_API_KEY)
    }
    
    # Initialize clients as needed
    perplexity_client = None
    anthropic_client = None
    
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
    
    # Check Qdrant status
    if agent and agent.memory_manager:
        api_status["qdrant"] = True
    
    # Fetch model information from each platform
    venice_models = []
    perplexity_models = []
    anthropic_models = []
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
                           model_performance=model_performance,
                           strategy=strategy)
