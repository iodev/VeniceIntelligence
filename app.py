import os
import logging
import json
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, Response, stream_with_context
from agent.core import Agent
from agent.memory import MemoryManager
from agent.models import VeniceClient
import config

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", os.urandom(24).hex())
app.wsgi_app = app.wsgi_app

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize the agent components
try:
    venice_client = VeniceClient(
        api_key=config.VENICE_API_KEY,
        openai_api_key=config.OPENAI_API_KEY
    )
    memory_manager = MemoryManager(
        qdrant_url=config.QDRANT_URL,
        qdrant_api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME,
        vector_size=config.QDRANT_VECTOR_SIZE,
        embedding_model=config.EMBEDDING_MODEL,
        venice_client=venice_client
    )
    agent = Agent(
        venice_client=venice_client,
        memory_manager=memory_manager,
        available_models=config.AVAILABLE_MODELS,
        default_model=config.DEFAULT_MODEL
    )
    logger.info("Agent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize agent: {str(e)}")
    agent = None

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
