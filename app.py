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

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat interactions with the agent"""
    if agent is None:
        return jsonify({"error": "Agent is not initialized"}), 500
    
    data = request.json
    query = data.get('query', '')
    system_prompt = data.get('system_prompt', 'You are a helpful AI assistant.')
    
    if not query:
        return jsonify({"error": "Empty query"}), 400
    
    try:
        # Get agent response
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
