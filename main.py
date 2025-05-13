import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize database base class
class Base(DeclarativeBase):
    pass

# Initialize Flask-SQLAlchemy
db = SQLAlchemy(model_class=Base)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", os.urandom(24).hex())

# Configure SQLAlchemy with PostgreSQL
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize the app with Flask-SQLAlchemy
db.init_app(app)

# Initialize all database models
with app.app_context():
    # Import models to register them with SQLAlchemy
    import models
    
    # Create all tables
    db.create_all()
    
    # Initialize default models if function exists
    try:
        from agent.core import init_default_models
        init_default_models()
        logger.info("Initialized default models")
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not initialize default models: {str(e)}")

# Import application routes after database initialization
import app as application