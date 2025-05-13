import logging
import os

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set up logging
loglevel = 'info'
errorlog = 'logs/gunicorn-error.log'
accesslog = 'logs/gunicorn-access.log'

# Disable signal handling for WINCH to prevent those messages
disable_winch_handling = True

# Bind to all interfaces
bind = "0.0.0.0:5000"

# Enable auto-reload for development
reload = True
reuse_port = True

# Handle custom log filters
class NoWinchFilter(logging.Filter):
    def filter(self, record):
        return not (record.getMessage().startswith('Handling signal: winch'))

# Initialize logger
logger = logging.getLogger('gunicorn.error')
logger.addFilter(NoWinchFilter())

def on_starting(server):
    """
    Configure server logging to filter out winch signals
    """
    logger = logging.getLogger('gunicorn.error')
    logger.addFilter(NoWinchFilter())