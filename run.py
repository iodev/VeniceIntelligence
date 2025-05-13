#!/usr/bin/env python3
"""
Run script to start the application with gunicorn
"""
import os
import sys
import subprocess
import logging
import signal

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run")

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

def signal_handler(sig, frame):
    """
    Handle signals to properly shut down the gunicorn process
    """
    if sig == signal.SIGINT:
        logger.info("Caught SIGINT, shutting down...")
        sys.exit(0)
    elif sig == signal.SIGTERM:
        logger.info("Caught SIGTERM, shutting down...")
        sys.exit(0)

def main():
    """
    Run the gunicorn server with proper configuration
    """
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Configure gunicorn command
    cmd = [
        "gunicorn",
        "--bind", "0.0.0.0:5000",
        "--reuse-port",
        "--reload",
        "--log-file", "logs/gunicorn.log",
        "--log-level", "info",
        "--access-logfile", "logs/access.log",
        "--error-logfile", "logs/error.log",
        "main:app"
    ]
    
    logger.info(f"Starting server with command: {' '.join(cmd)}")
    
    try:
        # Start gunicorn with subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for the process to complete
        stdout, stderr = process.communicate()
        
        # Check if there were errors
        if process.returncode != 0:
            logger.error(f"Gunicorn exited with code {process.returncode}")
            if stderr:
                logger.error(f"Stderr: {stderr.decode('utf-8')}")
        else:
            logger.info("Gunicorn process completed successfully")
            
    except Exception as e:
        logger.error(f"Error running gunicorn: {str(e)}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()