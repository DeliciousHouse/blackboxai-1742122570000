#!/usr/bin/env python3

import json
import logging.config
import os
from pathlib import Path

from server.api import app, start_api
from server.schema_discovery import SchemaDiscovery

# Set up logging
logging.config.fileConfig('config/logging.conf')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from file."""
    try:
        config_path = Path('config/config.json')
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        return {}

def init_database():
    """Initialize database schema."""
    try:
        schema_discovery = SchemaDiscovery()
        if not schema_discovery.create_schema():
            logger.error("Failed to initialize database schema")
            return False
        logger.info("Database schema initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False

def main():
    """Main entry point."""
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")

        # Initialize database
        if not init_database():
            logger.error("Failed to initialize application")
            return

        # Start API server
        host = config.get('api', {}).get('host', '0.0.0.0')
        port = config.get('api', {}).get('port', 5000)
        debug = config.get('api', {}).get('debug', False)

        logger.info(f"Starting API server on {host}:{port}")
        start_api(host=host, port=port, debug=debug)

    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")

if __name__ == '__main__':
    main()
