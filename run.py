#!/usr/bin/env python3
import os
import sys
import logging
import argparse
from server.db import test_connection
from server.schema_discovery import SchemaDiscovery
from server.api import app

def setup_logging():
    """Configure logging for the application."""
    logging.config.fileConfig('config/logging.conf')
    logger = logging.getLogger(__name__)
    return logger

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import flask
        import pymysql
        import numpy
        import scipy
        import sklearn
    except ImportError as e:
        print(f"Missing dependency: {str(e)}")
        print("Please install all required dependencies using: pip install -r requirements.txt")
        sys.exit(1)

def check_configuration():
    """Check if configuration files exist and are valid."""
    required_files = [
        'config/config.json',
        'config/logging.conf'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Missing configuration file: {file_path}")
            sys.exit(1)

def validate_database():
    """Validate database connection and schema."""
    try:
        # Test database connection
        if not test_connection():
            print("Failed to connect to database. Please check your database configuration.")
            sys.exit(1)
        
        # Validate schema
        schema_discovery = SchemaDiscovery()
        schema = schema_discovery.discover_schema()
        if not schema_discovery.validate_schema(schema):
            print("Invalid database schema. Please check your database setup.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Database validation failed: {str(e)}")
        sys.exit(1)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Home Assistant 3D Blueprint Generator')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Home Assistant 3D Blueprint Generator")
    
    try:
        # Check dependencies
        logger.info("Checking dependencies...")
        check_dependencies()
        
        # Check configuration
        logger.info("Checking configuration...")
        check_configuration()
        
        # Validate database
        logger.info("Validating database...")
        validate_database()
        
        # Start the Flask application
        logger.info(f"Starting server on {args.host}:{args.port}")
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
        
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
