#!/usr/bin/env python3

import json
import logging.config
import os
from pathlib import Path
from server.db import init_sqlite_db
import threading
import time

# Set up logging first
logging.config.fileConfig('config/logging.conf')
logger = logging.getLogger(__name__)

from server.api import app, start_api
from server.schema_discovery import SchemaDiscovery

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
        # First, initialize SQLite database (for write operations)
        logger.info("Initializing SQLite database...")
        if not init_sqlite_db():
            logger.error("Failed to initialize SQLite database")
            return False
        logger.info("SQLite database initialized successfully")

        # Then, validate MariaDB schema (for read-only operations)
        try:
            schema_discovery = SchemaDiscovery()
            schema = schema_discovery.discover_schema()

            # Just validate schema - don't try to create it (read-only)
            if not schema_discovery.validate_schema(schema):
                logger.warning("Home Assistant database schema is not as expected. " +
                              "Some queries may fail but read-only operations should work.")
            else:
                logger.info("Home Assistant database schema validated successfully")
        except Exception as e:
            logger.warning(f"Home Assistant database schema validation failed: {str(e)}")
            logger.warning("Continuing with SQLite only - some features may be limited")

        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False

def start_processing_scheduler():
    """Start a background thread that periodically processes Bluetooth data."""
    from server.bluetooth_processor import BluetoothProcessor
    from server.blueprint_generator import BlueprintGenerator

    bluetooth_processor = BluetoothProcessor()
    blueprint_generator = BlueprintGenerator()

    def process_loop():
        while True:
            try:
                logger.info("Running scheduled Bluetooth data processing")
                result = bluetooth_processor.process_bluetooth_sensors()

                # Log the results - FIXED
                logger.info(f"Processed {result.get('processed', 0)} entities, "
                           f"found {len(result.get('device_positions', {}))} device positions, "
                           f"detected {len(result.get('room_list', []))} rooms")

                # Generate blueprint if we have enough data
                if result:
                    logger.info("Generating blueprint from processed data")
                    blueprint = blueprint_generator.generate_blueprint(
                        positions=result.get('device_positions', {}),  # Changed from 'positions'
                        rooms=result.get('room_list', [])  # Changed from 'rooms'
                    )
                    logger.info(f"Blueprint generated with {len(blueprint.get('rooms', []))} rooms")

            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}")

            # Sleep for 5 minutes before next processing
            time.sleep(300)

    # Start the processing thread
    processing_thread = threading.Thread(target=process_loop, daemon=True)
    processing_thread.start()
    logger.info("Background processing scheduler started")


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

        start_processing_scheduler()
        logger.info("Background processing started")

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