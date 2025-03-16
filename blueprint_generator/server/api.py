import json
import logging
from typing import Dict, Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

from .blueprint_generator import BlueprintGenerator
from .bluetooth_processor import BluetoothProcessor
from .db import test_connection
from .schema_discovery import SchemaDiscovery

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize components
blueprint_generator = BlueprintGenerator()
bluetooth_processor = BluetoothProcessor()
schema_discovery = SchemaDiscovery()

# Set up logging
logger = logging.getLogger(__name__)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        db_status = test_connection()
        
        # Check schema
        schema_status = schema_discovery.validate_schema(
            schema_discovery.discover_schema()
        )

        status = {
            'status': 'healthy' if db_status and schema_status else 'unhealthy',
            'database': 'connected' if db_status else 'disconnected',
            'schema': 'valid' if schema_status else 'invalid'
        }
        
        return jsonify(status), 200 if status['status'] == 'healthy' else 503

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

@app.route('/api/blueprint', methods=['GET'])
def get_blueprint():
    """Get the latest blueprint."""
    try:
        blueprint = blueprint_generator.get_latest_blueprint()
        if blueprint:
            return jsonify(blueprint)
        return jsonify({'error': 'No blueprint found'}), 404

    except Exception as e:
        logger.error(f"Failed to get blueprint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/blueprint/generate', methods=['POST'])
def generate_blueprint():
    """Generate a new blueprint."""
    try:
        time_window = request.json.get('time_window', 300)
        blueprint = blueprint_generator.generate_blueprint(time_window)
        
        if blueprint:
            return jsonify(blueprint)
        return jsonify({'error': 'Failed to generate blueprint'}), 400

    except Exception as e:
        logger.error(f"Blueprint generation failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/blueprint/update', methods=['POST'])
def update_blueprint():
    """Update blueprint with manual changes."""
    try:
        blueprint_data = request.json
        if blueprint_generator.update_blueprint(blueprint_data):
            return jsonify({'status': 'success'})
        return jsonify({'error': 'Failed to update blueprint'}), 400

    except Exception as e:
        logger.error(f"Blueprint update failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sensor/reading', methods=['POST'])
def process_reading():
    """Process a new sensor reading."""
    try:
        data = request.json
        result = bluetooth_processor.process_reading(
            sensor_id=data['sensor_id'],
            rssi=data['rssi'],
            device_id=data['device_id'],
            sensor_location=data['sensor_location']
        )
        
        if result:
            return jsonify({'status': 'success'})
        return jsonify({'error': 'Failed to process reading'}), 400

    except Exception as e:
        logger.error(f"Failed to process sensor reading: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/positions', methods=['GET'])
def get_positions():
    """Get current device positions."""
    try:
        time_window = request.args.get('time_window', 300, type=int)
        positions = bluetooth_processor.estimate_positions(time_window)
        
        if positions:
            return jsonify(positions)
        return jsonify({'error': 'No positions found'}), 404

    except Exception as e:
        logger.error(f"Failed to get positions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/schema/validate', methods=['GET'])
def validate_schema():
    """Validate database schema."""
    try:
        current_schema = schema_discovery.discover_schema()
        is_valid = schema_discovery.validate_schema(current_schema)
        
        return jsonify({
            'valid': is_valid,
            'schema': current_schema
        })

    except Exception as e:
        logger.error(f"Schema validation failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/schema/create', methods=['POST'])
def create_schema():
    """Create database schema."""
    try:
        success = schema_discovery.create_schema()
        if success:
            return jsonify({'status': 'success'})
        return jsonify({'error': 'Failed to create schema'}), 500

    except Exception as e:
        logger.error(f"Schema creation failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

def start_api(host: str = '0.0.0.0', port: int = 8000, debug: bool = False):
    """Start the Flask API server."""
    app.run(host=host, port=port, debug=debug)
