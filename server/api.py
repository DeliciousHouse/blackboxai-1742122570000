import json
import logging.config
from datetime import datetime
from typing import Dict, Optional, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS
from db import execute_query, execute_write_query
from schema_discovery import SchemaDiscovery
from bluetooth_processor import BluetoothProcessor
from blueprint_generator import BlueprintGenerator

# Setup logging
logging.config.fileConfig('config/logging.conf')
logger = logging.getLogger('server.api')

# Initialize Flask application
app = Flask(__name__)

# Load configuration
with open('config/config.json') as f:
    config = json.load(f)

# Configure CORS
CORS(app, resources={r"/*": {"origins": config['api']['cors_origins']}})

# Initialize components
schema_discovery = SchemaDiscovery()
bluetooth_processor = BluetoothProcessor()
blueprint_generator = BlueprintGenerator()

def get_sensor_readings() -> Tuple[Optional[list], Optional[str]]:
    """
    Retrieve recent sensor readings from the database.
    
    Returns:
        Tuple[Optional[list], Optional[str]]: (readings, error_message)
    """
    try:
        # Get table name from schema config
        table_name = schema_discovery.schema_config.get('sensor_table', 'bluetooth_readings')
        
        # Query recent readings
        query = f"""
            SELECT sensor_id, rssi, sensor_location, device_id, timestamp
            FROM {table_name}
            WHERE timestamp >= NOW() - INTERVAL 5 MINUTE
            ORDER BY timestamp DESC
        """
        
        readings = execute_query(query)
        return readings, None
        
    except Exception as e:
        error_msg = f"Error retrieving sensor readings: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

@app.route('/api/blueprint', methods=['GET'])
def get_blueprint():
    """
    Generate and return the current blueprint based on recent sensor readings.
    """
    try:
        # Get recent sensor readings
        readings, error = get_sensor_readings()
        if error:
            return jsonify({'error': error}), 500
        
        if not readings:
            return jsonify({'error': 'No recent sensor readings available'}), 404
        
        # Process readings to get device positions
        device_positions = bluetooth_processor.process_readings(readings)
        
        if not device_positions:
            return jsonify({'error': 'Could not determine device positions'}), 404
        
        # Generate blueprint
        blueprint = blueprint_generator.generate_blueprint(device_positions)
        
        # Add timestamp
        blueprint['metadata']['timestamp'] = datetime.now().isoformat()
        
        return jsonify(blueprint)
        
    except Exception as e:
        error_msg = f"Error generating blueprint: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/api/blueprint/update', methods=['POST'])
def update_blueprint():
    """
    Accept manual updates to the blueprint (room names, wall positions, etc.).
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Validate update data
        if not validate_update_data(data):
            return jsonify({'error': 'Invalid update data'}), 400
        
        # Store the manual updates
        store_manual_updates(data)
        
        # Return the updated blueprint
        return get_blueprint()
        
    except Exception as e:
        error_msg = f"Error updating blueprint: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

def validate_update_data(data: Dict) -> bool:
    """
    Validate manual update data.
    
    Args:
        data (Dict): Update data to validate
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        required_fields = ['rooms', 'walls']
        if not all(field in data for field in required_fields):
            logger.error("Missing required fields in update data")
            return False
        
        # Validate rooms
        for room in data['rooms']:
            required_room_fields = ['id', 'type', 'vertices', 'center']
            if not all(field in room for field in required_room_fields):
                logger.error(f"Missing required fields in room data: {room}")
                return False
        
        # Validate walls
        for wall in data['walls']:
            required_wall_fields = ['id', 'room1_id', 'room2_id', 'start', 'end']
            if not all(field in wall for field in required_wall_fields):
                logger.error(f"Missing required fields in wall data: {wall}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating update data: {str(e)}")
        return False

def store_manual_updates(data: Dict):
    """
    Store manual updates in the database.
    
    Args:
        data (Dict): Update data to store
    """
    try:
        # Store room updates
        for room in data['rooms']:
            query = """
                INSERT INTO manual_updates (update_type, entity_id, data)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE data = VALUES(data)
            """
            params = ('room', room['id'], json.dumps(room))
            execute_write_query(query, params)
        
        # Store wall updates
        for wall in data['walls']:
            query = """
                INSERT INTO manual_updates (update_type, entity_id, data)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE data = VALUES(data)
            """
            params = ('wall', wall['id'], json.dumps(wall))
            execute_write_query(query, params)
        
        logger.info("Manual updates stored successfully")
        
    except Exception as e:
        logger.error(f"Error storing manual updates: {str(e)}")
        raise

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    """
    try:
        # Check database connection
        execute_query("SELECT 1")
        
        # Check schema discovery
        schema = schema_discovery.discover_schema()
        if not schema:
            raise Exception("Schema discovery failed")
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        error_msg = f"Health check failed: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'status': 'unhealthy',
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

def main():
    """
    Main function to run the Flask application.
    """
    try:
        # Discover and validate database schema
        schema = schema_discovery.discover_schema()
        if not schema_discovery.validate_schema(schema):
            raise Exception("Invalid database schema")
        
        # Start Flask application
        app.run(
            host=config['api']['host'],
            port=config['api']['port'],
            debug=config['api']['debug']
        )
        
    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
