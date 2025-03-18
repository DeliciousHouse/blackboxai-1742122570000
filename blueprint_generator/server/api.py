import json
import logging
from typing import Dict, Optional

from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS
import os

from .blueprint_generator import BlueprintGenerator
from .bluetooth_processor import BluetoothProcessor
from .db import test_connection, execute_query
from .schema_discovery import SchemaDiscovery
from .ha_client import HomeAssistantClient
import uuid

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize components
blueprint_generator = BlueprintGenerator()
bluetooth_processor = BluetoothProcessor()
schema_discovery = SchemaDiscovery()
ha_client = HomeAssistantClient()

# Set up logging
logger = logging.getLogger(__name__)

# Add root route handler
@app.route('/', methods=['GET'])
def index():
    """Serve the main page or redirect to API documentation."""
    try:
        # Check if templates directory exists and has index.html
        if os.path.exists(os.path.join(app.root_path, 'templates', 'index.html')):
            return render_template('index.html')
        else:
            # Create a simple HTML response
            return """
            <html>
                <head>
                    <title>3D Blueprint Generator</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                        h1 { color: #2c3e50; }
                        .container { max-width: 800px; margin: 0 auto; }
                        .endpoint { background: #f8f9fa; padding: 10px; margin-bottom: 10px; border-radius: 4px; }
                        code { background: #e9ecef; padding: 2px 4px; border-radius: 3px; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>3D Blueprint Generator</h1>
                        <p>API is running successfully. The following endpoints are available:</p>

                        <div class="endpoint">
                            <strong>GET /api/health</strong> - Check API health
                        </div>

                        <div class="endpoint">
                            <strong>GET /api/blueprint</strong> - Get the latest blueprint
                        </div>

                        <div class="endpoint">
                            <strong>POST /api/blueprint/generate</strong> - Generate a new blueprint
                        </div>

                        <div class="endpoint">
                            <strong>GET /api/positions</strong> - Get current device positions
                        </div>

                        <p>For more details, see the API documentation in Home Assistant.</p>
                    </div>
                </body>
            </html>
            """
    except Exception as e:
        logger.error(f"Error serving index page: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

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

@app.route('/api/scan', methods=['GET'])
def scan_entities():
    """Scan for relevant entities and report availability."""
    try:
        ha_client = HomeAssistantClient()

        # Very liberal scan - any entities that might be related
        all_entities = ha_client.find_entities_by_pattern([""], [])  # Get ALL entities
        ble_entities = [e for e in all_entities if 'ble' in e['entity_id'].lower()]
        distance_entities = [e for e in all_entities if 'distance' in e['entity_id'].lower()]
        position_entities = [e for e in all_entities if any(p in e['entity_id'].lower()
                                                       for p in ['position', 'bermuda', 'tracker', 'mmwave'])]

        # Organize by category
        entity_data = {
            'ble_entities': [e['entity_id'] for e in ble_entities],
            'position_entities': [e['entity_id'] for e in position_entities],
            'distance_entities': [e['entity_id'] for e in distance_entities],
            'sample_entities': [e['entity_id'] for e in all_entities[:10]]  # First 10 entities
        }

        return jsonify({
            'status': 'success',
            'entities_found': len(ble_entities) + len(position_entities) + len(distance_entities),
            'total_entities': len(all_entities),
            'entities': entity_data
        })
    except Exception as e:
        logger.error(f"Failed to scan entities: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/devices', methods=['GET'])
def get_devices():
    """Get list of tracked devices."""
    try:
        # Query unique device IDs from readings
        query = "SELECT DISTINCT device_id, MAX(timestamp) as last_seen FROM bluetooth_readings GROUP BY device_id ORDER BY last_seen DESC"
        results = execute_query(query)

        devices = [{'id': row[0], 'last_seen': row[1].isoformat() if row[1] else None} for row in results]
        return jsonify({'devices': devices})
    except Exception as e:
        logger.error(f"Failed to get devices: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/blueprint/generate', methods=['POST'])
def generate_blueprint():
    """Generate a new blueprint from collected data."""
    try:
        # Call the blueprint generator with correct method name
        time_window = request.json.get('time_window', 300) if request.json else 300
        result = blueprint_generator.generate_blueprint(time_window)

        return jsonify({
            'status': 'success',
            'message': 'Blueprint generation started',
            'job_id': str(uuid.uuid4())  # Generate a job ID
        })
    except Exception as e:
        logger.error(f"Failed to generate blueprint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/blueprint', methods=['GET'])
def get_blueprint():
    """Get the latest blueprint."""
    try:
        # Use correct method name
        blueprint = blueprint_generator.get_latest_blueprint()
        if (blueprint):
            return jsonify(blueprint)
        return jsonify({'error': 'No blueprint found'}), 404
    except Exception as e:
        logger.error(f"Failed to get blueprint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/blueprint/status', methods=['GET'])
def get_blueprint_status():
    """Get the status of the blueprint generation."""
    try:
        # Get the status of the generation process
        status = blueprint_generator.get_status()

        return jsonify(status)
    except Exception as e:
        logger.error(f"Failed to get blueprint status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/fix-schema', methods=['POST'])
def fix_schema():
    """Fix schema issues manually."""
    try:
        # Use the fix_schema_validation method
        schema_discovery = SchemaDiscovery()
        result = schema_discovery.fix_schema_validation()

        return jsonify({
            'success': result,
            'message': 'Schema fix attempted'
        })
    except Exception as e:
        logger.error(f"Schema fix failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/visualize', methods=['GET'])
def visualize_blueprint():
    """Render a visualization of the latest blueprint."""
    try:
        # Get the latest blueprint data
        blueprint = blueprint_generator.get_latest_blueprint()
        if not blueprint:
            return render_template('no_blueprint.html')

        return render_template('visualize.html', blueprint=blueprint)
    except Exception as e:
        logger.error(f"Failed to visualize blueprint: {str(e)}")
        return render_template('error.html', error=str(e))

@app.route('/api/sync/bermuda', methods=['POST'])
def sync_bermuda():
    """Sync positions from Bermuda Trilateration."""
    try:
        positions = ha_client.get_bermuda_positions()

        if not positions:
            return jsonify({'message': 'No Bermuda positions found'}), 404

        # Store positions in database
        for position in positions:
            execute_write_query("""
            INSERT INTO device_positions
            (device_id, position_data, source, timestamp)
            VALUES (%s, %s, 'bermuda', NOW())
            ON DUPLICATE KEY UPDATE position_data = %s, timestamp = NOW()
            """, (position['device_id'], json.dumps(position['position']), json.dumps(position['position'])))

        return jsonify({
            'success': True,
            'count': len(positions),
            'message': f'Synced {len(positions)} positions from Bermuda'
        })

    except Exception as e:
        logger.error(f"Bermuda sync failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sync/esp32-ble', methods=['POST'])
def sync_esp32_ble():
    """Sync data from ESP32 BLE Monitor."""
    try:
        devices = ha_client.get_private_ble_devices()

        if not devices:
            return jsonify({'message': 'No ESP32 BLE Monitor devices found'}), 404

        # Store readings in database
        count = 0
        for device in devices:
            if device.get('rssi') and device.get('mac'):
                execute_write_query("""
                INSERT INTO w_readings
                (timestamp, sensor_id, rssi, device_id, sensor_location)
                VALUES (NOW(), %s, %s, %s, %s)
                """, ('esp32_monitor', device['rssi'], device['mac'],
                      json.dumps({'x': 0, 'y': 0, 'z': 0})))
                count += 1

        return jsonify({
            'success': True,
            'count': count,
            'message': f'Synced {count} ESP32 BLE Monitor devices'
        })

    except Exception as e:
        logger.error(f"ESP32 BLE sync failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

# AI-related endpoints

@app.route('/api/ai/status', methods=['GET'])
def get_ai_status():
    """Get the status of all AI models."""
    try:
        # Initialize AI processor if needed
        ai_processor = blueprint_generator.ai_processor

        # Get status of all models
        status = ai_processor.get_models_status()

        return jsonify(status)
    except Exception as e:
        logger.error(f"Failed to get AI status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/train/rssi-distance', methods=['POST'])
def train_rssi_distance_model():
    """Train the RSSI-to-distance model."""
    try:
        # Get training parameters from request
        params = request.json or {}

        # Initialize AI processor
        ai_processor = blueprint_generator.ai_processor

        # Train the model
        result = ai_processor.train_rssi_distance_model(
            model_type=params.get('model_type', 'random_forest'),
            test_size=params.get('test_size', 0.2),
            features=params.get('features', ['rssi']),
            hyperparams=params.get('hyperparams', {})
        )

        return jsonify({
            'success': result.get('success', False),
            'metrics': result.get('metrics', {}),
            'message': 'RSSI-to-distance model training completed'
        })
    except Exception as e:
        logger.error(f"RSSI-to-distance model training failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/train/room-clustering', methods=['POST'])
def train_room_clustering_model():
    """Configure the room clustering model."""
    try:
        # Get configuration parameters from request
        params = request.json or {}

        # Initialize AI processor
        ai_processor = blueprint_generator.ai_processor

        # Configure the model
        result = ai_processor.configure_room_clustering(
            algorithm=params.get('algorithm', 'dbscan'),
            eps=params.get('eps', 2.0),
            min_samples=params.get('min_samples', 3),
            features=params.get('features', ['x', 'y', 'z']),
            temporal_weight=params.get('temporal_weight', 0.2)
        )

        return jsonify({
            'success': result.get('success', False),
            'message': 'Room clustering model configuration completed'
        })
    except Exception as e:
        logger.error(f"Room clustering model configuration failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/train/wall-prediction', methods=['POST'])
def train_wall_prediction_model():
    """Train the wall prediction neural network."""
    try:
        # Get training parameters from request
        params = request.json or {}

        # Initialize AI processor
        ai_processor = blueprint_generator.ai_processor

        # Train the model
        result = ai_processor.train_wall_prediction_model(
            model_type=params.get('model_type', 'cnn'),
            training_data=params.get('training_data', []),
            epochs=params.get('epochs', 50),
            batch_size=params.get('batch_size', 32),
            learning_rate=params.get('learning_rate', 0.001)
        )

        return jsonify({
            'success': result.get('success', False),
            'metrics': result.get('metrics', {}),
            'message': 'Wall prediction model training completed'
        })
    except Exception as e:
        logger.error(f"Wall prediction model training failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/train/blueprint-refinement', methods=['POST'])
def train_blueprint_refinement_model():
    """Train the blueprint refinement model."""
    try:
        # Get training parameters from request
        params = request.json or {}

        # Initialize AI processor
        ai_processor = blueprint_generator.ai_processor

        # Train the model
        result = ai_processor.train_blueprint_refinement_model(
            feedback_data=params.get('feedback_data', []),
            reward_weights=params.get('reward_weights', {
                'room_size': 0.3,
                'wall_alignment': 0.4,
                'flow_efficiency': 0.3
            }),
            learning_rate=params.get('learning_rate', 0.01),
            discount_factor=params.get('discount_factor', 0.9)
        )

        return jsonify({
            'success': result.get('success', False),
            'message': 'Blueprint refinement model training completed'
        })
    except Exception as e:
        logger.error(f"Blueprint refinement model training failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug', methods=['GET'])
def debug_ha_connection():
    """Test Home Assistant API connection and entity detection."""
    try:
        ha_client = HomeAssistantClient()

        # Basic API test
        url = f"{ha_client.base_url}/api/config"
        try:
            response = requests.get(url, headers=ha_client.headers)
            ha_status = {
                "connected": response.status_code == 200,
                "status_code": response.status_code,
                "base_url": ha_client.base_url,
                "token_provided": bool(ha_client.token),
                "headers": list(ha_client.headers.keys())
            }
        except Exception as e:
            ha_status = {"error": str(e), "connected": False}

        # Try to get ANY entities
        try:
            raw_entities = ha_client.find_entities_by_pattern([""], None)
            raw_count = len(raw_entities)
            # Get first 5 entities for inspection
            sample_entities = [e['entity_id'] for e in raw_entities[:5]] if raw_entities else []
        except Exception as e:
            raw_count = -1
            sample_entities = []
            logger.error(f"Error getting raw entities: {str(e)}", exc_info=True)

        # Specific tests for your entity types
        specific_tests = {
            "ble_test": len(ha_client.find_entities_by_pattern(['ble'], [])),
            "bluetooth_test": len(ha_client.find_entities_by_pattern(['bluetooth'], [])),
            "distance_test": len(ha_client.find_entities_by_pattern(['distance'], [])),
            "sensor_ble_test": len(ha_client.find_entities_by_pattern(['ble'], ['sensor'])),
            "any_starting_with_sensor": len(ha_client.find_entities_by_pattern([""], ["sensor"]))
        }

        return jsonify({
            "ha_status": ha_status,
            "entity_scan": {
                "total_entities": raw_count,
                "sample_entities": sample_entities,
                "specific_tests": specific_tests
            }
        })

    except Exception as e:
        logger.error(f"Debug endpoint failed: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/ai/data/rssi-distance', methods=['POST'])
def add_rssi_distance_data():
    """Add training data for RSSI-to-distance model."""
    try:
        # Get data from request
        data = request.json
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Invalid data format'}), 400

        required_fields = ['rssi', 'distance', 'device_id', 'sensor_id']
        if not all(field in data for field in required_fields):
            return jsonify({'error': f'Missing required fields: {required_fields}'}), 400

        # Initialize AI processor
        ai_processor = blueprint_generator.ai_processor

        # Save the training data
        result = ai_processor.save_rssi_distance_sample(
            device_id=data['device_id'],
            sensor_id=data['sensor_id'],
            rssi=data['rssi'],
            distance=data['distance'],
            tx_power=data.get('tx_power'),
            frequency=data.get('frequency'),
            environment_type=data.get('environment_type')
        )

        return jsonify({
            'success': result,
            'message': 'RSSI-to-distance training data saved'
        })
    except Exception as e:
        logger.error(f"Failed to save RSSI-to-distance data: {str(e)}")
        return jsonify({'error': str(e)}), 500

def start_api(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """Start the Flask API server."""
    app.run(host=host, port=port, debug=debug)

@app.route('/api/debug/entities', methods=['GET'])
def debug_entities():
    """Debug endpoint for entity detection."""
    try:
        ha_client = HomeAssistantClient()
        result = {
            "connection": {
                "url": ha_client.base_url,
                "token_provided": bool(ha_client.token),
                "headers": list(ha_client.headers.keys())
            },
            "entities": {
                "all": [],
                "ble": [],
                "distance": [],
                "position": []
            },
            "test_queries": {}
        }

        # Test direct API call
        try:
            test_url = f"{ha_client.base_url}/api/states"
            response = requests.get(test_url, headers=ha_client.headers)
            result["connection"]["test_status"] = response.status_code

            # Get all entities
            all_states = response.json()
            result["entities"]["total_count"] = len(all_states)

            # Sample first 10 entities
            result["entities"]["all"] = [e["entity_id"] for e in all_states[:10]]

            # Find entities matching your example patterns
            example_patterns = [
                "apple_watch", "iphone", "ble_distance", "mmwave", "bermuda"
            ]

            for pattern in example_patterns:
                matches = []
                for state in all_states:
                    if pattern in state["entity_id"].lower():
                        matches.append(state["entity_id"])
                result["test_queries"][pattern] = matches[:5]  # Just show first 5

            # Find BLE entities
            for state in all_states:
                entity_id = state["entity_id"].lower()

                if "_ble" in entity_id or entity_id.endswith("_ble"):
                    result["entities"]["ble"].append(state["entity_id"])

                if "distance" in entity_id:
                    result["entities"]["distance"].append(state["entity_id"])

                if any(p in entity_id for p in ["position", "bermuda", "tracker", "mmwave"]):
                    result["entities"]["position"].append(state["entity_id"])

        except Exception as e:
            result["connection"]["error"] = str(e)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Debug endpoint failed: {str(e)}")
        return jsonify({"error": str(e)}), 500