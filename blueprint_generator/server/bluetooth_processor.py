import json
import logging
import math
import random
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN

from .db import get_sensor_readings, save_sensor_reading
from .ai_processor import AIProcessor

logger = logging.getLogger(__name__)

def load_global_config(config_path=None):
    """Load global configuration from file or use defaults."""
    if hasattr(load_global_config, 'cached_config'):
        return load_global_config.cached_config

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'processing_params': {
                'distance_calculation': {
                    'reference_power': -59,
                    'path_loss_exponent': 3.0
                },
                'rssi_threshold': -85,
                'minimum_sensors': 3,
                'accuracy_threshold': 15.0,  # Increased from 7.0
                'use_ml_distance': True,
                'ml_fallback_threshold': 3.0
            },
            'blueprint_validation': {
                'min_room_area': 4,
                'max_room_area': 100,
                'min_room_dimension': 1.5,
                'max_room_dimension': 15,
                'min_wall_thickness': 0.1,
                'max_wall_thickness': 0.5,
                'min_ceiling_height': 2.2,
                'max_ceiling_height': 4.0
            },
            'ai_settings': {
                'use_ml_wall_prediction': True,
                'use_ml_blueprint_refinement': True,
                'training_data_collection': True
            },
            'unit_conversion': {
                'home_assistant_uses_feet': True  # Set to True if HA uses feet
            }
        }

    # Cache the config so we don't reload it every time
    load_global_config.cached_config = config
    return config

def convert_to_meters(value, config=None):
    """Convert a value to meters based on config settings."""
    if config is None:
        config = load_global_config()

    if config.get('unit_conversion', {}).get('home_assistant_uses_feet', True):
        return float(value) * 0.3048  # Convert feet to meters
    return float(value)  # Already in meters

class BluetoothProcessor:
    """Process Bluetooth signals for position estimation."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Bluetooth processor."""
        self.config = load_global_config(config_path)
        self.reference_power = self.config['processing_params']['distance_calculation']['reference_power']
        self.path_loss_exponent = self.config['processing_params']['distance_calculation']['path_loss_exponent']
        self.rssi_threshold = self.config['processing_params']['rssi_threshold']
        self.minimum_sensors = self.config['processing_params']['minimum_sensors']
        self.accuracy_threshold = self.config['processing_params']['accuracy_threshold']

        # Initialize AI processor for ML-based distance estimation
        self.ai_processor = AIProcessor(config_path)
        self.use_ml_distance = self.config['processing_params'].get('use_ml_distance', True)

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from file or use defaults."""
        if config_path:
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            'processing_params': {
                'distance_calculation': {
                    'reference_power': -59,
                    'path_loss_exponent': 3.0
                },
                'rssi_threshold': -85,
                'minimum_sensors': 3,
                'accuracy_threshold': 15.0,
                'use_ml_distance': True,
                'ml_fallback_threshold': 3.0  # Confidence threshold for ML model
            }
        }

    def get_initial_sensor_positions(self) -> Dict:
        """Generate initial sensor positions based on Home Assistant areas/rooms."""
        from .ha_client import HomeAssistantClient
        ha_client = HomeAssistantClient()

        try:
            # Get areas using the WebSocket method in ha_client.py
            areas = ha_client.get_areas()

            # Create a simple grid layout
            area_positions = {}
            grid_size = math.ceil(math.sqrt(len(areas)))

            for i, area in enumerate(areas):
                row = i // grid_size
                col = i % grid_size
                area_positions[area["area_id"]] = {
                    'x': col * 5,
                    'y': row * 5,
                    'z': 0,
                    'name': area["name"]
                }

            # Get devices and their area assignments
            entities = requests.get(f"{ha_client.base_url}/api/states", headers=ha_client.headers).json()
            registry = safe_json_request(f"{ha_client.base_url}/api/config/entity_registry", headers=ha_client.headers)

            # Create a map of entity_id to area_id
            entity_areas = {}
            for item in registry:
                if 'entity_id' in item and 'area_id' in item and item['area_id']:
                    entity_areas[item['entity_id']] = item['area_id']

            # Create sensor positions based on area assignments
            sensor_positions = {}
            for entity in entities:
                entity_id = entity.get('entity_id')
                if entity_id in entity_areas and entity_areas[entity_id] in area_positions:
                    area = area_positions[entity_areas[entity_id]]
                    sensor_positions[entity_id] = {
                        'x': area['x'] + random.uniform(-1.5, 1.5),
                        'y': area['y'] + random.uniform(-1.5, 1.5),
                        'z': 0
                    }

            return sensor_positions

        except Exception as e:
            logger.error(f"Failed to generate initial sensor positions: {str(e)}")
            return {
                "default_sensor_1": {'x': 0, 'y': 0, 'z': 0},
                "default_sensor_2": {'x': 5, 'y': 0, 'z': 0},
                "default_sensor_3": {'x': 0, 'y': 5, 'z': 0},
                "default_sensor_4": {'x': 5, 'y': 5, 'z': 0}
            }

    def process_reading(
        self,
        sensor_id: str,
        rssi: int,
        device_id: str,
        sensor_location: Dict[str, float],
        known_distance: Optional[float] = None,
        tx_power: Optional[int] = None,
        frequency: Optional[float] = None,
        environment_type: Optional[str] = None
    ) -> bool:
        """Process and save a new sensor reading."""
        if rssi < self.rssi_threshold:
            logger.debug(f"RSSI {rssi} below threshold {self.rssi_threshold}")
            return False

        # Save the reading to the database
        result = save_sensor_reading(sensor_id, rssi, device_id, sensor_location)

        # If we have a known distance, save it as training data for the ML model
        if known_distance is not None and result:
            try:
                # Save training data for RSSI-to-distance model
                self.ai_processor.save_rssi_distance_sample(
                    device_id=device_id,
                    sensor_id=sensor_id,
                    rssi=rssi,
                    distance=known_distance,
                    tx_power=tx_power,
                    frequency=frequency,
                    environment_type=environment_type
                )
                logger.debug(f"Saved training data: RSSI {rssi} -> {known_distance}m")
            except Exception as e:
                logger.warning(f"Failed to save training data: {str(e)}")

        return result

    def process_bluetooth_sensors(self) -> Dict:
        """Process Bluetooth sensors from Home Assistant and prepare data for blueprint generation."""
        from .ha_client import HomeAssistantClient

        try:
            logger.info("Starting Bluetooth sensor processing from Home Assistant")

            # Get data from Home Assistant
            ha_client = HomeAssistantClient()
            ble_devices = ha_client.get_private_ble_devices()
            position_entities = ha_client.get_bermuda_positions()

            # Get entity area assignments for room hints using WebSocket
            registry = ha_client.get_entity_registry_websocket()

            # Get areas (for room hints)
            areas = ha_client.get_areas()

            # Use position entities directly instead of calculating from RSSI
            device_positions = {}
            for entity_id, data in position_entities.items():
                if 'attributes' in data and 'position' in data['attributes']:
                    position = data['attributes']['position']

                    # Convert units if needed (feet → meters)
                    x = float(position.get('x', 0))
                    y = float(position.get('y', 0))
                    z = float(position.get('z', 0))

                    if self.config.get('unit_conversion', {}).get('home_assistant_uses_feet', False):
                        x *= 0.3048
                        y *= 0.3048
                        z *= 0.3048

                    device_positions[entity_id] = {
                        'x': x,
                        'y': y,
                        'z': z,
                        'accuracy': float(position.get('accuracy', 5.0)),
                        'source': 'bermuda'
                    }

            # Apply movement pattern detection
            movement_patterns = self.ai_processor.detect_movement_patterns()
            for device_id, pattern in movement_patterns.items():
                if device_id in device_positions and pattern.get('static', False):
                    if 'accuracy' in device_positions[device_id]:
                        device_positions[device_id]['accuracy'] *= 0.8

            # Apply spatial memory
            device_positions = self.ai_processor.apply_spatial_memory(device_positions)

            # Room detection
            rooms = self.detect_rooms(device_positions)

            # Save positions to database
            self.save_device_positions_to_db(device_positions)

            return {
                "processed": len(position_entities),
                "devices": len(device_positions),
                "rooms": len(rooms),
                "device_positions": device_positions,
                "rooms": rooms
            }
        except Exception as e:
            logger.error(f"Error processing Bluetooth sensors: {e}")
            return {"error": str(e)}

    def estimate_positions(self, time_window: int = 300) -> Dict[str, Dict[str, float]]:
        """Estimate positions of all devices in the time window."""
        readings = get_sensor_readings(time_window)
        if not readings:
            return {}

        # Group readings by device
        devices = {}
        for reading in readings:
            device_id = reading['device_id']
            if device_id not in devices:
                devices[device_id] = []
            devices[device_id].append(reading)

        # Estimate position for each device
        positions = {}
        for device_id, device_readings in devices.items():
            if len(device_readings) >= self.minimum_sensors:
                position = self._trilaterate(device_readings)
                if position:
                    positions[device_id] = position

        return positions

    def save_device_positions_to_db(self, device_positions):
        """Save device positions to database for blueprint generation."""
        try:
            from .db import get_sqlite_connection
            conn = get_sqlite_connection()
            cursor = conn.cursor()

            # Add positions to database
            for device_id, position in device_positions.items():
                position_json = json.dumps(position)

                # Insert new position
                cursor.execute('''
                INSERT INTO device_positions
                (device_id, position_data, source, accuracy, timestamp)
                VALUES (?, ?, ?, ?, datetime('now'))
                ''', (device_id, position_json, 'bluetooth', position.get('accuracy', 0)))

            conn.commit()
            conn.close()
            logger.info(f"Saved {len(device_positions)} device positions to database")
            return True
        except Exception as e:
            logger.error(f"Error saving device positions: {e}")
            return False

    def _rssi_to_distance(self, rssi: int, sensor_id: Optional[str] = None) -> float:
        """Convert RSSI to distance using ML model or path loss model."""
        try:
            # Use ML model if enabled
            if self.use_ml_distance:
                # Better environment type detection
                environment_type = 'indoor'  # Default assumption

                # Check entity_id for better location hints
                if sensor_id:
                    if any(outdoor_term in sensor_id.lower() for outdoor_term in
                           ['outdoor', 'exterior', 'outside', 'yard', 'garden', 'patio']):
                        environment_type = 'outdoor'

                # Try to get distance from ML model
                distance = self.ai_processor.estimate_distance(
                    rssi=rssi,
                    environment_type=environment_type
                )

                # If we got a valid distance, return it
                if distance > 0:
                    logger.debug(f"Using ML model for distance estimation: RSSI {rssi} -> {distance:.2f}m")
                    return min(distance, 30.0)  # Cap at reasonable maximum

        except Exception as e:
            logger.warning(f"Error using ML model for distance estimation: {str(e)}")

        # Safety checks before physics calculation
        try:
            # Ensure valid RSSI and cap extreme values
            if not isinstance(rssi, (int, float)):
                rssi = -70  # Default if not a number

            # Cap RSSI at reasonable bounds
            rssi = max(min(rssi, -30), -100)

            # Calculate exponent with protection against overflow
            exponent = (self.reference_power - rssi) / (10 * self.path_loss_exponent)

            # Prevent extreme values that would cause overflow
            if exponent > 6:  # 10^6 is million meters = 1000km (more than enough)
                logger.warning(f"Capping extreme distance calculation: RSSI={rssi}, exponent={exponent}")
                return 100.0  # Cap distance at 100m

            # Safe calculation
            distance = 10 ** exponent

            # Ensure reasonable result
            distance = min(max(distance, 0.1), 100.0)  # Between 10cm and 100m

            logger.debug(f"Using physics model for distance estimation: RSSI {rssi} -> {distance:.2f}m")
            return distance

        except Exception as e:
            logger.warning(f"Physics-based distance calculation failed: {str(e)}")
            return 5.0  # Default reasonable distance

    def _trilaterate(self, readings: List[Dict]) -> Optional[Dict[str, float]]:
        """Estimate device position using trilateration."""
        if len(readings) < self.minimum_sensors:
            return None

        # Prepare sensor positions and distances
        positions = []
        distances = []
        for reading in readings:
            sensor_loc = json.loads(reading['sensor_location'])
            positions.append([sensor_loc['x'], sensor_loc['y'], sensor_loc['z']])
            distances.append(self._rssi_to_distance(reading['rssi'], reading.get('sensor_id')))

        positions = np.array(positions)
        distances = np.array(distances)

        # Define objective function for minimization
        def objective(point):
            return sum((np.linalg.norm(positions - point, axis=1) - distances) ** 2)

        # Initial guess: centroid of sensor positions
        initial_guess = np.mean(positions, axis=0)

        # Minimize the objective function
        result = minimize(
            objective,
            initial_guess,
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )

        if not result.success:
            logger.warning("Position estimation optimization failed")
            return None

        # Calculate accuracy estimate
        accuracy = math.sqrt(result.fun / len(readings))
        if accuracy > self.accuracy_threshold:
            logger.warning(f"Position accuracy {accuracy} exceeds threshold {self.accuracy_threshold}")
            return None

        return {
            'x': float(result.x[0]),
            'y': float(result.x[1]),
            'z': float(result.x[2]),
            'accuracy': float(accuracy)
        }

    def detect_rooms(self, positions: Dict[str, Dict[str, float]]) -> List[Dict]:
        """Detect rooms based on device positions using ML clustering."""
        if not positions:
            return []

        try:
            # Try to use the ML-based room clustering
            rooms = self.ai_processor.detect_rooms_ml(positions)
            if rooms:
                logger.debug(f"Using ML-based room clustering: detected {len(rooms)} rooms")
                return rooms
        except Exception as e:
            logger.warning(f"ML-based room clustering failed: {str(e)}")

        # Fall back to basic DBSCAN clustering
        logger.debug("Falling back to basic DBSCAN clustering")

        # Extract position coordinates
        coords = np.array([[p['x'], p['y'], p['z']] for p in positions.values()])

        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=2.0, min_samples=3).fit(coords)

        # Group positions by cluster
        rooms = []
        for label in set(clustering.labels_):
            if label == -1:  # Skip noise points
                continue

            # Get positions in this cluster
            room_positions = coords[clustering.labels_ == label]

            # Calculate room properties
            min_coords = np.min(room_positions, axis=0)
            max_coords = np.max(room_positions, axis=0)
            center = np.mean(room_positions, axis=0)

            rooms.append({
                'center': {
                    'x': float(center[0]),
                    'y': float(center[1]),
                    'z': float(center[2])
                },
                'dimensions': {
                    'width': float(max_coords[0] - min_coords[0]),
                    'length': float(max_coords[1] - min_coords[1]),
                    'height': float(max_coords[2] - min_coords[2])
                },
                'bounds': {
                    'min': {
                        'x': float(min_coords[0]),
                        'y': float(min_coords[1]),
                        'z': float(min_coords[2])
                    },
                    'max': {
                        'x': float(max_coords[0]),
                        'y': float(max_coords[1]),
                        'z': float(max_coords[2])
                    }
                }
            })

        return rooms

# Add this helper function to your bluetooth_processor.py file
def safe_json_request(url, headers):
    """Safely get and parse JSON from a request, handling potential malformation."""
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            logger.warning(f"Request failed with status {response.status_code}: {url}")
            return []

        # Try to clean up the response if needed
        content = response.text.strip()
        if content.startswith('\n'):
            content = content.lstrip()

        # Handle potential HTML responses
        if content.startswith('<!DOCTYPE') or content.startswith('<html'):
            logger.warning(f"Got HTML response instead of JSON: {url}")
            return []

        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.debug(f"Response content preview: {response.text[:100]}...")
        return []
    except Exception as e:
        logger.error(f"Request error: {e}")
        return []
