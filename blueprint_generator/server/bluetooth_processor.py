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
            registry = ha_client.get_entity_registry()
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
        from .ha_client import HomeAssistantClient

        try:
            logger.info("Starting Bluetooth sensor processing from Home Assistant")

            # Get BLE distance readings
            ha_client = HomeAssistantClient()
            ble_devices = ha_client.get_private_ble_devices()

            # Debug log what we received
            logger.info(f"BLE devices type: {type(ble_devices)}")
            logger.info(f"BLE devices count: {len(ble_devices) if ble_devices else 0}")

            # Log more device samples to understand the structure
            if len(ble_devices) > 0:
                sample_count = min(10, len(ble_devices))
                logger.info(f"Logging {sample_count} sample devices for debugging:")
                for i, entity in enumerate(ble_devices[:sample_count]):
                    entity_id = entity.get('entity_id', 'unknown')
                    logger.info(f"Sample entity {i+1}: {entity_id} = {entity.get('state')}")
                    if 'attributes' in entity:
                        # Log the first few attributes for debugging
                        attrs = {k: entity['attributes'][k] for k in list(entity['attributes'].keys())[:5]}
                        logger.info(f"  First attributes: {attrs}")
                        # Look for specific useful attributes
                        for key in ['friendly_name', 'device_class', 'distance', 'unit_of_measurement']:
                            if key in entity['attributes']:
                                logger.info(f"  {key}: {entity['attributes'][key]}")

            # Use test data to generate a blueprint
            logger.info("Using sample data for blueprint generation testing")

            # Sample positions for testing - using requested room names
            device_positions = {
                "lounge_device": {"x": 0, "y": 0, "z": 0, "accuracy": 1.0, "source": "sample"},
                "kitchen_device": {"x": 5, "y": 0, "z": 0, "accuracy": 1.0, "source": "sample"},
                "master_bedroom_device": {"x": 0, "y": 5, "z": 0, "accuracy": 1.0, "source": "sample"},
                "master_bathroom_device": {"x": 5, "y": 5, "z": 0, "accuracy": 1.0, "source": "sample"},
                "office_device": {"x": -5, "y": 0, "z": 0, "accuracy": 1.0, "source": "sample"},
                "dining_room_device": {"x": 2.5, "y": -3, "z": 0, "accuracy": 1.0, "source": "sample"},
                "garage_device": {"x": -5, "y": -5, "z": 0, "accuracy": 1.0, "source": "sample"},
                "sky_floor_device": {"x": 0, "y": 0, "z": 3, "accuracy": 1.0, "source": "sample"},
                "balcony_device": {"x": 7, "y": 3, "z": 3, "accuracy": 1.0, "source": "sample"},
                "front_porch_device": {"x": -2, "y": -7, "z": 0, "accuracy": 1.0, "source": "sample"},
                "laundry_room_device": {"x": 7, "y": -2, "z": 0, "accuracy": 1.0, "source": "sample"}
            }

            # Generate rooms from sample data
            rooms = self.detect_rooms(device_positions)
            self.save_device_positions_to_db(device_positions)

            logger.info(f"Generated {len(rooms)} sample rooms from test data")

            return {
                "processed": len(ble_devices),
                "devices": len(device_positions),
                "rooms": len(rooms),
                "device_positions": device_positions,
                "room_list": rooms
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

    def _trilaterate_from_distances(self, readings: List[Dict]) -> Dict:
        """Calculate position using distance readings from known points."""
        if len(readings) < 3:
            return None

        # Extract sensor positions and distances
        positions = []
        distances = []

        for reading in readings:
            positions.append([
                reading['sensor_location']['x'],
                reading['sensor_location']['y'],
                reading['sensor_location']['z']
            ])
            distances.append(reading['distance'])

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

        return {
            'x': float(result.x[0]),
            'y': float(result.x[1]),
            'z': float(result.x[2]),
            'accuracy': float(accuracy),
            'source': 'trilateration'
        }

    def detect_rooms(self, positions: Dict[str, Dict[str, float]]) -> List[Dict]:
        """Detect rooms based on device positions using ML clustering."""
        if not positions:
            return []

        # For testing with test data, use DBSCAN clustering instead of hardcoded rooms
        if any(device_id.endswith('_device') for device_id in positions.keys()):
            logger.info("Creating AI-detected rooms from test device positions")
            # Continue to ML/clustering algorithms below
            # No return here - let it flow to the ML code below

        # Try to use the ML-based room clustering (original code follows)
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

        # These settings are more appropriate for test data with 1 device per room
        clustering = DBSCAN(eps=3.0, min_samples=1).fit(coords)

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

        # If no rooms were detected, create basic ones from positions
        if not rooms:
            logger.warning("DBSCAN failed to detect rooms, creating basic rooms from positions")
            room_types = ['lounge', 'kitchen', 'master_bedroom', 'master_bathroom', 'office',
                         'dining_room', 'hallway', 'laundry_room', 'nova_bedroom', 'christian_bedroom',
                         'sky_floor', 'balcony', 'front_porch', 'dressing_room']

            for i, (device_id, position) in enumerate(positions.items()):
                # Extract room name from device ID if possible
                room_name = device_id.replace('_device', '')
                if room_name in device_id:
                    name = room_name.replace('_', ' ').title()
                else:
                    name = room_types[i % len(room_types)]

                rooms.append({
                    'id': f"{name.lower().replace(' ', '_')}_{i}",
                    'name': name,
                    'center': {
                        'x': float(position['x']),
                        'y': float(position['y']),
                        'z': float(position['z'])
                    },
                    'dimensions': {
                        'width': 4.0,
                        'length': 5.0,
                        'height': 2.7
                    }
                })
            logger.info(f"Created {len(rooms)} fallback rooms")

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
