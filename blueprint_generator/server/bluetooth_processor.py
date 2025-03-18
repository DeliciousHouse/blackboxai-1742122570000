import json
import logging
import math
import random
import requests
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN

from .db import get_sensor_readings, save_sensor_reading
from .ai_processor import AIProcessor

logger = logging.getLogger(__name__)

class BluetoothProcessor:
    """Process Bluetooth signals for position estimation."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Bluetooth processor."""
        self.config = self._load_config(config_path)
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
                    'path_loss_exponent': 2.0
                },
                'rssi_threshold': -75,
                'minimum_sensors': 3,
                'accuracy_threshold': 1.0,
                'use_ml_distance': True,
                'ml_fallback_threshold': 0.5  # Confidence threshold for ML model
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
            registry = requests.get(f"{ha_client.base_url}/api/config/entity_registry", headers=ha_client.headers).json()

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

            logger.info(f"Found {len(ble_devices)} BLE devices and {len(position_entities)} position entities in Home Assistant")

            # Debug the first few entities found to verify detection
            if ble_devices:
                logger.info(f"Sample BLE entities: {[d.get('entity_id') for d in ble_devices[:3]]}")
            if position_entities:
                logger.info(f"Sample position entities: {[p.get('entity_id') for p in position_entities[:3]]}")

            if not ble_devices and not position_entities:
                logger.warning("No BLE devices or position entities found - check entity detection patterns")
                return {"error": "No devices found", "processed": 0}

            # Process the BLE device data
            device_positions = {}

            # First, use any direct position entities we have (most accurate)
            for entity in position_entities:
                device_id = entity.get('device_id')
                position = entity.get('position')
                if device_id and position and all(k in position for k in ['x', 'y', 'z']):
                    device_positions[device_id] = {
                        'x': float(position['x']),
                        'y': float(position['y']),
                        'z': float(position.get('z', 0)),
                        'accuracy': 0.5,  # Assume fairly accurate
                        'source': 'position_entity'
                    }
                    logger.debug(f"Using direct position for {device_id}: {position}")

            # If we have devices with known RSSI/distance but no position, estimate positions
            ble_device_map = {}
            for device in ble_devices:
                device_id = device.get('mac')
                if not device_id:
                    continue

                entity_id = device.get('entity_id')
                sensor_location = {'x': 0, 'y': 0, 'z': 0}  # Default value
                if device_id not in ble_device_map:
                    ble_device_map[device_id] = []


                # Try to get location from entity attributes or known mapping
                if 'sensor_location' in device.get('attributes', {}):
                    # Use attributes if available
                    sensor_attrs = device['attributes']['sensor_location']
                    sensor_location = {
                        'x': float(sensor_attrs.get('x', 0)),
                        'y': float(sensor_attrs.get('y', 0)),
                        'z': float(sensor_attrs.get('z', 0))
                    }
                else:
                    # Get dynamically generated sensor positions based on HA areas
                    if not hasattr(self, '_sensor_positions'):
                        self._sensor_positions = self.get_initial_sensor_positions()

                    entity_id = device.get('entity_id')
                    if entity_id in self._sensor_positions:
                        sensor_location = self._sensor_positions[entity_id]
                    else:
                        # Try partial matching
                        for sensor_id, location in self._sensor_positions.items():
                            if sensor_id in entity_id or entity_id in sensor_id:
                                sensor_location = location
                                logger.debug(f"Using partial match for {entity_id}: {sensor_id}")
                                break

                # If distance is available, process it
                if 'distance' in device and device['distance'] is not None:
                    rssi = device.get('rssi', -80)  # Use provided RSSI or default
                    # Store reading with device, sensor, and location info
                    ble_device_map[device_id].append({
                        'device_id': device_id,
                        'entity_id': entity_id,
                        'rssi': rssi,
                        'sensor_id': f"sensor_{len(ble_device_map[device_id])}",
                        'sensor_location': json.dumps(sensor_location)
                    })

            # Process readings for position estimation if needed
            for device_id, readings in ble_device_map.items():
                # Only process devices not already positioned
                if device_id not in device_positions and len(readings) >= self.minimum_sensors:
                    position = self._trilaterate(readings)
                    if position:
                        device_positions[device_id] = position
                        device_positions[device_id]['source'] = 'trilateration'

            # Detect rooms based on device positions
            rooms = self.detect_rooms(device_positions)
            logger.info(f"Detected {len(rooms)} rooms from {len(device_positions)} device positions")

            # Train AI models if we have enough data
            if len(ble_devices) > 5:
                try:
                    logger.info("Training AI models with collected data")
                    # This trains models based on data we've seen
                    self.ai_processor.train_models()
                except Exception as e:
                    logger.error(f"Failed to train models: {str(e)}")

            return {
                "processed": len(ble_devices) + len(position_entities),
                "devices": len(device_positions),
                "rooms": len(rooms),
                "device_positions": device_positions,
                "rooms": rooms
            }

        except Exception as e:
            logger.error(f"Error processing Bluetooth sensors: {str(e)}", exc_info=True)
            return {"error": str(e), "processed": 0}

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
