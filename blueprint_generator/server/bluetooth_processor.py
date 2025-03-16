import json
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN

from .db import get_sensor_readings, save_sensor_reading

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
                'accuracy_threshold': 1.0
            }
        }

    def process_reading(
        self,
        sensor_id: str,
        rssi: int,
        device_id: str,
        sensor_location: Dict[str, float]
    ) -> bool:
        """Process and save a new sensor reading."""
        if rssi < self.rssi_threshold:
            logger.debug(f"RSSI {rssi} below threshold {self.rssi_threshold}")
            return False

        return save_sensor_reading(sensor_id, rssi, device_id, sensor_location)

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

    def _rssi_to_distance(self, rssi: int) -> float:
        """Convert RSSI to distance using path loss model."""
        return 10 ** ((self.reference_power - rssi) / (10 * self.path_loss_exponent))

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
            distances.append(self._rssi_to_distance(reading['rssi']))

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
        """Detect rooms based on device positions using DBSCAN clustering."""
        if not positions:
            return []

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
