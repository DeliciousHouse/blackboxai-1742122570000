import json
import math
import logging.config
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Setup logging
logging.config.fileConfig('config/logging.conf')
logger = logging.getLogger('server.bluetooth_processor')

@dataclass
class SensorReading:
    sensor_id: str
    rssi: float
    sensor_location: Dict[str, float]  # {'x': float, 'y': float, 'z': float}
    device_id: str
    timestamp: str

@dataclass
class ProcessingParams:
    rssi_threshold: float
    minimum_sensors: int
    accuracy_threshold: float
    reference_power: float
    path_loss_exponent: float

class BluetoothProcessor:
    def __init__(self, config_path: str = 'config/config.json'):
        """
        Initialize BluetoothProcessor with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        self.params = self._load_processing_params()

    def _load_processing_params(self) -> ProcessingParams:
        """
        Load processing parameters from config file.
        
        Returns:
            ProcessingParams: Processing parameters
        """
        try:
            with open(self.config_path) as f:
                config = json.load(f)
            params = config['processing_params']
            
            return ProcessingParams(
                rssi_threshold=params['rssi_threshold'],
                minimum_sensors=params['minimum_sensors'],
                accuracy_threshold=params['accuracy_threshold'],
                reference_power=params['distance_calculation']['reference_power'],
                path_loss_exponent=params['distance_calculation']['path_loss_exponent']
            )
        except Exception as e:
            logger.error(f"Error loading processing parameters: {str(e)}")
            raise

    def rssi_to_distance(self, rssi: float) -> float:
        """
        Convert RSSI value to estimated distance using the log-distance path loss model.
        
        Args:
            rssi (float): RSSI value in dBm
        
        Returns:
            float: Estimated distance in meters
        """
        try:
            # Log-distance path loss model
            # distance = 10 ^ ((|RSSI| - |Reference Power|) / (10 * Path Loss Exponent))
            distance = 10 ** ((abs(rssi) - abs(self.params.reference_power)) / 
                            (10 * self.params.path_loss_exponent))
            return distance
        except Exception as e:
            logger.error(f"Error converting RSSI to distance: {str(e)}")
            raise

    def trilaterate(self, sensor_readings: List[SensorReading]) -> Optional[Dict[str, float]]:
        """
        Perform trilateration using sensor readings to determine device position.
        
        Args:
            sensor_readings (List[SensorReading]): List of sensor readings
        
        Returns:
            Optional[Dict[str, float]]: Estimated position {x, y, z} or None if insufficient data
        """
        try:
            # Filter readings based on RSSI threshold
            valid_readings = [
                reading for reading in sensor_readings 
                if reading.rssi >= self.params.rssi_threshold
            ]

            if len(valid_readings) < self.params.minimum_sensors:
                logger.warning(
                    f"Insufficient valid sensor readings. Need {self.params.minimum_sensors}, "
                    f"got {len(valid_readings)}"
                )
                return self._default_estimation(valid_readings)

            # Convert RSSI to distances
            distances = [self.rssi_to_distance(reading.rssi) for reading in valid_readings]
            
            # Extract sensor positions
            positions = [
                [reading.sensor_location['x'], 
                 reading.sensor_location['y'], 
                 reading.sensor_location['z']]
                for reading in valid_readings
            ]

            # Perform trilateration using least squares optimization
            estimated_position = self._least_squares_optimization(positions, distances)
            
            if estimated_position is None:
                logger.warning("Trilateration failed to converge")
                return self._default_estimation(valid_readings)

            return {
                'x': float(estimated_position[0]),
                'y': float(estimated_position[1]),
                'z': float(estimated_position[2])
            }

        except Exception as e:
            logger.error(f"Error during trilateration: {str(e)}")
            return self._default_estimation(sensor_readings)

    def _least_squares_optimization(
        self, 
        sensor_positions: List[List[float]], 
        distances: List[float]
    ) -> Optional[np.ndarray]:
        """
        Perform least squares optimization to estimate device position.
        
        Args:
            sensor_positions (List[List[float]]): List of sensor positions
            distances (List[float]): List of distances from sensors to device
        
        Returns:
            Optional[np.ndarray]: Estimated position or None if optimization fails
        """
        try:
            # Convert to numpy arrays
            positions = np.array(sensor_positions)
            distances = np.array(distances)

            # Initial guess: centroid of sensor positions
            initial_guess = np.mean(positions, axis=0)
            
            # Define the objective function to minimize
            def objective(point):
                calculated_distances = np.sqrt(np.sum((positions - point) ** 2, axis=1))
                return np.sum((calculated_distances - distances) ** 2)

            # Perform optimization
            from scipy.optimize import minimize
            result = minimize(
                objective,
                initial_guess,
                method='Nelder-Mead',
                options={'maxiter': 1000}
            )

            if result.success:
                return result.x
            else:
                logger.warning(f"Optimization failed: {result.message}")
                return None

        except Exception as e:
            logger.error(f"Error in least squares optimization: {str(e)}")
            return None

    def _default_estimation(
        self, 
        readings: List[SensorReading]
    ) -> Dict[str, float]:
        """
        Provide a fallback position estimation when trilateration fails.
        
        Args:
            readings (List[SensorReading]): Available sensor readings
        
        Returns:
            Dict[str, float]: Estimated position using fallback method
        """
        try:
            if not readings:
                logger.warning("No readings available for default estimation")
                return {'x': 0.0, 'y': 0.0, 'z': 0.0}

            # Find the sensor with strongest signal
            strongest_reading = max(readings, key=lambda r: r.rssi)
            
            # Estimate rough position near the strongest sensor
            distance = self.rssi_to_distance(strongest_reading.rssi)
            position = strongest_reading.sensor_location

            # Add some uncertainty to the position
            position['x'] += distance * 0.5  # Arbitrary offset
            position['y'] += distance * 0.5
            
            logger.info(
                f"Using default estimation near sensor {strongest_reading.sensor_id} "
                f"with RSSI {strongest_reading.rssi}"
            )
            
            return position

        except Exception as e:
            logger.error(f"Error in default estimation: {str(e)}")
            return {'x': 0.0, 'y': 0.0, 'z': 0.0}

    def process_readings(
        self, 
        readings: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """
        Process a batch of sensor readings and estimate positions for all devices.
        
        Args:
            readings (List[Dict]): List of raw sensor readings from database
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary mapping device IDs to positions
        """
        try:
            # Group readings by device_id
            devices = {}
            for reading in readings:
                sensor_reading = SensorReading(
                    sensor_id=reading['sensor_id'],
                    rssi=reading['rssi'],
                    sensor_location=json.loads(reading['sensor_location']),
                    device_id=reading['device_id'],
                    timestamp=reading['timestamp']
                )
                
                if sensor_reading.device_id not in devices:
                    devices[sensor_reading.device_id] = []
                devices[sensor_reading.device_id].append(sensor_reading)

            # Process each device's readings
            results = {}
            for device_id, device_readings in devices.items():
                position = self.trilaterate(device_readings)
                if position:
                    results[device_id] = position
                    logger.info(f"Successfully estimated position for device {device_id}")
                else:
                    logger.warning(f"Failed to estimate position for device {device_id}")

            return results

        except Exception as e:
            logger.error(f"Error processing readings batch: {str(e)}")
            raise

def main():
    """
    Main function to test Bluetooth processing functionality.
    """
    try:
        processor = BluetoothProcessor()
        
        # Example sensor readings
        sample_readings = [
            {
                'sensor_id': 'sensor1',
                'rssi': -65,
                'sensor_location': json.dumps({'x': 0.0, 'y': 0.0, 'z': 0.0}),
                'device_id': 'device1',
                'timestamp': '2024-01-20 12:00:00'
            },
            {
                'sensor_id': 'sensor2',
                'rssi': -70,
                'sensor_location': json.dumps({'x': 3.0, 'y': 0.0, 'z': 0.0}),
                'device_id': 'device1',
                'timestamp': '2024-01-20 12:00:00'
            },
            {
                'sensor_id': 'sensor3',
                'rssi': -75,
                'sensor_location': json.dumps({'x': 0.0, 'y': 3.0, 'z': 0.0}),
                'device_id': 'device1',
                'timestamp': '2024-01-20 12:00:00'
            }
        ]

        # Process readings
        results = processor.process_readings(sample_readings)
        logger.info(f"Processing results: {results}")

    except Exception as e:
        logger.error(f"Test processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
