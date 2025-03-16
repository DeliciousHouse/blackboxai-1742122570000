import pytest
import json
import numpy as np
from server.bluetooth_processor import BluetoothProcessor, SensorReading, ProcessingParams

@pytest.fixture
def bluetooth_processor():
    """Create a BluetoothProcessor instance for testing."""
    return BluetoothProcessor()

@pytest.fixture
def sample_readings():
    """Create sample sensor readings for testing."""
    return [
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

@pytest.fixture
def processing_params():
    """Create processing parameters for testing."""
    return ProcessingParams(
        rssi_threshold=-80,
        minimum_sensors=3,
        accuracy_threshold=1.0,
        reference_power=-59,
        path_loss_exponent=2.0
    )

def test_load_processing_params(bluetooth_processor):
    """Test loading processing parameters from config."""
    params = bluetooth_processor._load_processing_params()
    assert isinstance(params, ProcessingParams)
    assert params.rssi_threshold is not None
    assert params.minimum_sensors > 0
    assert params.accuracy_threshold > 0
    assert params.reference_power is not None
    assert params.path_loss_exponent > 0

def test_rssi_to_distance(bluetooth_processor):
    """Test RSSI to distance conversion."""
    # Test with various RSSI values
    test_cases = [
        (-59, 1.0),  # Reference power should give ~1m
        (-65, 2.0),  # Weaker signal should give larger distance
        (-71, 4.0),  # Even weaker signal
    ]
    
    for rssi, expected_distance in test_cases:
        distance = bluetooth_processor.rssi_to_distance(rssi)
        assert distance > 0
        assert abs(distance - expected_distance) < 0.5  # Allow some margin of error

def test_trilaterate(bluetooth_processor, sample_readings):
    """Test trilateration with known sensor positions."""
    # Convert sample readings to SensorReading objects
    readings = [
        SensorReading(
            sensor_id=reading['sensor_id'],
            rssi=reading['rssi'],
            sensor_location=json.loads(reading['sensor_location']),
            device_id=reading['device_id'],
            timestamp=reading['timestamp']
        )
        for reading in sample_readings
    ]
    
    # Perform trilateration
    position = bluetooth_processor.trilaterate(readings)
    
    # Verify result
    assert position is not None
    assert 'x' in position
    assert 'y' in position
    assert 'z' in position
    assert isinstance(position['x'], float)
    assert isinstance(position['y'], float)
    assert isinstance(position['z'], float)

def test_trilaterate_insufficient_data(bluetooth_processor):
    """Test trilateration with insufficient sensor data."""
    # Create readings with only two sensors
    readings = [
        SensorReading(
            sensor_id='sensor1',
            rssi=-65,
            sensor_location={'x': 0.0, 'y': 0.0, 'z': 0.0},
            device_id='device1',
            timestamp='2024-01-20 12:00:00'
        ),
        SensorReading(
            sensor_id='sensor2',
            rssi=-70,
            sensor_location={'x': 3.0, 'y': 0.0, 'z': 0.0},
            device_id='device1',
            timestamp='2024-01-20 12:00:00'
        )
    ]
    
    # Should return default estimation
    position = bluetooth_processor.trilaterate(readings)
    assert position is not None
    assert position['x'] == 0.0
    assert position['y'] == 0.0
    assert position['z'] == 0.0

def test_least_squares_optimization(bluetooth_processor):
    """Test least squares optimization for position estimation."""
    # Create test data
    sensor_positions = [
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0]
    ]
    distances = [2.0, 2.0, 2.0]
    
    result = bluetooth_processor._least_squares_optimization(sensor_positions, distances)
    
    assert result is not None
    assert len(result) == 3
    assert all(isinstance(x, float) for x in result)

def test_default_estimation(bluetooth_processor, sample_readings):
    """Test default position estimation."""
    readings = [
        SensorReading(
            sensor_id=reading['sensor_id'],
            rssi=reading['rssi'],
            sensor_location=json.loads(reading['sensor_location']),
            device_id=reading['device_id'],
            timestamp=reading['timestamp']
        )
        for reading in sample_readings
    ]
    
    position = bluetooth_processor._default_estimation(readings)
    assert position is not None
    assert 'x' in position
    assert 'y' in position
    assert 'z' in position

def test_process_readings(bluetooth_processor, sample_readings):
    """Test processing multiple readings."""
    results = bluetooth_processor.process_readings(sample_readings)
    
    assert results is not None
    assert isinstance(results, dict)
    assert 'device1' in results
    assert isinstance(results['device1'], dict)
    assert 'x' in results['device1']
    assert 'y' in results['device1']
    assert 'z' in results['device1']

def test_process_readings_multiple_devices(bluetooth_processor):
    """Test processing readings from multiple devices."""
    # Create readings for multiple devices
    readings = [
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
            'device_id': 'device2',
            'timestamp': '2024-01-20 12:00:00'
        }
    ]
    
    results = bluetooth_processor.process_readings(readings)
    assert len(results) == 2
    assert 'device1' in results
    assert 'device2' in results

def test_invalid_rssi_values(bluetooth_processor):
    """Test handling of invalid RSSI values."""
    readings = [
        {
            'sensor_id': 'sensor1',
            'rssi': 0,  # Invalid RSSI (too strong)
            'sensor_location': json.dumps({'x': 0.0, 'y': 0.0, 'z': 0.0}),
            'device_id': 'device1',
            'timestamp': '2024-01-20 12:00:00'
        }
    ]
    
    results = bluetooth_processor.process_readings(readings)
    assert results['device1'] is not None  # Should use default estimation

def test_invalid_sensor_location(bluetooth_processor):
    """Test handling of invalid sensor location data."""
    with pytest.raises(json.JSONDecodeError):
        readings = [
            {
                'sensor_id': 'sensor1',
                'rssi': -65,
                'sensor_location': 'invalid_json',
                'device_id': 'device1',
                'timestamp': '2024-01-20 12:00:00'
            }
        ]
        bluetooth_processor.process_readings(readings)

def test_optimization_convergence(bluetooth_processor):
    """Test optimization convergence with various initial conditions."""
    # Create test data with known solution
    sensor_positions = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0]
    ])
    target_point = np.array([1.0, 1.0, 0.0])
    distances = np.sqrt(np.sum((sensor_positions - target_point) ** 2, axis=1))
    
    result = bluetooth_processor._least_squares_optimization(sensor_positions.tolist(), distances.tolist())
    
    assert result is not None
    assert np.allclose(result, target_point, atol=0.5)  # Should be close to target point
