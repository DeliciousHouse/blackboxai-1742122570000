import os
import pytest
import json
from datetime import datetime
from flask import Flask
from server.api import app as flask_app
from server.db import get_db_connection, execute_write_query
from server.schema_discovery import SchemaDiscovery
from server.bluetooth_processor import BluetoothProcessor
from server.blueprint_generator import BlueprintGenerator

@pytest.fixture(scope='session')
def app():
    """Create a Flask application for testing."""
    flask_app.config.update({
        'TESTING': True,
        'SERVER_NAME': 'localhost:5000'
    })
    return flask_app

@pytest.fixture(scope='session')
def client(app):
    """Create a test client for the Flask application."""
    return app.test_client()

@pytest.fixture(scope='session')
def runner(app):
    """Create a test CLI runner for the Flask application."""
    return app.test_cli_runner()

@pytest.fixture(scope='session')
def test_config():
    """Load test configuration."""
    config_path = os.path.join(os.path.dirname(__file__), 'test_config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

@pytest.fixture(autouse=True)
def setup_test_db(test_config):
    """
    Set up test database before each test and clean up after.
    This fixture runs automatically for each test.
    """
    # Create test tables
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Create bluetooth_readings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bluetooth_readings (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    sensor_id VARCHAR(50) NOT NULL,
                    rssi INT NOT NULL,
                    device_id VARCHAR(50) NOT NULL,
                    sensor_location JSON NOT NULL
                )
            """)
            
            # Create manual_updates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS manual_updates (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    update_type VARCHAR(10) NOT NULL,
                    entity_id VARCHAR(50) NOT NULL,
                    data JSON NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_entity (update_type, entity_id)
                )
            """)
        conn.commit()
    
    yield
    
    # Cleanup after tests
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS bluetooth_readings")
            cursor.execute("DROP TABLE IF EXISTS manual_updates")
        conn.commit()

@pytest.fixture
def sample_sensor_readings():
    """Create and return sample sensor readings."""
    readings = [
        {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sensor_id': 'sensor1',
            'rssi': -65,
            'device_id': 'device1',
            'sensor_location': json.dumps({'x': 0.0, 'y': 0.0, 'z': 0.0})
        },
        {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sensor_id': 'sensor2',
            'rssi': -70,
            'device_id': 'device1',
            'sensor_location': json.dumps({'x': 3.0, 'y': 0.0, 'z': 0.0})
        },
        {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sensor_id': 'sensor3',
            'rssi': -75,
            'device_id': 'device1',
            'sensor_location': json.dumps({'x': 0.0, 'y': 3.0, 'z': 0.0})
        }
    ]
    
    # Insert readings into database
    insert_query = """
        INSERT INTO bluetooth_readings 
        (timestamp, sensor_id, rssi, device_id, sensor_location)
        VALUES (%s, %s, %s, %s, %s)
    """
    
    for reading in readings:
        execute_write_query(
            insert_query,
            (
                reading['timestamp'],
                reading['sensor_id'],
                reading['rssi'],
                reading['device_id'],
                reading['sensor_location']
            )
        )
    
    return readings

@pytest.fixture
def schema_discovery():
    """Create a SchemaDiscovery instance for testing."""
    return SchemaDiscovery('tests/test_config.json')

@pytest.fixture
def bluetooth_processor():
    """Create a BluetoothProcessor instance for testing."""
    return BluetoothProcessor('tests/test_config.json')

@pytest.fixture
def blueprint_generator():
    """Create a BlueprintGenerator instance for testing."""
    return BlueprintGenerator('tests/test_config.json')

@pytest.fixture
def sample_device_positions():
    """Return sample device positions for testing."""
    return {
        'device1': {'x': 0.0, 'y': 0.0, 'z': 1.5},
        'device2': {'x': 3.0, 'y': 0.0, 'z': 1.5},
        'device3': {'x': 0.0, 'y': 4.0, 'z': 1.5},
        'device4': {'x': 3.0, 'y': 4.0, 'z': 1.5}
    }

@pytest.fixture
def sample_blueprint_update():
    """Return sample blueprint update data for testing."""
    return {
        'rooms': [
            {
                'id': 'room_0',
                'type': 'LIVING',
                'vertices': [[0,0,0], [5,0,0], [5,4,0], [0,4,0]],
                'center': {'x': 2.5, 'y': 2.0, 'z': 0.0},
                'area': 20.0,
                'height': 2.5
            }
        ],
        'walls': [
            {
                'id': 'wall_0',
                'room1_id': 'room_0',
                'room2_id': 'room_1',
                'start': [0,0,0],
                'end': [5,0,0],
                'height': 2.5,
                'thickness': 0.2
            }
        ]
    }

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as an API test"
    )
    config.addinivalue_line(
        "markers", "db: mark test as a database test"
    )
    config.addinivalue_line(
        "markers", "bluetooth: mark test as a bluetooth processing test"
    )
    config.addinivalue_line(
        "markers", "blueprint: mark test as a blueprint generation test"
    )

@pytest.fixture
def mock_db_connection(mocker):
    """Mock database connection for unit tests."""
    mock_connection = mocker.patch('server.db.get_db_connection')
    mock_cursor = mocker.MagicMock()
    mock_connection.return_value.__enter__.return_value = mock_cursor
    return mock_connection, mock_cursor

@pytest.fixture
def mock_bluetooth_processor(mocker):
    """Mock BluetoothProcessor for unit tests."""
    return mocker.patch('server.bluetooth_processor.BluetoothProcessor')

@pytest.fixture
def mock_blueprint_generator(mocker):
    """Mock BlueprintGenerator for unit tests."""
    return mocker.patch('server.blueprint_generator.BlueprintGenerator')
