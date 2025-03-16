import os
import json
import pytest
from typing import Dict, Any
from contextlib import contextmanager
from server.db import execute_write_query, execute_query

def load_test_config() -> Dict[str, Any]:
    """
    Load test configuration from test_config.json.
    
    Returns:
        Dict[str, Any]: Test configuration dictionary
    """
    config_path = os.path.join(os.path.dirname(__file__), 'test_config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

@contextmanager
def setup_test_database():
    """
    Context manager to set up and tear down test database.
    Creates necessary tables and cleans up after tests.
    """
    try:
        # Create test tables
        create_tables = [
            """
            CREATE TABLE IF NOT EXISTS bluetooth_readings (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                sensor_id VARCHAR(50) NOT NULL,
                rssi INT NOT NULL,
                device_id VARCHAR(50) NOT NULL,
                sensor_location JSON NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS manual_updates (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                update_type VARCHAR(10) NOT NULL,
                entity_id VARCHAR(50) NOT NULL,
                data JSON NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_entity (update_type, entity_id)
            )
            """
        ]
        
        for query in create_tables:
            execute_write_query(query)
        
        yield
        
    finally:
        # Cleanup tables
        cleanup_queries = [
            "DROP TABLE IF EXISTS bluetooth_readings",
            "DROP TABLE IF EXISTS manual_updates"
        ]
        
        for query in cleanup_queries:
            execute_write_query(query)

@pytest.fixture
def test_config():
    """Fixture to provide test configuration."""
    return load_test_config()

@pytest.fixture
def test_db():
    """Fixture to provide test database setup and teardown."""
    with setup_test_database():
        yield

def insert_test_sensor_readings(readings):
    """
    Insert test sensor readings into the database.
    
    Args:
        readings (List[Dict]): List of sensor reading dictionaries
    """
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

def clear_test_data():
    """Clear all test data from the database."""
    queries = [
        "DELETE FROM bluetooth_readings",
        "DELETE FROM manual_updates"
    ]
    
    for query in queries:
        execute_write_query(query)

def get_test_sensor_readings():
    """
    Get all test sensor readings from the database.
    
    Returns:
        List[Dict]: List of sensor reading dictionaries
    """
    return execute_query("SELECT * FROM bluetooth_readings")

def get_test_manual_updates():
    """
    Get all test manual updates from the database.
    
    Returns:
        List[Dict]: List of manual update dictionaries
    """
    return execute_query("SELECT * FROM manual_updates")

# Common test data
SAMPLE_DEVICE_POSITIONS = {
    'device1': {'x': 0.0, 'y': 0.0, 'z': 1.5},
    'device2': {'x': 3.0, 'y': 0.0, 'z': 1.5},
    'device3': {'x': 0.0, 'y': 4.0, 'z': 1.5},
    'device4': {'x': 3.0, 'y': 4.0, 'z': 1.5}
}

SAMPLE_ROOM_DATA = {
    'id': 'room_0',
    'type': 'LIVING',
    'vertices': [[0,0,0], [5,0,0], [5,4,0], [0,4,0]],
    'center': {'x': 2.5, 'y': 2.0, 'z': 0.0},
    'area': 20.0,
    'height': 2.5
}

SAMPLE_WALL_DATA = {
    'id': 'wall_0',
    'room1_id': 'room_0',
    'room2_id': 'room_1',
    'start': [0,0,0],
    'end': [5,0,0],
    'height': 2.5,
    'thickness': 0.2
}

# Test validation helpers
def validate_room_structure(room):
    """
    Validate the structure of a room dictionary.
    
    Args:
        room (Dict): Room dictionary to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_fields = ['id', 'type', 'vertices', 'center', 'area', 'height']
    return all(field in room for field in required_fields)

def validate_wall_structure(wall):
    """
    Validate the structure of a wall dictionary.
    
    Args:
        wall (Dict): Wall dictionary to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_fields = ['id', 'room1_id', 'room2_id', 'start', 'end', 'height', 'thickness']
    return all(field in wall for field in required_fields)

def validate_blueprint_structure(blueprint):
    """
    Validate the structure of a blueprint dictionary.
    
    Args:
        blueprint (Dict): Blueprint dictionary to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check top-level structure
    required_fields = ['version', 'rooms', 'walls', 'metadata']
    if not all(field in blueprint for field in required_fields):
        return False
    
    # Check rooms
    if not isinstance(blueprint['rooms'], list):
        return False
    for room in blueprint['rooms']:
        if not validate_room_structure(room):
            return False
    
    # Check walls
    if not isinstance(blueprint['walls'], list):
        return False
    for wall in blueprint['walls']:
        if not validate_wall_structure(wall):
            return False
    
    # Check metadata
    required_metadata = ['total_area', 'room_count', 'timestamp']
    return all(field in blueprint['metadata'] for field in required_metadata)
