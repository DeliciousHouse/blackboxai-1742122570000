import pytest
import json
from datetime import datetime
from flask import url_for
from server.api import app
from server.db import execute_write_query

@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def sample_sensor_data():
    """Create sample sensor data in the database."""
    # Create test table
    create_table_query = """
    CREATE TABLE IF NOT EXISTS bluetooth_readings (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        timestamp DATETIME NOT NULL,
        sensor_id VARCHAR(50) NOT NULL,
        rssi INT NOT NULL,
        device_id VARCHAR(50) NOT NULL,
        sensor_location JSON NOT NULL
    )
    """
    execute_write_query(create_table_query)
    
    # Insert test data
    insert_query = """
    INSERT INTO bluetooth_readings 
    (timestamp, sensor_id, rssi, device_id, sensor_location)
    VALUES (%s, %s, %s, %s, %s)
    """
    
    test_data = [
        (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sensor1',
            -65,
            'device1',
            json.dumps({'x': 0.0, 'y': 0.0, 'z': 0.0})
        ),
        (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sensor2',
            -70,
            'device1',
            json.dumps({'x': 3.0, 'y': 0.0, 'z': 0.0})
        ),
        (
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sensor3',
            -75,
            'device1',
            json.dumps({'x': 0.0, 'y': 3.0, 'z': 0.0})
        )
    ]
    
    for data in test_data:
        execute_write_query(insert_query, data)
    
    yield
    
    # Cleanup
    execute_write_query("DROP TABLE IF EXISTS bluetooth_readings")

def test_get_blueprint(client, sample_sensor_data):
    """Test GET /api/blueprint endpoint."""
    response = client.get('/api/blueprint')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'rooms' in data
    assert 'walls' in data
    assert 'metadata' in data
    assert 'timestamp' in data['metadata']
    
    # Verify rooms structure
    for room in data['rooms']:
        assert 'id' in room
        assert 'vertices' in room
        assert 'center' in room
        assert 'area' in room
        assert 'height' in room
        assert 'type' in room
    
    # Verify walls structure
    for wall in data['walls']:
        assert 'id' in wall
        assert 'room1_id' in wall
        assert 'room2_id' in wall
        assert 'start' in wall
        assert 'end' in wall
        assert 'height' in wall
        assert 'thickness' in wall

def test_update_blueprint(client):
    """Test POST /api/blueprint/update endpoint."""
    update_data = {
        'rooms': [
            {
                'id': 'room_0',
                'type': 'LIVING',
                'vertices': [[0,0,0], [5,0,0], [5,4,0], [0,4,0]],
                'center': {'x': 2.5, 'y': 2.0, 'z': 0.0}
            }
        ],
        'walls': [
            {
                'id': 'wall_0',
                'room1_id': 'room_0',
                'room2_id': 'room_1',
                'start': [0,0,0],
                'end': [5,0,0]
            }
        ]
    }
    
    response = client.post(
        '/api/blueprint/update',
        data=json.dumps(update_data),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'rooms' in data
    assert 'walls' in data

def test_update_blueprint_invalid_data(client):
    """Test POST /api/blueprint/update with invalid data."""
    invalid_data = {
        'rooms': []  # Missing required fields
    }
    
    response = client.post(
        '/api/blueprint/update',
        data=json.dumps(invalid_data),
        content_type='application/json'
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_health_check(client):
    """Test GET /api/health endpoint."""
    response = client.get('/api/health')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'timestamp' in data

def test_not_found(client):
    """Test 404 error handler."""
    response = client.get('/nonexistent-endpoint')
    assert response.status_code == 404
    
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'Not found'

def test_get_blueprint_no_readings(client):
    """Test GET /api/blueprint with no sensor readings."""
    # Ensure no readings exist
    execute_write_query("DROP TABLE IF EXISTS bluetooth_readings")
    execute_write_query("""
        CREATE TABLE bluetooth_readings (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME NOT NULL,
            sensor_id VARCHAR(50) NOT NULL,
            rssi INT NOT NULL,
            device_id VARCHAR(50) NOT NULL,
            sensor_location JSON NOT NULL
        )
    """)
    
    response = client.get('/api/blueprint')
    assert response.status_code == 404
    
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'No recent sensor readings available'

def test_update_blueprint_non_json(client):
    """Test POST /api/blueprint/update with non-JSON data."""
    response = client.post(
        '/api/blueprint/update',
        data='not json data',
        content_type='text/plain'
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'Request must be JSON'

def test_concurrent_updates(client):
    """Test handling concurrent updates to the blueprint."""
    update_data = {
        'rooms': [
            {
                'id': 'room_0',
                'type': 'LIVING',
                'vertices': [[0,0,0], [5,0,0], [5,4,0], [0,4,0]],
                'center': {'x': 2.5, 'y': 2.0, 'z': 0.0}
            }
        ],
        'walls': []
    }
    
    # Make multiple concurrent requests
    responses = []
    for _ in range(5):
        response = client.post(
            '/api/blueprint/update',
            data=json.dumps(update_data),
            content_type='application/json'
        )
        responses.append(response)
    
    # Verify all requests were successful
    for response in responses:
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'rooms' in data
        assert 'walls' in data

def test_large_blueprint(client):
    """Test handling a large blueprint update."""
    # Create a large blueprint with many rooms and walls
    rooms = []
    walls = []
    
    for i in range(100):
        rooms.append({
            'id': f'room_{i}',
            'type': 'LIVING',
            'vertices': [[i,0,0], [i+1,0,0], [i+1,1,0], [i,1,0]],
            'center': {'x': i+0.5, 'y': 0.5, 'z': 0.0}
        })
        
        if i > 0:
            walls.append({
                'id': f'wall_{i}',
                'room1_id': f'room_{i-1}',
                'room2_id': f'room_{i}',
                'start': [i,0,0],
                'end': [i,1,0]
            })
    
    update_data = {
        'rooms': rooms,
        'walls': walls
    }
    
    response = client.post(
        '/api/blueprint/update',
        data=json.dumps(update_data),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data['rooms']) == 100
    assert len(data['walls']) == 99

def test_invalid_room_type(client):
    """Test update with invalid room type."""
    update_data = {
        'rooms': [
            {
                'id': 'room_0',
                'type': 'INVALID_TYPE',  # Invalid room type
                'vertices': [[0,0,0], [5,0,0], [5,4,0], [0,4,0]],
                'center': {'x': 2.5, 'y': 2.0, 'z': 0.0}
            }
        ],
        'walls': []
    }
    
    response = client.post(
        '/api/blueprint/update',
        data=json.dumps(update_data),
        content_type='application/json'
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
