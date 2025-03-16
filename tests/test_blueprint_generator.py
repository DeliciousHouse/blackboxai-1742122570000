import pytest
import json
import numpy as np
from server.blueprint_generator import BlueprintGenerator, ValidationThresholds

@pytest.fixture
def blueprint_generator():
    """Create a BlueprintGenerator instance for testing."""
    return BlueprintGenerator()

@pytest.fixture
def sample_device_positions():
    """Create sample device positions for testing."""
    return {
        'device1': {'x': 0.0, 'y': 0.0, 'z': 1.5},
        'device2': {'x': 3.0, 'y': 0.0, 'z': 1.5},
        'device3': {'x': 0.0, 'y': 4.0, 'z': 1.5},
        'device4': {'x': 3.0, 'y': 4.0, 'z': 1.5}
    }

@pytest.fixture
def validation_thresholds():
    """Create validation thresholds for testing."""
    return ValidationThresholds(
        min_room_area=4.0,
        max_room_area=100.0,
        min_room_dimension=1.5,
        max_room_dimension=15.0,
        min_wall_thickness=0.1,
        max_wall_thickness=0.5,
        min_ceiling_height=2.2,
        max_ceiling_height=4.0
    )

def test_load_validation_thresholds(blueprint_generator):
    """Test loading validation thresholds from config."""
    thresholds = blueprint_generator._load_validation_thresholds()
    assert isinstance(thresholds, ValidationThresholds)
    assert thresholds.min_room_area > 0
    assert thresholds.max_room_area > thresholds.min_room_area
    assert thresholds.min_room_dimension > 0
    assert thresholds.max_room_dimension > thresholds.min_room_dimension
    assert thresholds.min_wall_thickness > 0
    assert thresholds.max_wall_thickness > thresholds.min_wall_thickness
    assert thresholds.min_ceiling_height > 0
    assert thresholds.max_ceiling_height > thresholds.min_ceiling_height

def test_generate_blueprint(blueprint_generator, sample_device_positions):
    """Test generating a blueprint from device positions."""
    blueprint = blueprint_generator.generate_blueprint(sample_device_positions)
    
    assert isinstance(blueprint, dict)
    assert 'version' in blueprint
    assert 'rooms' in blueprint
    assert 'walls' in blueprint
    assert 'metadata' in blueprint
    
    # Check rooms
    assert len(blueprint['rooms']) > 0
    for room in blueprint['rooms']:
        assert 'id' in room
        assert 'vertices' in room
        assert 'center' in room
        assert 'area' in room
        assert 'height' in room
        assert 'type' in room
    
    # Check walls
    for wall in blueprint['walls']:
        assert 'id' in wall
        assert 'room1_id' in wall
        assert 'room2_id' in wall
        assert 'start' in wall
        assert 'end' in wall
        assert 'height' in wall
        assert 'thickness' in wall

def test_generate_rooms(blueprint_generator, sample_device_positions):
    """Test room generation from point cloud."""
    points = np.array([[pos['x'], pos['y'], pos['z']] 
                      for pos in sample_device_positions.values()])
    
    rooms = blueprint_generator._generate_rooms(points)
    
    assert isinstance(rooms, list)
    assert len(rooms) > 0
    
    for room in rooms:
        assert 'id' in room
        assert 'vertices' in room
        assert 'center' in room
        assert 'area' in room
        assert 'height' in room
        assert isinstance(room['vertices'], list)
        assert len(room['vertices']) >= 3  # At least 3 vertices for a room

def test_identify_room_types(blueprint_generator):
    """Test room type identification."""
    test_rooms = [
        {
            'id': 'room_0',
            'vertices': [[0,0,0], [5,0,0], [5,5,0], [0,5,0]],
            'center': {'x': 2.5, 'y': 2.5, 'z': 0},
            'area': 25.0,
            'height': 2.5,
            'type': None
        },
        {
            'id': 'room_1',
            'vertices': [[0,0,0], [2,0,0], [2,2,0], [0,2,0]],
            'center': {'x': 1, 'y': 1, 'z': 0},
            'area': 4.0,
            'height': 2.5,
            'type': None
        }
    ]
    
    rooms = blueprint_generator._identify_room_types(test_rooms)
    
    assert len(rooms) == 2
    assert rooms[0]['type'] == 'LIVING'  # Larger room
    assert rooms[1]['type'] == 'BATHROOM'  # Smaller room

def test_generate_walls(blueprint_generator):
    """Test wall generation between rooms."""
    test_rooms = [
        {
            'id': 'room_0',
            'vertices': [[0,0,0], [3,0,0], [3,3,0], [0,3,0]],
            'height': 2.5
        },
        {
            'id': 'room_1',
            'vertices': [[3,0,0], [6,0,0], [6,3,0], [3,3,0]],
            'height': 2.5
        }
    ]
    
    walls = blueprint_generator._generate_walls(test_rooms)
    
    assert isinstance(walls, list)
    assert len(walls) > 0
    
    for wall in walls:
        assert 'id' in wall
        assert 'room1_id' in wall
        assert 'room2_id' in wall
        assert 'start' in wall
        assert 'end' in wall
        assert 'height' in wall
        assert 'thickness' in wall
        assert wall['thickness'] >= blueprint_generator.thresholds.min_wall_thickness
        assert wall['thickness'] <= blueprint_generator.thresholds.max_wall_thickness

def test_validate_blueprint(blueprint_generator):
    """Test blueprint validation."""
    valid_blueprint = {
        'rooms': [
            {
                'id': 'room_0',
                'area': 20.0,
                'height': 2.5,
                'vertices': [[0,0,0], [5,0,0], [5,4,0], [0,4,0]]
            }
        ],
        'walls': [
            {
                'id': 'wall_0',
                'thickness': 0.2,
                'height': 2.5,
                'start': [0,0,0],
                'end': [5,0,0]
            }
        ]
    }
    
    assert blueprint_generator._validate_blueprint(valid_blueprint) is True

def test_validate_blueprint_invalid_room_area(blueprint_generator):
    """Test blueprint validation with invalid room area."""
    invalid_blueprint = {
        'rooms': [
            {
                'id': 'room_0',
                'area': 1.0,  # Too small
                'height': 2.5,
                'vertices': [[0,0,0], [1,0,0], [1,1,0], [0,1,0]]
            }
        ],
        'walls': []
    }
    
    assert blueprint_generator._validate_blueprint(invalid_blueprint) is False

def test_validate_blueprint_invalid_ceiling_height(blueprint_generator):
    """Test blueprint validation with invalid ceiling height."""
    invalid_blueprint = {
        'rooms': [
            {
                'id': 'room_0',
                'area': 20.0,
                'height': 5.0,  # Too high
                'vertices': [[0,0,0], [5,0,0], [5,4,0], [0,4,0]]
            }
        ],
        'walls': []
    }
    
    assert blueprint_generator._validate_blueprint(invalid_blueprint) is False

def test_validate_blueprint_invalid_wall_thickness(blueprint_generator):
    """Test blueprint validation with invalid wall thickness."""
    invalid_blueprint = {
        'rooms': [
            {
                'id': 'room_0',
                'area': 20.0,
                'height': 2.5,
                'vertices': [[0,0,0], [5,0,0], [5,4,0], [0,4,0]]
            }
        ],
        'walls': [
            {
                'id': 'wall_0',
                'thickness': 1.0,  # Too thick
                'height': 2.5,
                'start': [0,0,0],
                'end': [5,0,0]
            }
        ]
    }
    
    assert blueprint_generator._validate_blueprint(invalid_blueprint) is False

def test_empty_device_positions(blueprint_generator):
    """Test handling of empty device positions."""
    with pytest.raises(ValueError):
        blueprint_generator.generate_blueprint({})

def test_single_device_position(blueprint_generator):
    """Test handling of single device position."""
    single_position = {
        'device1': {'x': 0.0, 'y': 0.0, 'z': 1.5}
    }
    
    blueprint = blueprint_generator.generate_blueprint(single_position)
    assert isinstance(blueprint, dict)
    assert len(blueprint['rooms']) > 0

def test_complex_layout(blueprint_generator):
    """Test handling of complex room layout."""
    complex_positions = {
        'device1': {'x': 0.0, 'y': 0.0, 'z': 1.5},
        'device2': {'x': 3.0, 'y': 0.0, 'z': 1.5},
        'device3': {'x': 0.0, 'y': 4.0, 'z': 1.5},
        'device4': {'x': 3.0, 'y': 4.0, 'z': 1.5},
        'device5': {'x': 6.0, 'y': 0.0, 'z': 1.5},
        'device6': {'x': 6.0, 'y': 4.0, 'z': 1.5}
    }
    
    blueprint = blueprint_generator.generate_blueprint(complex_positions)
    assert isinstance(blueprint, dict)
    assert len(blueprint['rooms']) > 1
    assert len(blueprint['walls']) > 1
