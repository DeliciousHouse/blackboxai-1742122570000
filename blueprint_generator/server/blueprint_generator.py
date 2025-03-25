import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid

import numpy as np
from scipy.spatial import Delaunay

from .bluetooth_processor import BluetoothProcessor
from .ai_processor import AIProcessor
from .db import save_blueprint_update, execute_query, execute_write_query

logger = logging.getLogger(__name__)

class BlueprintGenerator:
    """Generate 3D blueprints from room detection data."""

    # Class variable to hold the single instance
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super(BlueprintGenerator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        """Initialize only once."""
        if getattr(self, '_initialized', False):
            return

        # Your existing initialization code
        self.bluetooth_processor = BluetoothProcessor(config_path)
        self.ai_processor = AIProcessor(config_path)
        self.config = self._load_config(config_path)
        self.validation = self.config['blueprint_validation']
        self.status = {"state": "idle", "progress": 0}
        self.latest_job_id = None
        self.latest_generated_blueprint = None  # In-memory blueprint storage

        # Initialize AI database tables if needed
        self.ai_processor._create_tables()

        self._initialized = True

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from file or use defaults."""
        if config_path:
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
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
            }
        }

    def generate_blueprint(self, device_positions=None, rooms=None):
        """Generate a 3D blueprint based on device positions and detected rooms."""
        try:
            # Debug info - print what's being passed in
            logger.info(f"Generate blueprint called with: {len(device_positions) if device_positions else 0} positions, {len(rooms) if rooms else 0} rooms")

            # Skip database loading if we already have data
            if device_positions and rooms:
                logger.info("Using provided positions and rooms, skipping database loading")
            else:
                # If no positions are provided, load from database
                if not device_positions:
                    device_positions = self.get_device_positions_from_db()

                # If no rooms are provided, try to detect them
                if not rooms and device_positions:
                    bluetooth_processor = BluetoothProcessor()
                    rooms = bluetooth_processor.detect_rooms(device_positions)

            # Now proceed with blueprint generation
            if not rooms:  # If still no rooms, check if we have positions to work with
                if not device_positions:
                    logger.warning("No valid positions found for blueprint generation")
                    return {}

                # Debug what positions we have
                logger.info(f"Have {len(device_positions)} positions but no rooms. Position keys: {list(device_positions.keys())}")

                # Try one more time to create rooms from positions directly
                bluetooth_processor = BluetoothProcessor()
                rooms = bluetooth_processor.detect_rooms(device_positions)

                if not rooms:
                    logger.warning("Failed to generate rooms from available positions")
                    return {}

            # Now we should have rooms to generate a blueprint
            logger.info(f"Generating blueprint with {len(rooms)} rooms")

            # Generate walls between rooms
            walls = self._generate_walls(rooms, device_positions)
            logger.info(f"Generated {len(walls)} walls between rooms")

            # Create the blueprint structure
            blueprint = {
                'version': '1.0',
                'generated_at': datetime.now().isoformat(),
                'rooms': rooms,
                'walls': walls,  # Add walls to the blueprint
                'floors': self._group_rooms_into_floors(rooms),
                'metadata': {
                    'device_count': len(device_positions),
                    'room_count': len(rooms),
                    'wall_count': len(walls)
                }
            }

            # Validate before saving
            if not self._validate_blueprint(blueprint):
                logger.warning("Generated blueprint failed validation")
                # Create a minimal valid blueprint
                blueprint = self._create_minimal_valid_blueprint(rooms)

            # Store in memory before trying database
            self.latest_generated_blueprint = blueprint

            # Save blueprint to database (fixing method name)
            self._save_blueprint(blueprint)  # Changed from _save_blueprint_to_db

            logger.info(f"Final blueprint has {len(blueprint.get('rooms', []))} rooms and {len(blueprint.get('walls', []))} walls")

            # If blueprint is somehow empty but we have rooms, create a minimal blueprint
            if not blueprint.get('rooms') and rooms:
                logger.warning("Blueprint is empty but we have rooms - creating minimal blueprint")
                minimal_blueprint = {
                    'version': '1.0',
                    'generated_at': datetime.now().isoformat(),
                    'rooms': rooms,
                    'walls': [],
                    'floors': [{'level': 0, 'rooms': [r.get('id', f'room_{i}') for i, r in enumerate(rooms)]}],
                    'metadata': {
                        'device_count': len(device_positions),
                        'room_count': len(rooms),
                        'is_minimal': True
                    }
                }
                return minimal_blueprint

            return blueprint

        except Exception as e:
            logger.error(f"Error generating blueprint: {e}")
            logger.info(f"Stored in memory: {self.latest_generated_blueprint is not None}, with {len(self.latest_generated_blueprint.get('rooms', []))} rooms")
            import traceback
            logger.error(traceback.format_exc())  # More detailed error information
            return {}

    def _create_minimal_valid_blueprint(self, rooms):
        """Create a minimal valid blueprint when validation fails."""
        # Basic structure with just rooms
        return {
            'version': '1.0',
            'generated_at': datetime.now().isoformat(),
            'rooms': rooms,
            'walls': [],
            'floors': [{'level': 0, 'rooms': [r['id'] for r in rooms]}],
            'metadata': {
                'room_count': len(rooms),
                'is_minimal': True
            }
        }

    def _generate_walls(self, rooms: List[Dict], positions: Optional[Dict[str, Dict[str, float]]] = None) -> List[Dict]:
        """Generate walls between rooms using AI prediction when available."""
        if not rooms:
            return []

        # Check if ML wall prediction is enabled
        use_ml = self.config.get('ai_settings', {}).get('use_ml_wall_prediction', True)

        if use_ml and positions:
            try:
                # Try to use the ML-based wall prediction
                walls = self.ai_processor.predict_walls(positions, rooms)
                if walls:
                    logger.debug(f"Using ML-based wall prediction: generated {len(walls)} walls")
                    return walls
            except Exception as e:
                logger.warning(f"ML-based wall prediction failed: {str(e)}")

        # Fall back to Delaunay triangulation
        logger.debug("Using Delaunay triangulation for wall generation")

        walls = []

        # Extract room vertices
        vertices = []
        for room in rooms:
            bounds = room['bounds']
            vertices.extend([
                [bounds['min']['x'], bounds['min']['y']],
                [bounds['min']['x'], bounds['max']['y']],
                [bounds['max']['x'], bounds['min']['y']],
                [bounds['max']['x'], bounds['max']['y']]
            ])

        if len(vertices) < 3:
            return walls

        # Create Delaunay triangulation
        vertices = np.array(vertices)
        tri = Delaunay(vertices)

        # Extract edges
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
                edges.add(edge)

        # Convert edges to walls
        for edge in edges:
            p1 = vertices[edge[0]]
            p2 = vertices[edge[1]]

            # Calculate wall properties
            length = np.linalg.norm(p2 - p1)
            angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

            # Skip walls that are too short or too long
            if length < self.validation['min_room_dimension'] or \
               length > self.validation['max_room_dimension']:
                continue

            walls.append({
                'start': {'x': float(p1[0]), 'y': float(p1[1])},
                'end': {'x': float(p2[0]), 'y': float(p2[1])},
                'thickness': self.validation['min_wall_thickness'],
                'height': self.validation['min_ceiling_height'],
                'angle': float(angle)
            })

        return walls

    def _generate_walls_between_rooms(self, rooms):
        """Generate walls between rooms using more realistic algorithms."""
        walls = []

        try:
            # First identify rooms on the same floor
            rooms_by_floor = {}
            for room in rooms:
                floor = int(room['center']['z'] // 3)  # Assuming 3m floor height
                if floor not in rooms_by_floor:
                    rooms_by_floor[floor] = []
                rooms_by_floor[floor].append(room)

            # For each floor, generate walls between adjacent rooms
            for floor, floor_rooms in rooms_by_floor.items():
                logger.info(f"Generating walls for floor {floor} with {len(floor_rooms)} rooms")  # Add this line
                if len(floor_rooms) <= 1:
                    continue

                # Use Delaunay triangulation to find potential adjacent rooms
                from scipy.spatial import Delaunay
                import numpy as np

                # Extract room centers
                centers = []
                for room in floor_rooms:
                    centers.append([room['center']['x'], room['center']['y']])

                # Handle case with too few rooms
                if len(centers) < 3:
                    # Just connect them with a wall
                    if len(centers) == 2:
                        r1, r2 = floor_rooms[0], floor_rooms[1]
                        wall_height = min(r1['dimensions']['height'], r2['dimensions']['height'])
                        walls.append({
                            'start': {'x': r1['center']['x'], 'y': r1['center']['y']},
                            'end': {'x': r2['center']['x'], 'y': r2['center']['y']},
                            'height': wall_height,
                            'thickness': 0.2
                        })
                    continue

                # Create Delaunay triangulation
                tri = Delaunay(np.array(centers))

                # Generate walls from triangulation edges
                edges = set()
                for simplex in tri.simplices:
                    for i in range(3):
                        edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
                        edges.add(edge)

                # Create walls from edges
                for i, j in edges:
                    r1, r2 = floor_rooms[i], floor_rooms[j]

                    # Check if rooms are too far apart
                    dx = r1['center']['x'] - r2['center']['x']
                    dy = r1['center']['y'] - r2['center']['y']
                    distance = (dx**2 + dy**2)**0.5

                    # Skip if too far apart
                    if distance > 10:  # Adjust threshold as needed
                        continue

                    # Create wall
                    wall_height = min(r1['dimensions']['height'], r2['dimensions']['height'])
                    walls.append({
                        'start': {'x': r1['center']['x'], 'y': r1['center']['y']},
                        'end': {'x': r2['center']['x'], 'y': r2['center']['y']},
                        'height': wall_height,
                        'thickness': 0.2
                    })
        except Exception as e:
            logger.error(f"Wall generation failed: {str(e)}")

        logger.info(f"Generated {len(walls)} walls between rooms")
        return walls

    def _validate_blueprint(self, blueprint: Dict) -> bool:
        """Validate generated blueprint."""
        try:
            # Log the validation criteria
            logger.debug(f"Validation criteria: {self.validation}")

            # Validate rooms
            for room in blueprint['rooms']:
                # Log room data for debugging
                logger.debug(f"Validating room: {room['id']}")
                logger.debug(f"Room dimensions: {room['dimensions']}")

                # Check dimensions
                dims = room['dimensions']
                if dims['width'] < self.validation['min_room_dimension'] or \
                   dims['width'] > self.validation['max_room_dimension'] or \
                   dims['length'] < self.validation['min_room_dimension'] or \
                   dims['length'] > self.validation['max_room_dimension'] or \
                   dims['height'] < self.validation['min_ceiling_height'] or \
                   dims['height'] > self.validation['max_ceiling_height']:
                    logger.error(f"Room {room['id']} dimensions out of valid range: width={dims['width']}, length={dims['length']}, height={dims['height']}")
                    return False

                # Check area
                area = dims['width'] * dims['length']
                if area < self.validation['min_room_area'] or \
                   area > self.validation['max_room_area']:
                    logger.error(f"Room {room['id']} area out of valid range: {area}")
                    return False

            # Validate walls
            for idx, wall in enumerate(blueprint['walls']):
                # Log wall data
                logger.debug(f"Validating wall {idx}: {wall}")

                # Check thickness
                if wall['thickness'] < self.validation['min_wall_thickness'] or \
                   wall['thickness'] > self.validation['max_wall_thickness']:
                    logger.error(f"Wall {idx} thickness out of valid range: {wall['thickness']}")
                    return False

                # Check height
                if wall['height'] < self.validation['min_ceiling_height'] or \
                   wall['height'] > self.validation['max_ceiling_height']:
                    logger.error(f"Wall {idx} height out of valid range: {wall['height']}")
                    return False

            logger.info("Blueprint validation passed successfully")
            return True

        except Exception as e:
            logger.error(f"Blueprint validation failed with exception: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def get_latest_blueprint(self) -> Optional[Dict]:
        """Get the latest saved blueprint, with fallback to in-memory."""
        # First check our in-memory blueprint
        if hasattr(self, 'latest_generated_blueprint') and self.latest_generated_blueprint:
            logger.info(f"Using in-memory blueprint with {len(self.latest_generated_blueprint.get('rooms', []))} rooms")
            return self.latest_generated_blueprint

        # Otherwise try database (existing code)
        try:
            logger.info("No in-memory blueprint, checking database")

            # Get multiple blueprints to ensure we find the latest one
            query = """
            SELECT data, created_at, id FROM blueprints
            ORDER BY created_at DESC, id DESC LIMIT 5
            """
            results = execute_query(query)

            if not results:
                logger.warning("No blueprints found in database")
                return None

            logger.info(f"Found {len(results)} blueprints in database")

            # Sort results by created_at to ensure we get the latest
            latest_blueprint = None
            latest_created_at = None

            for result in results:
                # Extract data based on result format
                if isinstance(result, tuple):
                    blueprint_data, created_at, blueprint_id = result
                elif isinstance(result, dict):
                    blueprint_data = result.get('data')
                    created_at = result.get('created_at')
                    blueprint_id = result.get('id')
                else:
                    continue

                logger.info(f"Examining blueprint {blueprint_id} from {created_at}")

                # Parse JSON data
                try:
                    if isinstance(blueprint_data, str):
                        blueprint = json.loads(blueprint_data)
                    else:
                        blueprint = blueprint_data

                    # Verify this is a valid blueprint with rooms
                    if 'rooms' in blueprint and isinstance(blueprint['rooms'], list) and len(blueprint['rooms']) > 0:
                        if latest_created_at is None or (created_at and created_at > latest_created_at):
                            latest_blueprint = blueprint
                            latest_created_at = created_at
                            logger.info(f"Found valid blueprint from {created_at} with {len(blueprint['rooms'])} rooms")
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Error parsing blueprint {blueprint_id}: {e}")

            if latest_blueprint:
                logger.info(f"Using blueprint from {latest_created_at} with {len(latest_blueprint.get('rooms', []))} rooms")
                return latest_blueprint
            else:
                logger.warning("No valid blueprints found")
                return None
        except Exception as e:
            logger.error(f"Failed to get latest blueprint: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def update_blueprint(self, blueprint_data: Dict) -> bool:
        """Update blueprint with manual changes."""
        if not self._validate_blueprint(blueprint_data):
            logger.error("Updated blueprint failed validation")
            return False

        return save_blueprint_update(blueprint_data)

    def get_status(self):
        """Get the current status of blueprint generation."""
        return self.status

    def _refine_blueprint(self, blueprint: Dict) -> Dict:
        """Refine the blueprint using AI techniques."""
        try:
            logger.debug("Applying AI-based blueprint refinement")
            refined_blueprint = self.ai_processor.refine_blueprint(blueprint)

            # Validate the refined blueprint
            if self._validate_blueprint(refined_blueprint):
                logger.debug("AI refinement successful")
                return refined_blueprint
            else:
                logger.warning("AI-refined blueprint failed validation, using original")
                return blueprint

        except Exception as e:
            logger.warning(f"Blueprint refinement failed: {str(e)}")
            return blueprint

    def _save_blueprint(self, blueprint):
        """Save blueprint to database."""
        try:
            # Convert blueprint to JSON string
            blueprint_json = json.dumps(blueprint)

            # Get current timestamp
            current_time = datetime.now().isoformat()

            # Insert into database with explicit timestamp
            query = """
            INSERT INTO blueprints (data, status, created_at)
            VALUES (%s, 'active', %s)
            """
            execute_write_query(query, (blueprint_json, current_time))
            logger.info(f"Blueprint saved successfully to database with timestamp {current_time}")
            return True

        except Exception as e:
            logger.error(f"Failed to save blueprint: {str(e)}")
            return False

    def get_device_positions_from_db(self):
        """Get the latest device positions from the database."""
        try:
            logger.info("Loading device positions from database")
            query = """
            SELECT device_id, position_data, source, timestamp
            FROM device_positions
            WHERE timestamp = (SELECT MAX(timestamp) FROM device_positions)
            """
            results = execute_query(query)

            positions = {}
            for row in results:
                if isinstance(row, tuple):
                    device_id, position_data, source, timestamp = row
                else:
                    device_id = row.get('device_id')
                    position_data = row.get('position_data')
                    source = row.get('source')
                    timestamp = row.get('timestamp')

                # Parse position data from JSON
                try:
                    if isinstance(position_data, str):
                        position = json.loads(position_data)
                    else:
                        position = position_data

                    # Ensure all required fields exist
                    if all(k in position for k in ['x', 'y', 'z']):
                        positions[device_id] = {
                            'x': float(position['x']),
                            'y': float(position['y']),
                            'z': float(position['z']),
                            'accuracy': float(position.get('accuracy', 1.0)),  # Get accuracy from the position JSON
                            'source': source or position.get('source', 'unknown')
                        }
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Error parsing position data for {device_id}: {e}")

            logger.info(f"Loaded {len(positions)} device positions from database")
            return positions
        except Exception as e:
            logger.error(f"Error loading device positions from database: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _group_rooms_into_floors(self, rooms: List[Dict]) -> List[Dict]:
        """Group rooms into floors based on their z-coordinate."""
        if not rooms:
            return []

        # Group rooms by their z-coordinate (floor level)
        floors_dict = {}
        for room in rooms:
            # Use the minimum z-coordinate as the floor level
            floor_level = int(room['bounds']['min']['z'])

            if floor_level not in floors_dict:
                floors_dict[floor_level] = []

            floors_dict[floor_level].append(room['id'])

        # Convert to list of floor objects
        floors = []
        for level, room_ids in sorted(floors_dict.items()):
            floors.append({
                'level': level,
                'rooms': room_ids,
                'height': 3.0  # Default floor height
            })

        logger.info(f"Grouped {len(rooms)} rooms into {len(floors)} floors")
        return floors

# Add this to your main.py or blueprint_generator.py
def ensure_reference_positions():
    """Make sure we have at least some reference positions in the database"""
    from .db import get_sqlite_connection

    conn = get_sqlite_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM device_positions")
    count = cursor.fetchone()[0]
    conn.close()

    if count == 0:
        logger.info("No device positions in database, creating initial reference points")
        # Create at least 3 reference points for the system to work with
        default_positions = {
            "reference_point_1": {"x": 0, "y": 0, "z": 0},
            "reference_point_2": {"x": 5, "y": 0, "z": 0},
            "reference_point_3": {"x": 0, "y": 5, "z": 0}
        }
        from .bluetooth_processor import save_device_position
        for device_id, position in default_positions.items():
            position['source'] = 'initial_setup'
            position['accuracy'] = 1.0
            save_device_position(device_id, position)
        return default_positions
    return None
