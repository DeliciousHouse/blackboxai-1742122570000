import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid

import numpy as np
from scipy.spatial import Delaunay

from .bluetooth_processor import BluetoothProcessor
from .ai_processor import AIProcessor
from .db import get_latest_blueprint, save_blueprint_update, execute_query, execute_write_query

logger = logging.getLogger(__name__)

class BlueprintGenerator:
    """Generate 3D blueprints from room detection data."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the blueprint generator."""
        self.bluetooth_processor = BluetoothProcessor(config_path)
        self.ai_processor = AIProcessor(config_path)
        self.config = self._load_config(config_path)
        self.validation = self.config['blueprint_validation']
        self.status = {"state": "idle", "progress": 0}
        self.latest_job_id = None

        # Initialize AI database tables if needed
        self.ai_processor._create_tables()

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

    def generate_blueprint(self, time_window=3600):
        """Generate a blueprint using position data from integrations."""
        try:
            job_id = str(uuid.uuid4())
            self.latest_job_id = job_id
            self.status = {"state": "processing", "progress": 10, "job_id": job_id}

            # First check for Bermuda positions
            query = """
            SELECT device_id, position_data FROM device_positions
            WHERE source = 'bermuda' AND timestamp > NOW() - INTERVAL %s SECOND
            """
            bermuda_positions = {row[0]: json.loads(row[1]) for row in execute_query(query, (time_window,))}

            # Basic blueprint structure
            blueprint = {
                "id": job_id,
                "timestamp": datetime.now().isoformat(),
                "rooms": [
                    {"id": "lounge", "name": "Lounge", "dimensions": {"length": 5, "width": 4, "height": 2.5}}
                ],
                "walls": [],
                "devices": []
            }

            # Add devices from Bermuda positions
            if bermuda_positions:
                logger.info(f"Using {len(bermuda_positions)} positions from Bermuda")
                for device_id, position in bermuda_positions.items():
                    blueprint["devices"].append({
                        "id": device_id,
                        "position": position,
                        "source": "bermuda"
                    })
            else:
                # Fall back to raw readings if no Bermuda positions
                logger.warning("No valid positions found for blueprint generation")

                # Get device positions using the bluetooth processor
                positions = self.bluetooth_processor.estimate_positions(time_window)
                if positions:
                    # Detect rooms
                    rooms = self.bluetooth_processor.detect_rooms(positions)
                    if rooms:
                        blueprint['rooms'] = rooms
                        blueprint['positions'] = positions

                        # Generate walls using AI-enhanced prediction
                        blueprint['walls'] = self._generate_walls(rooms, positions)

                        # Apply AI-based blueprint refinement if enabled
                        if self.config.get('ai_settings', {}).get('use_ml_blueprint_refinement', True):
                            blueprint = self._refine_blueprint(blueprint)

            # Store blueprint in database
            self._save_blueprint(blueprint)

            self.status = {"state": "completed", "progress": 100, "job_id": job_id}
            return {"job_id": job_id}

        except Exception as e:
            self.status = {"state": "error", "error": str(e)}
            logger.error(f"Blueprint generation failed: {str(e)}")
            raise

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

    def _validate_blueprint(self, blueprint: Dict) -> bool:
        """Validate generated blueprint."""
        try:
            # Validate rooms
            for room in blueprint['rooms']:
                # Check dimensions
                dims = room['dimensions']
                if dims['width'] < self.validation['min_room_dimension'] or \
                   dims['width'] > self.validation['max_room_dimension'] or \
                   dims['length'] < self.validation['min_room_dimension'] or \
                   dims['length'] > self.validation['max_room_dimension'] or \
                   dims['height'] < self.validation['min_ceiling_height'] or \
                   dims['height'] > self.validation['max_ceiling_height']:
                    logger.error("Room dimensions out of valid range")
                    return False

                # Check area
                area = dims['width'] * dims['length']
                if area < self.validation['min_room_area'] or \
                   area > self.validation['max_room_area']:
                    logger.error("Room area out of valid range")
                    return False

            # Validate walls
            for wall in blueprint['walls']:
                # Check thickness
                if wall['thickness'] < self.validation['min_wall_thickness'] or \
                   wall['thickness'] > self.validation['max_wall_thickness']:
                    logger.error("Wall thickness out of valid range")
                    return False

                # Check height
                if wall['height'] < self.validation['min_ceiling_height'] or \
                   wall['height'] > self.validation['max_ceiling_height']:
                    logger.error("Wall height out of valid range")
                    return False

            return True

        except Exception as e:
            logger.error(f"Blueprint validation failed: {str(e)}")
            return False

    def get_latest_blueprint(self) -> Optional[Dict]:
        """Get the latest saved blueprint."""
        try:
            query = """
            SELECT data FROM blueprints
            ORDER BY created_at DESC LIMIT 1
            """
            result = execute_query(query)

            if not result:
                return None

            blueprint_data = result[0][0] if isinstance(result[0], tuple) else result[0].get('data')

            # If data is stored as a string, parse it
            if isinstance(blueprint_data, str):
                return json.loads(blueprint_data)
            return blueprint_data

        except Exception as e:
            logger.error(f"Failed to get latest blueprint: {str(e)}")
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

            # Insert into database
            query = """
            INSERT INTO blueprints (data, status)
            VALUES (%s, 'active')
            """
            execute_write_query(query, (blueprint_json,))

        except Exception as e:
            logger.error(f"Failed to save blueprint: {str(e)}")
