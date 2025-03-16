import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import Delaunay

from .bluetooth_processor import BluetoothProcessor
from .db import get_latest_blueprint, save_blueprint_update

logger = logging.getLogger(__name__)

class BlueprintGenerator:
    """Generate 3D blueprints from room detection data."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the blueprint generator."""
        self.bluetooth_processor = BluetoothProcessor(config_path)
        self.config = self._load_config(config_path)
        self.validation = self.config['blueprint_validation']

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
            }
        }

    def generate_blueprint(self, time_window: int = 300) -> Optional[Dict]:
        """Generate a new blueprint based on recent readings."""
        # Get device positions
        positions = self.bluetooth_processor.estimate_positions(time_window)
        if not positions:
            logger.warning("No valid positions found for blueprint generation")
            return None

        # Detect rooms
        rooms = self.bluetooth_processor.detect_rooms(positions)
        if not rooms:
            logger.warning("No rooms detected for blueprint generation")
            return None

        # Generate walls
        walls = self._generate_walls(rooms)

        # Validate blueprint
        blueprint = {
            'rooms': rooms,
            'walls': walls,
            'positions': positions
        }

        if not self._validate_blueprint(blueprint):
            logger.error("Generated blueprint failed validation")
            return None

        # Save blueprint
        if save_blueprint_update(blueprint):
            return blueprint
        return None

    def _generate_walls(self, rooms: List[Dict]) -> List[Dict]:
        """Generate walls between rooms."""
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
        return get_latest_blueprint()

    def update_blueprint(self, blueprint_data: Dict) -> bool:
        """Update blueprint with manual changes."""
        if not self._validate_blueprint(blueprint_data):
            logger.error("Updated blueprint failed validation")
            return False
        
        return save_blueprint_update(blueprint_data)
