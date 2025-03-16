import json
import logging.config
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Setup logging
logging.config.fileConfig('config/logging.conf')
logger = logging.getLogger('server.blueprint_generator')

@dataclass
class ValidationThresholds:
    min_room_area: float
    max_room_area: float
    min_room_dimension: float
    max_room_dimension: float
    min_wall_thickness: float
    max_wall_thickness: float
    min_ceiling_height: float
    max_ceiling_height: float

class BlueprintGenerator:
    def __init__(self, config_path: str = 'config/config.json'):
        """
        Initialize BlueprintGenerator with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        self.thresholds = self._load_validation_thresholds()
        self.room_types = {
            'LIVING': {'min_area': 15, 'typical_ratio': 1.5},
            'BEDROOM': {'min_area': 9, 'typical_ratio': 1.3},
            'KITCHEN': {'min_area': 8, 'typical_ratio': 1.4},
            'BATHROOM': {'min_area': 4, 'typical_ratio': 1.2},
            'HALLWAY': {'min_area': 4, 'typical_ratio': 2.5}
        }

    def _load_validation_thresholds(self) -> ValidationThresholds:
        """
        Load validation thresholds from config file.
        
        Returns:
            ValidationThresholds: Validation thresholds for blueprint generation
        """
        try:
            with open(self.config_path) as f:
                config = json.load(f)
            thresholds = config['blueprint_validation']
            
            return ValidationThresholds(
                min_room_area=thresholds['min_room_area'],
                max_room_area=thresholds['max_room_area'],
                min_room_dimension=thresholds['min_room_dimension'],
                max_room_dimension=thresholds['max_room_dimension'],
                min_wall_thickness=thresholds['min_wall_thickness'],
                max_wall_thickness=thresholds['max_wall_thickness'],
                min_ceiling_height=thresholds['min_ceiling_height'],
                max_ceiling_height=thresholds['max_ceiling_height']
            )
        except Exception as e:
            logger.error(f"Error loading validation thresholds: {str(e)}")
            raise

    def generate_blueprint(self, device_positions: Dict[str, Dict[str, float]]) -> Dict:
        """
        Generate a 3D home blueprint from device positions.
        
        Args:
            device_positions (Dict[str, Dict[str, float]]): Dictionary mapping device IDs to positions
        
        Returns:
            Dict: Generated blueprint in JSON format
        """
        try:
            # Extract points cloud from device positions
            points = np.array([[pos['x'], pos['y'], pos['z']] 
                             for pos in device_positions.values()])
            
            # Generate rooms from points
            rooms = self._generate_rooms(points)
            
            # Identify room types based on size and position
            rooms = self._identify_room_types(rooms)
            
            # Generate walls between rooms
            walls = self._generate_walls(rooms)
            
            # Create the blueprint structure
            blueprint = {
                'version': '1.0',
                'rooms': rooms,
                'walls': walls,
                'metadata': {
                    'total_area': sum(room['area'] for room in rooms),
                    'room_count': len(rooms),
                    'timestamp': None  # To be filled by the API layer
                }
            }
            
            # Validate the generated blueprint
            if self._validate_blueprint(blueprint):
                logger.info("Generated valid blueprint")
                return blueprint
            else:
                logger.error("Generated blueprint failed validation")
                raise ValueError("Generated blueprint failed validation")

        except Exception as e:
            logger.error(f"Error generating blueprint: {str(e)}")
            raise

    def _generate_rooms(self, points: np.ndarray) -> List[Dict]:
        """
        Generate rooms from point cloud data using clustering and convex hull.
        
        Args:
            points (np.ndarray): Array of 3D points
        
        Returns:
            List[Dict]: List of room definitions
        """
        try:
            # Use DBSCAN clustering to identify potential rooms
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=2.0, min_samples=3).fit(points[:, :2])  # Use only x,y coordinates
            
            rooms = []
            for cluster_id in set(clustering.labels_):
                if cluster_id == -1:  # Skip noise points
                    continue
                    
                # Get points for this cluster
                cluster_points = points[clustering.labels_ == cluster_id]
                
                # Calculate 2D convex hull for room boundary
                from scipy.spatial import ConvexHull
                hull = ConvexHull(cluster_points[:, :2])
                
                # Extract room properties
                vertices = cluster_points[hull.vertices]
                center = np.mean(vertices, axis=0)
                area = hull.area
                
                # Create room definition
                room = {
                    'id': f'room_{len(rooms)}',
                    'vertices': vertices.tolist(),
                    'center': {'x': float(center[0]), 'y': float(center[1]), 'z': float(center[2])},
                    'area': float(area),
                    'height': float(np.max(cluster_points[:, 2]) - np.min(cluster_points[:, 2])),
                    'type': None  # To be filled by _identify_room_types
                }
                
                rooms.append(room)
            
            return rooms

        except Exception as e:
            logger.error(f"Error generating rooms: {str(e)}")
            raise

    def _identify_room_types(self, rooms: List[Dict]) -> List[Dict]:
        """
        Identify room types based on size and position characteristics.
        
        Args:
            rooms (List[Dict]): List of room definitions
        
        Returns:
            List[Dict]: Updated room definitions with types
        """
        try:
            # Sort rooms by area (largest first)
            rooms = sorted(rooms, key=lambda r: r['area'], reverse=True)
            
            for room in rooms:
                # Calculate room dimensions
                vertices = np.array(room['vertices'])
                width = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
                length = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
                ratio = max(width, length) / min(width, length)
                
                # Identify room type based on area and ratio
                if room['area'] >= self.room_types['LIVING']['min_area']:
                    room['type'] = 'LIVING'
                elif ratio > 2.0:
                    room['type'] = 'HALLWAY'
                elif room['area'] >= self.room_types['BEDROOM']['min_area']:
                    room['type'] = 'BEDROOM'
                elif room['area'] >= self.room_types['KITCHEN']['min_area']:
                    room['type'] = 'KITCHEN'
                else:
                    room['type'] = 'BATHROOM'
            
            return rooms

        except Exception as e:
            logger.error(f"Error identifying room types: {str(e)}")
            raise

    def _generate_walls(self, rooms: List[Dict]) -> List[Dict]:
        """
        Generate walls between adjacent rooms.
        
        Args:
            rooms (List[Dict]): List of room definitions
        
        Returns:
            List[Dict]: List of wall definitions
        """
        try:
            walls = []
            processed_pairs = set()
            
            for i, room1 in enumerate(rooms):
                for j, room2 in enumerate(rooms[i+1:], i+1):
                    if (room1['id'], room2['id']) in processed_pairs:
                        continue
                        
                    # Find closest vertices between rooms
                    vertices1 = np.array(room1['vertices'])
                    vertices2 = np.array(room2['vertices'])
                    
                    # Calculate minimum distance between room vertices
                    distances = np.linalg.norm(vertices1[:, np.newaxis] - vertices2, axis=2)
                    min_distance = np.min(distances)
                    
                    # If rooms are adjacent (distance less than threshold)
                    if min_distance < 2.0:  # Threshold for adjacency
                        # Find the closest vertices
                        idx1, idx2 = np.unravel_index(np.argmin(distances), distances.shape)
                        
                        # Create wall definition
                        wall = {
                            'id': f'wall_{len(walls)}',
                            'room1_id': room1['id'],
                            'room2_id': room2['id'],
                            'start': vertices1[idx1].tolist(),
                            'end': vertices2[idx2].tolist(),
                            'height': min(room1['height'], room2['height']),
                            'thickness': self.thresholds.min_wall_thickness
                        }
                        
                        walls.append(wall)
                        processed_pairs.add((room1['id'], room2['id']))
            
            return walls

        except Exception as e:
            logger.error(f"Error generating walls: {str(e)}")
            raise

    def _validate_blueprint(self, blueprint: Dict) -> bool:
        """
        Validate the generated blueprint against thresholds.
        
        Args:
            blueprint (Dict): Generated blueprint
        
        Returns:
            bool: True if blueprint is valid, False otherwise
        """
        try:
            # Validate rooms
            for room in blueprint['rooms']:
                # Check room area
                if not (self.thresholds.min_room_area <= room['area'] <= self.thresholds.max_room_area):
                    logger.error(f"Room {room['id']} area ({room['area']}) outside valid range")
                    return False
                
                # Check room height
                if not (self.thresholds.min_ceiling_height <= room['height'] <= self.thresholds.max_ceiling_height):
                    logger.error(f"Room {room['id']} height ({room['height']}) outside valid range")
                    return False
                
                # Check room dimensions
                vertices = np.array(room['vertices'])
                width = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
                length = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
                
                if width < self.thresholds.min_room_dimension or length < self.thresholds.min_room_dimension:
                    logger.error(f"Room {room['id']} dimensions too small")
                    return False
                
                if width > self.thresholds.max_room_dimension or length > self.thresholds.max_room_dimension:
                    logger.error(f"Room {room['id']} dimensions too large")
                    return False
            
            # Validate walls
            for wall in blueprint['walls']:
                # Check wall thickness
                if not (self.thresholds.min_wall_thickness <= wall['thickness'] <= self.thresholds.max_wall_thickness):
                    logger.error(f"Wall {wall['id']} thickness ({wall['thickness']}) outside valid range")
                    return False
            
            return True

        except Exception as e:
            logger.error(f"Error validating blueprint: {str(e)}")
            return False

def main():
    """
    Main function to test blueprint generation functionality.
    """
    try:
        generator = BlueprintGenerator()
        
        # Example device positions
        sample_positions = {
            'device1': {'x': 0.0, 'y': 0.0, 'z': 1.5},
            'device2': {'x': 3.0, 'y': 0.0, 'z': 1.5},
            'device3': {'x': 0.0, 'y': 4.0, 'z': 1.5},
            'device4': {'x': 3.0, 'y': 4.0, 'z': 1.5}
        }
        
        # Generate blueprint
        blueprint = generator.generate_blueprint(sample_positions)
        logger.info(f"Generated blueprint: {json.dumps(blueprint, indent=2)}")

    except Exception as e:
        logger.error(f"Test blueprint generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
