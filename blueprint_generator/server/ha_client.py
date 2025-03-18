import requests
import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class HomeAssistantClient:
    """Client for interacting with Home Assistant API."""

    def __init__(self, base_url: Optional[str] = None, token: Optional[str] = None):
        """Initialize with Home Assistant connection details."""
        # Use provided values or get from environment (for add-on)
        self.base_url = base_url or self._get_base_url()
        self.token = token or self._get_token()
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
        logger.debug(f"Initialized HomeAssistantClient with base_url: {self.base_url}")

    def _get_base_url(self) -> str:
        """Get base URL from environment or use default."""
        # For Home Assistant add-on
        if os.environ.get('SUPERVISOR_TOKEN'):
            return "http://supervisor/core"

        # Check for options file
        options_path = '/data/options.json'
        if os.path.exists(options_path):
            try:
                with open(options_path) as f:
                    options = json.load(f)
                    if 'ha_url' in options:
                        return options['ha_url']
            except Exception as e:
                logger.error(f"Error reading options.json: {e}")

        # Default for development
        return "http://localhost:8123"

    def _get_token(self) -> str:
        """Get authentication token from environment."""
        # Try to get token from Home Assistant add-on environment
        supervisor_token = os.environ.get('SUPERVISOR_TOKEN')
        if supervisor_token:
            return supervisor_token

        # Check for options file
        options_path = '/data/options.json'
        if os.path.exists(options_path):
            try:
                with open(options_path) as f:
                    options = json.load(f)
                    if 'ha_token' in options:
                        return options['ha_token']
            except Exception as e:
                logger.error(f"Error reading options.json: {e}")

        # Return empty string if no token found (will fail authentication)
        return ""

    def find_entities_by_pattern(self, patterns: List[str], domains: List[str] = None) -> List[Dict]:
        """Find entities matching any of the patterns in their entity_id or attributes."""
        try:
            url = f"{self.base_url}/api/states"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            all_states = response.json()
            matching_entities = []

            for state in all_states:
                entity_id = state.get('entity_id', '')

                # Check domain filter if provided
                if domains and not any(entity_id.startswith(f"{domain}.") for domain in domains):
                    continue

                # Check for pattern matches
                if any(pattern.lower() in entity_id.lower() for pattern in patterns):
                    matching_entities.append({
                        'entity_id': entity_id,
                        'state': state.get('state'),
                        'attributes': state.get('attributes', {})
                    })

            return matching_entities

        except Exception as e:
            logger.error(f"Failed to find entities by pattern: {str(e)}")
            return []

    def get_bluetooth_devices(self) -> List[Dict]:
        """Get all bluetooth devices from Home Assistant."""
        try:
            url = f"{self.base_url}/api/states"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            all_states = response.json()
            bluetooth_devices = []

            for state in all_states:
                entity_id = state.get('entity_id', '')
                if entity_id.startswith('sensor.', 'binary_sensor.'):
                    bluetooth_devices.append({
                        'entity_id': entity_id,
                        'attributes': state.get('attributes', {}),
                        'state': state.get('state')
                    })

            return bluetooth_devices

        except Exception as e:
            logger.error(f"Failed to get bluetooth devices from Home Assistant: {str(e)}")
            return []

    def get_private_ble_devices(self) -> List[Dict]:
        """Get all BLE and distance-sensing devices with improved flexibility."""
        try:
            url = f"{self.base_url}/api/states"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            all_states = response.json()
            devices = []

            # Keywords that indicate relevant entities
            device_keywords = ['ble', 'bluetooth', 'watch', 'iphone', 'phone', 'mmwave']
            measurement_keywords = ['distance', 'rssi', 'signal', 'detection']

            for state in all_states:
                entity_id = state.get('entity_id', '').lower()
                attributes = state.get('attributes', {})
                friendly_name = attributes.get('friendly_name', entity_id)

                # Don't limit to just sensors - include device trackers too
                if entity_id.startswith(('sensor.', 'device_tracker.', 'binary_sensor.')):

                    # Check if entity contains both device and measurement keywords
                    has_device_keyword = any(keyword in entity_id for keyword in device_keywords)
                    has_measurement = any(keyword in entity_id for keyword in measurement_keywords)

                    # Either it needs both types of keywords, or it has 'ble' and is related to devices
                    if (has_device_keyword and has_measurement) or ('ble' in entity_id):
                        # Get the value - could be distance or RSSI
                        value = state.get('state')
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            if value == 'unavailable' or value == 'unknown':
                                value = -100  # Default for unavailable devices
                            else:
                                # Skip non-numeric values that aren't standard states
                                continue

                        # Try to extract a device identifier
                        parts = entity_id.split('_')
                        if len(parts) >= 2:
                            # Use part after domain as device identifier
                            device_type = parts[1]
                        else:
                            device_type = 'unknown'

                        # Take a better guess at whether this is distance or RSSI
                        is_distance = any(k in entity_id for k in ['distance', 'meters', 'range'])
                        is_rssi = any(k in entity_id for k in ['rssi', 'signal', 'strength'])

                        # Convert between distance and RSSI if needed
                        if is_distance:
                            distance = value
                            rssi = -59 + (value * -2)  # Simple conversion
                        elif is_rssi:
                            rssi = value
                            distance = max(0, ((-1 * (rssi + 59)) / 2))  # Reverse of above formula
                        else:
                            # If unclear, assume it's RSSI if negative, distance if positive
                            if value < 0:
                                rssi = value
                                distance = max(0, ((-1 * (rssi + 59)) / 2))
                            else:
                                distance = value
                                rssi = -59 + (value * -2)

                        devices.append({
                            'mac': device_type,  # Use extracted identifier
                            'rssi': rssi,
                            'entity_id': state.get('entity_id'),  # Use original case
                            'friendly_name': friendly_name,
                            'distance': distance
                        })

                        # Debug logging to see what we found
                        logger.debug(f"Found relevant device: {state.get('entity_id')} with value {value}")

            logger.info(f"Found {len(devices)} BLE/distance devices")
            return devices

        except Exception as e:
            logger.error(f"Failed to get BLE devices from Home Assistant: {str(e)}", exc_info=True)
            return []

    def get_bermuda_positions(self) -> List[Dict]:
        """Get device positions with more flexible matching."""
        try:
            url = f"{self.base_url}/api/states"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            all_states = response.json()
            positions = []

            # Keywords that indicate position entities
            position_keywords = ['position', 'location', 'tracker', 'bermuda', 'mmwave', 'coordinates']

            for state in all_states:
                entity_id = state.get('entity_id', '').lower()
                attributes = state.get('attributes', {})

                # More flexible matching - look for position indicators in entity ID or attributes
                has_position_keyword = any(keyword in entity_id for keyword in position_keywords)
                has_position_attrs = ('position' in attributes or
                                    ('x' in attributes and 'y' in attributes) or
                                    ('latitude' in attributes and 'longitude' in attributes))

                if has_position_keyword or has_position_attrs:
                    # Try different formats of position data
                    position_data = {}

                    # Check for nested position attribute
                    if 'position' in attributes and isinstance(attributes['position'], dict):
                        position_data = attributes['position']

                    # Check for top-level x,y,z coordinates
                    elif 'x' in attributes and 'y' in attributes:
                        position_data = {
                            'x': attributes.get('x', 0),
                            'y': attributes.get('y', 0),
                            'z': attributes.get('z', 0)
                        }

                    # Check for lat/long coordinates and convert to x/y (simplified)
                    elif 'latitude' in attributes and 'longitude' in attributes:
                        # Very simple conversion - just for demonstration
                        position_data = {
                            'x': (attributes.get('longitude', 0) * 100),
                            'y': (attributes.get('latitude', 0) * 100),
                            'z': 0
                        }

                    # Skip if no position data found
                    if not position_data:
                        continue

                    # Extract device ID from entity
                    device_id = state.get('entity_id').replace('device_tracker.', '').replace('sensor.', '')

                    # Try to get a numeric position
                    try:
                        position = {
                            'x': float(position_data.get('x', 0)),
                            'y': float(position_data.get('y', 0)),
                            'z': float(position_data.get('z', 0))
                        }

                        positions.append({
                            'device_id': device_id,
                            'position': position,
                            'entity_id': state.get('entity_id')  # Keep original entity ID for reference
                        })

                        # Debug logging
                        logger.debug(f"Found position entity: {state.get('entity_id')} at {position}")

                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert position data to float for {state.get('entity_id')}")
                        continue

            logger.info(f"Found {len(positions)} position entities")
            return positions

        except Exception as e:
            logger.error(f"Failed to get position data from Home Assistant: {str(e)}")
            return []

    def process_bluetooth_data(self, data: Dict) -> Dict:
        """Process and transform bluetooth data."""
        # This is a placeholder for data transformation logic
        # Implement based on specific requirements
        return data

    def get_sensors(self, domain: str = None, device_class: str = None) -> List[Dict]:
        """Get sensors matching domain and/or device_class."""
        try:
            url = f"{self.base_url}/api/states"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            all_states = response.json()
            matching_sensors = []

            for state in all_states:
                entity_id = state.get('entity_id', '')
                attributes = state.get('attributes', {})

                # Filter by domain
                if domain and not entity_id.startswith(f"{domain}."):
                    continue

                # Filter by device class
                if device_class and attributes.get('device_class') != device_class:
                    continue

                matching_sensors.append({
                    'entity_id': entity_id,
                    'state': state.get('state'),
                    'attributes': attributes
                })

            return matching_sensors

        except Exception as e:
            logger.error(f"Failed to get sensors from Home Assistant: {str(e)}")
            return []