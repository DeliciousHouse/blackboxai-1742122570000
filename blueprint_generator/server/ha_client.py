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
                if entity_id.startswith('bluetooth_tracker.'):
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
        """Get all BLE devices with distance or RSSI data."""
        try:
            url = f"{self.base_url}/api/states"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            all_states = response.json()
            devices = []

            for state in all_states:
                entity_id = state.get('entity_id', '')
                attributes = state.get('attributes', {})

                # Much more flexible matching for BLE entities
                if (entity_id.startswith('sensor.') and
                    ('_ble' in entity_id.lower() or entity_id.lower().endswith('_ble')) and
                    ('distance' in entity_id.lower() or 'rssi' in entity_id.lower())):

                    # Try to extract device identifier
                    parts = entity_id.split('_')
                    if len(parts) >= 2:
                        # Use the first part as device type (phone, watch, etc)
                        device_type = parts[1]

                        # Use the device name or friendly name
                        device_name = attributes.get('friendly_name', entity_id)

                        # Get distance/RSSI value
                        value = state.get('state')
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            value = -100  # Default if conversion fails

                        devices.append({
                            'mac': device_type,  # Use device type as identifier
                            'rssi': value if 'rssi' in entity_id.lower() else -59 + (value * -2),  # Convert distance to RSSI if needed
                            'entity_id': entity_id,
                            'friendly_name': device_name,
                            'distance': value if 'distance' in entity_id.lower() else None
                        })

            return devices

        except Exception as e:
            logger.error(f"Failed to get BLE devices from Home Assistant: {str(e)}")
            return []

    def get_bermuda_positions(self) -> List[Dict]:
        """Get device positions from Bermuda Trilateration."""
        try:
            url = f"{self.base_url}/api/states"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            all_states = response.json()
            positions = []

            for state in all_states:
                entity_id = state.get('entity_id', '')
                attributes = state.get('attributes', {})

                # More flexible matching for Bermuda entities
                if ('bermuda' in entity_id.lower() or
                    'position' in attributes or
                    ('x' in attributes and 'y' in attributes)):

                    # Extract position data from attributes
                    position_data = attributes.get('position', {})
                    if not position_data and 'x' in attributes and 'y' in attributes:
                        # If position is not nested, get from top level attributes
                        position_data = {
                            'x': attributes.get('x', 0),
                            'y': attributes.get('y', 0),
                            'z': attributes.get('z', 0)
                        }

                    # Extract device ID from entity
                    device_id = entity_id.replace('device_tracker.', '').replace('sensor.', '')

                    # Create position object
                    position = {
                        'x': float(position_data.get('x', 0)),
                        'y': float(position_data.get('y', 0)),
                        'z': float(position_data.get('z', 0))
                    }

                    positions.append({
                        'device_id': device_id,
                        'position': position
                    })

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