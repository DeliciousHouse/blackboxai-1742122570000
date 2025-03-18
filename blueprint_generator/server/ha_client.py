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

    def get_bermuda_positions(self) -> List[Dict]:
        """Get device positions from Bermuda Trilateration."""
        try:
            # First, find all bermuda trilateration entities
            url = f"{self.base_url}/api/states"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            all_states = response.json()
            positions = []

            for state in all_states:
                entity_id = state.get('entity_id', '')
                # Check for bermuda position entities
                if entity_id.startswith('sensor.bermuda_'):
                    attributes = state.get('attributes', {})
                    if 'position' in attributes:
                        device_id = entity_id.replace('sensor.bermuda_', '')
                        position_data = attributes.get('position', {})

                        # Extract position data
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
            logger.error(f"Failed to get Bermuda positions from Home Assistant: {str(e)}")
            return []

    def get_private_ble_devices(self) -> List[Dict]:
        """Get devices from ESP32 BLE Monitor."""
        try:
            url = f"{self.base_url}/api/states"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            all_states = response.json()
            devices = []

            for state in all_states:
                entity_id = state.get('entity_id', '')
                if entity_id.startswith('sensor.') and 'ble_monitor' in entity_id and 'rssi' in entity_id:
                    # Extract the MAC address from entity ID or attributes
                    attributes = state.get('attributes', {})
                    mac = attributes.get('mac', '')

                    # If mac isn't in attributes, try to extract from entity_id
                    if not mac:
                        parts = entity_id.split('_')
                        if len(parts) >= 3:
                            mac_parts = parts[2:-1]  # Skip 'sensor', 'ble_monitor', and 'rssi'
                            mac = '_'.join(mac_parts)

                    if mac:
                        rssi = state.get('state')
                        try:
                            rssi = int(rssi)
                        except (ValueError, TypeError):
                            rssi = -100  # Default value if conversion fails

                        devices.append({
                            'mac': mac,
                            'rssi': rssi,
                            'entity_id': entity_id,
                            'friendly_name': attributes.get('friendly_name', '')
                        })

            return devices

        except Exception as e:
            logger.error(f"Failed to get ESP32 BLE Monitor devices from Home Assistant: {str(e)}")
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