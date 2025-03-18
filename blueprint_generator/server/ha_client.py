import requests
import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class HomeAssistantClient:
    """Client for interacting with Home Assistant API."""

    def __init__(self, base_url: str, token: str):
        """Initialize with Home Assistant connection details."""
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }

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

class HomeAssistantClient:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }

    def get_bluetooth_devices(self):
        # ...existing code...
        pass

    def get_bermuda_positions(self):
        # ...existing code...
        pass

    def get_private_ble_devices(self):
        # ...existing code...
        pass

    def process_bluetooth_data(self, data):
        # ...existing code...
        pass
