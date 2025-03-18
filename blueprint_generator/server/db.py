import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import os
import mysql.connector
import pymysql
from pymysql.cursors import DictCursor

from .config import load_config

logger = logging.getLogger(__name__)
config = load_config()

def get_db_connection():
    """Get database connection with read-only permissions."""
    # Get credentials from environment or options
    config = {
        'host': os.environ.get('DB_HOST', 'core-mariadb'),
        'port': int(os.environ.get('DB_PORT', 3306)),
        'user': os.environ.get('DB_USER', 'homeassistant'),
        'password': os.environ.get('DB_PASSWORD', ''),
        'database': os.environ.get('DB_NAME', 'homeassistant')
    }

    # Connect with read-only mode if not creating tables
    connection = mysql.connector.connect(**config)

    # Execute SET SESSION TRANSACTION READ ONLY for read operations
    cursor = connection.cursor()
    cursor.execute("SET SESSION TRANSACTION READ ONLY")
    cursor.close()

    return connection

def test_connection() -> bool:
    """Test the database connection."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                return True
    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}")
        return False

def execute_query(query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
    """Execute a SELECT query and return results."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
    except Exception as e:
        logger.error(f"Query execution failed: {str(e)}")
        raise

def execute_write_query(query: str, params: Optional[Tuple] = None) -> int:
    """Execute an INSERT/UPDATE/DELETE query and return affected rows."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                affected_rows = cursor.execute(query, params)
                conn.commit()
                return affected_rows
    except Exception as e:
        logger.error(f"Write query execution failed: {str(e)}")
        raise

def get_sensor_readings(time_window: int = 300) -> List[Dict[str, Any]]:
    """Get recent sensor readings within the specified time window."""
    query = """
    SELECT * FROM bluetooth_readings
    WHERE timestamp >= NOW() - INTERVAL %s SECOND
    ORDER BY timestamp DESC
    """
    return execute_query(query, (time_window,))

def save_sensor_reading(
    sensor_id: str,
    rssi: int,
    device_id: str,
    sensor_location: Dict[str, float]
) -> bool:
    """Save a new sensor reading."""
    query = """
    INSERT INTO bluetooth_readings
    (timestamp, sensor_id, rssi, device_id, sensor_location)
    VALUES (NOW(), %s, %s, %s, %s)
    """
    try:
        execute_write_query(
            query,
            (sensor_id, rssi, device_id, json.dumps(sensor_location))
        )
        return True
    except Exception as e:
        logger.error(f"Failed to save sensor reading: {str(e)}")
        return False

def save_blueprint_update(blueprint_data: Dict[str, Any]) -> bool:
    """Save a blueprint update."""
    query = """
    INSERT INTO manual_updates
    (update_type, entity_id, data, timestamp)
    VALUES ('blueprint', %s, %s, NOW())
    ON DUPLICATE KEY UPDATE
    data = VALUES(data),
    timestamp = VALUES(timestamp)
    """
    try:
        execute_write_query(
            query,
            ('blueprint', json.dumps(blueprint_data))
        )
        return True
    except Exception as e:
        logger.error(f"Failed to save blueprint update: {str(e)}")
        return False

def get_latest_blueprint() -> Optional[Dict[str, Any]]:
    """Get the latest blueprint data."""
    query = """
    SELECT data FROM manual_updates
    WHERE update_type = 'blueprint'
    ORDER BY timestamp DESC
    LIMIT 1
    """
    try:
        results = execute_query(query)
        if results:
            return json.loads(results[0]['data'])
        return None
    except Exception as e:
        logger.error(f"Failed to get latest blueprint: {str(e)}")
        return None
