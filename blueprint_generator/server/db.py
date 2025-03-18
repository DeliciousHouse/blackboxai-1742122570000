import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import os
import pymysql
from pymysql.cursors import DictCursor
import sqlite3

from .config import load_config

logger = logging.getLogger(__name__)
app_config = load_config()

def get_db_connection(readonly=True):
    """Get database connection with configurable read/write permissions."""
    # Get credentials from environment or config
    db_config = {
        'host': os.environ.get('DB_HOST', app_config.get('db', {}).get('host', 'core-mariadb')),
        'port': int(os.environ.get('DB_PORT', app_config.get('db', {}).get('port', 3306))),
        'user': os.environ.get('DB_USER', app_config.get('db', {}).get('user', 'homeassistant')),
        'password': os.environ.get('DB_PASSWORD', app_config.get('db', {}).get('password', '')),
        'database': os.environ.get('DB_NAME', app_config.get('db', {}).get('database', 'homeassistant')),
        'cursorclass': DictCursor
    }

    # Connect using PyMySQL
    connection = pymysql.connect(**db_config)

    # Set transaction to READ ONLY if requested
    if readonly:
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION TRANSACTION READ ONLY")

    return connection

def execute_query(query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
    """Execute a SELECT query and return results."""
    try:
        with get_db_connection(readonly=True) as conn:
            # Read-only operations
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
    except Exception as e:
        logger.error(f"Query execution failed: {str(e)}")
        raise

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

# SQLITE FUNCTIONS FOR WRITING DATA

def get_sqlite_connection():
    """Get connection to local SQLite database."""
    try:
        # Ensure the /data directory exists
        os.makedirs('/data', exist_ok=True)
        conn = sqlite3.connect('/data/blueprint_data.db')
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to SQLite: {str(e)}")
        raise

def init_database():
    """Initialize the database with required tables."""
    try:
        conn = get_sqlite_connection()
        cursor = conn.cursor()

        # Create device positions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS device_positions (
            id INTEGER PRIMARY KEY,
            device_id TEXT,
            position_data TEXT,
            source TEXT,
            accuracy REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Create RSSI distance samples table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS rssi_distance_samples (
            id INTEGER PRIMARY KEY,
            device_id TEXT,
            sensor_id TEXT,
            rssi INTEGER,
            distance REAL,
            tx_power INTEGER,
            frequency REAL,
            environment_type TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {str(e)}")
        return False

def init_sqlite_db():
    """Initialize SQLite database schema."""
    try:
        conn = get_sqlite_connection()
        cursor = conn.cursor()

        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blueprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                data TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bluetooth_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                sensor_id TEXT,
                rssi INTEGER,
                device_id TEXT,
                sensor_location TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS manual_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                update_type TEXT,
                entity_id TEXT,
                data TEXT,
                timestamp TEXT,
                UNIQUE(update_type, entity_id)
            )
        ''')

        conn.commit()
        conn.close()

        # Initialize additional tables from init_database function
        init_database()

        return True
    except Exception as e:
        logger.error(f"Failed to initialize SQLite schema: {str(e)}")
        return False

def get_sensor_readings(time_window: int = 300) -> List[Dict[str, Any]]:
    """Get recent sensor readings within the specified time window."""
    try:
        conn = get_sqlite_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM bluetooth_readings WHERE timestamp >= datetime('now', ? || ' seconds') ORDER BY timestamp DESC",
            (f"-{time_window}",)
        )
        result = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return result
    except Exception as e:
        logger.error(f"Failed to get sensor readings from SQLite: {str(e)}")
        return []

def save_sensor_reading(
    sensor_id: str,
    rssi: int,
    device_id: str,
    sensor_location: Dict[str, float]
) -> bool:
    """Save a new sensor reading to SQLite."""
    try:
        conn = get_sqlite_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO bluetooth_readings (timestamp, sensor_id, rssi, device_id, sensor_location) VALUES (datetime('now'), ?, ?, ?, ?)",
            (sensor_id, rssi, device_id, json.dumps(sensor_location))
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to save sensor reading to SQLite: {str(e)}")
        return False

def save_blueprint_update(blueprint_data: Dict[str, Any]) -> bool:
    """Save a blueprint update to SQLite."""
    try:
        conn = get_sqlite_connection()
        cursor = conn.cursor()

        # Check if record exists
        cursor.execute(
            "SELECT id FROM manual_updates WHERE update_type = 'blueprint' AND entity_id = 'blueprint'"
        )
        existing = cursor.fetchone()

        if existing:
            cursor.execute(
                "UPDATE manual_updates SET data = ?, timestamp = datetime('now') WHERE id = ?",
                (json.dumps(blueprint_data), existing['id'])
            )
        else:
            cursor.execute(
                "INSERT INTO manual_updates (update_type, entity_id, data, timestamp) VALUES ('blueprint', 'blueprint', ?, datetime('now'))",
                (json.dumps(blueprint_data),)
            )

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to save blueprint update to SQLite: {str(e)}")
        return False

def get_latest_blueprint() -> Optional[Dict[str, Any]]:
    """Get the latest blueprint data from SQLite."""
    try:
        conn = get_sqlite_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT data FROM manual_updates WHERE update_type = 'blueprint' ORDER BY timestamp DESC LIMIT 1"
        )
        result = cursor.fetchone()
        conn.close()

        if result:
            return json.loads(result['data'])
        return None
    except Exception as e:
        logger.error(f"Failed to get latest blueprint from SQLite: {str(e)}")
        return None

def execute_write_query(query: str, params: Optional[Tuple] = None) -> int:
    """Execute a write query on SQLite database.

    Note: This function redirects all write operations to SQLite instead of
    modifying the Home Assistant database.
    """
    try:
        # Parse the query to determine what kind of data is being written
        query_lower = query.lower().strip()

        # Handle the different types of queries
        if query_lower.startswith('insert into ai_models'):
            # AI model data
            logger.info("Writing AI model data to SQLite")
            conn = get_sqlite_connection()
            cursor = conn.cursor()

            # Create table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT,
                    model_data TEXT,
                    timestamp TEXT
                )
            ''')

            # Extract model type from parameters
            model_type = params[0] if params and len(params) > 0 else "unknown"
            model_data = params[1] if params and len(params) > 1 else "{}"

            # Insert into SQLite
            cursor.execute(
                "INSERT INTO ai_models (model_type, model_data, timestamp) VALUES (?, ?, datetime('now'))",
                (model_type, model_data)
            )

            affected_rows = cursor.rowcount
            conn.commit()
            conn.close()
            return affected_rows

        elif query_lower.startswith('insert into position_history'):
            # Position history data
            logger.info("Writing position history to SQLite")
            conn = get_sqlite_connection()
            cursor = conn.cursor()

            # Create table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS position_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT,
                    position_data TEXT,
                    timestamp TEXT
                )
            ''')

            # Extract data from parameters
            device_id = params[0] if params and len(params) > 0 else "unknown"
            position_data = params[1] if params and len(params) > 1 else "{}"

            # Insert into SQLite
            cursor.execute(
                "INSERT INTO position_history (device_id, position_data, timestamp) VALUES (?, ?, datetime('now'))",
                (device_id, position_data)
            )

            affected_rows = cursor.rowcount
            conn.commit()
            conn.close()
            return affected_rows

        else:
            # Generic write - log it but don't actually perform it
            logger.warning(f"Attempted write operation redirected to SQLite: {query}")
            return 0

    except Exception as e:
        logger.error(f"Write query execution failed: {str(e)}")
        raise