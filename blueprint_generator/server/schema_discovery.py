import json
import logging
from typing import Dict, List, Optional

from .db import execute_query, execute_write_query

logger = logging.getLogger(__name__)

class SchemaDiscovery:
    """Discover and validate database schema for blueprint generation."""
    
    REQUIRED_TABLES = {
        'bluetooth_readings': {
            'columns': {
                'id': 'BIGINT AUTO_INCREMENT PRIMARY KEY',
                'timestamp': 'DATETIME NOT NULL',
                'sensor_id': 'VARCHAR(50) NOT NULL',
                'rssi': 'INT NOT NULL',
                'device_id': 'VARCHAR(50) NOT NULL',
                'sensor_location': 'JSON NOT NULL'
            },
            'indices': [
                'INDEX idx_timestamp (timestamp)',
                'INDEX idx_sensor (sensor_id)',
                'INDEX idx_device (device_id)'
            ]
        },
        'manual_updates': {
            'columns': {
                'id': 'BIGINT AUTO_INCREMENT PRIMARY KEY',
                'update_type': 'VARCHAR(10) NOT NULL',
                'entity_id': 'VARCHAR(50) NOT NULL',
                'data': 'JSON NOT NULL',
                'timestamp': 'DATETIME DEFAULT CURRENT_TIMESTAMP'
            },
            'indices': [
                'UNIQUE KEY unique_entity (update_type, entity_id)',
                'INDEX idx_timestamp (timestamp)'
            ]
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize schema discovery with optional config path."""
        self.config_path = config_path

    def discover_schema(self) -> Dict:
        """Discover current database schema."""
        schema = {}
        
        # Get tables
        tables = self._get_tables()
        for table in tables:
            table_name = table['TABLE_NAME']
            schema[table_name] = {
                'columns': self._get_columns(table_name),
                'indices': self._get_indices(table_name)
            }
        
        return schema

    def validate_schema(self, schema: Dict) -> bool:
        """Validate discovered schema against required schema."""
        try:
            for table, requirements in self.REQUIRED_TABLES.items():
                # Check if table exists
                if table not in schema:
                    logger.error(f"Missing required table: {table}")
                    return False

                # Check columns
                current_columns = schema[table]['columns']
                for col, col_type in requirements['columns'].items():
                    if col not in current_columns:
                        logger.error(f"Missing column {col} in table {table}")
                        return False

                # Check indices
                current_indices = schema[table]['indices']
                for required_index in requirements['indices']:
                    if not any(required_index in index for index in current_indices):
                        logger.error(f"Missing index in table {table}: {required_index}")
                        return False

            return True

        except Exception as e:
            logger.error(f"Schema validation failed: {str(e)}")
            return False

    def create_schema(self) -> bool:
        """Create required schema if it doesn't exist."""
        try:
            for table, requirements in self.REQUIRED_TABLES.items():
                # Create table
                columns = [f"{col} {col_type}" for col, col_type in requirements['columns'].items()]
                indices = requirements['indices']
                
                create_query = f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    {', '.join(columns)},
                    {', '.join(indices)}
                )
                """
                
                execute_write_query(create_query)
                logger.info(f"Created or verified table: {table}")

            return True

        except Exception as e:
            logger.error(f"Schema creation failed: {str(e)}")
            return False

    def _get_tables(self) -> List[Dict]:
        """Get list of tables in the database."""
        query = """
        SELECT TABLE_NAME 
        FROM information_schema.TABLES 
        WHERE TABLE_SCHEMA = DATABASE()
        """
        return execute_query(query)

    def _get_columns(self, table: str) -> Dict:
        """Get columns for a specific table."""
        query = """
        SELECT COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE, COLUMN_DEFAULT, EXTRA
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s
        """
        columns = execute_query(query, (table,))
        return {col['COLUMN_NAME']: self._format_column_type(col) for col in columns}

    def _get_indices(self, table: str) -> List[str]:
        """Get indices for a specific table."""
        query = """
        SELECT INDEX_NAME, COLUMN_NAME, NON_UNIQUE
        FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s
        """
        indices = execute_query(query, (table,))
        return [self._format_index(index) for index in indices]

    def _format_column_type(self, column: Dict) -> str:
        """Format column type string."""
        col_type = column['COLUMN_TYPE']
        nullable = "NOT NULL" if column['IS_NULLABLE'] == 'NO' else "NULL"
        default = f"DEFAULT {column['COLUMN_DEFAULT']}" if column['COLUMN_DEFAULT'] else ""
        extra = column['EXTRA']
        return f"{col_type} {nullable} {default} {extra}".strip()

    def _format_index(self, index: Dict) -> str:
        """Format index definition string."""
        if index['INDEX_NAME'] == 'PRIMARY':
            return 'PRIMARY KEY'
        unique = "" if index['NON_UNIQUE'] else "UNIQUE"
        return f"{unique} INDEX {index['INDEX_NAME']} ({index['COLUMN_NAME']})"
