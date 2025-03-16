import json
import logging.config
from typing import Dict, Optional
from db import get_db_connection, get_db_cursor

# Setup logging
logging.config.fileConfig('config/logging.conf')
logger = logging.getLogger('server.schema_discovery')

class SchemaDiscovery:
    def __init__(self, config_path: str = 'config/config.json'):
        """
        Initialize SchemaDiscovery with configuration path.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config_path = config_path
        self.schema_config = self._load_schema_config()
        self.discovered_schema = None

    def _load_schema_config(self) -> Dict:
        """
        Load schema configuration from config file.
        
        Returns:
            dict: Schema configuration
        """
        try:
            with open(self.config_path) as f:
                config = json.load(f)
            return config.get('db_schema', {})
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in configuration file: {self.config_path}")
            raise

    def discover_schema(self, table_name: Optional[str] = None) -> Dict:
        """
        Discover the schema of the specified table or use the configured table name.
        
        Args:
            table_name (str, optional): Name of the table to discover schema for
        
        Returns:
            dict: Dictionary containing column names and their types
        """
        if table_name is None:
            table_name = self.schema_config.get('sensor_table', 'bluetooth_readings')

        try:
            with get_db_connection() as connection:
                with get_db_cursor(connection) as cursor:
                    # Get table schema
                    cursor.execute(f"SHOW COLUMNS FROM {table_name};")
                    columns = cursor.fetchall()
                    
                    # Convert to dictionary format
                    schema = {
                        column['Field']: {
                            'type': column['Type'],
                            'null': column['Null'],
                            'key': column['Key'],
                            'default': column['Default'],
                            'extra': column['Extra']
                        } for column in columns
                    }
                    
                    logger.info(f"Discovered schema for table {table_name}: {schema}")
                    self.discovered_schema = schema
                    return schema

        except Exception as e:
            logger.error(f"Error discovering schema for table {table_name}: {str(e)}")
            raise

    def validate_schema(self, schema: Dict) -> bool:
        """
        Validate discovered schema against required columns and types.
        
        Args:
            schema (dict): Discovered schema to validate
        
        Returns:
            bool: True if schema is valid, False otherwise
        """
        required_columns = {
            'timestamp': 'datetime',
            'sensor_id': 'varchar',
            'rssi': 'int',
            'device_id': 'varchar',
            'sensor_location': 'json'
        }

        # Check for required columns
        for column, expected_type in required_columns.items():
            if column not in schema:
                logger.error(f"Required column '{column}' not found in schema")
                return False
            
            actual_type = schema[column]['type'].lower()
            if not actual_type.startswith(expected_type.lower()):
                logger.error(f"Column '{column}' has incorrect type. Expected: {expected_type}, Got: {actual_type}")
                return False

        logger.info("Schema validation successful")
        return True

    def apply_schema_overrides(self) -> Dict:
        """
        Apply any schema overrides from the configuration.
        
        Returns:
            dict: Schema with overrides applied
        """
        if not self.discovered_schema:
            raise ValueError("No schema discovered yet. Call discover_schema() first.")

        overrides = self.schema_config.get('column_overrides', {})
        schema = self.discovered_schema.copy()

        for column, override in overrides.items():
            if column in schema:
                schema[column]['type'] = override
                logger.info(f"Applied override for column {column}: {override}")
            else:
                logger.warning(f"Override specified for non-existent column: {column}")

        return schema

    def get_schema_sql(self) -> str:
        """
        Generate SQL CREATE TABLE statement for the current schema.
        
        Returns:
            str: SQL statement for creating the table
        """
        if not self.discovered_schema:
            raise ValueError("No schema discovered yet. Call discover_schema() first.")

        table_name = self.schema_config.get('sensor_table', 'bluetooth_readings')
        sql_parts = [f"CREATE TABLE {table_name} ("]
        
        for column, details in self.discovered_schema.items():
            sql_parts.append(f"  {column} {details['type']}")
            
            if details['null'] == 'NO':
                sql_parts[-1] += " NOT NULL"
            
            if details['default'] is not None:
                sql_parts[-1] += f" DEFAULT {details['default']}"
            
            if details['extra']:
                sql_parts[-1] += f" {details['extra']}"
            
            sql_parts[-1] += ","

        # Add primary key if it exists
        primary_key = next(
            (col for col, details in self.discovered_schema.items() 
             if details['key'] == 'PRI'),
            None
        )
        if primary_key:
            sql_parts.append(f"  PRIMARY KEY ({primary_key})")
        else:
            sql_parts[-1] = sql_parts[-1].rstrip(',')

        sql_parts.append(");")
        return "\n".join(sql_parts)

def main():
    """
    Main function to test schema discovery functionality.
    """
    try:
        schema_discovery = SchemaDiscovery()
        
        # Discover schema
        schema = schema_discovery.discover_schema()
        logger.info("Schema discovery completed")

        # Validate schema
        if schema_discovery.validate_schema(schema):
            logger.info("Schema validation passed")
        else:
            logger.error("Schema validation failed")

        # Apply overrides
        schema_with_overrides = schema_discovery.apply_schema_overrides()
        logger.info("Schema overrides applied")

        # Generate SQL
        create_sql = schema_discovery.get_schema_sql()
        logger.info(f"Generated SQL:\n{create_sql}")

    except Exception as e:
        logger.error(f"Schema discovery process failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
