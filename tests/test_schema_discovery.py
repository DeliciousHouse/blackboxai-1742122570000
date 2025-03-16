import pytest
import json
from server.schema_discovery import SchemaDiscovery
from server.db import execute_write_query

@pytest.fixture
def schema_discovery():
    """Create a SchemaDiscovery instance for testing."""
    return SchemaDiscovery()

@pytest.fixture
def test_table():
    """Create a temporary test table for schema discovery testing."""
    # Create test table
    create_table_query = """
    CREATE TEMPORARY TABLE test_bluetooth_readings (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        timestamp DATETIME NOT NULL,
        sensor_id VARCHAR(50) NOT NULL,
        rssi INT NOT NULL,
        device_id VARCHAR(50) NOT NULL,
        sensor_location JSON NOT NULL
    )
    """
    execute_write_query(create_table_query)
    yield "test_bluetooth_readings"

def test_load_schema_config(schema_discovery):
    """Test loading schema configuration."""
    config = schema_discovery._load_schema_config()
    assert isinstance(config, dict)
    assert 'sensor_table' in config
    assert 'column_overrides' in config

def test_discover_schema(schema_discovery, test_table):
    """Test schema discovery for test table."""
    schema = schema_discovery.discover_schema(test_table)
    
    # Verify schema structure
    assert isinstance(schema, dict)
    assert len(schema) == 6  # Number of columns in test table
    
    # Check specific columns
    assert 'timestamp' in schema
    assert 'sensor_id' in schema
    assert 'rssi' in schema
    assert 'device_id' in schema
    assert 'sensor_location' in schema
    
    # Verify column details
    assert schema['timestamp']['type'].lower().startswith('datetime')
    assert schema['sensor_id']['type'].lower().startswith('varchar')
    assert schema['rssi']['type'].lower().startswith('int')
    assert schema['device_id']['type'].lower().startswith('varchar')
    assert schema['sensor_location']['type'].lower().startswith('json')

def test_validate_schema(schema_discovery, test_table):
    """Test schema validation."""
    schema = schema_discovery.discover_schema(test_table)
    assert schema_discovery.validate_schema(schema) is True

def test_validate_schema_invalid():
    """Test schema validation with invalid schema."""
    schema_discovery = SchemaDiscovery()
    invalid_schema = {
        'timestamp': {'type': 'INVALID_TYPE'},
        'sensor_id': {'type': 'VARCHAR(50)'}
    }
    assert schema_discovery.validate_schema(invalid_schema) is False

def test_apply_schema_overrides(schema_discovery, test_table):
    """Test applying schema overrides."""
    # Discover initial schema
    schema_discovery.discover_schema(test_table)
    
    # Apply overrides
    overridden_schema = schema_discovery.apply_schema_overrides()
    
    # Verify overrides from config are applied
    config = schema_discovery._load_schema_config()
    overrides = config.get('column_overrides', {})
    
    for column, override in overrides.items():
        if column in overridden_schema:
            assert overridden_schema[column]['type'] == override

def test_get_schema_sql(schema_discovery, test_table):
    """Test generating SQL CREATE TABLE statement."""
    # Discover schema
    schema_discovery.discover_schema(test_table)
    
    # Get SQL
    sql = schema_discovery.get_schema_sql()
    
    # Verify SQL structure
    assert sql.startswith('CREATE TABLE')
    assert 'timestamp DATETIME' in sql
    assert 'sensor_id VARCHAR' in sql
    assert 'rssi INT' in sql
    assert 'device_id VARCHAR' in sql
    assert 'sensor_location JSON' in sql
    assert 'PRIMARY KEY' in sql

def test_schema_discovery_nonexistent_table(schema_discovery):
    """Test schema discovery with nonexistent table."""
    with pytest.raises(Exception):
        schema_discovery.discover_schema('nonexistent_table')

def test_schema_discovery_invalid_config_path():
    """Test schema discovery with invalid config path."""
    with pytest.raises(FileNotFoundError):
        SchemaDiscovery('nonexistent/config.json')

def test_schema_discovery_invalid_config_content(tmp_path):
    """Test schema discovery with invalid config content."""
    # Create temporary invalid config file
    config_path = tmp_path / "invalid_config.json"
    config_path.write_text("invalid json content")
    
    with pytest.raises(json.JSONDecodeError):
        SchemaDiscovery(str(config_path))

def test_schema_discovery_empty_table(schema_discovery):
    """Test schema discovery with empty table."""
    # Create empty test table
    create_table_query = """
    CREATE TEMPORARY TABLE empty_test_table (
        id INT PRIMARY KEY
    )
    """
    execute_write_query(create_table_query)
    
    schema = schema_discovery.discover_schema('empty_test_table')
    assert isinstance(schema, dict)
    assert len(schema) == 1
    assert 'id' in schema

def test_schema_discovery_all_column_types(schema_discovery):
    """Test schema discovery with various column types."""
    # Create test table with various column types
    create_table_query = """
    CREATE TEMPORARY TABLE test_types_table (
        id INT PRIMARY KEY,
        varchar_col VARCHAR(100),
        text_col TEXT,
        date_col DATE,
        datetime_col DATETIME,
        bool_col BOOLEAN,
        decimal_col DECIMAL(10,2),
        float_col FLOAT,
        json_col JSON
    )
    """
    execute_write_query(create_table_query)
    
    schema = schema_discovery.discover_schema('test_types_table')
    assert len(schema) == 9
    assert schema['varchar_col']['type'].lower().startswith('varchar')
    assert schema['text_col']['type'].lower().startswith('text')
    assert schema['date_col']['type'].lower().startswith('date')
    assert schema['datetime_col']['type'].lower().startswith('datetime')
    assert schema['bool_col']['type'].lower().startswith('tinyint')  # MySQL represents BOOLEAN as TINYINT
    assert schema['decimal_col']['type'].lower().startswith('decimal')
    assert schema['float_col']['type'].lower().startswith('float')
    assert schema['json_col']['type'].lower().startswith('json')
