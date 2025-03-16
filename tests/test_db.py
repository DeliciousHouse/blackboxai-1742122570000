import pytest
import pymysql
from server.db import (
    load_config,
    get_db_connection,
    execute_query,
    execute_write_query,
    test_connection
)

@pytest.fixture
def test_config():
    return {
        'host': 'localhost',
        'port': 3306,
        'user': 'home_assistant',
        'password': 'home_assistant_pass',
        'database': 'home_sensors'
    }

def test_load_config():
    """Test loading database configuration."""
    config = load_config()
    assert isinstance(config, dict)
    assert 'host' in config
    assert 'port' in config
    assert 'user' in config
    assert 'password' in config
    assert 'database' in config

def test_db_connection():
    """Test database connection establishment."""
    with get_db_connection() as conn:
        assert conn is not None
        assert isinstance(conn, pymysql.Connection)

def test_execute_query():
    """Test executing a select query."""
    result = execute_query("SELECT 1 as test")
    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]['test'] == 1

def test_execute_write_query():
    """Test executing an insert query."""
    # Create a temporary test table
    create_table_query = """
    CREATE TEMPORARY TABLE test_table (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(50)
    )
    """
    execute_write_query(create_table_query)
    
    # Insert test data
    insert_query = "INSERT INTO test_table (name) VALUES (%s)"
    affected_rows = execute_write_query(insert_query, ('test_name',))
    
    assert affected_rows == 1
    
    # Verify insertion
    result = execute_query("SELECT name FROM test_table WHERE id = 1")
    assert result[0]['name'] == 'test_name'

def test_connection_test():
    """Test the connection test function."""
    assert test_connection() is True

def test_invalid_query():
    """Test handling of invalid SQL query."""
    with pytest.raises(pymysql.Error):
        execute_query("SELECT * FROM nonexistent_table")
