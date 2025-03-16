import os
import json
import logging.config
import pymysql
from pymysql.cursors import DictCursor
from contextlib import contextmanager

# Setup logging
logging.config.fileConfig('config/logging.conf')
logger = logging.getLogger('server.db')

def load_config():
    """Load database configuration from config file."""
    try:
        config_path = 'config/config.json'
        with open(config_path) as f:
            config = json.load(f)
        return config['db']
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in configuration file: {config_path}")
        raise
    except KeyError:
        logger.error("Database configuration not found in config file")
        raise

@contextmanager
def get_db_connection():
    """
    Context manager for database connections.
    Ensures connections are properly closed after use.
    """
    config = load_config()
    connection = None
    try:
        connection = pymysql.connect(
            host=config['host'],
            port=config['port'],
            user=config['user'],
            password=config['password'],
            database=config['database'],
            cursorclass=DictCursor
        )
        logger.info("Database connection established successfully")
        yield connection
    except pymysql.Error as e:
        logger.error(f"Database connection error: {str(e)}")
        raise
    finally:
        if connection:
            connection.close()
            logger.debug("Database connection closed")

@contextmanager
def get_db_cursor(connection):
    """
    Context manager for database cursors.
    Ensures cursors are properly closed after use.
    """
    cursor = None
    try:
        cursor = connection.cursor()
        yield cursor
    finally:
        if cursor:
            cursor.close()
            logger.debug("Database cursor closed")

def execute_query(query, params=None):
    """
    Execute a single query and return all results.
    
    Args:
        query (str): SQL query to execute
        params (tuple, optional): Parameters to pass to the query
    
    Returns:
        list: Query results as a list of dictionaries
    """
    try:
        with get_db_connection() as connection:
            with get_db_cursor(connection) as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
                logger.debug(f"Query executed successfully: {query}")
                return results
    except pymysql.Error as e:
        logger.error(f"Query execution error: {str(e)}\nQuery: {query}\nParams: {params}")
        raise

def execute_write_query(query, params=None):
    """
    Execute a write query (INSERT, UPDATE, DELETE) with transaction support.
    
    Args:
        query (str): SQL query to execute
        params (tuple, optional): Parameters to pass to the query
    
    Returns:
        int: Number of affected rows
    """
    try:
        with get_db_connection() as connection:
            with get_db_cursor(connection) as cursor:
                cursor.execute(query, params)
                affected_rows = cursor.rowcount
                connection.commit()
                logger.debug(f"Write query executed successfully. Affected rows: {affected_rows}")
                return affected_rows
    except pymysql.Error as e:
        logger.error(f"Write query execution error: {str(e)}\nQuery: {query}\nParams: {params}")
        raise

def execute_many(query, params_list):
    """
    Execute the same query with different parameters multiple times.
    
    Args:
        query (str): SQL query to execute
        params_list (list): List of parameter tuples
    
    Returns:
        int: Number of affected rows
    """
    try:
        with get_db_connection() as connection:
            with get_db_cursor(connection) as cursor:
                affected_rows = cursor.executemany(query, params_list)
                connection.commit()
                logger.debug(f"Batch query executed successfully. Affected rows: {affected_rows}")
                return affected_rows
    except pymysql.Error as e:
        logger.error(f"Batch query execution error: {str(e)}\nQuery: {query}")
        raise

def test_connection():
    """
    Test the database connection and return status.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        with get_db_connection() as connection:
            with get_db_cursor(connection) as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return result is not None and result['1'] == 1
    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Test the database connection when module is run directly
    if test_connection():
        logger.info("Database connection test successful")
    else:
        logger.error("Database connection test failed")
