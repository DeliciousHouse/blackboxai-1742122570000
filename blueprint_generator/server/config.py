#!/usr/bin/env python3

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from options.json (HA provided) and config.json (app default)."""
    config = {}

    # Try to load app-specific config first
    app_config_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '..', 'config', 'config.json'))
    if app_config_path.exists():
        try:
            with open(app_config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded app config from {app_config_path}")
        except Exception as e:
            logger.error(f"Failed to load app config: {str(e)}")

    # Try to load Home Assistant options (these override app config)
    ha_options_path = Path('/data/options.json')
    if ha_options_path.exists():
        try:
            with open(ha_options_path, 'r') as f:
                ha_options = json.load(f)
            logger.info(f"Loaded HA options from {ha_options_path}")

            # Update DB configuration from HA options
            if 'db' not in config:
                config['db'] = {}

            # Map HA option names to app config
            config['db']['host'] = ha_options.get('database_host', config['db'].get('host'))
            config['db']['port'] = ha_options.get('database_port', config['db'].get('port'))
            config['db']['database'] = ha_options.get('database_name', config['db'].get('database'))
            config['db']['user'] = ha_options.get('database_username', config['db'].get('user'))
            config['db']['password'] = ha_options.get('database_password', config['db'].get('password', ''))

            # Map other HA options
            if 'processing_params' not in config:
                config['processing_params'] = {}

            config['processing_params']['rssi_threshold'] = ha_options.get('min_rssi',
                config['processing_params'].get('rssi_threshold'))
            config['processing_params']['update_interval'] = ha_options.get('update_interval',
                config['processing_params'].get('update_interval'))
        except Exception as e:
            logger.error(f"Failed to load HA options: {str(e)}")

    # Debug output
    logger.debug(f"Final config: {json.dumps(config)}")
    return config