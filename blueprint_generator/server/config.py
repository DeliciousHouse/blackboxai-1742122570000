#!/usr/bin/env python3

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from file."""
    try:
        config_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'config.json'))
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        return {}