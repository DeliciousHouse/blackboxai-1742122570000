#!/usr/bin/env bash
set -e

# Create log directory if it doesn't exist
mkdir -p /var/log/blueprint_generator

# Start nginx
nginx

# Start the Flask application
python3 /opt/blueprint_generator/run.py
