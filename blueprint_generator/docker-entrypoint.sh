#!/bin/sh
set -e

# Create log directory if it doesn't exist
mkdir -p /var/log/blueprint_generator

# Start nginx in the background
nginx

# Start the Flask application
cd /opt/blueprint_generator
exec python3 run.py
