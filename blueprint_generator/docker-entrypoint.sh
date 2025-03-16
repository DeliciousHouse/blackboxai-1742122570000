#!/bin/sh
set -e

# Create log directory if it doesn't exist
mkdir -p /var/log/blueprint_generator

# Start nginx in the background
if [ -x "/usr/sbin/nginx" ]; then
    /usr/sbin/nginx
elif [ -x "/usr/local/sbin/nginx" ]; then
    /usr/local/sbin/nginx
else
    echo "Error: nginx executable not found"
    exit 1
fi

# Start the Flask application
cd /opt/blueprint_generator
exec python3 run.py
