# syntax=docker/dockerfile:1

# Use build arguments
ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base-python:3.9-alpine3.16
FROM ${BUILD_FROM}

# Set build version argument
ARG BUILD_VERSION=1.0.15

# Install system dependencies
RUN \
    apk add --no-cache \
        python3 \
        py3-pip \
        nginx \
        mariadb-connector-c \
        mariadb-client \
        build-base \
        python3-dev \
        mariadb-dev \
        linux-headers \
        bluez \
        bluez-libs \
        openblas \
        lapack

# Set working directory
WORKDIR /opt/blueprint_generator

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir wheel setuptools && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Configure nginx
COPY nginx.conf /etc/nginx/http.d/default.conf

# Create log directory
RUN mkdir -p /var/log/blueprint_generator

# Make scripts executable
RUN chmod +x run.py docker-entrypoint.sh && \
    mv docker-entrypoint.sh /usr/local/bin/

# Expose port
EXPOSE 8000

# Set up entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Labels
LABEL \
    io.hass.name="3D Blueprint Generator" \
    io.hass.description="Generate 3D home blueprints from Bluetooth sensor data" \
    io.hass.type="addon" \
    io.hass.version="${BUILD_VERSION}" \
    maintainer="brendan3394@gmail.com"
