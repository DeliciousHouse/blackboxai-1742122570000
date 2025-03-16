FROM python:3.9-slim

# Install build dependencies and nginx
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libmariadb-dev \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install requirements
RUN pip install --upgrade pip wheel setuptools
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . /opt/blueprint_generator/
WORKDIR /opt/blueprint_generator

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Create log directories
RUN mkdir -p /var/log/blueprint_generator /var/log/nginx

# Make scripts executable and move entrypoint
RUN chmod +x /opt/blueprint_generator/run.py /opt/blueprint_generator/docker-entrypoint.sh
RUN cp /opt/blueprint_generator/docker-entrypoint.sh /usr/local/bin/

# Expose port
EXPOSE 8000

# Set up entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Labels for Home Assistant
LABEL \
    io.hass.name="3D Blueprint Generator" \
    io.hass.description="Generate 3D home blueprints from Bluetooth sensor data" \
    io.hass.type="addon" \
    io.hass.version="${BUILD_VERSION}" \
    maintainer="brendan3394@gmail.com"