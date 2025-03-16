ARG BUILD_FROM
FROM $BUILD_FROM

# Set shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install system dependencies
RUN \
    apk add --no-cache \
        python3 \
        py3-pip \
        mariadb-connector-c \
        mariadb-client \
        build-base \
        python3-dev \
        mariadb-dev \
        linux-headers \
        bluez \
        bluez-libs \
        git \
        nginx \
        openblas \
        openblas-dev \
        lapack-dev \
        gfortran \
        musl-dev \
        cmake \
        g++ \
        make

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OPENBLAS=/usr/lib/
ENV LAPACK=/usr/lib/
ENV BLAS=/usr/lib/

# Create app directory
WORKDIR /opt/blueprint_generator

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir wheel setuptools && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Configure nginx
COPY nginx.conf /etc/nginx/http.d/default.conf

# Create required directories and set permissions
RUN mkdir -p /var/log/blueprint_generator && \
    chmod +x /opt/blueprint_generator/run.py && \
    chmod +x /opt/blueprint_generator/docker-entrypoint.sh

# Expose port
EXPOSE 8000

# Set up entrypoint
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

# Labels
LABEL \
    io.hass.name="3D Blueprint Generator" \
    io.hass.description="Generate 3D home blueprints from Bluetooth sensor data" \
    io.hass.type="addon" \
    io.hass.version="${BUILD_VERSION}" \
    maintainer="brendan3394@gmail.com"
