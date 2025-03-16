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
        # Add BLAS/LAPACK dependencies
        openblas-dev \
        lapack-dev \
        gfortran \
        musl-dev \
        # Add additional build dependencies
        cmake \
        g++ \
        make \
        # Add numpy/scipy dependencies
        openblas \
        libstdc++

# Set environment variables for numpy/scipy build
ENV OPENBLAS=/usr/lib/
ENV LAPACK=/usr/lib/
ENV BLAS=/usr/lib/
ENV CFLAGS="-I/usr/include/openblas"
ENV LDFLAGS="-L/usr/lib/openblas -lopenblas"
ENV NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=0

# Copy local files
COPY . /opt/blueprint_generator

# Set working directory
WORKDIR /opt/blueprint_generator

# Install Python dependencies in specific order
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir wheel setuptools && \
    pip3 install --no-cache-dir numpy==1.21.2 && \
    pip3 install --no-cache-dir scipy==1.7.1 && \
    pip3 install --no-cache-dir -r requirements.txt

# Add nginx configuration
COPY nginx.conf /etc/nginx/http.d/default.conf

# Create required directories
RUN mkdir -p /var/log/blueprint_generator

# Make scripts executable
RUN chmod a+x /opt/blueprint_generator/run.py
RUN chmod a+x /opt/blueprint_generator/docker-entrypoint.sh

# Expose the port
EXPOSE 8000

# Copy and set up entry point
COPY docker-entrypoint.sh /
RUN chmod a+x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]

# Labels
LABEL \
    io.hass.name="3D Blueprint Generator" \
    io.hass.description="Generate 3D home blueprints from Bluetooth sensor data" \
    io.hass.type="addon" \
    io.hass.version="${BUILD_VERSION}" \
    maintainer="brendan3394@gmail.com"
