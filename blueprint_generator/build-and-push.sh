#!/bin/bash
# build-and-push.sh

set -e # Exit on error

# Load GitHub token
if [ -f ~/.github_token ]; then
  GITHUB_TOKEN=$(cat ~/.github_token)
else
  echo "Error: GitHub token file not found at ~/.github_token"
  exit 1
fi

# Get current version from config.yaml
CURRENT_VERSION=$(grep 'version:' config.yaml | sed 's/.*"\([0-9.]*\)".*/\1/')
echo "Current version: $CURRENT_VERSION"

# Increment version by 0.01
NEW_VERSION=$(awk -v ver="$CURRENT_VERSION" 'BEGIN { printf("%.2f", ver + 0.01) }')
echo "New version: $NEW_VERSION"

# Update version in config.yaml
sed -i "s/version: \"$CURRENT_VERSION\"/version: \"$NEW_VERSION\"/" config.yaml

# Update repository.json in root directory
cd ..
if [ -f repository.json ]; then
    echo "Updating repository.json..."
    # Update root version
    sed -i "s/\"version\": \"$CURRENT_VERSION\"/\"version\": \"$NEW_VERSION\"/" repository.json
    # Update addon version
    sed -i "s/\"version\": \"$CURRENT_VERSION\",/\"version\": \"$NEW_VERSION\",/" repository.json
else
    echo "Warning: repository.json not found in root directory."
fi

# Update build.yaml
cd blueprint_generator
if [ -f build.yaml ]; then
    echo "Updating build.yaml..."
    sed -i "s/org.opencontainers.image.version: \"$CURRENT_VERSION\"/org.opencontainers.image.version: \"$NEW_VERSION\"/" build.yaml
else
    echo "Warning: build.yaml not found."
fi

# Set version for building
VERSION=$NEW_VERSION
echo "Building version $VERSION"

# Build the image
docker build --platform linux/amd64 -t blueprint-generator:$VERSION .

# Tag for GitHub
docker tag blueprint-generator:$VERSION ghcr.io/delicioushouse/blueprint-generator-amd64:$VERSION
docker tag blueprint-generator:$VERSION ghcr.io/delicioushouse/blueprint-generator-amd64:latest

# Push to GitHub Container Registry
echo "Logging in to GitHub Container Registry..."
echo "$GITHUB_TOKEN" | docker login ghcr.io -u DeliciousHouse --password-stdin

# Push images
docker push ghcr.io/delicioushouse/blueprint-generator-amd64:$VERSION
docker push ghcr.io/delicioushouse/blueprint-generator-amd64:latest

# Add all changes in blueprint_generator directory
git config --local credential.helper store
git config --local credential.username DeliciousHouse
{
  echo "url=https://github.com"
  echo "username=DeliciousHouse"
  echo "password=${GITHUB_TOKEN}"
  echo ""
} | git credential approve

git add -A
git commit -m "Update Blueprint Generator to version $VERSION"
git push origin main

echo "Successfully built, pushed and updated version $VERSION"
