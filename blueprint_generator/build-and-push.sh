#!/bin/bash
# build-and-push.sh

set -e # Exit on error

# Set version from config.yaml
VERSION=$(grep 'version:' config.yaml | sed 's/.*"\([0-9.]*\)".*/\1/')
echo "Building version $VERSION"

# Build the image
docker build -t blueprint-generator:$VERSION .

# Tag for GitHub
docker tag blueprint-generator:$VERSION ghcr.io/delicioushouse/blueprint-generator-amd64:$VERSION
docker tag blueprint-generator:$VERSION ghcr.io/delicioushouse/blueprint-generator-amd64:latest

# Login to GitHub (you'll need to enter your PAT)
if [ -z "$GITHUB_TOKEN" ]; then
  echo "Please enter your GitHub Personal Access Token:"
  read -s GITHUB_TOKEN
fi
echo "$GITHUB_TOKEN" | docker login ghcr.io -u DeliciousHouse --password-stdin

# Push images
docker push ghcr.io/delicioushouse/blueprint-generator-amd64:$VERSION
docker push ghcr.io/delicioushouse/blueprint-generator-amd64:latest

# Update repository.json in root directory
cd ..
if [ -f repository.json ]; then
    echo "Updating repository.json..."
    # Check repository structure and update accordingly
    if grep -q '"addons"' repository.json; then
        jq --arg ver "$VERSION" '(.addons[] | select(.slug == "blueprint_generator")).version = $ver' repository.json > repository.json.new
        mv repository.json.new repository.json
        echo "Updated repository.json with version $VERSION (addons structure)"
    elif grep -q '"blueprints"' repository.json; then
        jq --arg ver "$VERSION" '(.blueprints[] | select(.slug == "blueprint_generator")).version = $ver' repository.json > repository.json.new
        mv repository.json.new repository.json
        echo "Updated repository.json with version $VERSION (blueprints structure)"
    else
        echo "Warning: Could not determine repository.json structure. Manual update required."
    fi
else
    echo "Warning: repository.json not found in root directory."
fi

# Commit and push changes to GitHub
git add blueprint_generator/config.yaml repository.json blueprint_generator/Dockerfile blueprint_generator/build-and-push.sh
git commit -m "Update Blueprint Generator to version $VERSION"
git push origin main

echo "Successfully built, pushed and updated version $VERSION"