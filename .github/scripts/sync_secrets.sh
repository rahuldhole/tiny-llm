#!/bin/bash

# Ensure GitHub CLI is installed
if ! command -v gh &> /dev/null
then
    echo "GitHub CLI (gh) not found. Please install it first."
    exit 1
fi

# Load .env file
if [ -f .env ]; then
    echo "Loading .env file..."
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ $key =~ ^#.* ]] || [[ -z $key ]] && continue
        
        echo "Setting secret: $key"
        echo "$value" | gh secret set "$key"
    done < .env
    echo "Done syncing secrets."
else
    echo ".env file not found. Create it from .env.example"
    exit 1
fi
