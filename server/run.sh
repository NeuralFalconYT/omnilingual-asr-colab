#!/bin/bash
# A helper script to run the translations API inside Docker container
# This ensures the correct working directory and environment
# Works for both local development and Hugging Face Spaces
#
# Environment variables:
# - USE_CHUNKING: Set to "false" to disable audio chunking (default: "true")

# Change to the server directory
cd $HOME/app/server

# Set up models directory - prefer /data for HF Spaces, fallback to local
if [ -d "/data" ] && [ -w "/data" ]; then
    echo "Using /data directory for persistent storage (HF Spaces)"
    export MODELS_DIR="/data/models"
    export FAIRSEQ2_CACHE_DIR="/data/models"
    mkdir -p "$MODELS_DIR"
    mkdir -p "$FAIRSEQ2_CACHE_DIR"
    chmod 755 "$MODELS_DIR" 2>/dev/null || true
    chmod 755 "$FAIRSEQ2_CACHE_DIR" 2>/dev/null || true
else
    echo "Using local models directory"
    export MODELS_DIR="$HOME/app/models"
    export FAIRSEQ2_CACHE_DIR="$HOME/app/models"
    mkdir -p "$MODELS_DIR"
    mkdir -p "$FAIRSEQ2_CACHE_DIR"
    chmod 755 "$MODELS_DIR"
    chmod 755 "$FAIRSEQ2_CACHE_DIR"
fi

# Debug: Check directory permissions
echo "Models directory: $MODELS_DIR"
echo "FAIRSEQ2 cache directory: $FAIRSEQ2_CACHE_DIR"
echo "Current user: $(whoami)"
echo "Models directory exists: $([ -d "$MODELS_DIR" ] && echo "yes" || echo "no")"
echo "Models directory writable: $([ -w "$MODELS_DIR" ] && echo "yes" || echo "no")"
echo "FAIRSEQ2 cache directory exists: $([ -d "$FAIRSEQ2_CACHE_DIR" ] && echo "yes" || echo "no")"
echo "FAIRSEQ2 cache directory writable: $([ -w "$FAIRSEQ2_CACHE_DIR" ] && echo "yes" || echo "no")"
if [ -d "$MODELS_DIR" ]; then
    echo "Models directory permissions: $(ls -la "$MODELS_DIR" 2>/dev/null || echo "cannot list")"
fi

# Export environment variables so Python scripts can use them
export MODELS_DIR
export FAIRSEQ2_CACHE_DIR

# Download models on startup (will be cached for subsequent runs)
echo "Ensuring MMS models are available in $MODELS_DIR..."
bash ./download_models.sh

# Add current directory to PYTHONPATH to make modules importable
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "Updated PYTHONPATH: $PYTHONPATH"

# Determine port - use PORT env var for HF Spaces, default to 7860
PORT=${PORT:-7860}
echo "Starting server on port $PORT"

# Start the Flask server with single worker to avoid multiple model loading
# Large ML models should use single worker to prevent OOM issues
# Increased timeout for long-running ML inference tasks
gunicorn --worker-tmp-dir /dev/shm server:app --access-logfile /dev/null --log-file /dev/stderr -b 0.0.0.0:$PORT \
    --worker-class gthread --workers 1 --threads 20 \
    --worker-connections 1000 --backlog 2048 --keep-alive 60 --timeout 600
