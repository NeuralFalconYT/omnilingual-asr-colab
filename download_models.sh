#!/bin/bash

# Script to download model files on container startup
#
# Note: FAIRSEQ2_ASSET_DIR is set in run.sh to use the HuggingFace persistent cache:
# - /data/models in HF Spaces (persistent across restarts)
# - $HOME/app/models for local development
# This allows models to be cached and reused between container restarts

# Use the MODELS_DIR environment variable set by run.sh
# Falls back to default if not set
MODELS_DIR="${MODELS_DIR:-/home/user/app/models}"

echo "Checking and downloading MMS models to $MODELS_DIR..."

# Create models directory if it doesn't exist with proper permissions
mkdir -p "$MODELS_DIR"
chmod 755 "$MODELS_DIR"

# Change to models directory
cd "$MODELS_DIR" || {
    echo "✗ Failed to change to models directory: $MODELS_DIR"
    echo "Current user: $(whoami)"
    echo "Directory permissions: $(ls -la $(dirname "$MODELS_DIR"))"
    exit 1
}

# Check if we can write to the directory
if [ ! -w "$MODELS_DIR" ]; then
    echo "✗ No write permission to models directory: $MODELS_DIR"
    echo "Current user: $(whoami)"
    echo "Directory permissions: $(ls -la "$MODELS_DIR")"
    exit 1
fi

# Function to download file if it doesn't exist
download_if_missing() {
    local url="$1"
    local filename="$2"

    if [ -f "$filename" ]; then
        echo "✓ $filename already exists, skipping download"
    else
        echo "Downloading $filename..."
        if wget -O "$filename" "$url"; then
            echo "✓ Successfully downloaded $filename"
        else
            echo "✗ Failed to download $filename"
            exit 1
        fi
    fi
}

# Download CTC alignment model files
echo "Downloading CTC alignment model files..."
download_if_missing "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/dictionary.txt" "ctc_alignment_mling_uroman_model_dict.txt"
download_if_missing "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt" "ctc_alignment_mling_uroman_model.pt"

echo "All model files are ready!"
