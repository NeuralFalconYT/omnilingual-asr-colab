import argparse
import logging
import os
import sys
import tempfile


# Set up basic logging first
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Using direct imports
from env_vars import API_LOG_LEVEL

import torch
from flask import Flask, jsonify, send_file, send_from_directory
from flask_cors import CORS
from inference.audio_chunker import AudioChunker
from inference.audio_sentence_alignment import AudioAlignment
from inference.mms_model_pipeline import MMSModel
from transcriptions_blueprint import transcriptions_blueprint

# Configure logging with imported level
logging.basicConfig(stream=sys.stdout, level=API_LOG_LEVEL)


_model_loaded = False
_model_loading = False


def load_model():
    """Load the MMS model on startup - only called once"""
    global _model_loaded, _model_loading

    # If model is already loaded, return it
    if _model_loaded:
        logger.info("Model already loaded, skipping load")
        return

    # If model is currently being loaded by another thread/process, wait
    if _model_loading:
        logger.info("Model is currently being loaded, waiting...")
        return

    try:
        _model_loading = True
        logger.info("Loading MMS model...")

        # Initialize other components
        AudioChunker()
        AudioAlignment()

        # Initialize the new pipeline-based MMS model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        MMSModel(device=device)

        logger.info("✓ MMS pipeline loaded successfully during server startup")

        _model_loaded = True
        logger.info(f"Models successfully loaded")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        _model_loaded = False
        return None
    finally:
        _model_loading = False


app = Flask(__name__)
app.register_blueprint(transcriptions_blueprint)
cors = CORS(
    app,
    resources={
        r"/*": {
            "origins": "*",
            "allow_headers": "*",
            "expose_headers": "*",
            "supports_credentials": True,
        }
    },
)

logger = logging.getLogger(__name__)
gunicorn_logger = logging.getLogger("gunicorn.error")
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

# Load model on startup - only once during app initialization
logger.info("Initializing application and loading model...")
if not _model_loaded:
    load_model()
else:
    logger.info("Model already loaded, skipping initialization")


# Frontend static file serving
@app.route("/")
def serve_frontend():
    """Serve the frontend index.html"""
    frontend_dist = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "frontend", "dist"
    )
    return send_file(os.path.join(frontend_dist, "index.html"))


@app.route("/assets/<path:filename>")
def serve_assets(filename):
    """Serve frontend static assets"""
    frontend_dist = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "frontend", "dist"
    )
    return send_from_directory(os.path.join(frontend_dist, "assets"), filename)


# Catch-all route for SPA routing - must be last
@app.route("/<path:path>")
def serve_spa(path):
    """Serve index.html for any unmatched routes (SPA routing)"""
    # If the path starts with 'api/', return 404 for API routes
    if path.startswith("api/"):
        return jsonify({"error": "API endpoint not found"}), 404

    # For all other paths, serve the frontend index.html
    frontend_dist = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "frontend", "dist"
    )
    return send_file(os.path.join(frontend_dist, "index.html"))


@app.errorhandler(404)
def handle_404(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def handle_500(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument("--debug", default=True, type=bool)
    args = parser.parse_args()

    logger.info(f"Starting Translations API on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
