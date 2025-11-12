import logging
import os

log_level = os.environ.get("API_LOG_LEVEL", "INFO")  # @see logging._nameToLevel
API_LOG_LEVEL = logging._nameToLevel.get(log_level)

# MMS Model Configuration
MODEL_NAME = os.environ.get("MODEL_NAME", "omniASR_LLM_7B")  # Model name for pipeline

# Audio Processing Configuration
USE_CHUNKING = os.environ.get("USE_CHUNKING", "true").lower() == "true"  # Whether to use audio chunking
