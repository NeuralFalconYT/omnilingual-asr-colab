"""
Pipeline-based MMS Model using the official MMS library.
This implementation uses Wav2Vec2LlamaInferencePipeline to avoid Seq2SeqBatch complexity.
"""

import logging
import os
import torch
from typing import List, Dict, Any, Optional
from omnilingual_asr.models.inference.pipeline import Wav2Vec2InferencePipeline
from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

from inference.audio_reading_tools import wav_to_bytes
from env_vars import MODEL_NAME

logger = logging.getLogger(__name__)


class MMSModel:
    """Pipeline-based MMS model wrapper using the official inference pipeline."""
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            logger.info("Creating new MMSModel singleton instance")
            cls._instance = super().__new__(cls)
        else:
            logger.info("Using existing MMSModel singleton instance")
        return cls._instance

    def __init__(self, model_card: str = None, device = None):
        """
        Initialize the MMS model with the official pipeline.

        Args:
            model_card: Model card to use (omniASR_LLM_1B, omniASR_LLM_300M, etc.)
                       If None, uses MODEL_NAME from environment variables
            device: Device to use (torch.device object, "cuda", "cpu", etc.)
        """
        # Only initialize once
        if self._initialized:
            return

        # Get model name from environment variable with default fallback
        self.model_card = model_card or MODEL_NAME
        self.device = device

        # Load the pipeline immediately during initialization
        self._load_pipeline()

        # Mark as initialized
        self._initialized = True

    def _load_pipeline(self):
        """Load the MMS pipeline during initialization."""
        logger.info(f"Loading MMS pipeline: {self.model_card}")
        logger.info(f"Target device: {self.device}")

        # Debug FAIRSEQ2_CACHE_DIR environment variable
        fairseq2_cache_dir = os.environ.get('FAIRSEQ2_CACHE_DIR')
        logger.info(f"DEBUG: FAIRSEQ2_CACHE_DIR = {fairseq2_cache_dir}")

        try:
            # Convert device to string if it's a torch.device object
            device_str = str(self.device) if hasattr(self.device, 'type') else str(self.device)
            self.pipeline = Wav2Vec2InferencePipeline(
                model_card=self.model_card,
                device=device_str
            )
            logger.info("✓ MMS pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MMS pipeline: {e}")
            raise

    def transcribe_audio(self, audio_tensor: torch.Tensor, batch_size: int = 1, language_with_scripts: List[str] = None) -> List[Dict[str, Any]]:
        """
        Transcribe audio tensor using the MMS pipeline.

        Args:
            audio_tensor: Audio tensor (1D waveform) to transcribe
            batch_size: Batch size for processing
            language_with_scripts: List of language_with_scripts codes for transcription (3-letter ISO codes with script)
                 If None, uses auto-detection

        Returns:
            List of transcription results
        """
        # Pipeline is already loaded during initialization, no need to check

        # Convert tensor to bytes for the pipeline
        logger.info(f"Converting tensor (shape: {audio_tensor.shape}) to bytes")
        # Move to CPU first if on GPU
        tensor_cpu = audio_tensor.cpu() if audio_tensor.is_cuda else audio_tensor
        # Convert to bytes using wav_to_bytes with 16kHz sample rate
        audio_bytes = wav_to_bytes(tensor_cpu, sample_rate=16000, format="wav")

        logger.info(f"Transcribing audio tensor with batch_size={batch_size}, language_with_scripts={language_with_scripts}")

        try:
            # Use the official pipeline transcribe method with a list containing the single audio bytes
            if language_with_scripts is not None:
                transcriptions = self.pipeline.transcribe([audio_bytes], batch_size=batch_size, lang=language_with_scripts)
            else:
                transcriptions = self.pipeline.transcribe([audio_bytes], batch_size=batch_size)

            logger.info(f"✓ Successfully transcribed audio tensor")
            return transcriptions

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    @classmethod
    def get_instance(cls, model_card: str = None, device = None):
        """
        Get the singleton instance of MMSModel.

        Args:
            model_card: Model card to use (omniASR_LLM_1B, omniASR_LLM_300M, etc.)
                       If None, uses MODEL_NAME from environment variables
            device: Device to use (torch.device object, "cuda", "cpu", etc.)

        Returns:
            MMSModel: The singleton instance
        """
        if cls._instance is None:
            cls._instance = cls(model_card=model_card, device=device)
        return cls._instance
