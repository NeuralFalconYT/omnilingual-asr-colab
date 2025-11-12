"""
Media Transcription Processor

Pipeline-focused transcription processor that maintains state through processing stages
while exposing intermediate results for flexibility and ensuring proper resource cleanup.
"""

import base64
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from audio_transcription import transcribe_full_audio_with_chunking
from convert_media_to_wav import convert_media_to_wav_from_bytes
from inference.audio_reading_tools import wav_to_bytes
from transcription_status import transcription_status


class MediaTranscriptionProcessor:
    """
    Pipeline-focused transcription processor that maintains state through processing stages
    while exposing intermediate results for flexibility and ensuring proper resource cleanup.
    """

    # Maximum duration (in seconds) before a transcription is considered stuck
    # MAX_TRANSCRIPTION_DURATION = 120  # 2 minutes

    # For long meetings (1 hour max)
    # MAX_TRANSCRIPTION_DURATION = 3600  

    # Or disable timeout entirely
    MAX_TRANSCRIPTION_DURATION = float("inf")


    def __init__(self, media_bytes: bytes, filename: str, language_with_script: str = None):
        """Initialize processor with media data and metadata."""
        # Core input data
        self.media_bytes = media_bytes
        self.original_filename = filename
        self.language_with_script = language_with_script

        # Processing state - lazy loaded
        self._temp_wav_path: Optional[str] = None
        self._audio_tensor: Optional[torch.Tensor] = None
        self._audio_numpy: Optional[np.ndarray] = None
        self._sample_rate: int = 16000
        self._duration: Optional[float] = None
        self._chunks: Optional[List] = None
        self._transcription_results: Optional[Dict] = None
        self._error: Optional[str] = None

        # Resource tracking for cleanup
        self._temp_files: List[str] = []
        self._cleanup_performed = False

        # Transcription status management
        self._status_initialized = False

    def start_transcription(self):
        """Initialize transcription status tracking."""
        if not self._status_initialized:
            transcription_status.start_transcription("transcribe", self.original_filename)
            self._status_initialized = True

    def update_progress(self, progress: float):
        """Update transcription progress."""
        transcription_status.update_progress(progress)

    @staticmethod
    def is_server_busy() -> bool:
        """
        Check if the server is currently busy with another transcription.

        This method includes timeout handling - if a transcription has been
        running too long, it will be force-finished.
        """
        status = MediaTranscriptionProcessor.get_server_status()
        return status.get("is_busy", False)

    @staticmethod
    def get_server_status() -> dict:
        """
        Get current server transcription status with timeout handling.

        If a transcription has been running longer than MAX_TRANSCRIPTION_DURATION,
        it will be force-finished to prevent the server from being stuck indefinitely.
        """
        status = transcription_status.get_status()

        # Check if transcription has been running too long
        if (status.get("is_busy", False) and
            status.get("duration_seconds", 0) > MediaTranscriptionProcessor.MAX_TRANSCRIPTION_DURATION):

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Force-finishing stuck transcription after {status.get('duration_seconds', 0):.1f}s "
                f"(max: {MediaTranscriptionProcessor.MAX_TRANSCRIPTION_DURATION}s). "
                f"Operation: {status.get('current_operation')}, "
                f"File: {status.get('current_filename')}"
            )

            # Force finish the transcription
            transcription_status.finish_transcription()

            # Get updated status
            status = transcription_status.get_status()
            status["force_finished"] = True
            status["reason"] = f"Transcription exceeded maximum duration of {MediaTranscriptionProcessor.MAX_TRANSCRIPTION_DURATION}s"

        return status

    def convert_media(self) -> 'MediaTranscriptionProcessor':
        """
        Stage 1: Convert media to standardized audio format.

        Returns:
            Self for method chaining
        """
        if self._temp_wav_path is not None:
            # Already converted
            return self

        logger = logging.getLogger(__name__)
        logger.info(f"Converting media file: {self.original_filename}")

        # Update progress if status is initialized
        if self._status_initialized:
            self.update_progress(0.1)

        try:
            # Convert media bytes to WAV and tensor
            temp_wav_path, audio_tensor = convert_media_to_wav_from_bytes(
                self.media_bytes, self.original_filename
            )

            # Store results and track temp file
            self._temp_wav_path = temp_wav_path
            self._audio_tensor = audio_tensor
            self._temp_files.append(temp_wav_path)

            # Calculate duration from tensor
            if audio_tensor is not None:
                self._duration = len(audio_tensor) / self._sample_rate

            logger.info(f"Media conversion completed: {self.original_filename} -> {self._duration:.2f}s")

            # Update progress if status is initialized
            if self._status_initialized:
                self.update_progress(0.2)

        except Exception as e:
            logger.error(f"Media conversion failed for {self.original_filename}: {str(e)}")

            # Provide user-friendly error message based on the error type
            if "ffmpeg returned error code" in str(e).lower():
                error_msg = (
                    f"Audio/video conversion failed for '{self.original_filename}'. "
                    f"The file may have an unsupported audio codec or be corrupted. "
                    f"Please try converting the file to a standard format (MP3, WAV, MP4) before uploading. "
                    f"For best results, use files with common codecs: "
                    f"Audio - AAC, MP3, PCM, FLAC; Video - H.264/AAC (MP4), standard codecs. "
                    f"Avoid proprietary, DRM-protected, or very old codec variants."
                )
            else:
                error_msg = f"Failed to process media file '{self.original_filename}'"

            error_msg += f"\nTechnical Details: {str(e)}"

            # Store the error for later retrieval
            self._error = error_msg
            raise RuntimeError(error_msg)

        return self

    def get_wav_path(self) -> str:
        """Get the temporary WAV file path (converts media if needed)."""
        if self._temp_wav_path is None:
            self.convert_media()
        return self._temp_wav_path

    def get_audio_tensor(self) -> torch.Tensor:
        """Get standardized audio tensor (converts media if needed)."""
        if self._audio_tensor is None:
            self.convert_media()
        return self._audio_tensor

    def get_audio_numpy(self) -> np.ndarray:
        """Get audio as numpy array (converted from tensor if needed)."""
        if self._audio_numpy is None:
            tensor = self.get_audio_tensor()
            if tensor is not None:
                # Convert to numpy, handling different tensor types
                if hasattr(tensor, 'cpu'):
                    self._audio_numpy = tensor.cpu().numpy()
                else:
                    self._audio_numpy = tensor.numpy()
            else:
                self._audio_numpy = np.array([])
        return self._audio_numpy

    @property
    def duration(self) -> float:
        """Get audio duration in seconds."""
        if self._duration is None:
            self.convert_media()
        return self._duration or 0.0

    @property
    def sample_rate(self) -> int:
        """Get audio sample rate."""
        return self._sample_rate

    def transcribe_full_pipeline(self) -> 'MediaTranscriptionProcessor':
        """
        Stage 2: Run the complete transcription pipeline with chunking.

        Returns:
            Self for method chaining
        """
        if self._transcription_results is not None:
            # Already transcribed
            return self

        logger = logging.getLogger(__name__)

        # Ensure media is converted
        wav_path = self.get_wav_path()

        logger.info(f"Starting transcription pipeline for: {self.original_filename}")

        # Get the preprocessed audio tensor instead of just the WAV path
        audio_tensor = self.get_audio_tensor()

        # Run the full transcription with chunking using the tensor
        self._transcription_results = transcribe_full_audio_with_chunking(
            audio_tensor=audio_tensor,
            sample_rate=self._sample_rate,
            language_with_script=self.language_with_script,
        )

        logger.info(f"Transcription completed: {self._transcription_results.get('num_chunks', 0)} chunks")

        # Update progress if status is initialized
        if self._status_initialized:
            self.update_progress(0.9)

        return self

    def get_results(self, include_preprocessed_audio: bool = False) -> Dict:
        """
        Get final transcription results (runs transcription if needed).

        Args:
            include_preprocessed_audio: Whether to include base64-encoded preprocessed WAV data

        Returns:
            Complete transcription results dictionary, optionally with preprocessed audio
        """
        if self._transcription_results is None:
            self.transcribe_full_pipeline()

        results = self._transcription_results or {}

        # Add preprocessed audio data if requested
        if include_preprocessed_audio and self._audio_tensor is not None:
            try:
                # Convert the preprocessed tensor to WAV bytes
                audio_tensor_cpu = self._audio_tensor.cpu() if self._audio_tensor.is_cuda else self._audio_tensor
                wav_bytes = wav_to_bytes(audio_tensor_cpu, sample_rate=self._sample_rate, format="wav")

                # Encode as base64
                audio_data_b64 = base64.b64encode(wav_bytes.tobytes()).decode('utf-8')

                results["preprocessed_audio"] = {
                    "data": audio_data_b64,
                    "format": "wav",
                    "sample_rate": self._sample_rate,
                    "duration": self.duration,
                    "size_bytes": len(wav_bytes)
                }

                logging.getLogger(__name__).info(f"Added preprocessed audio data: {len(wav_bytes)} bytes")

            except Exception as e:
                logging.getLogger(__name__).warning(f"Failed to include preprocessed audio data: {e}")

        return results

    def cleanup(self):
        """Clean up all temporary files and resources."""
        if self._cleanup_performed:
            return

        logger = logging.getLogger(__name__)

        # Clean up temporary files
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file}: {e}")

        # Finish transcription status - always call to ensure we don't get stuck
        # It's better to be safe than risk leaving the server in a busy state
        transcription_status.finish_transcription()
        self._status_initialized = False

        # Clear references to help garbage collection
        self._audio_tensor = None
        self._audio_numpy = None
        self._transcription_results = None
        self._chunks = None
        self._temp_files.clear()

        self._cleanup_performed = True
        logger.debug(f"Cleanup completed for: {self.original_filename}")

    def __enter__(self) -> 'MediaTranscriptionProcessor':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.cleanup()

    def __del__(self):
        """Destructor - final cleanup attempt."""
        if not self._cleanup_performed:

            self.cleanup()
