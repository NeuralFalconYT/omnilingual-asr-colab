import torch
import torchaudio
import numpy as np
import logging
import tempfile
import os
import threading
from typing import List, Tuple, Dict, Optional, Any
import silero_vad
import soundfile as sf
import librosa

logger = logging.getLogger(__name__)

TARGET_CHUNK_DURATION = 30.0
MIN_CHUNK_DURATION = 5.0
SAMPLE_RATE = 16000


class AudioChunker:
    """
    Handles audio chunking with different strategies:
    - 'none': Single chunk (no chunking)
    - 'vad': VAD-based intelligent chunking
    - 'static': Fixed-duration time-based chunking
    """

    _instance = None
    _instance_lock = threading.Lock()
    vad_model: Optional[Any]

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                # Check again after acquiring lock as the value could have been set
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    # Only load VAD model here since this only runs once
                    cls._instance.vad_model = cls.load_vad_model()
        return cls._instance

    @staticmethod
    def load_vad_model():
        """Load silero VAD model with error handling."""
        try:
            logger.info("Loading Silero VAD model...")
            vad_model = silero_vad.load_silero_vad()
            logger.info("✓ VAD model loaded successfully")
            return vad_model
        except Exception as e:
            logger.error(f"Failed to load VAD model: {e}")
            logger.warning("VAD chunking will fall back to time-based chunking")
            return None

    @torch.inference_mode()
    def chunk_audio(self, audio_tensor: torch.Tensor, sample_rate: int = SAMPLE_RATE, mode: str = "vad", chunk_duration: float = 30.0) -> List[Dict]:
        """
        Chunk audio tensor using specified strategy.

        Args:
            audio_tensor: Audio tensor (1D waveform)
            sample_rate: Sample rate of the audio tensor
            mode: Chunking mode - 'none', 'vad', or 'static'
            chunk_duration: Target duration for static chunking (seconds)

        Returns:
            List of chunk info dicts with uniform format:
            - start_time: Start time in seconds
            - end_time: End time in seconds
            - duration: Duration in seconds
            - audio_data: Audio tensor for this chunk
            - sample_rate: Sample rate
            - chunk_index: Index of this chunk
        """
        logger.info(f"Chunking audio tensor: {audio_tensor.shape} at {sample_rate}Hz (mode: {mode})")

        try:
            # Assert tensor is already 1D (should be preprocessed by MediaTranscriptionProcessor)
            assert len(audio_tensor.shape) == 1, f"Expected 1D audio tensor, got shape {audio_tensor.shape}"

            # Assert sample rate is already 16kHz (should be preprocessed by MediaTranscriptionProcessor)
            assert sample_rate == SAMPLE_RATE, f"Expected {SAMPLE_RATE}Hz sample rate, got {sample_rate}Hz"

            # Route to appropriate chunking strategy
            if mode == "none":
                return self._create_single_chunk(audio_tensor, sample_rate)
            elif mode == "vad":
                if self.vad_model is not None:
                    return self._chunk_with_vad(audio_tensor)
                else:
                    logger.warning("VAD model not available, falling back to static chunking")
                    return self._chunk_static(audio_tensor, chunk_duration)
            elif mode == "static":
                return self._chunk_static(audio_tensor, chunk_duration)
            else:
                raise ValueError(f"Unknown chunking mode: {mode}")

        except Exception as e:
            logger.error(f"Error chunking audio tensor: {e}")
            # Ultimate fallback to single chunk
            return self._create_single_chunk(audio_tensor, sample_rate)

    def _create_single_chunk(self, waveform: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> List[Dict]:
        """Create a single chunk containing the entire audio."""
        duration = len(waveform) / sample_rate

        return [{
            "start_time": 0.0,
            "end_time": duration,
            "duration": duration,
            "audio_data": waveform,
            "sample_rate": sample_rate,
            "chunk_index": 0,
        }]

    def _chunk_static(self, waveform: torch.Tensor, chunk_duration: float) -> List[Dict]:
        """Create fixed-duration chunks."""
        chunks = []
        total_samples = len(waveform)
        target_samples = int(chunk_duration * SAMPLE_RATE)

        start_sample = 0
        chunk_idx = 0

        while start_sample < total_samples:
            end_sample = min(start_sample + target_samples, total_samples)
            chunk_audio = waveform[start_sample:end_sample]
            duration = len(chunk_audio) / SAMPLE_RATE

            # Only add chunk if it meets minimum duration
            if duration >= MIN_CHUNK_DURATION:
                chunks.append({
                    "start_time": start_sample / SAMPLE_RATE,
                    "end_time": end_sample / SAMPLE_RATE,
                    "duration": duration,
                    "audio_data": chunk_audio,
                    "sample_rate": SAMPLE_RATE,
                    "chunk_index": chunk_idx,
                })
                chunk_idx += 1

            start_sample = end_sample

        logger.info(f"Created {len(chunks)} static chunks of ~{chunk_duration}s each")
        return chunks

    def _chunk_fallback(self, audio_path: str) -> List[Dict]:
        """Ultimate fallback - create single chunk using librosa (for file-based legacy method)."""
        try:
            logger.warning("Using librosa fallback for chunking")
            data, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            waveform = torch.from_numpy(data)
            return self._create_single_chunk(waveform, SAMPLE_RATE)
        except Exception as e:
            logger.error(f"All chunking methods failed: {e}")
            return []
    def _chunk_with_vad(self, waveform: torch.Tensor) -> List[Dict]:
        """Chunk audio using VAD for speech detection with uniform return format."""
        try:
            # VAD model expects tensor on CPU
            vad_waveform = waveform.cpu() if waveform.is_cuda else waveform

            # Get speech timestamps using VAD
            speech_timestamps = silero_vad.get_speech_timestamps(
                vad_waveform,
                self.vad_model,
                sampling_rate=SAMPLE_RATE,
                min_speech_duration_ms=500,  # Minimum speech segment
                min_silence_duration_ms=300,  # Minimum silence to split
                window_size_samples=1536,
                speech_pad_ms=100,  # Padding around speech
            )

            logger.info(f"Found {len(speech_timestamps)} speech segments")

            # Create chunks based on speech segments and target duration
            # Pass original waveform (with device preserved) to chunk creation
            chunks = self._create_chunks_from_speech_segments(
                waveform, speech_timestamps
            )

            logger.info(f"Created {len(chunks)} audio chunks using VAD")
            return chunks

        except Exception as e:
            logger.error(f"VAD chunking failed: {e}")
            return self._chunk_static(waveform, TARGET_CHUNK_DURATION)
    def _create_chunks_from_speech_segments(
        self, waveform: torch.Tensor, speech_segments: List[Dict]
    ) -> List[Dict]:
        """Create chunks that respect speech boundaries and target duration with uniform format."""
        if not speech_segments:
            logger.warning(
                "No speech segments found, falling back to static chunking"
            )
            return self._chunk_static(waveform, TARGET_CHUNK_DURATION)

        chunks = []
        current_chunk_start = 0
        target_samples = int(TARGET_CHUNK_DURATION * SAMPLE_RATE)
        total_samples = len(waveform)
        chunk_idx = 0

        while current_chunk_start < total_samples:
            # Calculate target end for this chunk
            target_chunk_end = current_chunk_start + target_samples

            # If this would be the last chunk or close to it, just take the rest
            if target_chunk_end >= total_samples or (
                total_samples - target_chunk_end
            ) < (target_samples * 0.3):
                chunk_end = total_samples
            else:
                # Find the best place to end this chunk using VAD, but ensure continuous coverage
                chunk_end = self._find_best_chunk_end_continuous(
                    speech_segments,
                    current_chunk_start,
                    target_chunk_end,
                    total_samples,
                )

            # Create chunk with uniform format
            chunk_audio = waveform[current_chunk_start:chunk_end]
            duration = len(chunk_audio) / SAMPLE_RATE

            chunks.append({
                "start_time": current_chunk_start / SAMPLE_RATE,
                "end_time": chunk_end / SAMPLE_RATE,
                "duration": duration,
                "audio_data": chunk_audio,
                "sample_rate": SAMPLE_RATE,
                "chunk_index": chunk_idx,
            })

            logger.info(
                f"Created chunk {chunk_idx + 1}: {current_chunk_start/SAMPLE_RATE:.2f}s - {chunk_end/SAMPLE_RATE:.2f}s ({duration:.2f}s)"
            )
            chunk_idx += 1

            # Move to next chunk - IMPORTANT: start exactly where this chunk ended
            current_chunk_start = chunk_end

        # Verify total coverage
        total_audio_duration = len(waveform) / SAMPLE_RATE
        total_chunks_duration = sum(chunk["duration"] for chunk in chunks)
        logger.info(
            f"Audio chunking complete: {len(chunks)} chunks covering {total_chunks_duration:.2f}s of {total_audio_duration:.2f}s total audio"
        )

        if (
            abs(total_chunks_duration - total_audio_duration) > 0.01
        ):  # Allow 10ms tolerance
            logger.error(
                f"Duration mismatch: chunks={total_chunks_duration:.2f}s, original={total_audio_duration:.2f}s"
            )
        else:
            logger.info("✓ Perfect audio coverage achieved")

        return chunks

    def _find_best_chunk_end_continuous(
        self,
        speech_segments: List[Dict],
        chunk_start: int,
        target_end: int,
        total_samples: int,
    ) -> int:
        """Find the best place to end a chunk while ensuring continuous coverage."""

        # Don't go beyond the audio
        target_end = min(target_end, total_samples)

        # Look for a good break point within a reasonable window around target
        search_window = int(SAMPLE_RATE * 3)  # 3 second window
        search_start = max(chunk_start, target_end - search_window)
        search_end = min(total_samples, target_end + search_window)

        best_end = target_end
        best_score = 0

        # Look for speech segment boundaries within the search window
        for segment in speech_segments:
            segment_start = segment["start"]
            segment_end = segment["end"]

            # Check if segment end is in our search window
            if search_start <= segment_end <= search_end:
                # Score based on how close to target and if it's a good break point
                distance_score = 1.0 - abs(segment_end - target_end) / search_window

                # Prefer segment ends (natural pauses)
                boundary_score = 1.0

                total_score = distance_score * boundary_score

                if total_score > best_score:
                    best_score = total_score
                    best_end = segment_end

        # Ensure we don't go beyond audio bounds
        best_end = min(int(best_end), total_samples)

        # Ensure we make progress (don't end before we started)
        if best_end <= chunk_start:
            best_end = min(target_end, total_samples)

        return best_end

    def _find_best_chunk_end(
        self,
        speech_segments: List[Dict],
        start_idx: int,
        chunk_start: int,
        target_end: int,
    ) -> int:
        """Find the best place to end a chunk (at silence, near target duration)."""

        best_end = target_end

        # Look for speech segments that could provide good break points
        for i in range(start_idx, len(speech_segments)):
            segment = speech_segments[i]
            segment_start = segment["start"]
            segment_end = segment["end"]

            # If segment starts after our target end, use the gap before it
            if segment_start > target_end:
                best_end = min(target_end, segment_start)
                break

            # If segment ends near our target, use the end of the segment
            if abs(segment_end - target_end) < SAMPLE_RATE * 5:  # Within 5 seconds
                best_end = segment_end
                break

            # If segment extends way past target, look for a good break point
            if segment_end > target_end + SAMPLE_RATE * 10:  # 10+ seconds past
                # Try to find a silence gap within the segment or use target
                best_end = target_end
                break

        return int(best_end)

    def save_chunk_to_file(self, chunk: Dict, output_path: str) -> str:
        """Save a chunk to a temporary audio file."""
        try:
            # Convert tensor to numpy if needed
            audio_data = chunk["audio_data"]
            if isinstance(audio_data, torch.Tensor):
                # Move to CPU first if on GPU, then convert to numpy
                audio_data = audio_data.cpu().numpy()

            # Save to file
            sf.write(output_path, audio_data, chunk["sample_rate"])
            return output_path

        except Exception as e:
            logger.error(f"Failed to save chunk to file: {e}")
            raise
