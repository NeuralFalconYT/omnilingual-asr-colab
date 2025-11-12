from __future__ import annotations

# Standard library imports
import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple

# Third-party imports
import librosa
import numpy as np
import soundfile as sf
import torch
import uroman

# fairseq2 imports
from inference.align_utils import get_uroman_tokens
from inference.audio_chunker import AudioChunker

from inference.audio_reading_tools import wav_to_bytes

# Import AudioAlignment and its config classes
from inference.audio_sentence_alignment import AudioAlignment
from inference.mms_model_pipeline import MMSModel
from inference.text_normalization import text_normalize
from transcription_status import transcription_status
from env_vars import USE_CHUNKING

# Constants
SAMPLE_RATE = 16000

logger = logging.getLogger(__name__)


def transcribe_single_chunk(audio_tensor: torch.Tensor, sample_rate: int = 16000, language_with_script: str = None):
    """
    Basic transcription pipeline for a single audio chunk using MMS model pipeline.
    This is the lowest-level transcription function that handles individual audio segments.

    Args:
        audio_tensor (torch.Tensor): Audio tensor (1D waveform)
        sample_rate (int): Sample rate of the audio tensor
        language_with_script (str): language_with_script for transcription (3-letter ISO codes like "eng", "spa") with script

    Returns:
        str: Transcribed text
    """

    logger.info("Starting complete audio transcription pipeline...")

    try:
        logger.info("Using pipeline transcription...")
        # Use the singleton model instance
        model = MMSModel.get_instance()

        # Transcribe using pipeline - convert tensor to list format
        lang_list = [language_with_script] if language_with_script else None
        results = model.transcribe_audio(audio_tensor, batch_size=1, language_with_scripts=lang_list)
        result = results[0] if results else {}

        # Convert pipeline result to expected format
        if isinstance(result, dict) and 'text' in result:
            transcription_text = result['text']
        elif isinstance(result, str):
            transcription_text = result
        else:
            transcription_text = str(result)

        if not transcription_text.strip():
            logger.warning("Pipeline returned empty transcription")
            return ""

        logger.info(f"✓ Pipeline transcription successful: '{transcription_text}'")

        # Return the transcription text
        return transcription_text

    except Exception as e:
        logger.error(f"Error in transcription pipeline: {str(e)}", exc_info=True)
        raise


def perform_forced_alignment(
    audio_tensor: torch.Tensor,
    transcription_tokens: List[str],
    device,
    sample_rate: int = 16000,
) -> List[Dict]:
    """
    Perform forced alignment using the AudioAlignment class from audio_sentence_alignment.py.
    Uses the provided audio tensor directly.

    Args:
        audio_tensor (torch.Tensor): Audio tensor (1D waveform)
        transcription_tokens (List[str]): List of tokens from transcription
        device: Device for computation
        sample_rate (int): Audio sample rate

    Returns:
        List[Dict]: List of segments with timestamps and text
    """

    try:
        logger.info(f"Starting forced alignment with audio tensor")
        logger.info(f"Audio shape: {audio_tensor.shape}, sample_rate: {sample_rate}")
        logger.info(f"Tokens to align: {transcription_tokens}")

        # Use the provided audio tensor directly
        # Convert to the format expected by AudioAlignment.get_one_row_alignments
        if hasattr(audio_tensor, "cpu"):
            # If it's a torch tensor, use it directly
            alignment_tensor = audio_tensor.float()
        else:
            # If it's numpy, convert to tensor
            alignment_tensor = torch.from_numpy(audio_tensor).float()

        # Ensure it's 1D (flatten if needed)
        if len(alignment_tensor.shape) > 1:
            alignment_tensor = alignment_tensor.flatten()

        # Convert audio tensor to bytes format expected by AudioAlignment
        # Use wav_to_bytes to create proper audio bytes
        # Move tensor to CPU first to avoid CUDA tensor to numpy conversion error
        audio_tensor_cpu = alignment_tensor.cpu() if alignment_tensor.is_cuda else alignment_tensor

        audio_arr = wav_to_bytes(
            audio_tensor_cpu, sample_rate=sample_rate, format="wav"
        )

        # logger.info(
        #     f"Converted audio to bytes: shape={audio_arr.shape}, dtype={audio_arr.dtype}"
        # )
        logger.info(f"Converted audio to bytes: {len(audio_arr)} bytes")

        # Preprocess tokens for MMS alignment model using the same approach as TextRomanizer
        # The MMS alignment model expects romanized tokens in the same format as text_sentences_tokens
        try:
            # Join tokens back to text for uroman processing
            transcription_text = " ".join(transcription_tokens)

            # Create uroman instance and process the text the same way as TextRomanizer
            uroman_instance = uroman.Uroman()

            # Step 1: Normalize the text first using text_normalize function (same as TextRomanizer)
            normalized_text = text_normalize(transcription_text.strip(), "en")

            # Step 2: Get uroman tokens using the same function as TextRomanizer
            # This creates character-level tokens with spaces between characters
            uroman_tokens_str = get_uroman_tokens(
                [normalized_text], uroman_instance, "en"
            )[0]

            # Step 3: Split by spaces to get individual character tokens (same as real MMS pipeline)
            alignment_tokens = uroman_tokens_str.split()

            logger.info(f"Original tokens: {transcription_tokens}")
            logger.info(f"Original text: '{transcription_text}'")
            logger.info(f"Normalized text: '{normalized_text}'")
            logger.info(f"Uroman tokens string: '{uroman_tokens_str}'")
            logger.info(
                f"Alignment tokens (count={len(alignment_tokens)}): {alignment_tokens[:20]}..."
            )

            # Additional debugging - check for any unusual characters
            for i, token in enumerate(alignment_tokens[:10]):  # Check first 10 tokens
                logger.info(
                    f"Token {i}: '{token}' (length={len(token)}, chars={[c for c in token]})"
                )

        except Exception as e:
            logger.warning(
                f"Failed to preprocess tokens with TextRomanizer approach: {e}"
            )
            logger.exception("Full error traceback:")
            # Fallback: use simple character-level tokenization
            transcription_text = " ".join(transcription_tokens).lower()
            # Simple character-level tokenization as fallback
            alignment_tokens = []
            for char in transcription_text:
                if char == " ":
                    alignment_tokens.append(" ")
                else:
                    alignment_tokens.append(char)
            logger.info(f"Using fallback character tokens: {alignment_tokens[:20]}...")

        logger.info(
            f"Using {len(alignment_tokens)} alignment tokens for forced alignment"
        )

        # Create AudioAlignment instance
        logger.info("Creating AudioAlignment instance...")
        alignment = AudioAlignment()

        # Perform alignment using get_one_row_alignments
        logger.info("Performing alignment...")
        logger.info(f"About to call get_one_row_alignments with:")
        # logger.info(f"  audio_arr type: {type(audio_arr)}, shape: {audio_arr.shape}")
        logger.info(f"audio_arr type: {type(audio_arr)}")
        logger.info(
            f"  alignment_tokens type: {type(alignment_tokens)}, length: {len(alignment_tokens)}"
        )
        logger.info(
            f"  First 10 tokens: {alignment_tokens[:10] if len(alignment_tokens) >= 10 else alignment_tokens}"
        )

        # Check for any problematic characters in tokens
        for i, token in enumerate(alignment_tokens[:5]):
            token_chars = [ord(c) for c in str(token)]
            logger.info(f"  Token {i} '{token}' char codes: {token_chars}")

        # Check if tokens contain any RTL characters that might cause the LTR assertion
        rtl_chars = []
        for i, token in enumerate(alignment_tokens):
            for char in str(token):
                # Check for Arabic, Hebrew, and other RTL characters
                if (
                    "\u0590" <= char <= "\u08ff"
                    or "\ufb1d" <= char <= "\ufdff"
                    or "\ufe70" <= char <= "\ufeff"
                ):
                    rtl_chars.append((i, token, char, ord(char)))

        if rtl_chars:
            logger.warning(f"Found RTL characters in tokens: {rtl_chars[:10]}...")

        try:
            audio_segments = alignment.get_one_row_alignments(
                audio_arr, sample_rate, alignment_tokens
            )

        except Exception as alignment_error:
            logger.error(f"Alignment failed with error: {alignment_error}")
            logger.error(f"Error type: {type(alignment_error)}")

            # Try to provide more context about the error
            if "ltr" in str(alignment_error).lower():
                logger.error("LTR assertion error detected. This might be due to:")
                logger.error("1. RTL characters in the input tokens")
                logger.error(
                    "2. Incorrect token format - tokens should be individual characters"
                )
                logger.error("3. Unicode normalization issues")

                # Try a simple ASCII-only fallback
                logger.info("Attempting ASCII-only fallback...")
                ascii_tokens = []
                for token in alignment_tokens:
                    # Keep only ASCII characters
                    ascii_token = "".join(c for c in str(token) if ord(c) < 128)
                    if ascii_token:
                        ascii_tokens.append(ascii_token)

                logger.info(
                    f"ASCII tokens (count={len(ascii_tokens)}): {ascii_tokens[:20]}..."
                )

                try:
                    audio_segments = alignment.get_one_row_alignments(
                        audio_arr, ascii_tokens
                    )
                    alignment_tokens = ascii_tokens  # Update for later use
                    logger.info("ASCII fallback successful!")
                except Exception as ascii_error:
                    logger.error(f"ASCII fallback also failed: {ascii_error}")
                    raise alignment_error
            else:
                raise

        logger.info(
            f"Alignment completed, got {len(audio_segments)} character segments"
        )

        # Debug: Log the actual structure of audio_segments
        if audio_segments:
            logger.info("=== Audio Segments Debug Info ===")
            logger.info(f"Total segments: {len(audio_segments)}")

            # Print ALL audio segments for complete debugging
            logger.info("=== ALL AUDIO SEGMENTS ===")
            for i, segment in enumerate(audio_segments):
                logger.info(f"Segment {i}: {segment}")
                if i > 0 and i % 20 == 0:  # Print progress every 20 segments
                    logger.info(
                        f"... printed {i+1}/{len(audio_segments)} segments so far..."
                    )
            logger.info("=== End All Audio Segments ===")
            logger.info("=== End Audio Segments Debug ===")

        # Convert character-level segments back to word-level segments
        # Use the actual alignment timings to preserve silence and natural timing
        aligned_segments = []

        logger.info(
            f"Converting {len(audio_segments)} character segments to word segments"
        )
        logger.info(f"Original tokens: {transcription_tokens}")
        logger.info(f"Alignment tokens: {alignment_tokens[:20]}...")

        # Validate that we have segments and tokens
        if not audio_segments or not transcription_tokens:
            logger.warning("No audio segments or transcription tokens available")
            return []

        # Get actual timing from character segments
        if audio_segments:
            # Use the known segment keys from audio_sentence_alignment
            start_key, duration_key = "segment_start_sec", "segment_duration"

            first_segment = audio_segments[0]
            last_segment = audio_segments[-1]

            total_audio_duration = last_segment.get(start_key, 0) + last_segment.get(
                duration_key, 0
            )
            logger.info(
                f"Total audio duration from segments: {total_audio_duration:.3f}s"
            )
        else:
            total_audio_duration = 0.0
            start_key, duration_key = "segment_start_sec", "segment_duration"

        # Strategy: Group character segments by words using the actual alignment timing
        # This preserves the natural timing including silences from the forced alignment

        # First, reconstruct the alignment character sequence
        alignment_char_sequence = "".join(alignment_tokens)
        transcription_text = "".join(
            transcription_tokens
        )  # Remove spaces for character matching

        logger.info(f"Alignment sequence length: {len(alignment_char_sequence)}")
        logger.info(f"Transcription length: {len(transcription_text)}")

        # Create word boundaries based on romanized alignment tokens
        # We need to map each original word to its position in the romanized sequence
        word_boundaries = []
        alignment_pos = 0

        # Process each word individually to get its romanized representation
        for word in transcription_tokens:
            try:
                # Get romanized version of this individual word
                normalized_word = text_normalize(word.strip(), "en")
                uroman_word_str = get_uroman_tokens([normalized_word], uroman_instance, "en")[0]
                romanized_word_tokens = uroman_word_str.split()

                word_start = alignment_pos
                word_end = alignment_pos + len(romanized_word_tokens)
                word_boundaries.append((word_start, word_end))
                alignment_pos = word_end

                logger.info(f"Word '{word}' -> romanized tokens {romanized_word_tokens} -> positions {word_start}-{word_end}")

            except Exception as e:
                logger.warning(f"Failed to romanize word '{word}': {e}")
                # Fallback: estimate based on character length ratio
                estimated_length = max(1, int(len(word) * len(alignment_tokens) / len(transcription_text)))
                word_start = alignment_pos
                word_end = min(alignment_pos + estimated_length, len(alignment_tokens))
                word_boundaries.append((word_start, word_end))
                alignment_pos = word_end

                logger.info(f"Word '{word}' (fallback) -> estimated positions {word_start}-{word_end}")

        logger.info(f"Word boundaries (romanized): {word_boundaries[:5]}...")
        logger.info(f"Total alignment tokens used: {alignment_pos}/{len(alignment_tokens)}")

        # Map each word to its character segments using the boundaries
        for word_idx, (word, (word_start, word_end)) in enumerate(
            zip(transcription_tokens, word_boundaries)
        ):
            # Find character segments that belong to this word
            word_segments = []

            # Map word character range to alignment token indices
            # Since alignment_tokens might be slightly different due to normalization,
            # we'll be flexible and use a range around the expected positions
            start_idx = max(0, min(word_start, len(audio_segments) - 1))
            end_idx = min(word_end, len(audio_segments))

            # Ensure we don't go beyond available segments
            for seg_idx in range(start_idx, end_idx):
                if seg_idx < len(audio_segments):
                    word_segments.append(audio_segments[seg_idx])

            if word_segments:
                # Use actual timing from the character segments for this word
                start_times = [seg.get(start_key, 0) for seg in word_segments]
                end_times = [
                    seg.get(start_key, 0) + seg.get(duration_key, 0)
                    for seg in word_segments
                ]

                start_time = min(start_times) if start_times else 0
                end_time = max(end_times) if end_times else start_time + 0.1
                duration = end_time - start_time

                # Ensure minimum duration
                if duration < 0.05:  # Minimum 50ms
                    duration = 0.05
                    end_time = start_time + duration

                logger.debug(
                    f"Word '{word}' (segments {start_idx}-{end_idx}, {len(word_segments)} segs): {start_time:.3f}s - {end_time:.3f}s ({duration:.3f}s)"
                )
            else:
                logger.warning(
                    f"No segments found for word '{word}' at position {word_start}-{word_end}"
                )
                # Fallback: use proportional timing if no segments found
                if total_audio_duration > 0 and len(transcription_text) > 0:
                    start_proportion = word_start / len(transcription_text)
                    end_proportion = word_end / len(transcription_text)
                    start_time = start_proportion * total_audio_duration
                    end_time = end_proportion * total_audio_duration
                    duration = end_time - start_time
                else:
                    # Ultimate fallback
                    word_duration = 0.5
                    start_time = word_idx * word_duration
                    end_time = start_time + word_duration
                    duration = word_duration

                logger.debug(
                    f"Word '{word}' (fallback): {start_time:.3f}s - {end_time:.3f}s"
                )

            aligned_segments.append(
                {
                    "text": word,
                    "start": start_time,
                    "end": end_time,
                    "duration": duration,
                }
            )

        # Validate segments don't overlap but preserve natural gaps/silences
        for i in range(1, len(aligned_segments)):
            prev_end = aligned_segments[i - 1]["end"]
            current_start = aligned_segments[i]["start"]

            if current_start < prev_end:
                # Only fix actual overlaps, don't force adjacency
                gap = prev_end - current_start
                logger.debug(
                    f"Overlap detected: segment {i-1} ends at {prev_end:.3f}s, segment {i} starts at {current_start:.3f}s (overlap: {gap:.3f}s)"
                )

                # Fix overlap by adjusting current segment start to previous end
                aligned_segments[i]["start"] = prev_end
                aligned_segments[i]["duration"] = (
                    aligned_segments[i]["end"] - aligned_segments[i]["start"]
                )
                logger.debug(
                    f"Fixed overlap for segment {i}: adjusted start to {prev_end:.3f}s"
                )
            else:
                # Log natural gaps (this is normal and expected)
                gap = current_start - prev_end
                if gap > 0.1:  # Log gaps > 100ms
                    logger.debug(
                        f"Natural gap preserved: {gap:.3f}s between segments {i-1} and {i}"
                    )

        logger.info(f"Forced alignment completed: {len(aligned_segments)} segments")
        return aligned_segments

    except Exception as e:
        logger.error(f"Error in forced alignment: {str(e)}", exc_info=True)

        # Fallback: create uniform timestamps based on audio tensor length
        logger.info("Using fallback uniform timestamps")
        try:
            # Calculate duration from the audio tensor
            total_duration = (
                len(audio_tensor) / sample_rate
                if len(audio_tensor) > 0
                else len(transcription_tokens) * 0.5
            )
        except:
            total_duration = len(transcription_tokens) * 0.5  # Fallback

        segment_duration = (
            total_duration / len(transcription_tokens) if transcription_tokens else 1.0
        )

        fallback_segments = []
        for i, token in enumerate(transcription_tokens):
            start_time = i * segment_duration
            end_time = (i + 1) * segment_duration

            fallback_segments.append(
                {
                    "text": token,
                    "start": start_time,
                    "end": end_time,
                    "duration": segment_duration,
                }
            )

        logger.info(
            f"Using fallback uniform timestamps: {len(fallback_segments)} segments"
        )
        return fallback_segments


def transcribe_with_word_alignment(audio_tensor: torch.Tensor, sample_rate: int = 16000, language_with_script: str = None) -> Dict:
    """
    Transcription pipeline that includes word-level timing through forced alignment.
    Adds precise word-level timestamps to the basic transcription capability.

    Args:
        audio_tensor (torch.Tensor): Audio tensor (1D waveform)
        sample_rate (int): Sample rate of the audio tensor
        language_with_script (str): language_with_script code for transcription (3-letter ISO codes like "eng", "spa") with script

    Returns:
        Dict: Transcription results with alignment information including word-level timestamps
    """

    try:
        # Get model and device first

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Get the transcription results
        transcription_text = transcribe_single_chunk(audio_tensor, sample_rate=sample_rate, language_with_script=language_with_script)

        if not transcription_text:
            return {
                "transcription": "",
                "tokens": [],
                "aligned_segments": [],
                "total_duration": 0.0,
            }

        # Tokenize the transcription for alignment
        tokens = transcription_text.split()

        # Perform forced alignment using the original audio tensor
        logger.info("Performing forced alignment with original audio tensor...")
        aligned_segments = perform_forced_alignment(audio_tensor, tokens, device, sample_rate)

        # Calculate total duration
        total_duration = aligned_segments[-1]["end"] if aligned_segments else 0.0

        result = {
            "transcription": transcription_text,
            "tokens": tokens,
            "aligned_segments": aligned_segments,
            "total_duration": total_duration,
            "num_segments": len(aligned_segments),
        }

        logger.info(
            f"Transcription with alignment completed: {len(aligned_segments)} segments, {total_duration:.2f}s total"
        )
        return result

    except Exception as e:
        logger.error(f"Error in transcription with alignment: {str(e)}", exc_info=True)
        # Return basic transcription without alignment
        try:
            transcription_text = transcribe_single_chunk(audio_tensor, sample_rate=sample_rate, language_with_script=language_with_script)
            tokens = transcription_text.split() if transcription_text else []

            return {
                "transcription": transcription_text,
                "tokens": tokens,
                "aligned_segments": [],
                "total_duration": 0.0,
                "alignment_error": str(e),
            }
        except Exception as e2:
            logger.error(f"Error in fallback transcription: {str(e2)}", exc_info=True)
            return {
                "transcription": "",
                "tokens": [],
                "aligned_segments": [],
                "total_duration": 0.0,
                "error": str(e2),
            }


def _validate_and_adjust_segments(
    aligned_segments: List[Dict],
    chunk_start_time: float,
    chunk_audio_tensor: torch.Tensor,
    chunk_sample_rate: int,
    chunk_duration: float,
    chunk_index: int
) -> List[Dict]:
    """
    Private helper function to validate and adjust segment timestamps to global timeline.

    Args:
        aligned_segments: Raw segments from forced alignment (local chunk timeline)
        chunk_start_time: Start time of this chunk in global timeline
        chunk_audio_tensor: Audio tensor for this chunk (to get actual duration)
        chunk_sample_rate: Sample rate of the chunk
        chunk_duration: Reported duration of the chunk
        chunk_index: Index of this chunk for debugging

    Returns:
        List of validated segments with global timeline timestamps
    """
    adjusted_segments = []

    # Get the actual audio duration from the chunk tensor instead of the potentially incorrect chunk duration
    actual_chunk_duration = len(chunk_audio_tensor) / chunk_sample_rate if len(chunk_audio_tensor) > 0 else chunk_duration

    for segment in aligned_segments:
        original_start = segment["start"]
        original_end = segment["end"]

        # Validate that segment timestamps are within chunk boundaries
        if original_start < 0:
            logger.warning(
                f"Segment '{segment['text']}' has negative start time {original_start:.3f}s, clipping to 0"
            )
            original_start = 0

        if original_end > actual_chunk_duration + 1.0:  # Allow 1s buffer for alignment errors
            logger.warning(
                f"Segment '{segment['text']}' end time {original_end:.3f}s exceeds actual chunk duration {actual_chunk_duration:.3f}s, clipping"
            )
            original_end = actual_chunk_duration

        if original_start >= original_end:
            logger.warning(
                f"Segment '{segment['text']}' has invalid timing {original_start:.3f}s-{original_end:.3f}s, using fallback"
            )
            # Use proportional timing based on segment position using actual chunk duration
            segment_index = len(adjusted_segments)
            total_segments = len(aligned_segments)
            if total_segments > 0:
                segment_proportion = segment_index / total_segments
                next_proportion = (segment_index + 1) / total_segments
                original_start = segment_proportion * actual_chunk_duration
                original_end = next_proportion * actual_chunk_duration
            else:
                original_start = 0
                original_end = 0.5

        # Create segment with absolute timeline
        adjusted_segment = {
            "text": segment["text"],
            "start": original_start + chunk_start_time,  # Global timeline
            "end": original_end + chunk_start_time,    # Global timeline
            "duration": original_end - original_start,
            "chunk_index": chunk_index,
            "original_start": original_start,  # Local chunk time
            "original_end": original_end,     # Local chunk time
        }

        adjusted_segments.append(adjusted_segment)

        logger.debug(
            f"Segment '{segment['text']}': {original_start:.3f}-{original_end:.3f} -> {adjusted_segment['start']:.3f}-{adjusted_segment['end']:.3f}"
        )

    logger.info(
        f"Adjusted {len(adjusted_segments)} segments to absolute timeline (chunk starts at {chunk_start_time:.2f}s)"
    )

    return adjusted_segments


def transcribe_full_audio_with_chunking(
    audio_tensor: torch.Tensor, sample_rate: int = 16000, chunk_duration: float = 30.0, language_with_script: str = None, progress_callback=None
) -> Dict:
    """
    Complete audio transcription pipeline that handles any length audio with intelligent chunking.
    This is the full-featured transcription function that can process both short and long audio files.

    Chunking mode is controlled by USE_CHUNKING environment variable:
    - USE_CHUNKING=false: No chunking (single chunk mode)
    - USE_CHUNKING=true (default): VAD-based intelligent chunking

    Args:
        audio_tensor: Audio tensor (1D waveform)
        sample_rate: Sample rate of the audio tensor
        chunk_duration: Target chunk duration in seconds (for static chunking)
        language_with_script: {Language code}_{script} for transcription
        progress_callback: Optional callback for progress updates

    Returns:
        Dict with full transcription and segment information including word-level timestamps
    """

    try:
        logger.info(f"Starting long-form transcription: tensor shape {audio_tensor.shape} at {sample_rate}Hz")
        logger.info(f"USE_CHUNKING = {USE_CHUNKING}")

        # Initialize chunker
        chunker = AudioChunker()

        # Determine chunking mode based on USE_CHUNKING setting
        chunking_mode = "vad" if USE_CHUNKING else "none"

        # Chunk the audio using the new unified interface
        # Ensure tensor is 1D before chunking (squeeze any extra dimensions)
        if len(audio_tensor.shape) > 1:
            logger.info(f"Squeezing audio tensor from {audio_tensor.shape} to 1D")
            audio_tensor_1d = audio_tensor.squeeze()
        else:
            audio_tensor_1d = audio_tensor

        chunks = chunker.chunk_audio(audio_tensor_1d, sample_rate=sample_rate, mode=chunking_mode, chunk_duration=chunk_duration)

        if not chunks:
            logger.warning("No audio chunks created")
            return {
                "transcription": "",
                "chunks": [],
                "total_duration": 0.0,
                "error": "No audio content detected",
            }

        logger.info(f"Processing {len(chunks)} audio chunks (mode: {chunking_mode})")

        # Validate chunk continuity
        for i, chunk in enumerate(chunks):
            logger.info(
                f"Chunk {i+1}: {chunk['start_time']:.2f}s - {chunk['end_time']:.2f}s ({chunk['duration']:.2f}s)"
            )
            if i > 0:
                prev_end = chunks[i - 1]["end_time"]
                current_start = chunk["start_time"]
                gap = current_start - prev_end
                if abs(gap) > 0.1:  # More than 100ms gap/overlap
                    logger.warning(
                        f"Gap/overlap between chunks {i} and {i+1}: {gap:.3f}s"
                    )

        # Process each chunk - now all chunks have uniform format!
        all_segments = []
        full_transcription_parts = []
        total_duration = 0.0
        chunk_details = []

        for i, chunk in enumerate(chunks):
            logger.info(
                f"Processing chunk {i+1}/{len(chunks)} ({chunk['duration']:.1f}s, {chunk['start_time']:.1f}s-{chunk['end_time']:.1f}s)"
            )

            try:
                # Process this chunk using tensor-based transcription pipeline
                # Use the chunk's audio_data tensor directly - no more file operations!
                chunk_audio_tensor = chunk["audio_data"]
                chunk_sample_rate = chunk["sample_rate"]

                chunk_result = transcribe_with_word_alignment(
                    audio_tensor=chunk_audio_tensor,
                    sample_rate=chunk_sample_rate,
                    language_with_script=language_with_script
                )

                # Process alignment results - uniform handling for all chunk types
                chunk_segments = []
                chunk_start_time = chunk["start_time"]
                chunk_duration = chunk["duration"]

                if chunk_result.get("aligned_segments"):
                    logger.info(
                        f"Chunk {i+1} has {len(chunk_result['aligned_segments'])} segments"
                    )

                    chunk_segments = _validate_and_adjust_segments(
                        aligned_segments=chunk_result["aligned_segments"],
                        chunk_start_time=chunk_start_time,
                        chunk_audio_tensor=chunk_audio_tensor,
                        chunk_sample_rate=chunk_sample_rate,
                        chunk_duration=chunk_duration,
                        chunk_index=i
                    )

                all_segments.extend(chunk_segments)
                logger.info(f"Chunk {i+1} processed {len(chunk_segments)} valid segments")

                # Add to full transcription
                chunk_transcription = ""
                if chunk_result.get("transcription"):
                    chunk_transcription = chunk_result["transcription"]
                    full_transcription_parts.append(chunk_transcription)

                # Store detailed chunk information
                chunk_detail = {
                    "chunk_index": i,
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                    "duration": chunk["duration"],
                    "transcription": chunk_transcription,
                    "num_segments": len(chunk_segments),
                    "segments": chunk_segments,
                }
                chunk_details.append(chunk_detail)

                total_duration = max(total_duration, chunk["end_time"])

                # Update progress linearly from 0.1 to 0.9 based on chunk processing
                progress = 0.1 + (0.8 * (i + 1) / len(chunks))
                transcription_status.update_progress(progress)

                logger.info(
                    f"Chunk {i+1} processed: '{chunk_transcription}' ({len(chunk_segments)} segments)"
                )

            except Exception as chunk_error:
                logger.error(f"Error processing chunk {i+1}: {chunk_error}")
                # Continue with next chunk

        # Combine results
        full_transcription = " ".join(full_transcription_parts)

        # Validate segment continuity
        logger.info("Validating segment continuity...")
        for i in range(1, len(all_segments)):
            prev_end = all_segments[i - 1]["end"]
            current_start = all_segments[i]["start"]
            gap = current_start - prev_end
            if abs(gap) > 1.0:  # More than 1 second gap
                logger.warning(f"Large gap between segments {i-1} and {i}: {gap:.3f}s")

        result = {
            "transcription": full_transcription,
            "aligned_segments": all_segments,
            "chunks": [
                {
                    "chunk_index": chunk_detail["chunk_index"],
                    "start_time": chunk_detail["start_time"],
                    "end_time": chunk_detail["end_time"],
                    "duration": chunk_detail["duration"],
                    "transcription": chunk_detail["transcription"],
                    "num_segments": chunk_detail["num_segments"],
                }
                for chunk_detail in chunk_details
            ],
            "chunk_details": chunk_details,  # Full details including segments per chunk
            "total_duration": total_duration,
            "num_chunks": len(chunks),
            "num_segments": len(all_segments),
            "status": "success",
        }

        logger.info(
            f"Long-form transcription completed: {len(chunks)} chunks, {total_duration:.1f}s total"
        )
        logger.info(f"Total segments: {len(all_segments)}")

        # Log chunk timing summary
        for chunk_detail in chunk_details:
            logger.info(
                f"Chunk {chunk_detail['chunk_index']}: {chunk_detail['start_time']:.2f}-{chunk_detail['end_time']:.2f}s, {chunk_detail['num_segments']} segments"
            )

        return result

    except Exception as e:
        logger.error(f"Error in long-form transcription: {str(e)}", exc_info=True)
        return {
            "transcription": "",
            "chunks": [],
            "total_duration": 0.0,
            "error": str(e),
        }


