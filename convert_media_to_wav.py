"""
Media to WAV Converter Module

Converts various media formats (m4a, mp3, mp4, etc.) to standardized WAV files
and PyTorch tensors for audio transcription pipelines.

Standardization:
- 16kHz sample rate
- Mono channel (merged if multi-channel)
- Layer normalized
- bfloat16 dtype tensor
- Fail-fast error handling
"""

import os
import tempfile
from pathlib import Path
from typing import Tuple, Union, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from pydub import AudioSegment
from pydub.utils import which


# Constants
TARGET_SAMPLE_RATE = 16000
TARGET_DTYPE = torch.bfloat16


def verify_ffmpeg_installation():
    """Verify FFmpeg is available for pydub operations."""
    if not which("ffmpeg"):
        raise RuntimeError(
            "FFmpeg not found. Please install FFmpeg for media format support. "
            "On Ubuntu: sudo apt install ffmpeg"
        )


def layer_norm(tensor: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Apply layer normalization to audio tensor."""
    # Simple layer normalization: (x - mean) / std
    mean = tensor.mean()
    std = tensor.std()
    if std == 0:
        return tensor - mean
    return (tensor - mean) / std


def detect_media_format(file_path: str) -> str:
    """Detect media format from file extension."""
    file_path = Path(file_path)
    extension = file_path.suffix.lower()

    supported_formats = {
        '.wav': 'wav',
        '.mp3': 'mp3',
        '.m4a': 'm4a',
        '.aac': 'aac',
        '.flac': 'flac',
        '.ogg': 'ogg',
        '.wma': 'wma',
        '.mp4': 'mp4',
        '.avi': 'avi',
        '.mov': 'mov',
        '.mkv': 'mkv'
    }

    # Return known format or just pass through the extension without the dot
    # Let downstream processing handle unknown formats with detailed error messages
    return supported_formats.get(extension, extension[1:] if extension.startswith('.') else extension)


def convert_to_wav_with_pydub(input_path: str, output_path: str, format_hint: str = None):
    """Convert media file to WAV using pydub (FFmpeg backend)."""
    verify_ffmpeg_installation()

    # Load audio file - pydub auto-detects format or use hint
    if format_hint:
        audio = AudioSegment.from_file(input_path, format=format_hint)
    else:
        # Let pydub auto-detect
        audio = AudioSegment.from_file(input_path)

    # Convert to WAV format with standard settings
    # pydub will handle the initial conversion, librosa will do the final processing
    audio.export(output_path, format="wav")


def process_wav_to_standard_format(wav_path: str) -> Tuple[np.ndarray, int]:
    """Process WAV file to standard format using librosa."""
    # Load the WAV file with librosa (handles resampling better than pydub)
    data, fs = librosa.load(wav_path, sr=None)  # Load at original sample rate first

    # Resample to target sample rate if needed
    if fs != TARGET_SAMPLE_RATE:
        data = librosa.resample(data, orig_sr=fs, target_sr=TARGET_SAMPLE_RATE)

    # Handle multi-channel audio by merging to mono
    if len(data.shape) > 1:
        # Average across channels
        data = np.mean(data, axis=0)

    # Ensure it's a 1D array
    data = np.asarray(data, dtype=np.float32)

    return data, TARGET_SAMPLE_RATE


def create_normalized_tensor(audio_data: np.ndarray) -> torch.Tensor:
    """Convert numpy audio data to normalized PyTorch tensor with device handling."""
    # Convert to bf16 tensor and normalize
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = torch.Tensor(audio_data).to(torch.bfloat16)
    data = layer_norm(data, data.shape)
    data = data.unsqueeze(0).to(device)

    return data


def convert_media_to_wav(
    input_path: str,
    output_dir: Optional[str] = None,
    keep_temp_wav: bool = True
) -> Tuple[str, torch.Tensor]:
    """
    Convert media file to standardized WAV file and normalized tensor.

    Args:
        input_path: Path to input media file
        output_dir: Directory for output WAV file (default: temp directory)
        keep_temp_wav: Whether to keep the temporary WAV file

    Returns:
        Tuple of (wav_file_path, normalized_tensor)

    Raises:
        ValueError: If file format is unsupported
        RuntimeError: If FFmpeg is not available
        FileNotFoundError: If input file doesn't exist
    """

    # Validate input file
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    input_path = os.path.abspath(input_path)

    # Detect format
    media_format = detect_media_format(input_path)

    # Setup output path
    if output_dir is None:
        output_dir = tempfile.gettempdir()

    # Create output filename
    input_name = Path(input_path).stem
    output_wav_path = os.path.join(output_dir, f"{input_name}_converted.wav")

    # Step 1: Convert to WAV using pydub (handles format conversion)
    if media_format == 'wav':
        # Already WAV, but still process through pydub to normalize format
        convert_to_wav_with_pydub(input_path, output_wav_path, 'wav')
    else:
        # Convert from other format to WAV
        convert_to_wav_with_pydub(input_path, output_wav_path, media_format)

    # Step 2: Process WAV to standard format using librosa
    audio_data, sample_rate = process_wav_to_standard_format(output_wav_path)

    # Step 3: Create normalized tensor
    normalized_tensor = create_normalized_tensor(audio_data)

    # Step 4: Save the processed audio back to WAV file
    # Overwrite the temp WAV with the processed version
    sf.write(output_wav_path, audio_data, sample_rate)

    return output_wav_path, normalized_tensor


def convert_media_to_wav_from_bytes(
    media_bytes: bytes,
    original_filename: str,
    output_dir: Optional[str] = None
) -> Tuple[str, torch.Tensor]:
    """
    Convert media from bytes to WAV file and tensor.

    Args:
        media_bytes: Raw media file bytes
        original_filename: Original filename for format detection
        output_dir: Directory for output files

    Returns:
        Tuple of (wav_file_path, normalized_tensor)
    """

    # Create temporary input file
    input_extension = Path(original_filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=input_extension) as temp_input:
        temp_input.write(media_bytes)
        temp_input_path = temp_input.name

    # Convert using the main function
    wav_path, tensor = convert_media_to_wav(temp_input_path, output_dir)

    # Clean up temporary input file
    os.unlink(temp_input_path)

    return wav_path, tensor


# Utility function for getting audio info
def get_media_info(file_path: str) -> dict:
    """Get information about media file."""
    verify_ffmpeg_installation()

    audio = AudioSegment.from_file(file_path)

    return {
        "duration_seconds": len(audio) / 1000.0,
        "frame_rate": audio.frame_rate,
        "channels": audio.channels,
        "sample_width": audio.sample_width,
        "format": detect_media_format(file_path)
    }


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) != 2:
        print("Usage: python convert_media_to_wav.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    print(f"Converting {input_file}...")
    wav_path, tensor = convert_media_to_wav(input_file)

    print(f"✓ WAV file: {wav_path}")
    print(f"✓ Tensor shape: {tensor.shape}")
    print(f"✓ Tensor dtype: {tensor.dtype}")
    print(f"✓ Tensor device: {tensor.device}")

    # Show media info
    info = get_media_info(input_file)
    print(f"✓ Media info: {info}")
