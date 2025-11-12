import io

import numpy as np
import soundfile as sf
import torch
from numpy.typing import NDArray


def wav_to_bytes(
    wav: torch.Tensor | NDArray, sample_rate: int = 16_000, format: str = "wav"
) -> NDArray[np.int8]:
    """Convert audio tensor to bytes using soundfile directly."""
    # Convert to numpy if torch tensor
    if isinstance(wav, torch.Tensor):
        if wav.is_cuda:
            wav = wav.cpu()
        # Convert to float32 first (numpy doesn't support bfloat16)
        if wav.dtype != torch.float32:
            wav = wav.float()
        wav = wav.numpy()

    # Ensure float32 dtype for numpy arrays
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)

    # Handle shape: soundfile expects (samples,) for mono or (samples, channels) for multi-channel
    if wav.ndim == 1:
        # Already correct shape for mono
        pass
    elif wav.ndim == 2:
        # If shape is (channels, samples), transpose to (samples, channels)
        if wav.shape[0] < wav.shape[1]:
            wav = wav.T

    # Create buffer and write using soundfile directly
    buffer = io.BytesIO()

    # Map format string to soundfile format
    sf_format = format.upper() if format.lower() in ['wav', 'flac', 'ogg'] else 'WAV'
    subtype = 'PCM_16' if sf_format == 'WAV' else None

    # Write to buffer
    sf.write(buffer, wav, sample_rate, format=sf_format, subtype=subtype)

    buffer.seek(0)
    return np.frombuffer(buffer.getvalue(), dtype=np.int8)
    # return buffer.read()

