import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def combine_video_with_subtitles(
    video_file_path: str,
    subtitle_content: str,
    subtitle_format: str = "srt",
    output_format: str = "mp4",
    language: str = "eng",
) -> str:
    """
    Combine video file with subtitle content using FFmpeg.

    Args:
        video_file_path: Path to the input video file
        subtitle_content: String content of the subtitles (SRT or WebVTT)
        subtitle_format: Format of subtitles ("srt" or "webvtt")
        output_format: Output container format ("mp4" or "mkv")
        language: Language code for subtitle track

    Returns:
        Path to the output video file with embedded subtitles
    """

    # Create temporary files
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=f".{subtitle_format}", delete=False
    ) as sub_file:
        sub_file.write(subtitle_content)
        subtitle_file_path = sub_file.name

    # Generate output filename
    input_path = Path(video_file_path)
    output_path = (
        input_path.parent / f"{input_path.stem}_with_subtitles.{output_format}"
    )

    try:
        if output_format.lower() == "mkv":
            # MKV has better subtitle support
            if subtitle_format.lower() == "webvtt":
                codec = "webvtt"
            else:
                codec = "srt"

            cmd = [
                "ffmpeg",
                "-y",  # -y to overwrite output file
                "-i",
                video_file_path,
                "-i",
                subtitle_file_path,
                "-c:v",
                "copy",  # Copy video stream
                "-c:a",
                "copy",  # Copy audio stream
                "-c:s",
                codec,  # Subtitle codec
                "-metadata:s:s:0",
                f"language={language}",
                str(output_path),
            ]
        else:
            # MP4 format
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_file_path,
                "-i",
                subtitle_file_path,
                "-c:v",
                "copy",  # Copy video stream
                "-c:a",
                "copy",  # Copy audio stream
                "-c:s:0",
                "mov_text",  # MP4 subtitle format
                "-map",
                "0:v",  # Map video from first input
                "-map",
                "0:a",  # Map audio from first input
                "-map",
                "1:s",  # Map subtitles from second input
                "-metadata:s:s:0",
                f"language={language}",
                "-disposition:s:0",
                "default",  # Make subtitles default
                str(output_path),
            ]

        # Execute FFmpeg command
        logger.info(f"Executing FFmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Log FFmpeg output for debugging
        if result.stdout:
            logger.debug(f"FFmpeg stdout: {result.stdout}")
        if result.stderr:
            logger.debug(f"FFmpeg stderr: {result.stderr}")

        logger.info(f"FFmpeg completed successfully, output file: {output_path}")

        return str(output_path)

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
    finally:
        # Clean up temporary subtitle file
        try:
            os.unlink(subtitle_file_path)
        except OSError:
            pass


def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available on the system."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_audio_from_video(video_file_path: str, output_audio_path: str = None) -> str:
    """
    Extract audio from video file using FFmpeg.

    Args:
        video_file_path: Path to the input video file
        output_audio_path: Path for output audio file (optional)

    Returns:
        Path to the extracted audio file
    """
    if not check_ffmpeg_available():
        raise RuntimeError("FFmpeg not found. Please install FFmpeg.")

    # Generate output filename if not provided
    if output_audio_path is None:
        input_path = Path(video_file_path)
        output_audio_path = str(input_path.with_suffix('.wav'))

    try:
        # FFmpeg command to extract audio
        # -vn: disable video stream
        # -acodec pcm_s16le: use 16-bit PCM encoding
        # -ar 16000: set sample rate to 16kHz (optimal for speech recognition)
        # -ac 1: mono audio (single channel)
        cmd = [
            "ffmpeg",
            "-i", video_file_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", "16000",  # 16kHz sample rate
            "-ac", "1",  # Mono
            "-y",  # Overwrite output file if it exists
            output_audio_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Audio extracted successfully to: {output_audio_path}")
        return output_audio_path

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg audio extraction failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg.")


def get_video_info(video_file_path: str) -> dict:
    """Get basic information about a video file."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            video_file_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    except json.JSONDecodeError:
        return {}
