import json
import logging
import os
import tempfile

import torch
from audio_transcription import perform_forced_alignment
from media_transcription_processor import MediaTranscriptionProcessor
from transcription_status import transcription_status
from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

from env_vars import API_LOG_LEVEL, MODEL_NAME
from flask import Blueprint, jsonify, request, send_file
from video_utils import check_ffmpeg_available, combine_video_with_subtitles

transcriptions_blueprint = Blueprint(
    "transcriptions_blueprint",
    __name__,
)

logger = logging.getLogger(__name__)
logger.level = API_LOG_LEVEL
logging.getLogger("boto3").setLevel(API_LOG_LEVEL)
logging.getLogger("botocore").setLevel(API_LOG_LEVEL)

MAX_SHORTFORM_DURATION = 10  # seconds


@transcriptions_blueprint.route("/health")
def health():
    """Comprehensive health check endpoint"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    ffmpeg_available = check_ffmpeg_available()

    # Get transcription status
    transcription_info = MediaTranscriptionProcessor.get_server_status()

    # Get GPU details if CUDA is available
    gpu_info = {}
    if cuda_available:
        gpu_info = {
            "gpu_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.device_count() > 0
                else "Unknown"
            ),
        }

        # Add GPU memory information
        try:
            current_device = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated(current_device)
            memory_reserved = torch.cuda.memory_reserved(current_device)
            memory_total = torch.cuda.get_device_properties(current_device).total_memory

            gpu_info.update(
                {
                    "gpu_memory_allocated_mb": round(memory_allocated / 1024 / 1024, 1),
                    "gpu_memory_reserved_mb": round(memory_reserved / 1024 / 1024, 1),
                    "gpu_memory_total_mb": round(memory_total / 1024 / 1024, 1),
                    "gpu_memory_free_mb": round(
                        (memory_total - memory_reserved) / 1024 / 1024, 1
                    ),
                }
            )
        except Exception as e:
            logger.warning(f"Could not get GPU memory info: {e}")

    return {
        "status": "healthy",
        "message": "MMS Transcription API is running",
        "version": "1.0.0",
        "service": "mms-transcription",
        "device": str(device),
        "cuda_available": cuda_available,
        "ffmpeg_available": ffmpeg_available,
        "transcription_status": transcription_info,
        **gpu_info,
    }


@transcriptions_blueprint.route("/supported-languages")
def get_supported_languages():
    """Get list of supported languages for transcription"""
    try:
        return jsonify({
            "supported_languages": supported_langs,
        })
    except Exception as e:
        logger.error(f"Error getting supported languages: {str(e)}")
        return jsonify({
            "error": "Could not retrieve supported languages",
            "message": str(e)
        }), 500


@transcriptions_blueprint.route("/status")
def get_transcription_status():
    """Get current transcription status"""
    return jsonify(MediaTranscriptionProcessor.get_server_status())


@transcriptions_blueprint.route("/transcribe", methods=["POST"])
def transcribe_audio():
    """Transcribe media using the MMS model with intelligent chunking for all audio/video files"""
    try:
        # Check if server is busy
        if MediaTranscriptionProcessor.is_server_busy():
            status = MediaTranscriptionProcessor.get_server_status()
            return (
                jsonify(
                    {
                        "error": "Server is currently processing another transcription",
                        "status": "busy",
                        "current_operation": status.get("current_operation"),
                    }
                ),
                503,
            )

        # Check if media file is provided
        if "media" not in request.files:
            return jsonify({"error": "No media file provided"}), 400

        media_file = request.files["media"]
        if media_file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Get optional language parameter
        language_with_script = request.form.get("language", None)

        if language_with_script:
            logger.info(f"Language specified: {language_with_script}")
        else:
            logger.info("No language specified, using auto-detection")

        # Get optional include_preprocessed parameter (from form data or query string)
        include_preprocessed = (
            request.form.get("include_preprocessed", "false").lower() == "true" or
            request.args.get("include_preprocessed", "false").lower() == "true"
        )
        if include_preprocessed:
            logger.info("Preprocessed audio will be included in response")

        # Mark as busy and start transcription
        # This will be handled by the processor

        # Read file bytes once
        media_bytes = media_file.read()

        try:
            # Use the MediaTranscriptionProcessor with context manager for automatic cleanup
            with MediaTranscriptionProcessor(media_bytes, media_file.filename, language_with_script) as processor:
                # Start transcription status tracking
                processor.start_transcription()

                # Stage 1: Convert media (this also calculates duration and updates progress)
                processor.convert_media()
                logger.info(f"Media conversion completed for: {media_file.filename}")

                # Stage 2: Run full transcription pipeline (this also updates progress)
                processor.transcribe_full_pipeline()

                # Get final results with optional preprocessed audio
                results = processor.get_results(include_preprocessed_audio=include_preprocessed)

                logger.info(f"Transcription completed: {results.get('num_chunks', 0)} chunks")

                # Format response
                response = {
                    "transcription": results.get("transcription", ""),
                    "aligned_segments": results.get("aligned_segments", []),
                    "chunks": results.get("chunks", []),
                    "total_duration": results.get("total_duration", 0.0),
                    "num_chunks": results.get("num_chunks", 0),
                    "num_segments": results.get("num_segments", 0),
                    "model": MODEL_NAME,
                    "device": str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")),
                    "status": results.get("status", "success"),
                }

                # Add preprocessed audio if it was included in results
                if "preprocessed_audio" in results:
                    response["preprocessed_audio"] = results["preprocessed_audio"]

                if "error" in results:
                    response["error"] = results["error"]
                    logger.error(f"Transcription response with error: {response}")
                    return jsonify(response), 500

                # Print out the complete response for debugging
                logger.info("=== TRANSCRIBE RESPONSE ===")
                # logger.info(f"Full response: {json.dumps(response, indent=2)}")
                logger.info("=== END TRANSCRIBE RESPONSE ===")

                return jsonify(response)
                # Context manager automatically handles cleanup and status finalization here

        except Exception as e:
            logger.error(f"Media conversion/transcription error: {str(e)}")
            return jsonify({"error": f"Media processing failed: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500


@transcriptions_blueprint.route("/combine-video-subtitles", methods=["POST"])
def combine_video_subtitles():
    """Combine video with subtitles using FFmpeg"""
    try:
        # Check if server is busy
        if MediaTranscriptionProcessor.is_server_busy():
            status = MediaTranscriptionProcessor.get_server_status()
            return (
                jsonify(
                    {
                        "error": "Server is currently processing another request",
                        "status": "busy",
                        "current_operation": status.get("current_operation"),
                    }
                ),
                503,
            )

        # Check required fields
        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        if "subtitles" not in request.form:
            return jsonify({"error": "No subtitles provided"}), 400

        video_file = request.files["video"]
        subtitles = request.form["subtitles"]

        if video_file.filename == "":
            return jsonify({"error": "No video file selected"}), 400

        # Get optional parameters
        subtitle_format = request.form.get("format", "srt")  # srt or webvtt
        output_format = request.form.get("output_format", "mp4")  # mp4 or mkv
        language = request.form.get("language", "eng")

        # Mark as busy and start processing
        transcription_status.start_transcription("combine_video", video_file.filename)

        try:
            transcription_status.update_progress(0.1)

            # Save the uploaded video file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.filename)[1]) as temp_video:
                video_file.save(temp_video.name)
                temp_video_path = temp_video.name

            transcription_status.update_progress(0.3)

            try:
                # Combine video with subtitles using video_utils function
                output_path = combine_video_with_subtitles(
                    temp_video_path, subtitles, subtitle_format, output_format, language
                )

                transcription_status.update_progress(0.9)

                logger.info(f"Video combination completed: {output_path}")

                # Return the combined video file
                return send_file(
                    output_path,
                    as_attachment=True,
                    download_name=f"{video_file.filename.rsplit('.', 1)[0]}_with_subtitles.{output_format}",
                    mimetype=f"video/{output_format}",
                )

            finally:
                # Clean up temporary video file
                try:
                    os.unlink(temp_video_path)
                except OSError:
                    pass

        finally:
            # Mark transcription as finished
            transcription_status.finish_transcription()

    except Exception as e:
        transcription_status.finish_transcription()
        logger.error(f"Video combination error: {str(e)}")
        return jsonify({"error": f"Video combination failed: {str(e)}"}), 500
