import logging
import threading
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)


class TranscriptionStatus:
    """Simple transcription status tracker"""

    def __init__(self):
        self.is_busy = False
        self.current_operation = None
        self.current_filename = None
        self.started_at = None
        self.progress = 0.0
        self.lock = threading.Lock()
        self.total_completed = 0

    def start_transcription(self, operation_type: str, filename: str = None):
        """Mark transcription as started"""
        with self.lock:
            self.is_busy = True
            self.current_operation = operation_type
            self.current_filename = filename
            self.started_at = datetime.now()
            self.progress = 0.0
            logger.info(f"Started {operation_type} transcription for {filename or 'unknown file'}")

    def update_progress(self, progress: float):
        """Update transcription progress (0.0 to 1.0)"""
        with self.lock:
            self.progress = max(0.0, min(1.0, progress))

    def finish_transcription(self):
        """Mark transcription as finished"""
        with self.lock:
            self.is_busy = False
            self.current_operation = None
            self.current_filename = None
            self.started_at = None
            self.progress = 0.0
            self.total_completed += 1
            logger.info("Transcription finished")

    def get_status(self) -> Dict:
        """Get current status for API response"""
        with self.lock:
            status = {"is_busy": self.is_busy, "total_completed": self.total_completed}

            if self.is_busy:
                duration = (
                    (datetime.now() - self.started_at).total_seconds()
                    if self.started_at
                    else 0
                )
                status.update(
                    {
                        "current_operation": self.current_operation,
                        "current_filename": self.current_filename,
                        "progress": self.progress,
                        "duration_seconds": round(duration, 1),
                    }
                )

            return status


# Global status instance
transcription_status = TranscriptionStatus()
