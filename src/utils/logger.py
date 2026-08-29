"""
IBVAP Logging Utility
Structured logging for detection events, alerts, and system status.
"""
import json
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path


class IBVAPLogger:
    """Structured logger for IBVAP events."""

    def __init__(self, log_dir: str = "logs", level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.level = getattr(logging, level.upper(), logging.INFO)

        # Set up standard logger
        self.logger = logging.getLogger("ibvap")
        self.logger.setLevel(self.level)

        # Console handler
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(self.level)
            fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%H:%M:%S"
            )
            ch.setFormatter(fmt)
            self.logger.addHandler(ch)

    def info(self, msg: str, **kwargs):
        """Log info message."""
        self.logger.info(self._format(msg, kwargs))

    def warning(self, msg: str, **kwargs):
        """Log warning."""
        self.logger.warning(self._format(msg, kwargs))

    def error(self, msg: str, **kwargs):
        """Log error."""
        self.logger.error(self._format(msg, kwargs))

    def alert(self, event_type: str, severity: str, explanation: str, **kwargs):
        """Log an alert event."""
        msg = f"[{severity.upper()}] {event_type}: {explanation}"
        if severity in ("critical", "high"):
            self.logger.error(self._format(msg, kwargs))
        else:
            self.logger.warning(self._format(msg, kwargs))

    def detection(self, frame_id: int, num_detections: int, **kwargs):
        """Log detection results."""
        self.logger.info(f"Frame {frame_id}: {num_detections} detections")

    def signal_loss(self, camera_id: str, message: str):
        """Log signal loss event."""
        self.logger.error(f"SIGNAL LOSS [{camera_id}]: {message}")

    def chain(self, total_events: int, valid: bool):
        """Log hash chain status."""
        status = "VALID" if valid else "BROKEN"
        self.logger.info(f"Hash chain: {status} ({total_events} events)")

    def _format(self, msg: str, kwargs: dict) -> str:
        """Format message with extra context."""
        if kwargs:
            context = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            return f"{msg} [{context}]"
        return msg


# Global logger instance
log = IBVAPLogger()
