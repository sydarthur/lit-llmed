"""Simple logger wrapper for GCP compatibility."""

import logging
from typing import Any


class SimpleLogger:
    """Logger wrapper that handles structured logging gracefully."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        if kwargs:
            extra_info = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            self.logger.info(f"{message} - {extra_info}")
        else:
            self.logger.info(message)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional structured data."""
        if kwargs:
            extra_info = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            self.logger.error(f"{message} - {extra_info}")
        else:
            self.logger.error(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        if kwargs:
            extra_info = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            self.logger.warning(f"{message} - {extra_info}")
        else:
            self.logger.warning(message)


def get_logger(name: str) -> SimpleLogger:
    """Get a simple logger instance."""
    return SimpleLogger(name)