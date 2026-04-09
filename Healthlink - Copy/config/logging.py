"""
Simple logging configuration for HealthLink.
Uses Python's built-in logging - no external dependencies needed.
"""
import logging
import sys


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure application logging with Python's built-in logger.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("healthlink")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level.upper()))

    # Simple, clean text formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(f"Logging initialized at {log_level} level")

    return logger


def get_logger(name: str = "healthlink") -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (defaults to "healthlink")

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
