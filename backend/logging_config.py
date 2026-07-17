"""
Sets up centralized, structured logging 
"""

import sys
from pathlib import Path

from loguru import logger



from backend.config import settings


def setup_logging() -> None:
    """
    Configure loguru logger with console and file sinks
    """
    # Remove default handler
    logger.remove()

    # Console handler (colorized, stderr)
    logger.add(
        sys.stderr,   format=settings.LOG_FORMAT,
        level=settings.LOG_LEVEL,  colorize=True,
        backtrace=True,   diagnose=True,
    )

    # File handler (all logs, rotated)
    log_dir = settings.PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    logger.add(
        log_dir / "kubesage_{time:YYYY-MM-DD}.log",
        format=settings.LOG_FORMAT,
        level="DEBUG",    rotation="10 MB",
        retention="30 days",   compression="gz",
        backtrace=True, diagnose=False,
    )

    # Error-only file
    logger.add(
        log_dir / "kubesage_errors_{time:YYYY-MM-DD}.log",
        format=settings.LOG_FORMAT,
        level="ERROR",rotation="5 MB",
        retention="90 days",     backtrace=True,
        diagnose=True,
    )

    logger.info("=" * 40)
    logger.info("KubeSage logging initialized ..... ha ")
    logger.info(f"Log level: {settings.LOG_LEVEL}")
    logger.info(f"Project root: {settings.PROJECT_ROOT}")
    logger.info("=" * 50)


_logger_initialized = False
_initialization_lock = False


def get_logger(name: str):
    """
    get a logger instance bound to the given module name


    """
    global _logger_initialized
    if not _logger_initialized:
        setup_logging()
        _logger_initialized = True
    return logger.bind(module=name)
