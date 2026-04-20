"""
Logging utilities for UACPY

Provides clean, timestamped logging with levels (INFO, WARN, ERROR)
"""

from datetime import datetime, timezone
from typing import Optional
from enum import Enum


class LogLevel(Enum):
    """Log levels for UACPY"""
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    DEBUG = "DEBUG"


class Logger:
    """
    Clean logger for UACPY models

    Provides timestamped logging with levels.
    Format: [HH:MM:SS DD/MM/YYYY UTC] [LEVEL] [Model] Message

    Parameters
    ----------
    name : str
        Logger name (typically model name)
    verbose : bool
        Enable/disable logging output
    min_level : LogLevel, optional
        Minimum level to display. Default is INFO.

    Examples
    --------
    >>> logger = Logger("Bellhop", verbose=True)
    >>> logger.info("Starting simulation")
    [14:23:45 21/11/2025 UTC] [INFO] [Bellhop] Starting simulation

    >>> logger.warn("Using approximation")
    [14:23:46 21/11/2025 UTC] [WARN] [Bellhop] Using approximation

    >>> logger.error("Simulation failed")
    [14:23:47 21/11/2025 UTC] [ERROR] [Bellhop] Simulation failed
    """

    def __init__(
        self,
        name: str,
        verbose: bool = False,
        min_level: LogLevel = LogLevel.INFO
    ):
        self.name = name
        self.verbose = verbose
        self.min_level = min_level

        # Level ordering for filtering
        self._level_order = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARN: 2,
            LogLevel.ERROR: 3,
        }

    def _get_timestamp(self) -> str:
        """Get current UTC timestamp in required format"""
        now = datetime.now(timezone.utc)
        return now.strftime("%Y/%m/%d %H:%M:%S UTC")

    def _should_log(self, level: LogLevel) -> bool:
        """Check if message should be logged based on level"""
        if not self.verbose:
            return False
        return self._level_order[level] >= self._level_order[self.min_level]

    def _log(self, level: LogLevel, message: str):
        """Internal logging method"""
        if self._should_log(level):
            timestamp = self._get_timestamp()
            print(f"[{timestamp}] [{level.value}] [{self.name}] {message}")

    def info(self, message: str):
        """Log info message"""
        self._log(LogLevel.INFO, message)

    def warn(self, message: str):
        """Log warning message"""
        self._log(LogLevel.WARN, message)

    def error(self, message: str):
        """Log error message"""
        self._log(LogLevel.ERROR, message)

    def debug(self, message: str):
        """Log debug message (only if min_level=DEBUG)"""
        self._log(LogLevel.DEBUG, message)


def create_logger(name: str, verbose: bool = False) -> Logger:
    """
    Create a logger for a model

    Parameters
    ----------
    name : str
        Model name
    verbose : bool
        Enable verbose output

    Returns
    -------
    logger : Logger
        Configured logger instance

    Examples
    --------
    >>> logger = create_logger("Bellhop", verbose=True)
    >>> logger.info("Model initialized")
    """
    return Logger(name, verbose=verbose)
