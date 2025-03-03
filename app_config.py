"""
Handles application configuration.
"""

import sys
from pathlib import Path
from typing import Dict
import toml
from loguru import logger
from loguru._logger import Logger


class ConfigLoader:
    """
    Loads configuration from a TOML file.
    """

    def __init__(self, config_file: str):
        self.config_file = config_file

    def load_config(self) -> Dict[str, any]:
        """Loads config from file"""
        return toml.load(self.config_file)


class LogLoader:
    """
    Configures the application logger.
    """

    def __init__(self, log_folder: str = "logs", log_retention_days: int = 10):
        """
        Constructor for the LogLoader class.
        """
        self.log_folder = log_folder
        self.log_retention_days = log_retention_days
        self._validate_log_folder()

    def _validate_log_folder(self):
        """Ensure the log directory exists"""
        log_dir = Path(self.log_folder)
        log_dir.mkdir(exist_ok=True, parents=True)

    def validate_log_level(self, level: str) -> bool:
        """
        Validate the logging level string.
        :param level: The log level string to check.
        Returns: True if valid, False otherwise
        """
        valid_levels = {
            "TRACE",
            "DEBUG",
            "INFO",
            "SUCCESS",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

        return level.upper() in valid_levels

    def configure_logger(self, log_file: str, log_format: str, log_level: str) -> Logger:
        """
        Configures logging for the application.
        param log_file (str): The log file name.
        param log_format (str): The log format.
        param log_level (str): The log level.
        Returns: Logger: Configured logger instance.
        Raises: ValueError: If the provided log level is invalid.
        """

        if not self.validate_log_level(log_level):
            raise ValueError(f"Invalid logging level: {log_level} is not supported.")

        logger.remove()

        # Add file handler
        logger.add(
            Path(self.log_folder) / Path(log_file),
            rotation="50 MB",
            format=log_format,
            retention=f"{self.log_retention_days} days",
            level=log_level.upper(),
            backtrace=True,
            diagnose=True,
        )

        # Add stdout handler
        logger.add(sys.stdout, format=log_format, level=log_level.upper())
        return logger
