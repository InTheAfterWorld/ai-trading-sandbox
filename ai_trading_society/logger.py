"""
Logging configuration for AI Trading Society.

Provides structured logging with console and file output support.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "ai_trading_society",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Parameters
    ----------
    name : str
        Logger name.
    level : int
        Logging level (logging.DEBUG, logging.INFO, etc.).
    log_file : str, optional
        Specific log file name. If None and log_to_file is True,
        generates a timestamped filename.
    log_dir : str
        Directory for log files.
    console : bool
        Whether to output to console.

    Returns
    -------
    logger : logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        os.makedirs(log_dir, exist_ok=True)
        filepath = os.path.join(log_dir, log_file)
        file_handler = logging.FileHandler(filepath, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(name: str = "ai_trading_society") -> logging.Logger:
    """Get an existing logger or create a default one."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger = setup_logger(name=name, console=True)
    return logger


def create_run_log(
    mode: str = "classic",
    steps: int = 100,
    log_dir: str = "logs",
) -> logging.Logger:
    """
    Create a logger for a specific simulation run.

    Parameters
    ----------
    mode : str
        Simulation mode name for the filename.
    steps : int
        Number of steps (for filename).
    log_dir : str
        Directory for log files.

    Returns
    -------
    logger : logging.Logger
        Logger that writes to both console and a timestamped file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sim_{mode}_{steps}steps_{timestamp}.log"
    return setup_logger(
        name=f"ai_trading_society.run.{timestamp}",
        log_file=filename,
        log_dir=log_dir,
        console=True,
    )
