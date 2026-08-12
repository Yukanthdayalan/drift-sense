"""
Structured logging utilities for the Drift-Sense engine.
Ensures uniform log formatting and level management across all modules.
"""
import logging
import sys
from typing import Optional

def get_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Creates or retrieves a structured logger configured for industrial telemetry.
    
    Args:
        name: Name of the logger (typically __name__ of the calling module).
        level: Logging level threshold (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional absolute or relative file path to output logs to disk.
        
    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if get_logger is invoked multiple times for the same name
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(level)
    
    # Structured format suitable for both human reading and simple regex parsing
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler (standard output)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional persistence)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    # Isolate from root logger to prevent duplicated lines in embedded environments
    logger.propagate = False
    
    return logger
