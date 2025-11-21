"""Logging configuration and utilities for MAGE.

This module provides comprehensive logging functionality with:
- Colored console output
- File-based logging with rotation
- Structured logging for debugging
- Performance tracking
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import colorlog


class MAGELogger:
    """Custom logger for the MAGE system.
    
    Provides colored console logging and file-based logging with
    automatic rotation and structured output.
    """
    
    _instances: dict[str, logging.Logger] = {}
    _configured: bool = False
    
    @classmethod
    def get_logger(
        cls,
        name: str,
        log_level: str = "INFO",
        log_dir: Optional[Path] = None
    ) -> logging.Logger:
        """Get or create a logger instance.
        
        Args:
            name: Name of the logger (typically __name__)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files (default: ./logs)
            
        Returns:
            Configured logger instance
        """
        if name in cls._instances:
            return cls._instances[name]
        
        logger = logging.getLogger(name)
        
        # Configure logging once globally
        if not cls._configured:
            cls._configure_logging(log_level, log_dir)
            cls._configured = True
        
        logger.setLevel(getattr(logging, log_level.upper()))
        cls._instances[name] = logger
        
        return logger
    
    @classmethod
    def _configure_logging(
        cls,
        log_level: str = "INFO",
        log_dir: Optional[Path] = None
    ) -> None:
        """Configure the logging system.
        
        Args:
            log_level: Default logging level
            log_dir: Directory for log files
        """
        # Create log directory
        if log_dir is None:
            log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Console handler with colors
        console_handler = colorlog.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s%(reset)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            }
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler with rotation
        log_file = log_dir / f"mage_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
    
    @classmethod
    def reset_configuration(cls) -> None:
        """Reset the logging configuration.
        
        Useful for testing or reconfiguring logging at runtime.
        """
        cls._configured = False
        cls._instances.clear()
        
        # Remove all handlers from root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)


def log_function_call(logger: logging.Logger):
    """Decorator to log function calls with parameters and results.
    
    Args:
        logger: Logger instance to use
        
    Returns:
        Decorated function
        
    Example:
        @log_function_call(logger)
        def my_function(param1, param2):
            return param1 + param2
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(
                f"Calling {func.__name__}",
                extra={"args": args, "kwargs": kwargs}
            )
            try:
                result = func(*args, **kwargs)
                logger.debug(
                    f"{func.__name__} completed successfully",
                    extra={"result": str(result)[:100]}  # Truncate long results
                )
                return result
            except Exception as e:
                logger.error(
                    f"{func.__name__} raised exception: {e}",
                    extra={"exception_type": type(e).__name__},
                    exc_info=True
                )
                raise
        return wrapper
    return decorator


def log_performance(logger: logging.Logger):
    """Decorator to log function execution time.
    
    Args:
        logger: Logger instance to use
        
    Returns:
        Decorated function
        
    Example:
        @log_performance(logger)
        def expensive_function():
            # ... long running code ...
    """
    import time
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed = end_time - start_time
            
            logger.info(
                f"{func.__name__} executed in {elapsed:.4f} seconds",
                extra={"execution_time": elapsed, "function": func.__name__}
            )
            return result
        return wrapper
    return decorator
