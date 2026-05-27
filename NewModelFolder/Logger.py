"""
Logging configuration for the LSTM pipeline.
Provides structured logging with timestamps, log levels, and colored output.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[37m',       # White
        'SUCCESS': '\033[32m',    # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to the entire message
        if record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            reset = self.COLORS['RESET']
            
            # Color the level name
            record.levelname_colored = f"{color}{record.levelname:8s}{reset}"
            
            # Color the message
            record.msg = f"{color}{record.msg}{reset}"
            
            # If there are args, they'll be interpolated into the colored message
            if record.args:
                record.msg = record.msg % record.args
                record.args = None  # Prevent double formatting
        else:
            record.levelname_colored = f"{record.levelname:8s}"
        
        return super().format(record)


def setup_logger(name="pipeline", log_dir="logs", local=False):
    """
    Set up logger with console and file handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to store log files
        local: Whether running in local mode (affects log path)
    
    Returns:
        logger: Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create log directory
    if local:
        log_path = Path(log_dir)
    else:
        log_path = Path("/ceph/project/SW6-Group18-Abvaerk/logs")
    
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"Pipeline_{timestamp}.log"
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = ColoredFormatter(
        '%(asctime)s | %(levelname_colored)s | %(name)-21s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler without colors
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-21s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized. Logging to: {log_file}")
    
    return logger


def add_success_level():
    """Add custom SUCCESS log level between INFO and WARNING."""
    SUCCESS_LEVEL = 25  # Between INFO (20) and WARNING (30)
    logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")
    
    def success(self, message, *args, **kwargs):
        if self.isEnabledFor(SUCCESS_LEVEL):
            self._log(SUCCESS_LEVEL, message, args, **kwargs)
    
    logging.Logger.success = success


# Add SUCCESS level when module is imported
add_success_level()


# Example usage
if __name__ == "__main__":
    # Test the logger
    logger = setup_logger("test", local=True)
    
    logger.debug("This is a debug message")
    logger.info("Processing data...")
    logger.success("Server started successfully")
    logger.warning("Invalid configuration detected")
    logger.error("Failed to connect to database")