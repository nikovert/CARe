import logging
import sys

def configure_logging(log_file=None, log_level=logging.DEBUG):
    """Configure logging to both file and console with proper formatting."""
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    root_logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(name)s:%(lineno)d:%(levelname)s: %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)
    
    # Set levels for specific modules
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Set debug level for verification modules
    logging.getLogger('certreach.verification').setLevel(logging.DEBUG)
