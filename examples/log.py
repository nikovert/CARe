import logging
import sys
import os

def suppress_matplotlib_logging():
    """
    Suppress all logging from matplotlib by setting its logger level to WARNING.
    This prevents matplotlib's debug and info messages from appearing.
    """
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    # Also suppress some common matplotlib submodules
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.backends').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.ticker').setLevel(logging.WARNING)

def suppress_onnx_logging():
    """
    Suppress all logging from ONNX by setting its logger level to WARNING.
    This prevents ONNX's debug and info messages from appearing.
    """
    logging.getLogger('onnx').setLevel(logging.WARNING)
    logging.getLogger('onnxruntime').setLevel(logging.WARNING)
    logging.getLogger('onnxscript').setLevel(logging.WARNING)

def configure_logging(log_file=None, log_level=logging.INFO):
    """
    Configure the logging system using the hierarchical logging approach.
    
    This sets up the root logger with appropriate handlers and formatting.
    Individual modules can then create their own loggers using getLogger(__name__).
    
    Args:
        log_file: Optional file path to write logs to
        log_level: Logging level (default: INFO)
    """
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers to avoid duplicate messages
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(message)s',
        '%H:%M:%S'  # Format for time only (HH:MM:SS)
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log_file is provided
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress matplotlib logging
    suppress_matplotlib_logging()

    # Suppress ONNX logging
    suppress_onnx_logging()
