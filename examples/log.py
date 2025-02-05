import logging.config
import os

logger = logging.getLogger(__name__)

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
        'debug_console': {
            'level': 'DEBUG',
            'formatter': 'detailed',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False
        },
        'certreach': {  # Package logger
            'handlers': ['debug_console'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

class MakeFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=0):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logging.FileHandler.__init__(self, filename, mode, encoding, delay)

def configure_logging(log_file=None):
    """Configure logging with optional file output.
    
    Args:
        log_file (str, optional): Path to log file. If provided, adds file logging.
    """
    config = LOGGING_CONFIG.copy()
    
    if log_file is not None:
        # Add file handler configuration
        config['handlers']['file'] = {
            'level': 'DEBUG',
            'formatter': 'detailed',
            'class': 'log.MakeFileHandler',
            'filename': log_file,
            'mode': 'a',
        }
        # Add file handler to all loggers
        for logger in config['loggers'].values():
            logger['handlers'].append('file')

    # Apply configuration
    logging.config.dictConfig(config)
    
    logger = logging.getLogger(__name__)
    logger.info('Logging system configured successfully')
    if log_file:
        logger.debug(f'File logging enabled: {log_file}')
