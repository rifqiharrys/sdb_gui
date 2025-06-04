import os
import sys
import logging
import logging.config
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def setup_logging(log_file: str = 'sdb_gui.log', config: Optional[dict] = None) -> None:
    """Setup logging configuration
    
    Args:
        log_file: Path to log file
        config: Optional logging configuration dictionary
    """
    if config:
        try:
            config['handlers']['file']['filename'] = log_file
            logging.config.dictConfig(config)
            logger.info("Loaded custom logging configuration")
        except Exception as e:
            logger.error(f"Error loading logging config: {e}")
            _setup_default_logging(log_file)
    else:
        _setup_default_logging(log_file)

def _setup_default_logging(log_file: str) -> None:
    """Setup default logging if config fails"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )

def resource_path(relative_path: str) -> str:
    """Get absolute path to resource"""
    try:
        base_path = sys._MEIPASS # type: ignore
    except Exception:
        base_path = os.path.abspath(".")
    
    full_path = os.path.join(base_path, relative_path)
    if not os.path.exists(full_path):
        logger.error(f"Resource not found: {full_path}")
        raise FileNotFoundError(f"Required resource missing: {relative_path}")
    return full_path

def to_title(text: str) -> str:
    """Convert snake_case to Title Case"""
    return text.replace('_', ' ').title()