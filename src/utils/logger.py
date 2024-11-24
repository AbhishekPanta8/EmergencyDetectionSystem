import logging
import os
from datetime import datetime
from typing import Optional

def setup_logging(
    log_dir: str = 'logs',
    level: int = logging.INFO,
    filename: Optional[str] = None
) -> None:
    """Setup logging configuration"""
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Generate default filename if none provided
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'emergency_detection_{timestamp}.log'
    
    # Full path for log file
    log_path = os.path.join(log_dir, filename)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Setup basic configuration
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            # File handler
            logging.FileHandler(log_path),
            # Console handler
            logging.StreamHandler()
        ]
    )
    
    # Create logger instance
    logger = logging.getLogger(__name__)
    
    logger.info(f"Logging setup completed. Log file: {log_path}")
    
    # Log system info
    logger.info("System Information:")
    try:
        import platform
        logger.info(f"Python version: {platform.python_version()}")
        logger.info(f"Platform: {platform.platform()}")
    except Exception as e:
        logger.warning(f"Could not log system information: {str(e)}")

def log_experiment_params(params: dict) -> None:
    """Log experiment parameters"""
    logger = logging.getLogger(__name__)
    
    logger.info("Experiment Parameters:")
    for key, value in params.items():
        logger.info(f"{key}: {value}")