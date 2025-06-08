"""
Enhanced logging functionality for the bot
"""

import logging
import inspect
import traceback
from typing import Optional
from pathlib import Path
from datetime import datetime

def setup_logging(log_dir: str = "logs") -> None:
    """Set up enhanced logging with file output"""
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"bot_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def log_error(e: Exception, context: str = "") -> None:
    """Enhanced error logging with stack trace and line numbers"""
    logger = logging.getLogger(__name__)
    
    current_frame = inspect.currentframe()
    calling_frame = inspect.getouterframes(current_frame)[1]
    error_location = f"{calling_frame.filename}:{calling_frame.lineno}"
    
    error_msg = f"""
ERROR DETAILS:
Location: {error_location}
Context: {context}
Error Type: {type(e).__name__}
Error Message: {str(e)}
Stack Trace:
{traceback.format_exc()}
"""
    logger.error(error_msg)

class BotLogger:
    """Class to handle bot-specific logging"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def info(self, message: str) -> None:
        """Log info message with enhanced context"""
        self.logger.info(message)
    
    def error(self, message: str, error: Optional[Exception] = None) -> None:
        """Log error message with enhanced context and optional exception details"""
        if error:
            log_error(error, message)
        else:
            self.logger.error(message)
    
    def warning(self, message: str) -> None:
        """Log warning message with enhanced context"""
        self.logger.warning(message)
    
    def debug(self, message: str) -> None:
        """Log debug message with enhanced context"""
        self.logger.debug(message) 