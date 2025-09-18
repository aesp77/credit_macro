"""
Logging configuration for CDS Monitor
Centralized logging setup for all modules
"""
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import json


def setup_logger(
    name: str = "cds_monitor",
    level: str = "INFO",
    log_dir: str = "logs",
    console: bool = True,
    file: bool = True,
    json_format: bool = False
) -> logging.Logger:
    """
    Setup logger with console and file handlers
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        console: Whether to log to console
        file: Whether to log to file
        json_format: Whether to use JSON formatting
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file:
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create file handler with rotation
        log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Also create an error-only file
        error_file = log_path / f"{name}_errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
    
    return logger


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 
                          'funcName', 'levelname', 'levelno', 'lineno', 
                          'module', 'msecs', 'message', 'pathname', 'process', 
                          'processName', 'relativeCreated', 'stack_info', 
                          'thread', 'threadName', 'exc_info', 'exc_text']:
                log_data[key] = value
        
        return json.dumps(log_data)


# Performance logger for tracking execution times
class PerformanceLogger:
    """Logger for tracking performance metrics"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger("performance")
        self._timers = {}
    
    def start_timer(self, name: str):
        """Start a named timer"""
        from time import time
        self._timers[name] = time()
        self.logger.debug(f"Timer started: {name}")
    
    def end_timer(self, name: str) -> float:
        """End a named timer and return elapsed time"""
        from time import time
        
        if name not in self._timers:
            self.logger.warning(f"Timer not found: {name}")
            return 0
        
        elapsed = time() - self._timers[name]
        del self._timers[name]
        
        self.logger.info(f"Timer {name}: {elapsed:.3f} seconds")
        return elapsed
    
    def log_metrics(self, **metrics):
        """Log arbitrary metrics"""
        self.logger.info("Metrics", extra=metrics)


# Default logger setup
def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (default: caller's module)
        
    Returns:
        Logger instance
    """
    if name is None:
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'cds_monitor')
        else:
            name = 'cds_monitor'
    
    return logging.getLogger(name)


# Initialize default logger on import
default_logger = setup_logger()