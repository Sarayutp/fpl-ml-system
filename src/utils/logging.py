"""
Production logging configuration for FPL ML System.
Structured logging with multiple handlers and formatters.
"""

import logging
import logging.handlers
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import os
from contextvars import ContextVar

# Context variables for request tracking
request_id: ContextVar[str] = ContextVar('request_id', default='')
user_id: ContextVar[str] = ContextVar('user_id', default='')
operation: ContextVar[str] = ContextVar('operation', default='')


class StructuredFormatter(logging.Formatter):
    """
    Structured JSON formatter for production logging.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
        }
        
        # Add context variables
        if request_id.get():
            log_data['request_id'] = request_id.get()
        if user_id.get():  
            log_data['user_id'] = user_id.get()
        if operation.get():
            log_data['operation'] = operation.get()
        
        # Add extra fields from record
        extra_fields = getattr(record, '__dict__', {})
        for key, value in extra_fields.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 'processName',
                          'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                log_data[key] = value
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add stack info if present
        if record.stack_info:
            log_data['stack_info'] = record.stack_info
        
        return json.dumps(log_data, default=str, ensure_ascii=False)


class HumanReadableFormatter(logging.Formatter):
    """
    Human-readable formatter for development and console output.
    """
    
    def __init__(self):
        super().__init__()
        self.format_str = (
            "%(asctime)s | %(levelname)-8s | %(name)-20s | "
            "%(message)s"
        )
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record for human readability."""
        
        # Add context information if available
        context_parts = []
        if request_id.get():
            context_parts.append(f"req:{request_id.get()[:8]}")
        if user_id.get():
            context_parts.append(f"user:{user_id.get()}")
        if operation.get():
            context_parts.append(f"op:{operation.get()}")
        
        if context_parts:
            context_str = f"[{' | '.join(context_parts)}] "
            record.msg = f"{context_str}{record.msg}"
        
        formatter = logging.Formatter(self.format_str)
        formatted = formatter.format(record)
        
        # Add exception information with proper formatting
        if record.exc_info:
            formatted += "\n" + "".join(traceback.format_exception(*record.exc_info))
        
        return formatted


class FPLLogger:
    """
    Centralized logger for FPL ML System with context management.
    """
    
    def __init__(self, name: str, context: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(name)
        self.context = context or {}
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with context information."""
        
        # Merge context with kwargs
        log_kwargs = {**self.context, **kwargs}
        
        # Create extra dict for structured logging
        extra = {key: value for key, value in log_kwargs.items() 
                if key not in ['exc_info', 'stack_info']}
        
        self.logger.log(level, message, extra=extra, **{
            k: v for k, v in log_kwargs.items() 
            if k in ['exc_info', 'stack_info']
        })
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        kwargs['exc_info'] = True
        self._log_with_context(logging.ERROR, message, **kwargs)


def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    structured: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_console: bool = True
) -> None:
    """
    Setup comprehensive logging configuration for production.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        structured: Use structured JSON logging for files
        max_file_size: Maximum size for log files before rotation
        backup_count: Number of backup files to keep
        enable_console: Enable console logging
    """
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler for development/debugging
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(HumanReadableFormatter())
        root_logger.addHandler(console_handler)
    
    # Main application log file
    app_handler = logging.handlers.RotatingFileHandler(
        log_path / "fpl_system.log",
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    app_handler.setLevel(logging.DEBUG)
    
    if structured:
        app_handler.setFormatter(StructuredFormatter())
    else:
        app_handler.setFormatter(HumanReadableFormatter())
    
    root_logger.addHandler(app_handler)
    
    # Error-only log file
    error_handler = logging.handlers.RotatingFileHandler(
        log_path / "errors.log",
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(StructuredFormatter() if structured else HumanReadableFormatter())
    root_logger.addHandler(error_handler)
    
    # Performance log file
    perf_handler = logging.handlers.RotatingFileHandler(
        log_path / "performance.log",
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(StructuredFormatter())
    
    # Create performance logger
    perf_logger = logging.getLogger('performance')
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.INFO)
    perf_logger.propagate = False
    
    # ML model log file
    ml_handler = logging.handlers.RotatingFileHandler(
        log_path / "ml_models.log",
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    ml_handler.setLevel(logging.INFO)
    ml_handler.setFormatter(StructuredFormatter())
    
    # Create ML logger
    ml_logger = logging.getLogger('ml_models')
    ml_logger.addHandler(ml_handler)
    ml_logger.setLevel(logging.INFO)
    ml_logger.propagate = False
    
    # Agent operations log file
    agent_handler = logging.handlers.RotatingFileHandler(
        log_path / "agents.log",
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    agent_handler.setLevel(logging.INFO)
    agent_handler.setFormatter(StructuredFormatter())
    
    # Create agent logger
    agent_logger = logging.getLogger('agents')
    agent_logger.addHandler(agent_handler)
    agent_logger.setLevel(logging.INFO)
    agent_logger.propagate = False
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    
    # Log startup message
    logger = FPLLogger('system')
    logger.info("Logging system initialized", 
               log_level=level, 
               structured=structured,
               log_directory=str(log_path))


def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> FPLLogger:
    """
    Get a logger instance with optional context.
    
    Args:
        name: Logger name
        context: Default context to include in all log messages
        
    Returns:
        FPLLogger instance
    """
    return FPLLogger(name, context)


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(
            self.__class__.__module__ + '.' + self.__class__.__name__,
            context={'component': self.__class__.__name__}
        )


class PerformanceLogger:
    """
    Specialized logger for performance metrics and monitoring.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('performance')
    
    def log_operation_time(self, operation: str, duration: float, **kwargs):
        """Log operation timing."""
        self.logger.info(
            f"Operation completed: {operation}",
            extra={
                'operation': operation,
                'duration_seconds': duration,
                'metric_type': 'timing',
                **kwargs
            }
        )
    
    def log_ml_performance(self, model: str, metric: str, value: float, **kwargs):
        """Log ML model performance metrics."""
        self.logger.info(
            f"ML metric: {model} {metric}",
            extra={
                'model': model,
                'metric': metric,
                'value': value,
                'metric_type': 'ml_performance',
                **kwargs
            }
        )
    
    def log_api_call(self, endpoint: str, duration: float, status_code: int, **kwargs):
        """Log API call performance."""
        self.logger.info(
            f"API call: {endpoint}",
            extra={
                'endpoint': endpoint,
                'duration_seconds': duration,
                'status_code': status_code,
                'metric_type': 'api_performance',
                **kwargs
            }
        )
    
    def log_optimization_result(self, optimizer: str, solve_time: float, 
                              status: str, **kwargs):
        """Log optimization performance."""
        self.logger.info(
            f"Optimization: {optimizer}",
            extra={
                'optimizer': optimizer,
                'solve_time_seconds': solve_time,
                'status': status,
                'metric_type': 'optimization',
                **kwargs
            }
        )


# Context managers for request tracking
class LoggingContext:
    """Context manager for logging context variables."""
    
    def __init__(self, req_id: str = None, user: str = None, op: str = None):
        self.req_id = req_id
        self.user = user
        self.op = op
        self.tokens = {}
    
    def __enter__(self):
        if self.req_id:
            self.tokens['request_id'] = request_id.set(self.req_id)
        if self.user:
            self.tokens['user_id'] = user_id.set(self.user)
        if self.op:
            self.tokens['operation'] = operation.set(self.op)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for token in self.tokens.values():
            token.var.delete(token)


# Decorators for automatic logging
def log_performance(operation_name: str = None):
    """Decorator to automatically log function performance."""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            perf_logger = PerformanceLogger()
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                perf_logger.log_operation_time(op_name, duration, success=True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                perf_logger.log_operation_time(op_name, duration, success=False, error=str(e))
                raise
        
        return wrapper
    return decorator


def log_errors(logger_name: str = None):
    """Decorator to automatically log function errors."""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = get_logger(logger_name or func.__module__)
                logger.exception(
                    f"Error in {func.__name__}",
                    function=func.__name__,
                    args_count=len(args),
                    kwargs_keys=list(kwargs.keys())
                )
                raise
        
        return wrapper
    return decorator


# Initialize default logging configuration
if not logging.getLogger().handlers:
    # Only setup if not already configured
    setup_logging(
        level=os.getenv('LOG_LEVEL', 'INFO'),
        structured=os.getenv('LOG_STRUCTURED', 'true').lower() == 'true'
    )