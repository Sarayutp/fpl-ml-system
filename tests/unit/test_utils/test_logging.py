"""
Unit tests for logging utility (utils/logging.py).
Target: 90%+ coverage with comprehensive logging functionality testing.
"""

import pytest
import json
import logging
import tempfile
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
from datetime import datetime
import sys
import traceback

from src.utils.logging import (
    StructuredFormatter, HumanReadableFormatter, FPLLogger,
    setup_logging, get_logger, LoggerMixin, PerformanceLogger,
    LoggingContext, log_performance, log_errors,
    request_id, user_id, operation
)


@pytest.mark.unit
class TestStructuredFormatter:
    """Unit tests for StructuredFormatter."""
    
    def test_structured_formatter_basic(self):
        """Test basic structured formatting."""
        formatter = StructuredFormatter()
        
        record = logging.LogRecord(
            name='test_logger',
            level=logging.INFO,
            pathname='/test/path.py',
            lineno=100,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data['level'] == 'INFO'
        assert log_data['logger'] == 'test_logger'
        assert log_data['message'] == 'Test message'
        assert log_data['module'] == 'path'
        assert log_data['function'] == '<module>'
        assert log_data['line'] == 100
        assert 'timestamp' in log_data
        assert 'thread' in log_data
        assert 'thread_name' in log_data
    
    def test_structured_formatter_with_context(self):
        """Test structured formatting with context variables."""
        formatter = StructuredFormatter()
        
        # Set context variables
        req_token = request_id.set('req-123')
        user_token = user_id.set('user-456')
        op_token = operation.set('test_operation')
        
        try:
            record = logging.LogRecord(
                name='test_logger',
                level=logging.INFO,
                pathname='/test/path.py',
                lineno=100,
                msg='Test message with context',
                args=(),
                exc_info=None
            )
            
            formatted = formatter.format(record)
            log_data = json.loads(formatted)
            
            assert log_data['request_id'] == 'req-123'
            assert log_data['user_id'] == 'user-456'
            assert log_data['operation'] == 'test_operation'
            assert log_data['message'] == 'Test message with context'
        finally:
            # Clean up context
            request_id.reset(req_token)
            user_id.reset(user_token)
            operation.reset(op_token)
    
    def test_structured_formatter_with_exception(self):
        """Test structured formatting with exception information."""
        formatter = StructuredFormatter()
        
        try:
            raise ValueError("Test error")
        except ValueError:
            exc_info = sys.exc_info()
            
            record = logging.LogRecord(
                name='test_logger',
                level=logging.ERROR,
                pathname='/test/path.py',
                lineno=100,
                msg='Error occurred',
                args=(),
                exc_info=exc_info
            )
            
            formatted = formatter.format(record)
            log_data = json.loads(formatted)
            
            assert log_data['level'] == 'ERROR'
            assert log_data['message'] == 'Error occurred'
            assert 'exception' in log_data
            assert log_data['exception']['type'] == 'ValueError'
            assert log_data['exception']['message'] == 'Test error'
            assert isinstance(log_data['exception']['traceback'], list)
    
    def test_structured_formatter_with_extra_fields(self):
        """Test structured formatting with extra fields."""
        formatter = StructuredFormatter()
        
        record = logging.LogRecord(
            name='test_logger',
            level=logging.INFO,
            pathname='/test/path.py',
            lineno=100,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        # Add extra fields
        record.custom_field = 'custom_value'
        record.duration = 1.23
        record.success = True
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data['custom_field'] == 'custom_value'
        assert log_data['duration'] == 1.23
        assert log_data['success'] is True
    
    def test_structured_formatter_excludes_internal_fields(self):
        """Test that internal logging fields are excluded from output."""
        formatter = StructuredFormatter()
        
        record = logging.LogRecord(
            name='test_logger',
            level=logging.INFO,
            pathname='/test/path.py',
            lineno=100,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        # These fields should not be in the output
        assert 'args' not in log_data
        assert 'pathname' not in log_data
        assert 'filename' not in log_data
        assert 'created' not in log_data
        assert 'msecs' not in log_data


@pytest.mark.unit
class TestHumanReadableFormatter:
    """Unit tests for HumanReadableFormatter."""
    
    def test_human_readable_formatter_basic(self):
        """Test basic human-readable formatting."""
        formatter = HumanReadableFormatter()
        
        record = logging.LogRecord(
            name='test_logger',
            level=logging.INFO,
            pathname='/test/path.py',
            lineno=100,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        
        assert 'INFO' in formatted
        assert 'test_logger' in formatted
        assert 'Test message' in formatted
        assert '|' in formatted  # Check separator exists
    
    def test_human_readable_formatter_with_context(self):
        """Test human-readable formatting with context variables."""
        formatter = HumanReadableFormatter()
        
        # Set context variables
        req_token = request_id.set('req-123')
        user_token = user_id.set('user-456')
        op_token = operation.set('test_operation')
        
        try:
            record = logging.LogRecord(
                name='test_logger',
                level=logging.INFO,
                pathname='/test/path.py',
                lineno=100,
                msg='Test message',
                args=(),
                exc_info=None
            )
            
            formatted = formatter.format(record)
            
            assert 'req:req-123' in formatted
            assert 'user:user-456' in formatted
            assert 'op:test_operation' in formatted
            assert 'Test message' in formatted
        finally:
            # Clean up context
            request_id.reset(req_token)
            user_id.reset(user_token)
            operation.reset(op_token)
    
    def test_human_readable_formatter_with_exception(self):
        """Test human-readable formatting with exception."""
        formatter = HumanReadableFormatter()
        
        try:
            raise RuntimeError("Test runtime error")
        except RuntimeError:
            exc_info = sys.exc_info()
            
            record = logging.LogRecord(
                name='test_logger',
                level=logging.ERROR,
                pathname='/test/path.py',
                lineno=100,
                msg='Error occurred',
                args=(),
                exc_info=exc_info
            )
            
            formatted = formatter.format(record)
            
            assert 'ERROR' in formatted
            assert 'Error occurred' in formatted
            assert 'RuntimeError' in formatted
            assert 'Test runtime error' in formatted
            assert 'Traceback' in formatted


@pytest.mark.unit
class TestFPLLogger:
    """Unit tests for FPLLogger class."""
    
    def test_fpl_logger_initialization(self):
        """Test FPLLogger initialization."""
        logger = FPLLogger('test_logger')
        
        assert logger.logger.name == 'test_logger'
        assert logger.context == {}
        
        # Test with context
        context = {'component': 'test', 'version': '1.0'}
        logger_with_context = FPLLogger('test_logger_context', context)
        
        assert logger_with_context.context == context
    
    def test_fpl_logger_log_methods(self):
        """Test all FPLLogger logging methods."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            logger = FPLLogger('test_logger')
            
            # Test each log level
            logger.debug('Debug message', custom_field='debug')
            mock_logger.log.assert_called_with(
                logging.DEBUG, 'Debug message', 
                extra={'custom_field': 'debug'}
            )
            
            logger.info('Info message', custom_field='info')
            mock_logger.log.assert_called_with(
                logging.INFO, 'Info message', 
                extra={'custom_field': 'info'}
            )
            
            logger.warning('Warning message', custom_field='warning')
            mock_logger.log.assert_called_with(
                logging.WARNING, 'Warning message', 
                extra={'custom_field': 'warning'}
            )
            
            logger.error('Error message', custom_field='error')
            mock_logger.log.assert_called_with(
                logging.ERROR, 'Error message', 
                extra={'custom_field': 'error'}
            )
            
            logger.critical('Critical message', custom_field='critical')
            mock_logger.log.assert_called_with(
                logging.CRITICAL, 'Critical message', 
                extra={'custom_field': 'critical'}
            )
    
    def test_fpl_logger_exception_method(self):
        """Test FPLLogger exception method."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            logger = FPLLogger('test_logger')
            logger.exception('Exception occurred', custom_field='exception')
            
            mock_logger.log.assert_called_with(
                logging.ERROR, 'Exception occurred', 
                extra={'custom_field': 'exception'}, 
                exc_info=True
            )
    
    def test_fpl_logger_with_context_merge(self):
        """Test FPLLogger merges instance context with method kwargs."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            context = {'component': 'test', 'version': '1.0'}
            logger = FPLLogger('test_logger', context)
            
            logger.info('Test message', operation='test_op', duration=1.5)
            
            # Should merge context and kwargs
            mock_logger.log.assert_called_with(
                logging.INFO, 'Test message', 
                extra={
                    'component': 'test',
                    'version': '1.0',
                    'operation': 'test_op',
                    'duration': 1.5
                }
            )


@pytest.mark.unit
class TestSetupLogging:
    """Unit tests for setup_logging function."""
    
    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('logging.getLogger') as mock_get_logger:
                mock_root_logger = MagicMock()
                mock_get_logger.return_value = mock_root_logger
                
                setup_logging(
                    level="INFO",
                    log_dir=temp_dir,
                    structured=True,
                    enable_console=True
                )
                
                # Check that root logger was configured
                mock_root_logger.setLevel.assert_called_with(logging.INFO)
                mock_root_logger.handlers.clear.assert_called_once()
                
                # Check that handlers were added
                assert mock_root_logger.addHandler.call_count >= 3  # Console, app, error handlers
    
    @patch('pathlib.Path.mkdir')
    def test_setup_logging_creates_log_directory(self, mock_mkdir):
        """Test that setup_logging creates log directory."""
        log_dir = "/test/logs"
        
        with patch('logging.getLogger') as mock_get_logger:
            mock_root_logger = MagicMock()
            mock_get_logger.return_value = mock_root_logger
            
            setup_logging(log_dir=log_dir)
            
            # Check that directory creation was attempted
            mock_mkdir.assert_called_with(parents=True, exist_ok=True)
    
    def test_setup_logging_different_levels(self):
        """Test setup_logging with different log levels."""
        test_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in test_levels:
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch('logging.getLogger') as mock_get_logger:
                    mock_root_logger = MagicMock()
                    mock_get_logger.return_value = mock_root_logger
                    
                    setup_logging(level=level, log_dir=temp_dir)
                    
                    expected_level = getattr(logging, level.upper())
                    mock_root_logger.setLevel.assert_called_with(expected_level)
    
    def test_setup_logging_structured_vs_human_readable(self):
        """Test setup_logging with structured vs human-readable formatting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test structured logging
            with patch('logging.getLogger') as mock_get_logger:
                mock_root_logger = MagicMock()
                mock_get_logger.return_value = mock_root_logger
                
                setup_logging(log_dir=temp_dir, structured=True)
                
                # Verify structured formatter was used (by checking handler setup)
                assert mock_root_logger.addHandler.called
            
            # Test human-readable logging
            with patch('logging.getLogger') as mock_get_logger:
                mock_root_logger = MagicMock()
                mock_get_logger.return_value = mock_root_logger
                
                setup_logging(log_dir=temp_dir, structured=False)
                
                assert mock_root_logger.addHandler.called
    
    def test_setup_logging_console_disabled(self):
        """Test setup_logging with console logging disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('logging.getLogger') as mock_get_logger:
                mock_root_logger = MagicMock()
                mock_get_logger.return_value = mock_root_logger
                
                setup_logging(log_dir=temp_dir, enable_console=False)
                
                # Should still add file handlers but not console handler
                assert mock_root_logger.addHandler.called


@pytest.mark.unit
class TestGetLogger:
    """Unit tests for get_logger function."""
    
    def test_get_logger_basic(self):
        """Test basic get_logger functionality."""
        logger = get_logger('test_logger')
        
        assert isinstance(logger, FPLLogger)
        assert logger.logger.name == 'test_logger'
        assert logger.context == {}
    
    def test_get_logger_with_context(self):
        """Test get_logger with context."""
        context = {'component': 'test', 'version': '1.0'}
        logger = get_logger('test_logger', context)
        
        assert isinstance(logger, FPLLogger)
        assert logger.context == context


@pytest.mark.unit
class TestLoggerMixin:
    """Unit tests for LoggerMixin class."""
    
    def test_logger_mixin_initialization(self):
        """Test LoggerMixin initialization."""
        class TestClass(LoggerMixin):
            def __init__(self):
                super().__init__()
        
        test_instance = TestClass()
        
        assert hasattr(test_instance, 'logger')
        assert isinstance(test_instance.logger, FPLLogger)
        assert 'TestClass' in test_instance.logger.logger.name
        assert test_instance.logger.context['component'] == 'TestClass'
    
    def test_logger_mixin_with_inheritance(self):
        """Test LoggerMixin with inheritance."""
        class BaseClass:
            def __init__(self, value):
                self.value = value
        
        class TestClass(LoggerMixin, BaseClass):
            def __init__(self, value):
                super().__init__(value)
        
        test_instance = TestClass('test_value')
        
        assert test_instance.value == 'test_value'
        assert hasattr(test_instance, 'logger')
        assert isinstance(test_instance.logger, FPLLogger)


@pytest.mark.unit
class TestPerformanceLogger:
    """Unit tests for PerformanceLogger class."""
    
    def test_performance_logger_initialization(self):
        """Test PerformanceLogger initialization."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            perf_logger = PerformanceLogger()
            
            mock_get_logger.assert_called_with('performance')
            assert perf_logger.logger == mock_logger
    
    def test_performance_logger_operation_time(self):
        """Test PerformanceLogger log_operation_time method."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            perf_logger = PerformanceLogger()
            perf_logger.log_operation_time('test_operation', 1.25, custom_field='custom')
            
            mock_logger.info.assert_called_with(
                'Operation completed: test_operation',
                extra={
                    'operation': 'test_operation',
                    'duration_seconds': 1.25,
                    'metric_type': 'timing',
                    'custom_field': 'custom'
                }
            )
    
    def test_performance_logger_ml_performance(self):
        """Test PerformanceLogger log_ml_performance method."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            perf_logger = PerformanceLogger()
            perf_logger.log_ml_performance('xgboost', 'mse', 0.002, epoch=10)
            
            mock_logger.info.assert_called_with(
                'ML metric: xgboost mse',
                extra={
                    'model': 'xgboost',
                    'metric': 'mse',
                    'value': 0.002,
                    'metric_type': 'ml_performance',
                    'epoch': 10
                }
            )
    
    def test_performance_logger_api_call(self):
        """Test PerformanceLogger log_api_call method."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            perf_logger = PerformanceLogger()
            perf_logger.log_api_call('/api/players', 0.45, 200, method='GET')
            
            mock_logger.info.assert_called_with(
                'API call: /api/players',
                extra={
                    'endpoint': '/api/players',
                    'duration_seconds': 0.45,
                    'status_code': 200,
                    'metric_type': 'api_performance',
                    'method': 'GET'
                }
            )
    
    def test_performance_logger_optimization_result(self):
        """Test PerformanceLogger log_optimization_result method."""
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            perf_logger = PerformanceLogger()
            perf_logger.log_optimization_result('pulp', 2.1, 'optimal', variables=15)
            
            mock_logger.info.assert_called_with(
                'Optimization: pulp',
                extra={
                    'optimizer': 'pulp',
                    'solve_time_seconds': 2.1,
                    'status': 'optimal',
                    'metric_type': 'optimization',
                    'variables': 15
                }
            )


@pytest.mark.unit
class TestLoggingContext:
    """Unit tests for LoggingContext context manager."""
    
    def test_logging_context_basic(self):
        """Test LoggingContext basic functionality."""
        with LoggingContext(req_id='req-123', user='user-456', op='test_op'):
            assert request_id.get() == 'req-123'
            assert user_id.get() == 'user-456'
            assert operation.get() == 'test_op'
        
        # Context should be cleared after exiting
        assert request_id.get() == ''
        assert user_id.get() == ''
        assert operation.get() == ''
    
    def test_logging_context_partial(self):
        """Test LoggingContext with partial context."""
        with LoggingContext(req_id='req-123'):
            assert request_id.get() == 'req-123'
            assert user_id.get() == ''  # Not set
            assert operation.get() == ''  # Not set
        
        assert request_id.get() == ''  # Cleared after exit
    
    def test_logging_context_nested(self):
        """Test nested LoggingContext usage."""
        with LoggingContext(req_id='req-outer'):
            assert request_id.get() == 'req-outer'
            
            with LoggingContext(req_id='req-inner', user='user-123'):
                assert request_id.get() == 'req-inner'
                assert user_id.get() == 'user-123'
            
            # Should restore outer context
            assert request_id.get() == 'req-outer'
            assert user_id.get() == ''  # Inner context cleared
        
        assert request_id.get() == ''  # All cleared


@pytest.mark.unit
class TestLoggingDecorators:
    """Unit tests for logging decorators."""
    
    @patch('src.utils.logging.PerformanceLogger')
    def test_log_performance_decorator_success(self, mock_perf_logger_class):
        """Test log_performance decorator on successful function."""
        mock_perf_logger = MagicMock()
        mock_perf_logger_class.return_value = mock_perf_logger
        
        @log_performance('test_operation')
        def test_function(x, y):
            return x + y
        
        with patch('time.time', side_effect=[0.0, 1.5]):  # Start and end times
            result = test_function(2, 3)
        
        assert result == 5
        mock_perf_logger.log_operation_time.assert_called_with(
            'test_operation', 1.5, success=True
        )
    
    @patch('src.utils.logging.PerformanceLogger')
    def test_log_performance_decorator_error(self, mock_perf_logger_class):
        """Test log_performance decorator on function that raises error."""
        mock_perf_logger = MagicMock()
        mock_perf_logger_class.return_value = mock_perf_logger
        
        @log_performance('test_operation')
        def test_function():
            raise ValueError('Test error')
        
        with patch('time.time', side_effect=[0.0, 0.5]):  # Start and end times
            with pytest.raises(ValueError, match='Test error'):
                test_function()
        
        mock_perf_logger.log_operation_time.assert_called_with(
            'test_operation', 0.5, success=False, error='Test error'
        )
    
    @patch('src.utils.logging.PerformanceLogger')
    def test_log_performance_decorator_auto_name(self, mock_perf_logger_class):
        """Test log_performance decorator with automatic operation naming."""
        mock_perf_logger = MagicMock()
        mock_perf_logger_class.return_value = mock_perf_logger
        
        @log_performance()
        def test_function():
            return 'test_result'
        
        with patch('time.time', side_effect=[0.0, 0.1]):
            result = test_function()
        
        assert result == 'test_result'
        # Should use function module and name
        expected_op_name = f"{test_function.__module__}.{test_function.__name__}"
        mock_perf_logger.log_operation_time.assert_called_with(
            expected_op_name, 0.1, success=True
        )
    
    @patch('src.utils.logging.get_logger')
    def test_log_errors_decorator_success(self, mock_get_logger):
        """Test log_errors decorator on successful function."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        @log_errors('test_logger')
        def test_function():
            return 'success'
        
        result = test_function()
        
        assert result == 'success'
        # Logger should not be called for successful execution
        mock_logger.exception.assert_not_called()
    
    @patch('src.utils.logging.get_logger')
    def test_log_errors_decorator_error(self, mock_get_logger):
        """Test log_errors decorator on function that raises error."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        @log_errors('test_logger')
        def test_function(x, y, z=None):
            raise RuntimeError('Test runtime error')
        
        with pytest.raises(RuntimeError, match='Test runtime error'):
            test_function(1, 2, z='test')
        
        # Logger should be called with error details
        mock_get_logger.assert_called_with('test_logger')
        mock_logger.exception.assert_called_with(
            'Error in test_function',
            function='test_function',
            args_count=2,  # x, y
            kwargs_keys=['z']
        )
    
    @patch('src.utils.logging.get_logger')
    def test_log_errors_decorator_auto_logger(self, mock_get_logger):
        """Test log_errors decorator with automatic logger naming."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        @log_errors()  # No logger name provided
        def test_function():
            raise ValueError('Test value error')
        
        with pytest.raises(ValueError, match='Test value error'):
            test_function()
        
        # Should use function module as logger name
        mock_get_logger.assert_called_with(test_function.__module__)
        mock_logger.exception.assert_called_with(
            'Error in test_function',
            function='test_function',
            args_count=0,
            kwargs_keys=[]
        )


@pytest.mark.unit
class TestLoggingInitialization:
    """Unit tests for logging initialization."""
    
    @patch.dict('os.environ', {'LOG_LEVEL': 'DEBUG', 'LOG_STRUCTURED': 'false'})
    @patch('src.utils.logging.setup_logging')
    def test_logging_initialization_from_environment(self, mock_setup_logging):
        """Test that logging is initialized with environment variables."""
        # Re-import module to trigger initialization
        import importlib
        import src.utils.logging
        importlib.reload(src.utils.logging)
        
        # Should have called setup_logging with environment values
        mock_setup_logging.assert_called_with(level='DEBUG', structured=False)
    
    @patch('logging.getLogger')
    def test_logging_initialization_skip_if_already_configured(self, mock_get_logger):
        """Test that initialization is skipped if logger already has handlers."""
        mock_root_logger = MagicMock()
        mock_root_logger.handlers = [MagicMock()]  # Already has handlers
        mock_get_logger.return_value = mock_root_logger
        
        with patch('src.utils.logging.setup_logging') as mock_setup_logging:
            # Re-import module to trigger initialization check
            import importlib
            import src.utils.logging
            importlib.reload(src.utils.logging)
            
            # Should not call setup_logging since handlers already exist
            mock_setup_logging.assert_not_called()