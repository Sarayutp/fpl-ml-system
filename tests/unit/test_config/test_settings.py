"""
Unit tests for configuration management (settings.py).
Target: 90%+ coverage with comprehensive validation testing.
"""

import pytest
import os
from unittest.mock import patch, mock_open
from pydantic import ValidationError

from src.config.settings import FPLSettings, get_settings


@pytest.mark.unit
class TestFPLSettings:
    """Comprehensive unit tests for FPLSettings configuration."""
    
    def test_settings_initialization_with_defaults(self):
        """Test settings initialization with default values."""
        with patch.dict('os.environ', {'LLM_API_KEY': 'test-key', 'FPL_TEAM_ID': '12345'}):
            settings = FPLSettings()
            
            # Test default values
            assert settings.llm_provider == "openai"
            assert settings.llm_model == "gpt-4"
            assert settings.llm_base_url == "https://api.openai.com/v1"
            assert settings.fpl_api_base_url == "https://fantasy.premierleague.com/api"
            assert settings.database_url == "sqlite:///data/fpl.db"
            assert settings.model_cache_dir == "data/models"
            assert settings.xgboost_n_estimators == 100
            assert settings.xgboost_max_depth == 6
            assert settings.xgboost_learning_rate == 0.1
            assert settings.optimization_timeout == 300
            assert settings.max_players_per_team == 3
            assert settings.total_budget == 100.0
            assert settings.app_env == "development"
            assert settings.log_level == "INFO"
            assert settings.debug is False
            assert settings.cache_ttl == 900
            assert settings.enable_caching is True
            assert settings.notification_enabled is False
    
    def test_settings_initialization_with_environment_variables(self):
        """Test settings initialization with environment variables."""
        test_env = {
            'LLM_API_KEY': 'custom-openai-key',
            'FPL_TEAM_ID': '987654',
            'LLM_PROVIDER': 'anthropic',
            'LLM_MODEL': 'claude-3',
            'DATABASE_URL': 'postgresql://user:pass@localhost/fpl',
            'XGBOOST_N_ESTIMATORS': '200',
            'XGBOOST_MAX_DEPTH': '8',
            'XGBOOST_LEARNING_RATE': '0.05',
            'OPTIMIZATION_TIMEOUT': '600',
            'TOTAL_BUDGET': '105.0',
            'LOG_LEVEL': 'DEBUG',
            'DEBUG': 'true',
            'CACHE_TTL': '1800'
        }
        
        with patch.dict('os.environ', test_env, clear=False):
            settings = FPLSettings()
            
            assert settings.llm_api_key == 'custom-openai-key'
            assert settings.fpl_team_id == 987654
            assert settings.llm_provider == 'anthropic'
            assert settings.llm_model == 'claude-3'
            assert settings.database_url == 'postgresql://user:pass@localhost/fpl'
            assert settings.xgboost_n_estimators == 200
            assert settings.xgboost_max_depth == 8
            assert settings.xgboost_learning_rate == 0.05
            assert settings.optimization_timeout == 600
            assert settings.total_budget == 105.0
            assert settings.log_level == 'DEBUG'
            assert settings.debug is True
            assert settings.cache_ttl == 1800
    
    def test_settings_validation_empty_api_key(self):
        """Test validation fails for empty LLM API key."""
        with pytest.raises(ValidationError, match="LLM API key cannot be empty"):
            with patch.dict('os.environ', {'LLM_API_KEY': '', 'FPL_TEAM_ID': '12345'}):
                FPLSettings()
    
    def test_settings_validation_whitespace_api_key(self):
        """Test validation fails for whitespace-only API key."""
        with pytest.raises(ValidationError, match="LLM API key cannot be empty"):
            with patch.dict('os.environ', {'LLM_API_KEY': '   ', 'FPL_TEAM_ID': '12345'}):
                FPLSettings()
    
    def test_settings_validation_invalid_fpl_team_id(self):
        """Test validation fails for invalid FPL team ID."""
        with pytest.raises(ValidationError, match="FPL team ID must be a positive integer"):
            with patch.dict('os.environ', {'LLM_API_KEY': 'test-key', 'FPL_TEAM_ID': '0'}):
                FPLSettings()
        
        with pytest.raises(ValidationError, match="FPL team ID must be a positive integer"):
            with patch.dict('os.environ', {'LLM_API_KEY': 'test-key', 'FPL_TEAM_ID': '-1'}):
                FPLSettings()
    
    def test_settings_validation_invalid_budget(self):
        """Test validation fails for invalid total budget."""
        with pytest.raises(ValidationError, match="Total budget must be between £90M and £110M"):
            with patch.dict('os.environ', {
                'LLM_API_KEY': 'test-key', 
                'FPL_TEAM_ID': '12345', 
                'TOTAL_BUDGET': '89.0'
            }):
                FPLSettings()
        
        with pytest.raises(ValidationError, match="Total budget must be between £90M and £110M"):
            with patch.dict('os.environ', {
                'LLM_API_KEY': 'test-key', 
                'FPL_TEAM_ID': '12345', 
                'TOTAL_BUDGET': '111.0'
            }):
                FPLSettings()
    
    def test_settings_validation_invalid_learning_rate(self):
        """Test validation fails for invalid learning rate."""
        with pytest.raises(ValidationError, match="Learning rate must be between 0.01 and 1.0"):
            with patch.dict('os.environ', {
                'LLM_API_KEY': 'test-key', 
                'FPL_TEAM_ID': '12345', 
                'XGBOOST_LEARNING_RATE': '0.005'
            }):
                FPLSettings()
        
        with pytest.raises(ValidationError, match="Learning rate must be between 0.01 and 1.0"):
            with patch.dict('os.environ', {
                'LLM_API_KEY': 'test-key', 
                'FPL_TEAM_ID': '12345', 
                'XGBOOST_LEARNING_RATE': '1.5'
            }):
                FPLSettings()
    
    def test_settings_optional_fields(self):
        """Test optional fields can be None."""
        with patch.dict('os.environ', {'LLM_API_KEY': 'test-key', 'FPL_TEAM_ID': '12345'}):
            settings = FPLSettings()
            
            assert settings.fpl_email is None
            assert settings.fpl_password is None
            assert settings.llm_base_url is not None  # Has default
            assert settings.email_host is None
            assert settings.email_port is None
            assert settings.email_username is None
            assert settings.email_password is None
    
    def test_settings_with_notification_config(self):
        """Test settings with notification configuration."""
        test_env = {
            'LLM_API_KEY': 'test-key',
            'FPL_TEAM_ID': '12345',
            'NOTIFICATION_ENABLED': 'true',
            'EMAIL_HOST': 'smtp.gmail.com',
            'EMAIL_PORT': '587',
            'EMAIL_USERNAME': 'test@example.com',
            'EMAIL_PASSWORD': 'email_password'
        }
        
        with patch.dict('os.environ', test_env, clear=False):
            settings = FPLSettings()
            
            assert settings.notification_enabled is True
            assert settings.email_host == 'smtp.gmail.com'
            assert settings.email_port == 587
            assert settings.email_username == 'test@example.com'
            assert settings.email_password == 'email_password'
    
    def test_settings_case_insensitive(self):
        """Test settings are case insensitive."""
        test_env = {
            'llm_api_key': 'lower-case-key',  # lowercase
            'FPL_TEAM_ID': '12345',
            'llm_provider': 'anthropic',       # lowercase
            'DEBUG': 'True'                    # uppercase boolean
        }
        
        with patch.dict('os.environ', test_env, clear=False):
            settings = FPLSettings()
            
            assert settings.llm_api_key == 'lower-case-key'
            assert settings.llm_provider == 'anthropic'
            assert settings.debug is True
    
    def test_settings_extra_fields_ignored(self):
        """Test extra fields are ignored due to extra='ignore' config."""
        test_env = {
            'LLM_API_KEY': 'test-key',
            'FPL_TEAM_ID': '12345',
            'UNKNOWN_FIELD': 'should_be_ignored',
            'ANOTHER_RANDOM_FIELD': '42'
        }
        
        with patch.dict('os.environ', test_env, clear=False):
            # Should not raise an error due to unknown fields
            settings = FPLSettings()
            
            assert settings.llm_api_key == 'test-key'
            assert settings.fpl_team_id == 12345
            # Unknown fields should not be accessible
            assert not hasattr(settings, 'unknown_field')
            assert not hasattr(settings, 'another_random_field')


@pytest.mark.unit
class TestGetSettings:
    """Unit tests for get_settings function."""
    
    def test_get_settings_success(self):
        """Test successful settings retrieval."""
        with patch.dict('os.environ', {'LLM_API_KEY': 'test-key', 'FPL_TEAM_ID': '12345'}):
            settings = get_settings()
            
            assert isinstance(settings, FPLSettings)
            assert settings.llm_api_key == 'test-key'
            assert settings.fpl_team_id == 12345
    
    def test_get_settings_missing_api_key_error(self):
        """Test get_settings raises informative error for missing API key."""
        with patch.dict('os.environ', {'FPL_TEAM_ID': '12345'}, clear=False):
            with pytest.raises(ValueError, match="Failed to load settings"):
                get_settings()
    
    def test_get_settings_missing_team_id_error(self):
        """Test get_settings raises informative error for missing team ID."""
        with patch.dict('os.environ', {'LLM_API_KEY': 'test-key'}, clear=False):
            with pytest.raises(ValueError, match="Failed to load settings"):
                get_settings()
    
    def test_get_settings_error_message_contains_hints(self):
        """Test error messages contain helpful hints."""
        # Test API key hint
        with patch.dict('os.environ', {}, clear=True):
            try:
                get_settings()
                assert False, "Should have raised ValueError"
            except ValueError as e:
                error_msg = str(e)
                assert "Make sure to set LLM_API_KEY in your .env file" in error_msg


@pytest.mark.unit
class TestSettingsIntegration:
    """Integration tests for settings with .env file loading."""
    
    @patch("builtins.open", new_callable=mock_open, read_data="LLM_API_KEY=env_file_key\nFPL_TEAM_ID=67890")
    @patch("os.path.exists", return_value=True)
    def test_settings_loads_from_env_file(self, mock_exists, mock_file):
        """Test settings loads from .env file."""
        # Clear environment to test .env file loading
        with patch.dict('os.environ', {}, clear=True):
            with patch('dotenv.load_dotenv') as mock_load_dotenv:
                # Import fresh to trigger .env loading
                import importlib
                import src.config.settings
                importlib.reload(src.config.settings)
                
                mock_load_dotenv.assert_called_once()
    
    def test_settings_environment_variables_override_env_file(self):
        """Test environment variables take precedence over .env file."""
        with patch.dict('os.environ', {
            'LLM_API_KEY': 'env_var_key',
            'FPL_TEAM_ID': '54321'
        }, clear=False):
            settings = FPLSettings()
            
            # Environment variables should take precedence
            assert settings.llm_api_key == 'env_var_key'
            assert settings.fpl_team_id == 54321
    
    def test_settings_model_config(self):
        """Test model configuration settings."""
        with patch.dict('os.environ', {'LLM_API_KEY': 'test-key', 'FPL_TEAM_ID': '12345'}):
            settings = FPLSettings()
            
            # Test model_config attributes
            assert settings.model_config['case_sensitive'] is False
            assert settings.model_config['extra'] == 'ignore'
            assert settings.model_config['env_file'] == '.env'
            assert settings.model_config['env_file_encoding'] == 'utf-8'


@pytest.mark.unit 
class TestSettingsValidationEdgeCases:
    """Edge case tests for settings validation."""
    
    def test_settings_boundary_values_valid(self):
        """Test boundary values that should be valid."""
        test_env = {
            'LLM_API_KEY': 'test-key',
            'FPL_TEAM_ID': '1',  # Minimum valid value
            'TOTAL_BUDGET': '90.0',  # Minimum valid budget
            'XGBOOST_LEARNING_RATE': '0.01'  # Minimum valid learning rate
        }
        
        with patch.dict('os.environ', test_env, clear=False):
            settings = FPLSettings()
            
            assert settings.fpl_team_id == 1
            assert settings.total_budget == 90.0
            assert settings.xgboost_learning_rate == 0.01
    
    def test_settings_boundary_values_invalid(self):
        """Test boundary values that should be invalid."""
        # Test budget boundary
        with pytest.raises(ValidationError):
            with patch.dict('os.environ', {
                'LLM_API_KEY': 'test-key',
                'FPL_TEAM_ID': '12345',
                'TOTAL_BUDGET': '89.99'  # Just below minimum
            }):
                FPLSettings()
        
        # Test learning rate boundary
        with pytest.raises(ValidationError):
            with patch.dict('os.environ', {
                'LLM_API_KEY': 'test-key',
                'FPL_TEAM_ID': '12345',
                'XGBOOST_LEARNING_RATE': '0.009'  # Just below minimum
            }):
                FPLSettings()
    
    def test_settings_type_coercion(self):
        """Test automatic type coercion from environment variables."""
        test_env = {
            'LLM_API_KEY': 'test-key',
            'FPL_TEAM_ID': '12345',  # String to int
            'DEBUG': 'true',  # String to bool
            'CACHE_TTL': '1800',  # String to int
            'TOTAL_BUDGET': '95.5',  # String to float
            'XGBOOST_LEARNING_RATE': '0.15'  # String to float
        }
        
        with patch.dict('os.environ', test_env, clear=False):
            settings = FPLSettings()
            
            assert isinstance(settings.fpl_team_id, int)
            assert settings.fpl_team_id == 12345
            assert isinstance(settings.debug, bool)
            assert settings.debug is True
            assert isinstance(settings.cache_ttl, int)
            assert settings.cache_ttl == 1800
            assert isinstance(settings.total_budget, float)
            assert settings.total_budget == 95.5
            assert isinstance(settings.xgboost_learning_rate, float)
            assert settings.xgboost_learning_rate == 0.15
    
    def test_settings_field_descriptions(self):
        """Test that fields have proper descriptions."""
        with patch.dict('os.environ', {'LLM_API_KEY': 'test-key', 'FPL_TEAM_ID': '12345'}):
            settings = FPLSettings()
            
            # Get field info from model
            field_info = settings.model_fields
            
            assert 'description' in field_info['llm_provider']
            assert 'description' in field_info['fpl_team_id']
            assert 'description' in field_info['database_url']
            assert field_info['fpl_team_id']['description'] == "Your FPL team ID"
            assert field_info['llm_provider']['description'] == "LLM provider"