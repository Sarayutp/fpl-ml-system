"""
FPL ML System configuration management using pydantic-settings.
Follows main_agent_reference/settings.py patterns.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, ConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class FPLSettings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # LLM Configuration (following main_agent_reference pattern)
    llm_provider: str = Field(default="openai", description="LLM provider")
    llm_api_key: str = Field(..., description="API key for the LLM provider")
    llm_model: str = Field(default="gpt-4", description="Model name to use")
    llm_base_url: Optional[str] = Field(
        default="https://api.openai.com/v1", 
        description="Base URL for the LLM API"
    )
    
    # FPL-specific Configuration
    fpl_team_id: int = Field(..., description="Your FPL team ID")
    fpl_email: Optional[str] = Field(None, description="FPL account email for authentication")
    fpl_password: Optional[str] = Field(None, description="FPL account password for authentication")
    fpl_api_base_url: str = Field(
        default="https://fantasy.premierleague.com/api",
        description="FPL API base URL"
    )
    
    # Database Configuration  
    database_url: str = Field(
        default="sqlite:///data/fpl.db",
        description="Database connection URL"
    )
    
    # ML Model Configuration
    model_cache_dir: str = Field(
        default="data/models",
        description="Directory to store trained ML models"
    )
    xgboost_n_estimators: int = Field(default=100, description="XGBoost n_estimators")
    xgboost_max_depth: int = Field(default=6, description="XGBoost max_depth")
    xgboost_learning_rate: float = Field(default=0.1, description="XGBoost learning_rate")
    
    # Optimization Configuration
    optimization_timeout: int = Field(default=300, description="PuLP optimization timeout in seconds")
    max_players_per_team: int = Field(default=3, description="Max players from same team in FPL")
    total_budget: float = Field(default=100.0, description="Total FPL budget in millions")
    
    # Application Configuration
    app_env: str = Field(default="development", description="Application environment")
    log_level: str = Field(default="INFO", description="Logging level")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Cache Configuration
    cache_ttl: int = Field(default=900, description="Cache TTL in seconds (15 minutes)")
    enable_caching: bool = Field(default=True, description="Enable caching")
    
    # Notification Configuration
    notification_enabled: bool = Field(default=False, description="Enable notifications")
    email_host: Optional[str] = Field(None, description="SMTP email host")
    email_port: Optional[int] = Field(None, description="SMTP email port")
    email_username: Optional[str] = Field(None, description="SMTP email username")
    email_password: Optional[str] = Field(None, description="SMTP email password")
    
    @field_validator("llm_api_key", "fpl_team_id")
    @classmethod
    def validate_required_fields(cls, v, info):
        """Ensure critical fields are not empty."""
        field_name = info.field_name
        
        if field_name == "llm_api_key" and (not v or str(v).strip() == ""):
            raise ValueError("LLM API key cannot be empty")
        if field_name == "fpl_team_id" and (not v or v <= 0):
            raise ValueError("FPL team ID must be a positive integer")
        return v
    
    @field_validator("total_budget")
    @classmethod
    def validate_budget(cls, v):
        """Ensure budget is within valid FPL range."""
        if not (90.0 <= v <= 110.0):
            raise ValueError("Total budget must be between £90M and £110M")
        return v
    
    @field_validator("xgboost_learning_rate")
    @classmethod
    def validate_learning_rate(cls, v):
        """Ensure learning rate is valid."""
        if not (0.01 <= v <= 1.0):
            raise ValueError("Learning rate must be between 0.01 and 1.0")
        return v


# Global settings instance with error handling
try:
    settings = FPLSettings()
except Exception as e:
    # For testing, create settings with dummy values
    import os
    os.environ.setdefault("LLM_API_KEY", "test_key")
    os.environ.setdefault("FPL_TEAM_ID", "123456")
    settings = FPLSettings()
    print(f"Warning: Using test settings due to configuration error: {e}")


def get_settings() -> FPLSettings:
    """Get application settings with proper error handling."""
    try:
        return FPLSettings()
    except Exception as e:
        error_msg = f"Failed to load settings: {e}"
        if "llm_api_key" in str(e).lower():
            error_msg += "\nMake sure to set LLM_API_KEY in your .env file"
        if "fpl_team_id" in str(e).lower():
            error_msg += "\nMake sure to set FPL_TEAM_ID in your .env file"
        raise ValueError(error_msg) from e