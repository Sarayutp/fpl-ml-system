"""
Flexible provider configuration for LLM models.
Based on main_agent_reference/providers.py pattern.
"""

from typing import Optional
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.test import TestModel
from .settings import settings


def get_llm_model(model_choice: Optional[str] = None):
    """
    Get LLM model configuration based on environment variables.
    
    Args:
        model_choice: Optional override for model choice
    
    Returns:
        Configured LLM model (OpenAI or Test)
    """
    llm_choice = model_choice or settings.llm_model
    base_url = settings.llm_base_url
    api_key = settings.llm_api_key
    
    # For testing/demo without real API key, use TestModel
    if (not api_key or 
        api_key == "test-key-for-mock" or 
        settings.app_env == "test" or 
        getattr(settings, 'mock_fpl_api', False)):
        
        # Return a test model that simulates responses
        return TestModel()
    
    # Create provider based on configuration
    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    
    return OpenAIModel(llm_choice, provider=provider)


def get_model_info() -> dict:
    """
    Get information about current model configuration.
    
    Returns:
        Dictionary with model configuration info
    """
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_base_url": settings.llm_base_url,
        "fpl_team_id": settings.fpl_team_id,
        "app_env": settings.app_env,
        "debug": settings.debug,
        "database_url": settings.database_url,
        "model_cache_dir": settings.model_cache_dir,
    }


def validate_llm_configuration() -> bool:
    """
    Validate that LLM configuration is properly set.
    
    Returns:
        True if configuration is valid
    """
    try:
        # Check if we can create a model instance
        get_llm_model()
        return True
    except Exception as e:
        print(f"LLM configuration validation failed: {e}")
        return False


def validate_fpl_configuration() -> bool:
    """
    Validate that FPL configuration is properly set.
    
    Returns:
        True if FPL configuration is valid
    """
    try:
        # Check critical FPL settings
        if not settings.fpl_team_id or settings.fpl_team_id <= 0:
            print("FPL team ID is not properly configured")
            return False
        
        if not settings.fpl_api_base_url:
            print("FPL API base URL is not configured")
            return False
        
        return True
    except Exception as e:
        print(f"FPL configuration validation failed: {e}")
        return False


def validate_all_configuration() -> bool:
    """
    Validate all system configuration.
    
    Returns:
        True if all configuration is valid
    """
    llm_valid = validate_llm_configuration()
    fpl_valid = validate_fpl_configuration()
    
    return llm_valid and fpl_valid