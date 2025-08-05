"""
Pure tool functions package for FPL ML System.
Following main_agent_reference patterns for standalone functions.
"""

from .fpl_api import (
    get_bootstrap_data,
    get_player_data,
    get_team_picks,
    get_fixtures,
    get_live_gameweek_data,
    get_manager_history,
    batch_fetch_player_data,
)

from .ml_tools import (
    predict_player_points,
    optimize_team_selection,
    optimize_transfers,
    optimize_captain_selection,
    calculate_player_value,
    validate_model_performance,
)

__all__ = [
    # FPL API tools
    "get_bootstrap_data",
    "get_player_data", 
    "get_team_picks",
    "get_fixtures",
    "get_live_gameweek_data",
    "get_manager_history",
    "batch_fetch_player_data",
    
    # ML tools
    "predict_player_points",
    "optimize_team_selection",
    "optimize_transfers",
    "optimize_captain_selection",
    "calculate_player_value",
    "validate_model_performance",
]