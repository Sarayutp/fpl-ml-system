"""
CLI package for FPL ML System.
Comprehensive command-line interface with 30+ commands using Click and Rich.
"""

from .main import cli
from .commands import (
    team_commands,
    transfer_commands,
    player_commands,
    prediction_commands,
    data_commands,
    analysis_commands
)

__all__ = [
    "cli",
    "team_commands",
    "transfer_commands", 
    "player_commands",
    "prediction_commands",
    "data_commands",
    "analysis_commands"
]