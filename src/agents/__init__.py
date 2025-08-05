"""
Agents package for FPL ML System.
Following main_agent_reference patterns for multi-agent orchestration.
"""

from .fpl_manager import (
    fpl_manager_agent,
    FPLManagerDependencies,
    create_fpl_manager_agent
)

from .data_pipeline import (
    data_pipeline_agent,
    DataPipelineDependencies,
    create_data_pipeline_agent
)

from .ml_prediction import (
    ml_prediction_agent,
    MLPredictionDependencies,
    create_ml_prediction_agent
)

from .transfer_advisor import (
    transfer_advisor_agent,
    TransferAdvisorDependencies,
    create_transfer_advisor_agent
)

__all__ = [
    # FPL Manager Agent
    "fpl_manager_agent",
    "FPLManagerDependencies", 
    "create_fpl_manager_agent",
    
    # Data Pipeline Agent
    "data_pipeline_agent",
    "DataPipelineDependencies",
    "create_data_pipeline_agent",
    
    # ML Prediction Agent
    "ml_prediction_agent",
    "MLPredictionDependencies",
    "create_ml_prediction_agent",
    
    # Transfer Advisor Agent
    "transfer_advisor_agent",
    "TransferAdvisorDependencies",
    "create_transfer_advisor_agent",
]