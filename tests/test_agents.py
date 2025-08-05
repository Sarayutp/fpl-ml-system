"""
Agent Test Suite - Following TestModel patterns.
Tests for all Pydantic AI agents with async operation validation.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta
import json

from src.agents.fpl_manager import FPLManagerAgent
from src.agents.data_pipeline import DataPipelineAgent
from src.agents.ml_prediction import MLPredictionAgent
from src.agents.transfer_advisor import TransferAdvisorAgent
from src.models.data_models import Player, TransferSuggestion, PredictionResult


@pytest.mark.asyncio
class TestFPLManagerAgent:
    """Test suite for FPL Manager Agent following TestModel patterns."""
    
    async def test_agent_initialization(self, fpl_manager_deps):
        """Test FPL Manager agent initializes correctly."""
        agent = FPLManagerAgent()
        
        assert agent is not None
        assert hasattr(agent, 'model')
        assert hasattr(agent, 'system_prompt')
        
        # Test agent system prompt contains key instructions
        assert 'FPL expert' in agent.system_prompt
        assert 'transfer decisions' in agent.system_prompt
        assert 'optimization' in agent.system_prompt
    
    async def test_get_current_team_tool(self, fpl_manager_deps, mock_team_picks):
        """Test get_current_team tool functionality."""
        agent = FPLManagerAgent()
        
        with patch('src.data.fetchers.FPLAPIClient.get_team_picks') as mock_picks:
            mock_picks.return_value = mock_team_picks
            
            # Test tool execution
            result = await agent.run(
                "Get my current team analysis",
                deps=fpl_manager_deps
            )
            
            assert result is not None
            assert 'team' in str(result).lower()
            mock_picks.assert_called_once()
    
    async def test_optimize_team_tool(self, fpl_manager_deps, sample_players_data):
        """Test optimize_team tool with performance requirements."""
        agent = FPLManagerAgent()
        
        with patch('src.models.optimization.FPLOptimizer.optimize_team') as mock_opt:
            mock_opt.return_value = Mock(
                status='optimal',
                selected_players=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                total_cost=99.5,
                predicted_points=85.2
            )
            
            start_time = asyncio.get_event_loop().time()
            result = await agent.run(
                "Optimize my team for the next 3 gameweeks",
                deps=fpl_manager_deps
            )
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Test optimization completed within reasonable time
            assert execution_time < 10, f"Agent response took {execution_time:.2f}s, should be under 10s"
            assert result is not None
            mock_opt.assert_called_once()
    
    async def test_transfer_advice_tool(self, fpl_manager_deps):
        """Test transfer advice tool with ML integration."""
        agent = FPLManagerAgent()
        
        with patch('src.agents.ml_prediction.MLPredictionAgent.run') as mock_ml:
            mock_ml.return_value = "Player predictions indicate strong performance"
            
            with patch('src.agents.transfer_advisor.TransferAdvisorAgent.run') as mock_transfer:
                mock_transfer.return_value = "Transfer recommendation: Salah to Sterling"
                
                result = await agent.run(
                    "Give me transfer advice for this gameweek",
                    deps=fpl_manager_deps
                )
                
                assert result is not None
                assert 'transfer' in str(result).lower()
                mock_ml.assert_called()
                mock_transfer.assert_called()
    
    async def test_agent_error_handling(self, fpl_manager_deps):
        """Test agent handles errors gracefully."""
        agent = FPLManagerAgent()
        
        with patch('src.data.fetchers.FPLAPIClient.get_bootstrap_data') as mock_api:
            mock_api.side_effect = Exception("API Error")
            
            # Agent should handle errors without crashing
            result = await agent.run(
                "Get player statistics",
                deps=fpl_manager_deps
            )
            
            # Should return error information, not crash
            assert result is not None
            assert 'error' in str(result).lower() or 'unable' in str(result).lower()


@pytest.mark.asyncio
class TestDataPipelineAgent:
    """Test suite for Data Pipeline Agent following TestModel patterns."""
    
    async def test_agent_initialization(self, data_pipeline_deps):
        """Test Data Pipeline agent initializes correctly."""
        agent = DataPipelineAgent()
        
        assert agent is not None
        assert 'data quality' in agent.system_prompt
        assert 'FPL API' in agent.system_prompt
    
    async def test_fetch_bootstrap_data_tool(self, data_pipeline_deps, mock_fpl_api_response):
        """Test fetch_and_validate_bootstrap_data tool."""
        agent = DataPipelineAgent()
        
        with patch('src.data.fetchers.FPLAPIClient.get_bootstrap_data') as mock_api:
            mock_api.return_value = mock_fpl_api_response
            
            result = await agent.run(
                "Fetch and validate bootstrap data",
                deps=data_pipeline_deps
            )
            
            assert result is not None
            assert 'bootstrap' in str(result).lower()
            mock_api.assert_called_once()
    
    async def test_data_validation_quality(self, data_pipeline_deps, sample_players_data):
        """Test data validation meets quality standards."""
        agent = DataPipelineAgent()
        
        # Test data validation with quality metrics
        with patch('src.data.validators.DataValidator.validate_player_data') as mock_validator:
            mock_validator.return_value = {
                'is_valid': True,
                'completeness': 0.987,
                'accuracy': 0.992,
                'consistency': 0.978
            }
            
            result = await agent.run(
                "Validate player data quality",
                deps=data_pipeline_deps
            )
            
            assert result is not None
            mock_validator.assert_called()
    
    async def test_health_report_generation(self, data_pipeline_deps):
        """Test generate_data_health_report tool."""
        agent = DataPipelineAgent()
        
        result = await agent.run(
            "Generate comprehensive data health report",
            deps=data_pipeline_deps
        )
        
        assert result is not None
        assert 'health' in str(result).lower()
        # Should include key health metrics
        result_str = str(result).lower()
        assert any(metric in result_str for metric in ['uptime', 'data quality', 'freshness'])


@pytest.mark.asyncio 
class TestMLPredictionAgent:
    """Test suite for ML Prediction Agent following TestModel patterns."""
    
    async def test_agent_initialization(self, ml_prediction_deps):
        """Test ML Prediction agent initializes correctly."""
        agent = MLPredictionAgent()
        
        assert agent is not None
        assert 'machine learning' in agent.system_prompt.lower()
        assert 'predictions' in agent.system_prompt.lower()
    
    @pytest.mark.benchmark
    async def test_prediction_accuracy_benchmark(self, ml_prediction_deps, sample_ml_training_data, 
                                                performance_benchmarks, test_assertions):
        """Test ML predictions meet accuracy benchmarks."""
        agent = MLPredictionAgent()
        
        with patch('src.models.ml_models.PlayerPredictor') as mock_predictor:
            # Mock high-quality predictions
            mock_instance = Mock()
            mock_instance.predict_ensemble.return_value = [8.5, 6.2, 9.1, 7.8, 5.9]
            mock_instance.get_prediction_confidence.return_value = 0.85
            mock_predictor.return_value = mock_instance
            
            result = await agent.run(
                "Predict points for top 5 players next gameweek",
                deps=ml_prediction_deps
            )
            
            # Test that prediction results are returned
            assert result is not None
            assert 'predict' in str(result).lower()
            mock_instance.predict_ensemble.assert_called()
    
    async def test_captain_analysis_tool(self, ml_prediction_deps):
        """Test analyze_captain_options tool."""
        agent = MLPredictionAgent()
        
        with patch('src.models.ml_models.PlayerPredictor') as mock_predictor:
            mock_instance = Mock()
            mock_instance.predict_captain_scores.return_value = {
                'Salah': {'expected': 9.2, 'risk': 'medium'},
                'Haaland': {'expected': 8.8, 'risk': 'high'},
                'Kane': {'expected': 7.5, 'risk': 'low'}
            }
            mock_predictor.return_value = mock_instance
            
            result = await agent.run(
                "Analyze captain options for this gameweek",
                deps=ml_prediction_deps
            )
            
            assert result is not None
            assert 'captain' in str(result).lower()
            mock_instance.predict_captain_scores.assert_called()
    
    async def test_model_validation_tool(self, ml_prediction_deps, sample_ml_training_data):
        """Test validate_model_accuracy tool meets benchmarks."""
        agent = MLPredictionAgent()
        
        with patch('src.models.ml_models.PlayerPredictor.validate_model') as mock_validate:
            mock_validate.return_value = {
                'mse': 0.0025,  # Meets benchmark < 0.003
                'correlation': 0.72,  # Meets benchmark > 0.65
                'accuracy': 0.63  # Meets benchmark > 0.60
            }
            
            result = await agent.run(
                "Validate model accuracy against benchmarks",
                deps=ml_prediction_deps
            )
            
            assert result is not None
            assert 'validation' in str(result).lower()
            mock_validate.assert_called()


@pytest.mark.asyncio
class TestTransferAdvisorAgent:
    """Test suite for Transfer Advisor Agent following TestModel patterns."""
    
    async def test_agent_initialization(self, transfer_advisor_deps):
        """Test Transfer Advisor agent initializes correctly."""
        agent = TransferAdvisorAgent()
        
        assert agent is not None
        assert 'transfer' in agent.system_prompt.lower()
        assert 'optimization' in agent.system_prompt.lower()
    
    @pytest.mark.performance
    async def test_optimization_speed_benchmark(self, transfer_advisor_deps, sample_players_data,
                                              performance_benchmarks, test_assertions):
        """Test optimization meets speed benchmarks."""
        agent = TransferAdvisorAgent()
        
        with patch('src.models.optimization.FPLOptimizer.optimize_transfers') as mock_opt:
            mock_opt.return_value = Mock(
                status='optimal',
                recommended_transfers=[
                    Mock(player_out_id=1, player_in_id=50, expected_points_gain=2.3)
                ],
                solve_time=2.1  # Under 5 second benchmark
            )
            
            start_time = asyncio.get_event_loop().time()
            result = await agent.run(
                "Optimize transfers for next 4 gameweeks",
                deps=transfer_advisor_deps
            )
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Test optimization speed benchmark
            test_assertions.assert_optimization_performance(
                execution_time, 
                'optimal',
                performance_benchmarks
            )
            
            assert result is not None
            mock_opt.assert_called()
    
    async def test_single_transfer_optimization(self, transfer_advisor_deps):
        """Test optimize_single_transfer tool."""
        agent = TransferAdvisorAgent()
        
        with patch('src.models.optimization.FPLOptimizer.find_best_transfer') as mock_transfer:
            mock_transfer.return_value = Mock(
                player_out='Wilson',
                player_in='Haaland', 
                expected_gain=3.2,
                cost_change=2.5
            )
            
            result = await agent.run(
                "Find the best single transfer for my team",
                deps=transfer_advisor_deps
            )
            
            assert result is not None
            assert 'transfer' in str(result).lower()
            mock_transfer.assert_called()
    
    async def test_wildcard_timing_analysis(self, transfer_advisor_deps):
        """Test analyze_wildcard_timing tool."""
        agent = TransferAdvisorAgent()
        
        with patch('src.models.optimization.WildcardOptimizer.find_optimal_timing') as mock_wildcard:
            mock_wildcard.return_value = {
                'optimal_gameweek': 19,
                'expected_benefit': 15.8,
                'confidence': 0.78,
                'reasoning': ['Double gameweek approaching', 'Good fixture swing']
            }
            
            result = await agent.run(
                "When should I play my wildcard chip?",
                deps=transfer_advisor_deps
            )
            
            assert result is not None
            assert 'wildcard' in str(result).lower()
            mock_wildcard.assert_called()
    
    async def test_transfer_value_evaluation(self, transfer_advisor_deps):
        """Test evaluate_transfer_value tool."""
        agent = TransferAdvisorAgent()
        
        result = await agent.run(
            "Evaluate the value of transferring Salah to Sterling",
            deps=transfer_advisor_deps
        )
        
        assert result is not None
        assert 'value' in str(result).lower()
        # Should mention both players
        result_str = str(result).lower()
        assert 'salah' in result_str and 'sterling' in result_str


@pytest.mark.integration
class TestAgentIntegration:
    """Integration tests for multi-agent workflows."""
    
    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self, fpl_manager_deps, data_pipeline_deps, 
                                        ml_prediction_deps, transfer_advisor_deps):
        """Test complete analysis workflow across all agents."""
        
        # Initialize all agents
        fpl_manager = FPLManagerAgent()
        data_pipeline = DataPipelineAgent()
        ml_prediction = MLPredictionAgent()
        transfer_advisor = TransferAdvisorAgent()
        
        with patch('src.data.fetchers.FPLAPIClient.get_bootstrap_data'):
            with patch('src.models.ml_models.PlayerPredictor'):
                with patch('src.models.optimization.FPLOptimizer'):
                    
                    # Simulate full workflow
                    data_result = await data_pipeline.run(
                        "Update all data sources",
                        data_pipeline_deps
                    )
                    
                    ml_result = await ml_prediction.run(
                        "Generate predictions for all players",
                        ml_prediction_deps
                    )
                    
                    transfer_result = await transfer_advisor.run(
                        "Optimize transfers based on predictions", 
                        transfer_advisor_deps
                    )
                    
                    final_result = await fpl_manager.run(
                        "Provide comprehensive team analysis and recommendations",
                        fpl_manager_deps
                    )
                    
                    # All agents should complete successfully
                    assert data_result is not None
                    assert ml_result is not None  
                    assert transfer_result is not None
                    assert final_result is not None
    
    @pytest.mark.asyncio
    async def test_agent_communication_patterns(self, fpl_manager_deps):
        """Test agent delegation and communication patterns."""
        fpl_manager = FPLManagerAgent()
        
        with patch.object(fpl_manager, '_delegate_to_ml_agent') as mock_ml_delegate:
            mock_ml_delegate.return_value = "ML predictions completed"
            
            with patch.object(fpl_manager, '_delegate_to_transfer_agent') as mock_transfer_delegate:
                mock_transfer_delegate.return_value = "Transfer optimization completed"
                
                result = await fpl_manager.run(
                    "Give me complete analysis with ML predictions and transfer advice",
                    deps=fpl_manager_deps
                )
                
                # Verify delegation occurred
                mock_ml_delegate.assert_called()
                mock_transfer_delegate.assert_called()
                assert result is not None
    
    @pytest.mark.asyncio
    async def test_error_propagation_across_agents(self, fpl_manager_deps):
        """Test error handling in multi-agent scenarios."""
        fpl_manager = FPLManagerAgent()
        
        with patch('src.agents.ml_prediction.MLPredictionAgent.run') as mock_ml:
            mock_ml.side_effect = Exception("ML model error")
            
            # FPL Manager should handle ML agent errors gracefully
            result = await fpl_manager.run(
                "Get analysis with ML predictions",
                deps=fpl_manager_deps
            )
            
            # Should provide alternative analysis even if ML fails
            assert result is not None
            # Should indicate limitation due to ML unavailability
            result_str = str(result).lower()
            assert any(term in result_str for term in ['error', 'unable', 'unavailable', 'issue'])


@pytest.mark.performance
class TestAgentPerformance:
    """Performance benchmark tests for agents."""
    
    @pytest.mark.asyncio
    async def test_agent_response_time_benchmarks(self, fpl_manager_deps, performance_benchmarks):
        """Test agents meet response time benchmarks."""
        agent = FPLManagerAgent()
        
        start_time = asyncio.get_event_loop().time()
        result = await agent.run(
            "Get current team status",
            deps=fpl_manager_deps
        )
        execution_time = asyncio.get_event_loop().time() - start_time
        
        # Test response time benchmark
        max_response_time = performance_benchmarks['system_performance']['agent_response_time_seconds']
        assert execution_time <= max_response_time, \
            f"Agent response time {execution_time:.2f}s exceeds benchmark {max_response_time}s"
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_operations(self, fpl_manager_deps, ml_prediction_deps):
        """Test multiple agents can operate concurrently."""
        fpl_manager = FPLManagerAgent()
        ml_prediction = MLPredictionAgent()
        
        # Run agents concurrently
        tasks = [
            fpl_manager.run("Analyze team", deps=fpl_manager_deps),
            ml_prediction.run("Generate predictions", deps=ml_prediction_deps)
        ]
        
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        execution_time = asyncio.get_event_loop().time() - start_time
        
        # Concurrent execution should be efficient
        assert execution_time < 15, f"Concurrent execution took {execution_time:.2f}s, should be under 15s"
        
        # Both agents should complete
        assert len(results) == 2
        assert not isinstance(results[0], Exception)
        assert not isinstance(results[1], Exception)
    
    @pytest.mark.asyncio
    async def test_memory_usage_during_operation(self, fpl_manager_deps):
        """Test agent memory usage stays within bounds."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        agent = FPLManagerAgent()
        
        # Run multiple operations
        for i in range(5):
            result = await agent.run(
                f"Analysis operation {i}",
                deps=fpl_manager_deps
            )
            assert result is not None
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable (under 50MB for 5 operations)
        assert memory_increase < 50, f"Memory increased by {memory_increase:.1f}MB, should be under 50MB"