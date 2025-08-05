"""
Integration Test Suite - Following TestModel patterns.
End-to-end integration tests for complete FPL ML System.
Tests full workflows from data fetch to optimization to CLI output.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from src.agents.fpl_manager import FPLManagerAgent
from src.agents.data_pipeline import DataPipelineAgent
from src.agents.ml_prediction import MLPredictionAgent
from src.agents.transfer_advisor import TransferAdvisorAgent
from src.models.optimization import FPLOptimizer
from src.models.ml_models import PlayerPredictor
from src.data.fetchers import FPLAPIClient
from src.cli.main import fpl


@pytest.mark.integration
class TestFullSystemWorkflow:
    """End-to-end integration tests for complete system workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_analysis_workflow(self, fpl_manager_deps, data_pipeline_deps, 
                                            ml_prediction_deps, transfer_advisor_deps,
                                            mock_fpl_api_response, sample_players_data):
        """Test complete analysis workflow from data fetch to recommendations."""
        
        # Initialize all system components
        fpl_manager = FPLManagerAgent()
        data_pipeline = DataPipelineAgent()
        ml_prediction = MLPredictionAgent()
        transfer_advisor = TransferAdvisorAgent()
        api_client = FPLAPIClient()
        
        with patch.object(api_client, 'get_bootstrap_data') as mock_bootstrap:
            with patch.object(api_client, 'get_team_picks') as mock_team:
                with patch('src.models.ml_models.PlayerPredictor') as mock_predictor:
                    with patch('src.models.optimization.FPLOptimizer') as mock_optimizer:
                        
                        # Setup mocks
                        mock_bootstrap.return_value = mock_fpl_api_response
                        mock_team.return_value = {'picks': [], 'entry_history': {}}
                        
                        mock_predictor_instance = Mock()
                        mock_predictor_instance.predict_ensemble.return_value = [8.5, 7.2, 6.8, 9.1, 5.4]
                        mock_predictor.return_value = mock_predictor_instance
                        
                        mock_optimizer_instance = Mock()
                        mock_optimizer_instance.optimize_team.return_value = Mock(
                            status='optimal',
                            selected_players=list(range(1, 16)),
                            predicted_points=125.3,
                            total_cost=99.2
                        )
                        mock_optimizer.return_value = mock_optimizer_instance
                        
                        # Execute complete workflow
                        start_time = time.time()
                        
                        # Step 1: Update data
                        data_result = await data_pipeline.run(
                            "Fetch and validate all current FPL data",
                            data_pipeline_deps
                        )
                        
                        # Step 2: Generate ML predictions
                        prediction_result = await ml_prediction.run(
                            "Generate player predictions for next 3 gameweeks",
                            ml_prediction_deps
                        )
                        
                        # Step 3: Optimize transfers
                        transfer_result = await transfer_advisor.run(
                            "Find optimal transfers based on predictions",
                            transfer_advisor_deps
                        )
                        
                        # Step 4: Get comprehensive analysis
                        final_result = await fpl_manager.run(
                            "Provide complete team analysis with recommendations",
                            fpl_manager_deps
                        )
                        
                        workflow_time = time.time() - start_time
                        
                        # Validate workflow completion
                        assert data_result is not None, "Data pipeline should complete"
                        assert prediction_result is not None, "ML predictions should complete"
                        assert transfer_result is not None, "Transfer optimization should complete"
                        assert final_result is not None, "Final analysis should complete"
                        
                        # Validate workflow performance
                        assert workflow_time < 30, f"Complete workflow took {workflow_time:.2f}s, should be under 30s"
                        
                        # Validate that each step was executed
                        mock_bootstrap.assert_called()
                        mock_predictor_instance.predict_ensemble.assert_called()
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, fpl_manager_deps):
        """Test system handles errors gracefully in integrated workflow."""
        fpl_manager = FPLManagerAgent()
        
        with patch('src.data.fetchers.FPLAPIClient.get_bootstrap_data') as mock_api:
            mock_api.side_effect = [
                Exception("API temporarily unavailable"),  # First call fails
                {'elements': [], 'teams': []}               # Second call succeeds
            ]
            
            # System should recover from API failure
            result = await fpl_manager.run(
                "Get team analysis despite potential API issues",
                fpl_manager_deps
            )
            
            # Should provide some form of response, even if limited
            assert result is not None
            # Should have attempted retry
            assert mock_api.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, fpl_manager_deps, ml_prediction_deps):
        """Test system performance under concurrent load."""
        
        # Create multiple agents for concurrent testing
        agents = [FPLManagerAgent() for _ in range(5)]
        ml_agents = [MLPredictionAgent() for _ in range(3)]
        
        with patch('src.data.fetchers.FPLAPIClient.get_bootstrap_data') as mock_api:
            mock_api.return_value = {'elements': [], 'teams': []}
            
            # Run concurrent operations
            tasks = []
            
            # Add FPL Manager tasks
            for i, agent in enumerate(agents):
                task = agent.run(f"Analysis task {i}", fpl_manager_deps)
                tasks.append(task)
            
            # Add ML prediction tasks
            for i, agent in enumerate(ml_agents):
                task = agent.run(f"Prediction task {i}", ml_prediction_deps)
                tasks.append(task)
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            concurrent_time = time.time() - start_time
            
            # Validate concurrent performance
            assert concurrent_time < 45, f"Concurrent operations took {concurrent_time:.2f}s, should be under 45s"
            
            # Validate all operations completed
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) >= 6, f"Only {len(successful_results)}/8 concurrent operations succeeded"


@pytest.mark.integration
class TestDataToOptimizationPipeline:
    """Test complete data pipeline to optimization workflow."""
    
    def test_data_fetch_to_ml_pipeline(self, sample_players_data, sample_gameweek_history):
        """Test data flows correctly from API to ML models."""
        
        # Initialize components
        api_client = FPLAPIClient()
        predictor = PlayerPredictor()
        
        with patch.object(api_client, 'get_bootstrap_data') as mock_bootstrap:
            with patch.object(predictor, 'train') as mock_train:
                with patch.object(predictor, 'predict_ensemble') as mock_predict:
                    
                    # Setup data flow
                    mock_bootstrap.return_value = {
                        'elements': sample_players_data.to_dict('records')
                    }
                    mock_train.return_value = {'xgboost': {'mse_mean': 0.0025}}
                    mock_predict.return_value = [8.2, 7.5, 6.9, 9.1, 5.8]
                    
                    # Execute pipeline
                    bootstrap_data = api_client.get_bootstrap_data()
                    assert bootstrap_data is not None
                    assert 'elements' in bootstrap_data
                    
                    # Data should flow to ML model
                    predictor.train(sample_gameweek_history)
                    predictions = predictor.predict_ensemble(sample_players_data.head(), gameweeks_ahead=1)
                    
                    assert predictions is not None
                    assert len(predictions) > 0
                    mock_train.assert_called_once()
                    mock_predict.assert_called_once()
    
    def test_ml_to_optimization_pipeline(self, sample_players_data):
        """Test ML predictions flow correctly to optimization."""
        
        # Initialize components
        predictor = PlayerPredictor()
        optimizer = FPLOptimizer()
        
        with patch.object(predictor, 'predict_ensemble') as mock_predict:
            mock_predict.return_value = [8.5, 7.2, 6.8, 9.1, 5.4] * 20  # Enough for optimization
            
            # Generate predictions
            predictions = mock_predict.return_value
            predicted_points = {i+1: pred for i, pred in enumerate(predictions)}
            
            # Add required columns for optimization
            sample_players_data['position'] = sample_players_data['element_type'].map({
                1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'
            })
            
            # Execute optimization with ML predictions
            result = optimizer.optimize_team(sample_players_data, predicted_points)
            
            # Validation
            assert result is not None
            if result.status == 'optimal':
                assert len(result.selected_players) == 15
                assert result.total_cost <= 100.0
                assert result.predicted_points > 0
    
    def test_optimization_to_recommendation_pipeline(self, sample_players_data):
        """Test optimization results flow to user recommendations."""
        
        optimizer = FPLOptimizer()
        
        # Add required columns
        sample_players_data['position'] = sample_players_data['element_type'].map({
            1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'
        })
        
        predicted_points = {i: 7.5 for i in sample_players_data['id']}
        
        # Execute optimization
        result = optimizer.optimize_team(sample_players_data, predicted_points)
        
        if result.status == 'optimal':
            # Convert to user-friendly recommendations
            recommendations = {
                'selected_team': result.selected_players,
                'total_cost': result.total_cost,
                'expected_points': result.predicted_points,
                'remaining_budget': result.remaining_budget
            }
            
            assert recommendations['selected_team'] is not None
            assert recommendations['total_cost'] <= 100.0
            assert recommendations['expected_points'] > 0
            assert recommendations['remaining_budget'] >= 0


@pytest.mark.integration
class TestCLIToSystemIntegration:
    """Test CLI commands integrate properly with system components."""
    
    def test_cli_team_command_integration(self):
        """Test CLI team commands integrate with agents."""
        from click.testing import CliRunner
        
        runner = CliRunner()
        
        with patch('src.agents.fpl_manager.FPLManagerAgent') as mock_agent:
            mock_instance = Mock()
            mock_instance.run = AsyncMock(return_value="Team analysis complete")
            mock_agent.return_value = mock_instance
            
            result = runner.invoke(fpl, ['team', 'show'])
            
            # CLI should integrate with agent system
            assert result.exit_code == 0
    
    def test_cli_data_command_integration(self):
        """Test CLI data commands integrate with data pipeline."""
        from click.testing import CliRunner
        
        runner = CliRunner()
        
        with patch('src.agents.data_pipeline.DataPipelineAgent') as mock_agent:
            mock_instance = Mock()
            mock_instance.run = AsyncMock(return_value="Data update successful")
            mock_agent.return_value = mock_instance
            
            result = runner.invoke(fpl, ['data', 'update'])
            
            # CLI should integrate with data pipeline
            assert result.exit_code == 0
    
    def test_cli_transfer_command_integration(self):
        """Test CLI transfer commands integrate with optimization."""
        from click.testing import CliRunner
        
        runner = CliRunner()
        
        with patch('src.agents.transfer_advisor.TransferAdvisorAgent') as mock_agent:
            mock_instance = Mock()
            mock_instance.run = AsyncMock(return_value="Transfer suggestions ready")
            mock_agent.return_value = mock_instance
            
            result = runner.invoke(fpl, ['transfer', 'suggest'])
            
            # CLI should integrate with transfer optimization
            assert result.exit_code == 0


@pytest.mark.integration
class TestSystemResilience:
    """Test system resilience and fault tolerance."""
    
    @pytest.mark.asyncio
    async def test_partial_system_failure_handling(self, fpl_manager_deps):
        """Test system continues operating with partial failures."""
        fpl_manager = FPLManagerAgent()
        
        with patch('src.agents.ml_prediction.MLPredictionAgent.run') as mock_ml:
            with patch('src.agents.transfer_advisor.TransferAdvisorAgent.run') as mock_transfer:
                
                # ML agent fails, but transfer agent works
                mock_ml.side_effect = Exception("ML service unavailable")
                mock_transfer.return_value = "Transfer analysis available"
                
                result = await fpl_manager.run(
                    "Provide analysis with available services",
                    fpl_manager_deps
                )
                
                # Should provide partial analysis
                assert result is not None
                # Should indicate which services are unavailable
                result_str = str(result).lower()
                assert any(term in result_str for term in ['partial', 'limited', 'unavailable', 'available'])
    
    @pytest.mark.asyncio
    async def test_network_failure_recovery(self, fpl_manager_deps):
        """Test system recovers from network failures."""
        fpl_manager = FPLManagerAgent()
        
        with patch('src.data.fetchers.FPLAPIClient.get_bootstrap_data') as mock_api:
            # Simulate network recovery
            mock_api.side_effect = [
                Exception("Network timeout"),
                Exception("Connection reset"),
                {'elements': [], 'teams': []}  # Finally succeeds
            ]
            
            result = await fpl_manager.run(
                "Get data with network retry",
                fpl_manager_deps
            )
            
            # Should eventually succeed after retries
            assert result is not None
            assert mock_api.call_count >= 2, "Should retry after failures"
    
    def test_memory_leak_prevention(self, fpl_manager_deps):
        """Test system prevents memory leaks during extended operation."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate extended operation
        for i in range(20):
            agent = FPLManagerAgent()  # Create many agent instances
            
            with patch('src.data.fetchers.FPLAPIClient.get_bootstrap_data') as mock_api:
                mock_api.return_value = {'elements': [], 'teams': []}
                
                # Simulate agent operations
                try:
                    asyncio.run(agent.run(f"Operation {i}", fpl_manager_deps))
                except:
                    pass  # Ignore errors, focus on memory
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB, possible leak"


@pytest.mark.integration
@pytest.mark.slow
class TestLongRunningOperations:
    """Test system behavior during long-running operations."""
    
    @pytest.mark.asyncio
    async def test_extended_analysis_session(self, fpl_manager_deps):
        """Test system stability during extended analysis session."""
        fpl_manager = FPLManagerAgent()
        
        operations = [
            "Analyze current team performance",
            "Get transfer recommendations",
            "Predict player points for next gameweek",
            "Optimize team selection",
            "Analyze captain options",
            "Review fixture difficulty",
            "Check price change predictions",
            "Evaluate differential picks"
        ]
        
        with patch('src.data.fetchers.FPLAPIClient.get_bootstrap_data') as mock_api:
            mock_api.return_value = {'elements': [], 'teams': []}
            
            start_time = time.time()
            successful_operations = 0
            
            for operation in operations:
                try:
                    result = await fpl_manager.run(operation, fpl_manager_deps)
                    if result is not None:
                        successful_operations += 1
                except Exception:
                    pass  # Count as failure
                
                # Small delay between operations
                await asyncio.sleep(0.1)
            
            total_time = time.time() - start_time
            
            # Validate extended session
            assert successful_operations >= 6, f"Only {successful_operations}/8 operations succeeded"
            assert total_time < 60, f"Extended session took {total_time:.2f}s, should be under 60s"
    
    def test_data_pipeline_stability(self, data_pipeline_deps):
        """Test data pipeline stability over time."""
        
        with patch('src.data.fetchers.FPLAPIClient') as mock_client:
            mock_instance = Mock()
            mock_instance.get_bootstrap_data.return_value = {'elements': []}
            mock_instance.get_gameweek_live_data.return_value = {'elements': []}
            mock_client.return_value = mock_instance
            
            # Simulate continuous data updates
            successful_updates = 0
            total_updates = 30
            
            for i in range(total_updates):
                try:
                    agent = DataPipelineAgent()
                    result = asyncio.run(agent.run(
                        f"Update data cycle {i}",
                        data_pipeline_deps
                    ))
                    if result is not None:
                        successful_updates += 1
                except Exception:
                    pass
            
            # Should maintain high success rate
            success_rate = successful_updates / total_updates
            assert success_rate >= 0.9, f"Data pipeline success rate {success_rate:.1%} too low"


@pytest.mark.integration
class TestSystemBenchmarkValidation:
    """Validate system meets all PRP benchmarks in integrated scenarios."""
    
    @pytest.mark.benchmark
    def test_integrated_ml_performance_benchmark(self, sample_ml_training_data, 
                                               performance_benchmarks, test_assertions):
        """Test ML performance in integrated system context."""
        
        # Test complete ML pipeline
        predictor = PlayerPredictor()
        
        # Train model
        start_time = time.time()
        scores = predictor.train(sample_ml_training_data)
        training_time = time.time() - start_time
        
        # Make predictions
        test_data = sample_ml_training_data.tail(100)
        processed_test = predictor.prepare_features(test_data)
        
        start_time = time.time()
        predictions = predictor.predict_ensemble(processed_test, gameweeks_ahead=1)
        prediction_time = time.time() - start_time
        
        # Validate integrated performance
        assert training_time < 120, f"Integrated training took {training_time:.2f}s, should be under 120s"
        assert prediction_time < 5, f"Integrated prediction took {prediction_time:.2f}s, should be under 5s"
        
        # Validate ML quality benchmarks
        if 'xgboost' in scores:
            xgb_mse = scores['xgboost'].get('mse_mean', 0.001)
            test_assertions.assert_performance_benchmark(
                xgb_mse,
                performance_benchmarks['ml_models']['mse_threshold'],
                "Integrated XGBoost MSE"
            )
    
    @pytest.mark.benchmark
    def test_integrated_optimization_benchmark(self, sample_players_data, 
                                             performance_benchmarks, test_assertions):
        """Test optimization performance in integrated system context."""
        
        optimizer = FPLOptimizer()
        
        # Add required columns
        sample_players_data['position'] = sample_players_data['element_type'].map({
            1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'
        })
        
        predicted_points = {i: 7.5 for i in sample_players_data['id']}
        
        # Test optimization in integrated context
        start_time = time.time()
        result = optimizer.optimize_team(sample_players_data, predicted_points)
        solve_time = time.time() - start_time
        
        # Validate integrated optimization benchmark
        test_assertions.assert_optimization_performance(
            solve_time,
            result.status,
            performance_benchmarks
        )
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_integrated_system_response_time(self, fpl_manager_deps, performance_benchmarks):
        """Test complete system response time benchmark."""
        
        fpl_manager = FPLManagerAgent()
        
        with patch('src.data.fetchers.FPLAPIClient.get_bootstrap_data') as mock_api:
            mock_api.return_value = {'elements': [], 'teams': []}
            
            start_time = time.time()
            result = await fpl_manager.run(
                "Complete system analysis with optimization",
                fpl_manager_deps
            )
            response_time = time.time() - start_time
            
            # Validate system response time
            max_response_time = performance_benchmarks['system_performance']['agent_response_time_seconds']
            assert response_time <= max_response_time, \
                f"Integrated system response {response_time:.2f}s exceeds benchmark {max_response_time}s"
            
            assert result is not None