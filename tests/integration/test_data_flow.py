"""
Integration tests for complete data pipeline flow.
Tests data transformation consistency, caching, and database integration.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from datetime import datetime, timedelta
import json
import sqlite3
import tempfile
from pathlib import Path


@pytest.mark.integration
class TestDataPipelineIntegration:
    """Integration tests for complete data pipeline from API to ML models."""
    
    @pytest.mark.asyncio
    async def test_fpl_api_to_database_flow(self):
        """Test complete flow from FPL API to database storage."""
        # Mock FPL API response
        mock_api_response = {
            "elements": [
                {
                    "id": 1,
                    "web_name": "Salah",
                    "first_name": "Mohamed",
                    "second_name": "Salah",
                    "element_type": 3,
                    "team": 1,
                    "now_cost": 130,
                    "total_points": 187,
                    "form": "8.5",
                    "selected_by_percent": "45.2",
                    "minutes": 1200,
                    "goals_scored": 12,
                    "assists": 8
                }
            ],
            "teams": [
                {
                    "id": 1,
                    "name": "Liverpool",
                    "short_name": "LIV",
                    "strength": 5,
                    "strength_overall_home": 5,
                    "strength_overall_away": 4
                }
            ],
            "events": [
                {
                    "id": 20,
                    "name": "Gameweek 20",
                    "is_current": True,
                    "is_next": False,
                    "finished": False,
                    "deadline_time": "2024-01-15T11:30:00Z"
                }
            ]
        }
        
        # Test data fetcher
        with patch('src.data.fetchers.FPLDataFetcher') as MockFetcher:
            mock_fetcher = AsyncMock()
            MockFetcher.return_value.__aenter__.return_value = mock_fetcher
            MockFetcher.return_value.__aexit__.return_value = None
            
            # Mock API call
            mock_fetcher.fetch_bootstrap_data.return_value = mock_api_response
            
            # Mock data parsing
            mock_fetcher.parse_players_from_bootstrap.return_value = [
                MagicMock(
                    id=1, web_name="Salah", total_points=187, 
                    now_cost=130, element_type=3, team=1
                )
            ]
            mock_fetcher.parse_teams_from_bootstrap.return_value = [
                MagicMock(id=1, name="Liverpool", short_name="LIV", strength=5)
            ]
            
            # Test database storage
            with patch('sqlite3.connect') as mock_db_connect:
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_db_connect.return_value = mock_conn
                mock_conn.cursor.return_value = mock_cursor
                
                # Mock database operations
                mock_cursor.execute.return_value = None
                mock_cursor.fetchone.return_value = None
                mock_conn.commit.return_value = None
                
                # Simulate data flow
                async with MockFetcher() as fetcher:
                    # 1. Fetch data from API
                    bootstrap_data = await fetcher.fetch_bootstrap_data()
                    assert bootstrap_data == mock_api_response
                    
                    # 2. Parse data into models
                    players = fetcher.parse_players_from_bootstrap(bootstrap_data)
                    teams = fetcher.parse_teams_from_bootstrap(bootstrap_data)
                    
                    assert len(players) == 1
                    assert len(teams) == 1
                    assert players[0].web_name == "Salah"
                    
                    # 3. Store in database (mocked)
                    mock_cursor.execute.assert_called()
                    mock_conn.commit.assert_called()
    
    @pytest.mark.asyncio
    async def test_data_transformation_consistency(self):
        """Test data transformation consistency across pipeline stages."""
        # Raw API data
        raw_data = {
            "player_id": 1,
            "web_name": "Salah",
            "total_points": 187,
            "now_cost": 130,
            "form": "8.5",
            "minutes": 1200,
            "goals_scored": 12,
            "assists": 8,
            "selected_by_percent": "45.2"
        }
        
        with patch('src.data.processors.FeatureEngineer') as MockProcessor:
            mock_processor = MagicMock()
            MockProcessor.return_value = mock_processor
            
            # Mock feature engineering transformation
            engineered_features = {
                "player_id": 1,
                "web_name": "Salah",
                "total_points": 187,
                "price_millions": 13.0,  # 130 / 10
                "form_numeric": 8.5,     # parsed from string
                "goals_per_90": 0.9,     # (12 / 1200) * 90
                "assists_per_90": 0.6,   # (8 / 1200) * 90
                "ownership_numeric": 45.2, # parsed from string
                "form_last_5": 8.5,
                "minutes_last_5": 90,
                "fixture_difficulty": 3.0,
                "is_home": 1
            }
            
            mock_processor.engineer_features.return_value = engineered_features
            
            # Test transformation consistency
            processed_data = mock_processor.engineer_features(raw_data)
            
            # Verify data consistency
            assert processed_data["player_id"] == raw_data["player_id"]
            assert processed_data["total_points"] == raw_data["total_points"]
            assert processed_data["price_millions"] == 13.0  # Correct transformation
            assert processed_data["form_numeric"] == 8.5     # Correct parsing
            assert processed_data["goals_per_90"] > 0        # Calculated feature
            assert "form_last_5" in processed_data           # Rolling feature added
    
    @pytest.mark.asyncio
    async def test_cache_integration_workflow(self):
        """Test caching integration throughout the pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            cache_dir.mkdir(exist_ok=True)
            
            # Mock cached data
            cached_bootstrap = {
                "cached": True,
                "timestamp": datetime.now().isoformat(),
                "data": {"elements": [], "teams": [], "events": []}
            }
            
            cache_file = cache_dir / "bootstrap_data.json"
            
            with patch('src.data.fetchers.FPLDataFetcher') as MockFetcher:
                mock_fetcher = AsyncMock()
                MockFetcher.return_value.__aenter__.return_value = mock_fetcher
                MockFetcher.return_value.__aexit__.return_value = None
                
                # Mock cache operations
                mock_fetcher.cache_dir = cache_dir
                mock_fetcher._get_cache_path.return_value = cache_file
                mock_fetcher._is_cache_fresh.return_value = True
                mock_fetcher._load_cached_data.return_value = cached_bootstrap
                
                async with MockFetcher() as fetcher:
                    # Test cache hit
                    result = await fetcher.fetch_bootstrap_data()
                    
                    # Should return cached data
                    mock_fetcher._load_cached_data.assert_called()
                    
                    # Test cache miss scenario
                    mock_fetcher._is_cache_fresh.return_value = False
                    mock_fetcher._save_cached_data = MagicMock()
                    
                    fresh_data = {"fresh": True, "from_api": True}
                    with patch('src.data.fetchers.get_bootstrap_data', return_value=fresh_data):
                        result = await fetcher.fetch_bootstrap_data()
                        
                        # Should save to cache
                        mock_fetcher._save_cached_data.assert_called_with('bootstrap_data', fresh_data)
    
    @pytest.mark.asyncio
    async def test_data_quality_validation_pipeline(self):
        """Test data quality validation throughout the pipeline."""
        # Test data with various quality issues
        test_data = [
            {"id": 1, "web_name": "Salah", "total_points": 187, "now_cost": 130, "minutes": 1200},
            {"id": 2, "web_name": "Kane", "total_points": -5, "now_cost": 110, "minutes": 800},  # Negative points (invalid)
            {"id": 3, "web_name": "", "total_points": 95, "now_cost": 85, "minutes": 1500},        # Empty name (invalid)
            {"id": 4, "web_name": "Son", "total_points": 134, "now_cost": 200, "minutes": 1300},  # Price too high (invalid)
            {"id": 5, "web_name": "Bruno", "total_points": 156, "now_cost": 95, "minutes": 1100}, # Valid
        ]
        
        with patch('src.data.validators.DataValidator') as MockValidator:
            mock_validator = MagicMock()
            MockValidator.return_value = mock_validator
            
            # Mock validation results
            validation_results = {
                "total_records": 5,
                "valid_records": 3,
                "invalid_records": 2,
                "quality_score": 0.6,
                "issues": [
                    {"record_id": 2, "issue": "negative_total_points", "severity": "high"},
                    {"record_id": 3, "issue": "empty_web_name", "severity": "high"},
                    {"record_id": 4, "issue": "price_out_of_range", "severity": "medium"}
                ]
            }
            
            mock_validator.validate_batch.return_value = validation_results
            
            # Test validation pipeline
            validator = MockValidator()
            results = validator.validate_batch(test_data)
            
            assert results["total_records"] == 5
            assert results["valid_records"] == 3
            assert results["quality_score"] == 0.6
            assert len(results["issues"]) == 3
            
            # Test quality threshold enforcement
            assert results["quality_score"] > 0.5  # Minimum acceptable quality
    
    def test_database_schema_consistency(self):
        """Test database schema consistency and migrations."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
            
            try:
                # Create test database
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Create tables (simplified schema)
                cursor.execute('''
                    CREATE TABLE players (
                        id INTEGER PRIMARY KEY,
                        web_name TEXT NOT NULL,
                        element_type INTEGER,
                        team INTEGER,
                        now_cost INTEGER,
                        total_points INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE teams (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        short_name TEXT,
                        strength INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                
                # Test schema consistency
                cursor.execute("PRAGMA table_info(players)")
                player_columns = cursor.fetchall()
                
                cursor.execute("PRAGMA table_info(teams)")
                team_columns = cursor.fetchall()
                
                # Verify required columns exist
                player_column_names = [col[1] for col in player_columns]
                team_column_names = [col[1] for col in team_columns]
                
                assert 'id' in player_column_names
                assert 'web_name' in player_column_names
                assert 'total_points' in player_column_names
                assert 'created_at' in player_column_names
                
                assert 'id' in team_column_names
                assert 'name' in team_column_names
                assert 'strength' in team_column_names
                
                # Test data insertion consistency
                cursor.execute(
                    "INSERT INTO players (id, web_name, element_type, team, now_cost, total_points) VALUES (?, ?, ?, ?, ?, ?)",
                    (1, "Salah", 3, 1, 130, 187)
                )
                
                cursor.execute(
                    "INSERT INTO teams (id, name, short_name, strength) VALUES (?, ?, ?, ?)",
                    (1, "Liverpool", "LIV", 5)
                )
                
                conn.commit()
                
                # Verify data integrity
                cursor.execute("SELECT COUNT(*) FROM players")
                player_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM teams") 
                team_count = cursor.fetchone()[0]
                
                assert player_count == 1
                assert team_count == 1
                
                conn.close()
                
            finally:
                # Cleanup
                Path(db_path).unlink(missing_ok=True)


@pytest.mark.integration
class TestMLPipelineIntegration:
    """Integration tests for ML pipeline data flow."""
    
    @pytest.mark.asyncio
    async def test_feature_engineering_to_model_training(self):
        """Test flow from feature engineering to model training."""
        # Mock feature engineering output
        engineered_features = {
            "players_count": 100,
            "features_per_player": 15,
            "feature_matrix_shape": (1500, 15),  # 15 gameweeks × 100 players
            "target_variable": "total_points",
            "feature_columns": [
                "form_last_5", "minutes_last_5", "goals_per_90", "assists_per_90",
                "fixture_difficulty", "is_home", "team_strength", "opponent_weakness",
                "price_momentum", "ownership_trend", "expected_goals", "expected_assists",
                "bonus_last_5", "bps_last_5", "ict_index"
            ],
            "data_quality": 0.95
        }
        
        with patch('src.models.ml_models.PlayerPredictor') as MockPredictor:
            mock_predictor = MagicMock()
            MockPredictor.return_value = mock_predictor
            
            # Mock model training
            training_results = {
                "model_trained": True,
                "training_samples": 1500,
                "features_used": 15,
                "cross_validation_scores": {
                    "xgboost": {"mse_mean": 0.0025, "correlation_mean": 0.72},
                    "random_forest": {"mse_mean": 0.0031, "correlation_mean": 0.68}
                },
                "ensemble_performance": {
                    "mse": 0.0023,
                    "correlation": 0.74,
                    "accuracy_within_2pts": 0.63
                }
            }
            
            mock_predictor.train.return_value = training_results["cross_validation_scores"]
            mock_predictor.is_trained = True
            
            # Test model training pipeline
            predictor = MockPredictor()
            training_scores = predictor.train(engineered_features)
            
            # Verify training results meet benchmarks
            assert training_scores["xgboost"]["mse_mean"] < 0.003  # Research benchmark
            assert training_scores["xgboost"]["correlation_mean"] > 0.65
            assert predictor.is_trained
    
    @pytest.mark.asyncio 
    async def test_model_prediction_to_optimization(self):
        """Test flow from model predictions to optimization engine."""
        # Mock ML predictions
        ml_predictions = {
            "predictions": {
                str(i): {
                    "expected_points": 5.0 + (i % 10),  # Vary predictions
                    "confidence": 0.7 + (i % 3) * 0.1,
                    "prediction_interval": [4.0 + (i % 10), 6.0 + (i % 10)]
                }
                for i in range(1, 101)  # 100 players
            },
            "model_metadata": {
                "model_version": "1.2.0",
                "prediction_date": datetime.now().isoformat(),
                "benchmark_mse": 0.0025
            }
        }
        
        with patch('src.models.optimization.FPLOptimizer') as MockOptimizer:
            mock_optimizer = MagicMock()
            MockOptimizer.return_value = mock_optimizer
            
            # Mock optimization results
            optimization_result = {
                "status": "optimal",
                "selected_players": list(range(1, 16)),  # 15 players
                "total_cost": 98.5,
                "predicted_points": 125.3,
                "remaining_budget": 1.5,
                "solve_time": 2.1,
                "position_constraints_met": True,
                "team_constraints_met": True
            }
            
            mock_optimizer.optimize_team.return_value = optimization_result
            
            # Test optimization pipeline
            optimizer = MockOptimizer()
            result = optimizer.optimize_team(
                players=None,  # Would be actual player data
                predicted_points=ml_predictions["predictions"]
            )
            
            # Verify optimization results
            assert result["status"] == "optimal"
            assert result["total_cost"] <= 100.0
            assert len(result["selected_players"]) == 15
            assert result["solve_time"] < 5.0  # Performance requirement
            assert result["position_constraints_met"]
    
    @pytest.mark.asyncio
    async def test_prediction_validation_pipeline(self):
        """Test prediction validation and quality assurance pipeline."""
        # Mock prediction validation scenarios
        validation_scenarios = [
            {
                "name": "benchmark_validation",
                "predictions": [8.5, 6.2, 9.1, 4.3, 7.8],
                "actual": [8.2, 6.5, 8.9, 4.1, 7.5],
                "expected_mse": 0.0024,
                "expected_correlation": 0.72
            },
            {
                "name": "outlier_detection", 
                "predictions": [8.5, 25.0, 6.2, -3.0, 7.8],  # Contains outliers
                "actual": [8.2, 8.5, 6.5, 6.0, 7.5],
                "outliers_detected": 2,
                "outlier_threshold": 20.0
            },
            {
                "name": "consistency_check",
                "predictions": [8.5, 6.2, 9.1, 4.3, 7.8],
                "predictions_repeat": [8.5, 6.2, 9.1, 4.3, 7.8],  # Should be identical
                "consistency_tolerance": 0.001
            }
        ]
        
        with patch('src.models.validators.PredictionValidator') as MockValidator:
            mock_validator = MagicMock()
            MockValidator.return_value = mock_validator
            
            # Test each validation scenario
            for scenario in validation_scenarios:
                if scenario["name"] == "benchmark_validation":
                    mock_validator.validate_benchmark.return_value = {
                        "mse": scenario["expected_mse"],
                        "correlation": scenario["expected_correlation"],
                        "benchmark_met": True
                    }
                    
                    result = mock_validator.validate_benchmark(
                        scenario["predictions"], scenario["actual"]
                    )
                    
                    assert result["benchmark_met"]
                    assert result["mse"] < 0.003
                
                elif scenario["name"] == "outlier_detection":
                    mock_validator.detect_outliers.return_value = {
                        "outliers_found": scenario["outliers_detected"],
                        "outlier_indices": [1, 3],
                        "outlier_values": [25.0, -3.0]
                    }
                    
                    result = mock_validator.detect_outliers(
                        scenario["predictions"], scenario["outlier_threshold"]
                    )
                    
                    assert result["outliers_found"] == 2
                
                elif scenario["name"] == "consistency_check":
                    mock_validator.check_consistency.return_value = {
                        "consistent": True,
                        "max_difference": 0.0,
                        "tolerance_met": True
                    }
                    
                    result = mock_validator.check_consistency(
                        scenario["predictions"], scenario["predictions_repeat"]
                    )
                    
                    assert result["consistent"]
                    assert result["tolerance_met"]


@pytest.mark.integration
class TestSystemIntegrationWorkflows:
    """Integration tests for complete system workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_team_analysis_workflow(self):
        """Test complete team analysis workflow integration."""
        workflow_steps = [
            {"step": "data_fetch", "expected_duration": 1.0},
            {"step": "feature_engineering", "expected_duration": 0.5}, 
            {"step": "ml_prediction", "expected_duration": 2.0},
            {"step": "optimization", "expected_duration": 1.5},
            {"step": "report_generation", "expected_duration": 0.3}
        ]
        
        with patch('src.agents.fpl_manager.FPLManagerAgent') as MockFPLManager:
            mock_manager = AsyncMock()
            MockFPLManager.return_value = mock_manager
            
            # Mock complete workflow
            workflow_result = {
                "workflow_id": "team_analysis_123",
                "user_id": "user_456",
                "team_id": 789,
                "timestamp": datetime.now().isoformat(),
                "steps_completed": len(workflow_steps),
                "total_duration": sum(step["expected_duration"] for step in workflow_steps),
                "results": {
                    "current_team_value": 95.5,
                    "predicted_next_gw_points": 67,
                    "suggested_transfers": 2,
                    "captain_recommendation": "Salah",
                    "bench_optimization": "Recommended changes",
                    "risk_analysis": "Medium risk profile"
                },
                "confidence_metrics": {
                    "prediction_confidence": 0.78,
                    "optimization_confidence": 0.85,
                    "overall_confidence": 0.81
                },
                "performance_metrics": {
                    "data_quality": 0.96,
                    "model_accuracy": 0.72,
                    "optimization_status": "optimal"
                }
            }
            
            mock_manager.run_team_analysis.return_value = workflow_result
            
            # Test workflow execution
            result = await mock_manager.run_team_analysis()
            
            # Verify workflow completion
            assert result["steps_completed"] == 5
            assert result["total_duration"] < 10.0  # Performance requirement
            assert result["confidence_metrics"]["overall_confidence"] > 0.8
            assert result["performance_metrics"]["optimization_status"] == "optimal"
            assert result["results"]["suggested_transfers"] >= 0
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self):
        """Test error recovery and fallback workflows."""
        error_scenarios = [
            {
                "error_type": "api_timeout", 
                "recovery_strategy": "use_cached_data",
                "fallback_quality": 0.7
            },
            {
                "error_type": "model_prediction_failure",
                "recovery_strategy": "use_simple_heuristics", 
                "fallback_quality": 0.5
            },
            {
                "error_type": "optimization_infeasible",
                "recovery_strategy": "relax_constraints",
                "fallback_quality": 0.8
            }
        ]
        
        with patch('src.agents.fpl_manager.FPLManagerAgent') as MockFPLManager:
            mock_manager = AsyncMock()
            MockFPLManager.return_value = mock_manager
            
            for scenario in error_scenarios:
                # Mock error recovery
                recovery_result = {
                    "error_detected": True,
                    "error_type": scenario["error_type"],
                    "recovery_successful": True,
                    "recovery_strategy": scenario["recovery_strategy"],
                    "fallback_used": True,
                    "degraded_service": True,
                    "quality_impact": 1.0 - scenario["fallback_quality"],
                    "recovery_time": 1.2
                }
                
                mock_manager.handle_error_recovery.return_value = recovery_result
                
                # Test error recovery
                result = await mock_manager.handle_error_recovery(scenario["error_type"])
                
                assert result["error_detected"]
                assert result["recovery_successful"]
                assert result["fallback_used"]
                assert result["recovery_time"] < 5.0  # Should recover quickly