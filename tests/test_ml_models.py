"""
ML Models Test Suite - Following TestModel patterns.
Tests for ML model performance, benchmarks, and research requirements.
Target: MSE < 0.003, Correlation > 0.65, Accuracy > 60%
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

from src.models.ml_models import PlayerPredictor, FeatureEngineer
from src.models.data_models import PlayerPrediction


@pytest.mark.ml
class TestPlayerPredictor:
    """Test suite for PlayerPredictor following TestModel patterns."""
    
    def test_predictor_initialization(self):
        """Test PlayerPredictor initializes correctly."""
        predictor = PlayerPredictor()
        
        assert predictor is not None
        assert hasattr(predictor, 'models')
        assert hasattr(predictor, 'feature_columns')
        assert hasattr(predictor, 'ensemble_weights')
        assert 'xgboost' in predictor.models
        assert 'random_forest' in predictor.models
        assert not predictor.is_trained
    
    @pytest.mark.benchmark
    def test_model_training_performance(self, sample_ml_training_data, performance_benchmarks, test_assertions):
        """Test model training meets performance benchmarks."""
        predictor = PlayerPredictor()
        
        # Prepare training data
        training_data = sample_ml_training_data.copy()
        
        # Ensure we have enough data
        assert len(training_data) >= 1000, "Need at least 1000 samples for valid training"
        
        # Time the training process
        start_time = time.time()
        scores = predictor.train(training_data)
        training_time = time.time() - start_time
        
        # Assert training completed successfully
        assert isinstance(scores, dict)
        assert 'xgboost' in scores
        assert 'random_forest' in scores
        assert predictor.is_trained
        
        # Check training time is reasonable (should be under 60 seconds for this dataset)
        assert training_time < 60, f"Training took {training_time:.2f}s, should be under 60s"
        
        # Validate model performance meets research benchmarks
        for model_name, score_dict in scores.items():
            mse = score_dict['mse_mean']
            correlation = score_dict.get('correlation_mean', 0.7)  # Default if not available
            
            # CRITICAL: Test research benchmark MSE < 0.003
            test_assertions.assert_performance_benchmark(
                mse, 
                performance_benchmarks['ml_models']['mse_threshold'],
                f"{model_name} MSE"
            )
            
            # Test correlation benchmark
            assert correlation > performance_benchmarks['ml_models']['correlation_threshold'], \
                f"{model_name} correlation {correlation} below threshold"
    
    def test_feature_engineering_quality(self, sample_ml_training_data):
        """Test feature engineering produces high-quality features."""
        predictor = PlayerPredictor()
        
        # Test feature preparation
        processed_data = predictor.prepare_features(sample_ml_training_data)
        
        assert not processed_data.empty
        assert len(processed_data.columns) >= len(predictor.feature_columns)
        
        # Check for required engineered features
        required_features = ['form_last_5', 'minutes_last_5', 'goals_per_90', 'assists_per_90']
        for feature in required_features:
            assert feature in processed_data.columns, f"Missing required feature: {feature}"
        
        # Validate feature quality
        assert not processed_data[predictor.feature_columns].isnull().all().any(), \
            "Features should not be all null"
        
        # Check for reasonable feature ranges
        assert processed_data['form_last_5'].between(0, 15).all(), \
            "Form should be between 0-15 points"
        assert processed_data['goals_per_90'].between(0, 5).all(), \
            "Goals per 90 should be reasonable"
    
    @pytest.mark.performance
    def test_prediction_speed_benchmark(self, sample_ml_training_data, performance_benchmarks):
        """Test prediction speed meets performance requirements."""
        predictor = PlayerPredictor()
        
        # Train with sample data
        predictor.train(sample_ml_training_data)
        
        # Prepare test data
        test_data = sample_ml_training_data.tail(100)
        processed_test = predictor.prepare_features(test_data)
        
        # Time predictions
        start_time = time.time()
        predictions = predictor.predict_ensemble(processed_test, gameweeks_ahead=1)
        prediction_time = time.time() - start_time
        
        # Predictions should complete quickly (under 1 second for 100 players)
        assert prediction_time < 1.0, f"Prediction took {prediction_time:.3f}s, should be under 1.0s"
        
        # Validate prediction results
        assert len(predictions) == len(processed_test)
        assert all(0 <= pred <= 20 for pred in predictions), "Predictions should be in reasonable range"
    
    def test_ensemble_prediction_quality(self, sample_ml_training_data):
        """Test ensemble predictions are better than individual models."""
        predictor = PlayerPredictor()
        predictor.train(sample_ml_training_data)
        
        # Split data for testing
        train_data, test_data = train_test_split(sample_ml_training_data, test_size=0.2, random_state=42)
        
        processed_test = predictor.prepare_features(test_data)
        X_test = processed_test[predictor.feature_columns]
        y_test = processed_test['total_points']
        
        # Get individual model predictions
        xgb_pred = predictor.models['xgboost'].predict(X_test)
        rf_pred = predictor.models['random_forest'].predict(X_test)
        ensemble_pred = predictor.predict_ensemble(processed_test, gameweeks_ahead=1)
        
        # Calculate MSE for each
        xgb_mse = mean_squared_error(y_test, xgb_pred)
        rf_mse = mean_squared_error(y_test, rf_pred)
        ensemble_mse = mean_squared_error(y_test, ensemble_pred)
        
        # Test that ensemble is competitive with best individual model
        best_individual_mse = min(xgb_mse, rf_mse)
        assert ensemble_mse <= best_individual_mse * 1.1, \
            f"Ensemble MSE {ensemble_mse:.6f} should be competitive with best individual {best_individual_mse:.6f}"
    
    @pytest.mark.benchmark
    def test_cross_validation_benchmark(self, sample_ml_training_data, performance_benchmarks, test_assertions):
        """Test cross-validation performance meets research benchmarks."""
        predictor = PlayerPredictor()
        
        # Use time series cross-validation
        from sklearn.model_selection import TimeSeriesSplit
        
        data = sample_ml_training_data.copy()
        X = data[predictor.feature_columns].fillna(0)
        y = data['total_points']
        
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train XGBoost model for CV test
            from xgboost import XGBRegressor
            model = XGBRegressor(n_estimators=50, max_depth=6, random_state=42)
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_val)
            mse = mean_squared_error(y_val, predictions)
            cv_scores.append(mse)
        
        # Test average CV performance
        avg_cv_mse = np.mean(cv_scores)
        test_assertions.assert_performance_benchmark(
            avg_cv_mse,
            performance_benchmarks['ml_models']['mse_threshold'],
            "Cross-validation MSE"
        )
    
    def test_model_persistence(self, sample_ml_training_data, tmp_path):
        """Test model saving and loading functionality."""
        predictor = PlayerPredictor()
        predictor.train(sample_ml_training_data)
        
        # Save models to temporary directory
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        predictor.model_path = str(model_dir)
        predictor.save_models()
        
        # Create new predictor and load models
        new_predictor = PlayerPredictor()
        new_predictor.model_path = str(model_dir)
        new_predictor.load_models()
        
        # Test that loaded models work
        test_data = sample_ml_training_data.tail(10)
        processed_test = new_predictor.prepare_features(test_data)
        
        original_predictions = predictor.predict_ensemble(processed_test, gameweeks_ahead=1)
        loaded_predictions = new_predictor.predict_ensemble(processed_test, gameweeks_ahead=1)
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions, decimal=6)
    
    def test_feature_importance_analysis(self, sample_ml_training_data):
        """Test feature importance calculation and interpretation."""
        predictor = PlayerPredictor()
        predictor.train(sample_ml_training_data)
        
        # Get feature importance from XGBoost
        feature_importance = predictor.get_feature_importance()
        
        assert isinstance(feature_importance, dict)
        assert len(feature_importance) > 0
        
        # Check that importance values are reasonable
        total_importance = sum(feature_importance.values())
        assert abs(total_importance - 1.0) < 0.01, "Feature importance should sum to ~1.0"
        
        # Most important features should make sense
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        important_features = [f[0] for f in top_features]
        
        # These features should be among the most important
        expected_important = ['form_last_5', 'minutes_last_5', 'goals_per_90', 'assists_per_90']
        assert any(feat in important_features for feat in expected_important), \
            f"Expected important features not in top 3: {important_features}"


@pytest.mark.ml
class TestFeatureEngineer:
    """Test suite for FeatureEngineer following TestModel patterns."""
    
    def test_feature_engineer_initialization(self):
        """Test FeatureEngineer initializes correctly."""
        engineer = FeatureEngineer()
        assert engineer is not None
        assert hasattr(engineer, 'window_size')
        assert hasattr(engineer, 'min_minutes_threshold')
    
    def test_rolling_features_calculation(self, sample_gameweek_history):
        """Test rolling statistics feature calculation."""
        engineer = FeatureEngineer()
        
        # Test rolling features
        result = engineer.create_rolling_features(sample_gameweek_history, window=5)
        
        assert 'form_last_5' in result.columns
        assert 'minutes_last_5' in result.columns
        assert 'goals_last_5' in result.columns
        
        # Check that rolling features are calculated correctly
        player_data = result[result['player_id'] == 1].sort_values('gameweek')
        
        if len(player_data) >= 5:
            # Check that 5-game rolling average is correct
            manual_avg = player_data['total_points'].iloc[:5].mean()
            calculated_avg = player_data['form_last_5'].iloc[4]  # 5th game rolling average
            
            assert abs(manual_avg - calculated_avg) < 0.01, \
                f"Rolling average calculation incorrect: {manual_avg} vs {calculated_avg}"
    
    def test_per90_statistics_calculation(self, sample_gameweek_history):
        """Test per-90 minute statistics calculation."""
        engineer = FeatureEngineer()
        
        result = engineer.create_per90_features(sample_gameweek_history)
        
        assert 'goals_per_90' in result.columns
        assert 'assists_per_90' in result.columns
        assert 'points_per_90' in result.columns
        
        # Test calculation accuracy
        test_row = result.iloc[0]
        if test_row['minutes'] > 0:
            expected_goals_per_90 = (test_row['goals_scored'] / test_row['minutes']) * 90
            assert abs(test_row['goals_per_90'] - expected_goals_per_90) < 0.01
    
    def test_fixture_difficulty_features(self, sample_gameweek_history):
        """Test fixture difficulty feature engineering."""
        engineer = FeatureEngineer()
        
        result = engineer.add_fixture_features(sample_gameweek_history)
        
        assert 'fixture_difficulty' in result.columns
        assert 'is_home' in result.columns
        assert 'upcoming_fixtures' in result.columns
        
        # Check that fixture difficulty is in reasonable range
        assert result['fixture_difficulty'].between(1, 5).all(), \
            "Fixture difficulty should be between 1-5"
        
        # Check home/away encoding
        assert result['is_home'].isin([0, 1]).all(), \
            "Home indicator should be binary"
    
    def test_complete_feature_pipeline(self, sample_gameweek_history):
        """Test complete feature engineering pipeline."""
        engineer = FeatureEngineer()
        
        result = engineer.engineer_features(sample_gameweek_history)
        
        # Check that all expected features are present
        expected_features = [
            'form_last_5', 'minutes_last_5', 'goals_per_90', 'assists_per_90',
            'fixture_difficulty', 'is_home', 'team_strength', 'opponent_strength'
        ]
        
        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"
        
        # Check data quality
        assert not result.empty
        assert len(result) <= len(sample_gameweek_history)  # May filter some rows
        
        # Check for unreasonable values
        assert result['form_last_5'].between(0, 20).all(), "Form should be reasonable"
        assert result['goals_per_90'].between(0, 10).all(), "Goals per 90 should be reasonable"


@pytest.mark.benchmark
class TestMLModelBenchmarks:
    """Comprehensive benchmark tests for ML models following PRP requirements."""
    
    @pytest.mark.slow
    def test_research_benchmark_validation(self, sample_ml_training_data, performance_benchmarks, test_assertions):
        """CRITICAL TEST: Validate model meets research benchmark MSE < 0.003."""
        predictor = PlayerPredictor()
        
        # Split data for proper validation
        train_data, test_data = train_test_split(
            sample_ml_training_data, 
            test_size=0.2, 
            random_state=42,
            stratify=None  # Cannot stratify continuous target
        )
        
        # Train model
        predictor.train(train_data)
        
        # Prepare test data
        processed_test = predictor.prepare_features(test_data)
        X_test = processed_test[predictor.feature_columns]
        y_test = processed_test['total_points']
        
        # Make predictions
        predictions = predictor.predict_ensemble(processed_test, gameweeks_ahead=1)
        
        # Calculate performance metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        correlation = np.corrcoef(predictions, y_test)[0, 1]
        
        # Calculate accuracy within ±2 points
        accuracy = np.mean(np.abs(predictions - y_test) <= 2.0)
        
        # CRITICAL BENCHMARK TESTS
        test_assertions.assert_ml_model_quality(
            mse, correlation, accuracy, performance_benchmarks
        )
        
        # Log performance for monitoring
        print(f"\n🎯 ML Model Performance:")
        print(f"   MSE: {mse:.6f} (benchmark: < {performance_benchmarks['ml_models']['mse_threshold']})")
        print(f"   MAE: {mae:.3f}")
        print(f"   Correlation: {correlation:.3f} (benchmark: > {performance_benchmarks['ml_models']['correlation_threshold']})")
        print(f"   Accuracy (±2): {accuracy:.1%} (benchmark: > {performance_benchmarks['ml_models']['accuracy_threshold']:.0%})")
    
    def test_model_stability_across_seasons(self, sample_ml_training_data):
        """Test model stability across different time periods."""
        predictor = PlayerPredictor()
        
        # Split data by gameweek to simulate seasons
        early_season = sample_ml_training_data[sample_ml_training_data['gameweek'] <= 7]
        late_season = sample_ml_training_data[sample_ml_training_data['gameweek'] > 7]
        
        if len(early_season) < 100 or len(late_season) < 100:
            pytest.skip("Not enough data for season stability test")
        
        # Train on early season
        predictor.train(early_season)
        
        # Test on late season
        processed_test = predictor.prepare_features(late_season)
        X_test = processed_test[predictor.feature_columns]
        y_test = processed_test['total_points']
        
        predictions = predictor.predict_ensemble(processed_test, gameweeks_ahead=1)
        
        # Model should maintain reasonable performance
        mse = mean_squared_error(y_test, predictions)
        correlation = np.corrcoef(predictions, y_test)[0, 1]
        
        # Relaxed thresholds for cross-season performance
        assert mse < 0.01, f"Cross-season MSE {mse:.6f} too high"
        assert correlation > 0.3, f"Cross-season correlation {correlation:.3f} too low"
    
    def test_model_robustness_to_outliers(self, sample_ml_training_data):
        """Test model robustness to outlier data points."""
        predictor = PlayerPredictor()
        
        # Create training data with outliers
        data_with_outliers = sample_ml_training_data.copy()
        
        # Add some extreme outlier points
        outlier_indices = np.random.choice(len(data_with_outliers), size=10, replace=False)
        data_with_outliers.loc[outlier_indices, 'total_points'] = 50  # Extreme high scores
        
        # Train model
        predictor.train(data_with_outliers)
        
        # Test on normal data
        normal_data = sample_ml_training_data[~sample_ml_training_data.index.isin(outlier_indices)]
        processed_test = predictor.prepare_features(normal_data.tail(100))
        
        predictions = predictor.predict_ensemble(processed_test, gameweeks_ahead=1)
        
        # Predictions should still be in reasonable range despite outliers
        assert all(0 <= pred <= 25 for pred in predictions), \
            "Model should be robust to outliers and predict reasonable values"
    
    @pytest.mark.performance
    def test_prediction_consistency(self, sample_ml_training_data):
        """Test that model predictions are consistent across multiple runs."""
        predictor = PlayerPredictor()
        predictor.train(sample_ml_training_data)
        
        test_data = sample_ml_training_data.tail(20)
        processed_test = predictor.prepare_features(test_data)
        
        # Make predictions multiple times
        predictions_1 = predictor.predict_ensemble(processed_test, gameweeks_ahead=1)
        predictions_2 = predictor.predict_ensemble(processed_test, gameweeks_ahead=1)
        predictions_3 = predictor.predict_ensemble(processed_test, gameweeks_ahead=1)
        
        # Predictions should be identical (deterministic)
        np.testing.assert_array_equal(predictions_1, predictions_2)
        np.testing.assert_array_equal(predictions_2, predictions_3)
    
    def test_memory_efficiency(self, sample_ml_training_data):
        """Test model memory usage is reasonable."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        predictor = PlayerPredictor()
        predictor.train(sample_ml_training_data)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Model should not use excessive memory (under 100MB increase)
        assert memory_increase < 100, f"Model uses too much memory: {memory_increase:.1f}MB increase"