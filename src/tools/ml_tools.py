"""
Pure ML tool functions for agent integration.
Following main_agent_reference/tools.py patterns with proper error handling.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from ..models.ml_models import PlayerPredictor, FeatureEngineer
from ..models.optimization import FPLOptimizer, OptimizationConstraints
from ..models.data_models import PlayerPrediction, OptimizedTeam

logger = logging.getLogger(__name__)

# Global model instances (lazy-loaded)
_player_predictor = None
_fpl_optimizer = None


def get_player_predictor() -> PlayerPredictor:
    """Get or create player predictor instance."""
    global _player_predictor
    if _player_predictor is None:
        _player_predictor = PlayerPredictor()
        # Try to load existing models
        _player_predictor.load_models()
    return _player_predictor


def get_fpl_optimizer() -> FPLOptimizer:
    """Get or create FPL optimizer instance."""
    global _fpl_optimizer
    if _fpl_optimizer is None:
        _fpl_optimizer = FPLOptimizer()
    return _fpl_optimizer


async def predict_player_points(
    player_ids: List[int],
    historical_data: pd.DataFrame,
    gameweeks_ahead: int = 3
) -> Dict[int, PlayerPrediction]:
    """
    Pure function to predict player points for multiple gameweeks.
    
    Args:
        player_ids: List of FPL player IDs to predict
        historical_data: Historical player performance data
        gameweeks_ahead: Number of gameweeks to predict ahead
        
    Returns:
        Dictionary mapping player_id to PlayerPrediction
        
    Raises:
        ValueError: If input data is invalid
        Exception: If prediction fails
    """
    if not player_ids:
        raise ValueError("Player IDs list cannot be empty")
    
    if historical_data.empty:
        raise ValueError("Historical data cannot be empty")
    
    if not (1 <= gameweeks_ahead <= 8):
        raise ValueError("Gameweeks ahead must be between 1 and 8")
    
    logger.info(f"Predicting points for {len(player_ids)} players, {gameweeks_ahead} gameweeks ahead")
    
    try:
        predictor = get_player_predictor()
        
        # If model not trained, need historical data to train
        if not predictor.is_trained and len(historical_data) > 100:
            logger.info("Training predictor with provided historical data...")
            training_scores = predictor.train(historical_data)
            logger.info(f"Training complete. Average MSE: {np.mean([s['mse_mean'] for s in training_scores.values()]):.6f}")
        
        predictions = {}
        
        for player_id in player_ids:
            try:
                # Get player's recent data for prediction
                player_data = historical_data[historical_data['player_id'] == player_id]
                
                if player_data.empty:
                    logger.warning(f"No historical data found for player {player_id}")
                    # Create default prediction
                    predictions[player_id] = PlayerPrediction(
                        player_id=player_id,
                        gameweeks_ahead=gameweeks_ahead,
                        expected_points=[3.0] * gameweeks_ahead,  # Default average
                        confidence_intervals=[{"lower": 1.0, "upper": 8.0}] * gameweeks_ahead,
                        model_used="default",
                        feature_importance={}
                    )
                    continue
                
                # Prepare features
                processed_data = predictor.prepare_features(player_data)
                
                if processed_data.empty or len(processed_data) < 5:
                    logger.warning(f"Insufficient processed data for player {player_id}")
                    continue
                
                # Make prediction
                recent_features = processed_data.tail(1)  # Most recent gameweek
                
                if predictor.is_trained:
                    predicted_values = predictor.predict_ensemble(recent_features, gameweeks_ahead)
                    
                    # Create predictions for each gameweek
                    expected_points = []
                    confidence_intervals = []
                    
                    for week in range(gameweeks_ahead):
                        # Apply some decay for future weeks and add uncertainty
                        base_prediction = predicted_values[0] if len(predicted_values) > 0 else 3.0
                        decay_factor = 0.95 ** week  # Slight decay for future predictions
                        week_prediction = max(0, base_prediction * decay_factor)
                        
                        expected_points.append(float(week_prediction))
                        
                        # Calculate confidence intervals (simplified)
                        std_dev = max(1.5, week_prediction * 0.3)  # 30% standard deviation
                        confidence_intervals.append({
                            "lower": max(0, week_prediction - 1.96 * std_dev),
                            "upper": week_prediction + 1.96 * std_dev
                        })
                    
                    # Get feature importance (simplified)
                    feature_importance = {}
                    if hasattr(predictor.models['xgboost'], 'feature_importances_'):
                        importance_scores = predictor.models['xgboost'].feature_importances_
                        feature_names = predictor.feature_columns
                        for name, score in zip(feature_names[:5], importance_scores[:5]):  # Top 5
                            feature_importance[name] = float(score)
                    
                    predictions[player_id] = PlayerPrediction(
                        player_id=player_id,
                        gameweeks_ahead=gameweeks_ahead,
                        expected_points=expected_points,
                        confidence_intervals=confidence_intervals,
                        model_used="ensemble",
                        feature_importance=feature_importance
                    )
                else:
                    # Fallback to historical average if model not trained
                    avg_points = player_data['total_points'].tail(10).mean()
                    if pd.isna(avg_points):
                        avg_points = 3.0
                    
                    predictions[player_id] = PlayerPrediction(
                        player_id=player_id,
                        gameweeks_ahead=gameweeks_ahead,
                        expected_points=[float(avg_points)] * gameweeks_ahead,
                        confidence_intervals=[{"lower": 0.0, "upper": float(avg_points * 2)}] * gameweeks_ahead,
                        model_used="historical_average",
                        feature_importance={}
                    )
                
            except Exception as e:
                logger.error(f"Failed to predict for player {player_id}: {e}")
                continue
        
        logger.info(f"Successfully generated predictions for {len(predictions)} players")
        return predictions
        
    except Exception as e:
        logger.error(f"Player points prediction failed: {e}")
        raise Exception(f"Prediction failed: {str(e)}")


async def optimize_team_selection(
    players_df: pd.DataFrame,
    predicted_points: Dict[int, float],
    current_team: Optional[List[int]] = None,
    constraints: Optional[Dict[str, Any]] = None
) -> OptimizedTeam:
    """
    Pure function to optimize team selection using linear programming.
    
    Args:
        players_df: DataFrame with player data
        predicted_points: Dictionary mapping player_id to predicted points
        current_team: Current team for transfer optimization
        constraints: Custom optimization constraints
        
    Returns:
        OptimizedTeam with optimization results
        
    Raises:
        ValueError: If input data is invalid
        Exception: If optimization fails
    """
    if players_df.empty:
        raise ValueError("Players DataFrame cannot be empty")
    
    if not predicted_points:
        raise ValueError("Predicted points dictionary cannot be empty")
    
    required_cols = ['id', 'element_type', 'team', 'now_cost']
    missing_cols = [col for col in required_cols if col not in players_df.columns]
    if missing_cols:
        raise ValueError(f"Players DataFrame missing required columns: {missing_cols}")
    
    logger.info(f"Optimizing team selection with {len(players_df)} players")
    
    try:
        optimizer = get_fpl_optimizer()
        
        # Apply custom constraints if provided
        if constraints:
            custom_constraints = OptimizationConstraints()
            for key, value in constraints.items():
                if hasattr(custom_constraints, key):
                    setattr(custom_constraints, key, value)
            optimizer.constraints = custom_constraints
        
        # Run optimization
        result = optimizer.optimize_team(
            players_df=players_df,
            predicted_points=predicted_points,
            current_team=current_team
        )
        
        logger.info(f"Team optimization completed: {result.status}")
        if result.status == "optimal":
            logger.info(f"Selected {len(result.selected_players or [])} players, "
                       f"Total cost: £{result.total_cost:.1f}M, "
                       f"Predicted points: {result.predicted_points:.1f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Team optimization failed: {e}")
        raise Exception(f"Optimization failed: {str(e)}")


async def optimize_transfers(
    current_team: List[int],
    players_df: pd.DataFrame,
    predicted_points: Dict[int, float],
    free_transfers: int = 1,
    weeks_ahead: int = 4,
    chip_active: Optional[str] = None
) -> OptimizedTeam:
    """
    Pure function to optimize transfer decisions.
    
    Args:
        current_team: Current team player IDs
        players_df: Available players data
        predicted_points: Predicted points per player
        free_transfers: Number of free transfers available
        weeks_ahead: Planning horizon in gameweeks
        chip_active: Active chip (wildcard, free_hit, etc.)
        
    Returns:
        OptimizedTeam with transfer recommendations
        
    Raises:
        ValueError: If input data is invalid
        Exception: If optimization fails
    """
    if not current_team or len(current_team) != 15:
        raise ValueError("Current team must have exactly 15 players")
    
    if players_df.empty:
        raise ValueError("Players DataFrame cannot be empty")
    
    if not predicted_points:
        raise ValueError("Predicted points dictionary cannot be empty")
    
    if not (1 <= weeks_ahead <= 8):
        raise ValueError("Weeks ahead must be between 1 and 8")
    
    logger.info(f"Optimizing transfers for {weeks_ahead} weeks, {free_transfers} free transfers")
    
    try:
        optimizer = get_fpl_optimizer()
        
        result = optimizer.optimize_transfers(
            current_team=current_team,
            players_df=players_df,
            predicted_points=predicted_points,
            free_transfers=free_transfers,
            weeks_ahead=weeks_ahead,
            chip_active=chip_active
        )
        
        logger.info(f"Transfer optimization completed: {result.status}")
        if result.recommended_transfers:
            logger.info(f"Recommended {len(result.recommended_transfers)} transfers, "
                       f"Expected gain: +{result.expected_points_gain:.1f} points")
        
        return result
        
    except Exception as e:
        logger.error(f"Transfer optimization failed: {e}")
        raise Exception(f"Transfer optimization failed: {str(e)}")


async def optimize_captain_selection(
    playing_xi: List[int],
    predicted_points: Dict[int, float],
    ownership_data: Optional[Dict[int, float]] = None,
    risk_preference: str = "balanced"
) -> Dict[str, Any]:
    """
    Pure function to optimize captain selection.
    
    Args:
        playing_xi: List of playing XI player IDs
        predicted_points: Predicted points per player
        ownership_data: Player ownership percentages
        risk_preference: "safe", "balanced", or "differential"
        
    Returns:
        Dictionary with captain recommendations
        
    Raises:
        ValueError: If input data is invalid
    """
    if not playing_xi or len(playing_xi) != 11:
        raise ValueError("Playing XI must have exactly 11 players")
    
    if not predicted_points:
        raise ValueError("Predicted points dictionary cannot be empty")
    
    if risk_preference not in ["safe", "balanced", "differential"]:
        raise ValueError("Risk preference must be 'safe', 'balanced', or 'differential'")
    
    logger.info(f"Optimizing captain selection with {risk_preference} strategy")
    
    try:
        captain_options = []
        
        for player_id in playing_xi:
            if player_id not in predicted_points:
                continue
            
            expected_points = predicted_points[player_id]
            ownership = ownership_data.get(player_id, 50.0) if ownership_data else 50.0
            
            # Calculate captain score based on strategy
            if risk_preference == "safe":
                # Favor high ownership, high predicted points
                score = expected_points * 0.7 + (ownership / 100) * 0.3
            elif risk_preference == "differential":
                # Favor low ownership, high predicted points
                score = expected_points * 0.8 - (ownership / 100) * 0.2
            else:  # balanced
                # Equal weighting
                score = expected_points * 0.6 + (ownership / 100) * 0.1
            
            # Add some randomness for uncertainty
            uncertainty_factor = np.random.normal(1.0, 0.1)
            adjusted_score = score * uncertainty_factor
            
            captain_options.append({
                'player_id': player_id,
                'expected_points': expected_points,
                'ownership': ownership,
                'captain_score': adjusted_score,
                'risk_level': "High" if ownership < 20 else "Medium" if ownership < 50 else "Low"
            })
        
        # Sort by captain score
        captain_options.sort(key=lambda x: x['captain_score'], reverse=True)
        
        # Select top options
        top_options = captain_options[:3]
        
        logger.info(f"Captain optimization complete. Top choice: Player {top_options[0]['player_id']} "
                   f"({top_options[0]['expected_points']:.1f} pts, {top_options[0]['ownership']:.1f}% owned)")
        
        return {
            'status': 'optimal',
            'recommended_captain': top_options[0]['player_id'],
            'vice_captain': top_options[1]['player_id'] if len(top_options) > 1 else None,
            'all_options': top_options,
            'strategy': risk_preference
        }
        
    except Exception as e:
        logger.error(f"Captain optimization failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'recommended_captain': playing_xi[0] if playing_xi else None  # Fallback
        }


async def calculate_player_value(
    player_ids: List[int],
    predicted_points: Dict[int, float],
    player_costs: Dict[int, float],
    weeks_horizon: int = 5
) -> Dict[int, Dict[str, float]]:
    """
    Pure function to calculate player value metrics.
    
    Args:
        player_ids: List of player IDs to analyze
        predicted_points: Predicted points per player
        player_costs: Player costs in millions
        weeks_horizon: Analysis horizon in weeks
        
    Returns:
        Dictionary mapping player_id to value metrics
        
    Raises:
        ValueError: If input data is invalid
    """
    if not player_ids:
        raise ValueError("Player IDs list cannot be empty")
    
    if not predicted_points or not player_costs:
        raise ValueError("Predicted points and costs dictionaries cannot be empty")
    
    logger.info(f"Calculating value metrics for {len(player_ids)} players")
    
    try:
        value_metrics = {}
        
        for player_id in player_ids:
            if player_id not in predicted_points or player_id not in player_costs:
                continue
            
            points = predicted_points[player_id]
            cost = player_costs[player_id]
            
            # Calculate various value metrics
            points_per_million = points / cost if cost > 0 else 0
            
            # Project over horizon
            projected_points = points * weeks_horizon
            total_value = projected_points / cost if cost > 0 else 0
            
            # Calculate relative value (compared to position average)
            # This would be more sophisticated with position-specific data
            relative_value = points_per_million / 0.5  # Assume 0.5 is average PPM
            
            value_metrics[player_id] = {
                'points_per_million': points_per_million,
                'projected_points': projected_points,
                'total_value': total_value,
                'relative_value': relative_value,
                'cost': cost,
                'expected_points': points
            }
        
        logger.info(f"Value calculation complete for {len(value_metrics)} players")
        return value_metrics
        
    except Exception as e:
        logger.error(f"Player value calculation failed: {e}")
        return {}


async def validate_model_performance(
    predictor: Optional[PlayerPredictor] = None,
    test_data: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """
    Pure function to validate ML model performance.
    
    Args:
        predictor: PlayerPredictor instance to validate
        test_data: Test data for validation
        
    Returns:
        Dictionary with performance metrics
        
    Raises:
        ValueError: If validation data is invalid
    """
    if predictor is None:
        predictor = get_player_predictor()
    
    if not predictor.is_trained:
        raise ValueError("Predictor must be trained before validation")
    
    if test_data is None or test_data.empty:
        raise ValueError("Test data must be provided for validation")
    
    logger.info("Validating model performance...")
    
    try:
        # Prepare test features
        processed_test = predictor.prepare_features(test_data)
        
        if processed_test.empty:
            raise ValueError("No valid test data after processing")
        
        X_test = processed_test[predictor.feature_columns]
        y_test = processed_test['total_points']
        
        # Make predictions
        predictions = predictor.predict_ensemble(processed_test)
        
        # Calculate metrics
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        rmse = np.sqrt(mse)
        
        # Calculate correlation
        correlation = np.corrcoef(predictions, y_test)[0, 1] if len(predictions) > 1 else 0
        
        # Calculate accuracy (within 2 points)
        accuracy = np.mean(np.abs(predictions - y_test) <= 2.0)
        
        metrics = {
            'mse': float(mse),  
            'mae': float(mae),
            'rmse': float(rmse),
            'correlation': float(correlation),
            'accuracy_within_2': float(accuracy),
            'test_samples': len(y_test)
        }
        
        logger.info(f"Model validation complete. MSE: {mse:.6f}, MAE: {mae:.3f}, "
                   f"Correlation: {correlation:.3f}")
        
        # Check benchmark
        if mse < 0.003:
            logger.info("✅ Model meets research benchmark (MSE < 0.003)")
        else:
            logger.warning(f"⚠️ Model MSE {mse:.6f} exceeds research benchmark of 0.003")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return {'error': str(e)}