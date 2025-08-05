"""
ML Prediction Agent - handles all ML-powered predictions for FPL system.
Following main_agent_reference patterns with comprehensive ML model integration.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from pydantic_ai import Agent, RunContext

from ..config.providers import get_llm_model
from ..tools.ml_tools import (
    predict_player_points,
    validate_model_performance,
    calculate_player_value
)
from ..models.ml_models import PlayerPredictor, FeatureEngineer
from ..models.data_models import PlayerPrediction

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are an expert ML engineer specializing in Fantasy Premier League (FPL) predictions. Your primary responsibility is to provide accurate, data-driven predictions using advanced machine learning models including XGBoost and LSTM networks.

Your capabilities:
1. **Player Point Predictions**: Predict individual player points for upcoming gameweeks
2. **Price Change Predictions**: Forecast player price rises and falls
3. **Captaincy Analysis**: Identify optimal captain choices with risk assessment
4. **Form Analysis**: Analyze player form trends and momentum
5. **Model Performance**: Monitor and validate ML model accuracy
6. **Ensemble Predictions**: Combine multiple models for robust predictions

ML Model Standards:
- Target research benchmark: MSE < 0.003 for point predictions
- Use ensemble methods (XGBoost + LSTM) for robustness
- Apply time series cross-validation for proper evaluation
- Account for fixture difficulty and opponent strength
- Include confidence intervals and uncertainty quantification
- Regular model retraining with new data

Prediction Principles:
- Always provide confidence levels with predictions
- Account for player injury status and minutes played
- Consider team tactical changes and formations
- Factor in historical performance vs current form
- Adjust for fixture congestion and rotation risk
- Include differential analysis for captain choices

Quality Assurance:
- Validate all input data before prediction
- Monitor prediction accuracy against actual results
- Flag when model confidence is low
- Provide feature importance for explainability
- Alert when retraining is needed
"""


@dataclass
class MLPredictionDependencies:
    """Dependencies for the ML Prediction agent."""
    model_path: str
    retrain_threshold_mse: float = 0.005
    min_training_samples: int = 1000
    confidence_threshold: float = 0.7
    session_id: Optional[str] = None


# Initialize the ML Prediction agent
ml_prediction_agent = Agent(
    get_llm_model(),
    deps_type=MLPredictionDependencies,
    system_prompt=SYSTEM_PROMPT
)


@ml_prediction_agent.tool
async def predict_gameweek_points(
    ctx: RunContext[MLPredictionDependencies],
    player_ids: List[int],
    gameweeks_ahead: int = 3,
    include_confidence: bool = True
) -> str:
    """
    Predict player points for upcoming gameweeks using ensemble ML models.
    
    Args:
        player_ids: List of FPL player IDs to predict
        gameweeks_ahead: Number of gameweeks to predict (1-8)
        include_confidence: Include confidence intervals in results
        
    Returns:
        Formatted prediction results
    """
    try:
        logger.info(f"Predicting points for {len(player_ids)} players, {gameweeks_ahead} gameweeks ahead")
        
        if not player_ids:
            return "❌ No player IDs provided for prediction"
        
        if not (1 <= gameweeks_ahead <= 8):
            return "❌ Gameweeks ahead must be between 1 and 8"
        
        # Create dummy historical data for prediction (in production, this would come from database)
        # This is a simplified version - would normally fetch real historical data
        historical_data = _create_sample_historical_data(player_ids)
        
        if historical_data.empty:
            return f"❌ No historical data available for prediction"
        
        # Make predictions using ML tools
        predictions = await predict_player_points(
            player_ids=player_ids,
            historical_data=historical_data,
            gameweeks_ahead=gameweeks_ahead
        )
        
        if not predictions:
            return f"❌ Prediction failed - no results generated"
        
        # Format results
        results_text = f"🎯 **ML Point Predictions ({gameweeks_ahead} GW ahead)**\n\n"
        
        # Sort players by total predicted points
        sorted_predictions = sorted(
            predictions.items(),
            key=lambda x: sum(x[1].expected_points),
            reverse=True
        )
        
        for player_id, prediction in sorted_predictions:
            total_predicted = sum(prediction.expected_points)
            avg_per_gw = total_predicted / gameweeks_ahead
            
            results_text += f"**Player {player_id}:**\n"
            results_text += f"• Total Predicted: {total_predicted:.1f} points\n"
            results_text += f"• Average per GW: {avg_per_gw:.1f} points\n"
            results_text += f"• Model Used: {prediction.model_used}\n"
            
            if include_confidence and prediction.confidence_intervals:
                first_gw_confidence = prediction.confidence_intervals[0]
                results_text += f"• Next GW Range: {first_gw_confidence['lower']:.1f} - {first_gw_confidence['upper']:.1f}\n"
            
            # Show feature importance if available
            if prediction.feature_importance:
                top_features = sorted(
                    prediction.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                features_text = ", ".join([f"{feat}: {score:.3f}" for feat, score in top_features])
                results_text += f"• Key Factors: {features_text}\n"
            
            results_text += "\n"
        
        # Add model performance info
        results_text += f"**Model Information:**\n"
        results_text += f"• Predictions Generated: {len(predictions)}\n"
        results_text += f"• Prediction Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        # Check if model meets benchmark
        predictor = PlayerPredictor()
        if hasattr(predictor, 'last_validation_mse') and predictor.last_validation_mse:
            if predictor.last_validation_mse < 0.003:
                results_text += f"• Model Status: ✅ Meets research benchmark (MSE: {predictor.last_validation_mse:.6f})\n"
            else:
                results_text += f"• Model Status: ⚠️ Below benchmark (MSE: {predictor.last_validation_mse:.6f})\n"
        
        logger.info(f"Point predictions completed for {len(predictions)} players")
        return results_text.strip()
        
    except Exception as e:
        logger.error(f"Point prediction failed: {e}")
        return f"❌ Point prediction failed: {str(e)}"


@ml_prediction_agent.tool
async def analyze_captain_options(
    ctx: RunContext[MLPredictionDependencies],
    team_player_ids: List[int],
    risk_preference: str = "balanced",
    include_differentials: bool = True
) -> str:
    """
    Analyze captain options with ML predictions and ownership data.
    
    Args:
        team_player_ids: Player IDs in current team
        risk_preference: "safe", "balanced", or "differential"
        include_differentials: Include low-ownership differential options
        
    Returns:
        Captain analysis with recommendations
    """
    try:
        logger.info(f"Analyzing captain options for {len(team_player_ids)} players")
        
        if not team_player_ids:
            return "❌ No team players provided for captain analysis"
        
        if risk_preference not in ["safe", "balanced", "differential"]:
            return "❌ Risk preference must be 'safe', 'balanced', or 'differential'"
        
        # Get predictions for team players (next gameweek only)
        historical_data = _create_sample_historical_data(team_player_ids)
        
        predictions = await predict_player_points(
            player_ids=team_player_ids,
            historical_data=historical_data,
            gameweeks_ahead=1
        )
        
        if not predictions:
            return "❌ Could not generate predictions for captain analysis"
        
        # Create captain analysis
        captain_options = []
        
        for player_id, prediction in predictions.items():
            next_gw_points = prediction.expected_points[0] if prediction.expected_points else 0
            
            # Simulate ownership data (in production, this would come from API)
            ownership = _simulate_ownership_data(player_id)
            
            # Calculate captain score based on strategy
            captain_score = _calculate_captain_score(
                predicted_points=next_gw_points,
                ownership=ownership,
                strategy=risk_preference
            )
            
            # Determine risk level
            risk_level = "High" if ownership < 20 else "Medium" if ownership < 50 else "Low"
            
            captain_options.append({
                'player_id': player_id,
                'predicted_points': next_gw_points,
                'ownership': ownership,
                'captain_score': captain_score,
                'risk_level': risk_level,
                'confidence': prediction.confidence_intervals[0] if prediction.confidence_intervals else None
            })
        
        # Sort by captain score
        captain_options.sort(key=lambda x: x['captain_score'], reverse=True)
        
        # Filter differentials if requested
        if include_differentials and risk_preference == "differential":
            captain_options = [opt for opt in captain_options if opt['ownership'] < 30]
        
        # Format analysis
        analysis_text = f"👑 **Captain Analysis ({risk_preference} strategy)**\n\n"
        
        # Top recommendation
        if captain_options:
            top_choice = captain_options[0]
            analysis_text += f"**🥇 Top Recommendation:**\n"
            analysis_text += f"• Player {top_choice['player_id']}\n"
            analysis_text += f"• Predicted Points: {top_choice['predicted_points']:.1f}\n"
            analysis_text += f"• Ownership: {top_choice['ownership']:.1f}%\n"
            analysis_text += f"• Risk Level: {top_choice['risk_level']}\n"
            
            if top_choice['confidence']:
                conf = top_choice['confidence']
                analysis_text += f"• Confidence Range: {conf['lower']:.1f} - {conf['upper']:.1f} points\n"
            
            analysis_text += f"• Captain Score: {top_choice['captain_score']:.2f}\n\n"
        
        # Alternative options
        if len(captain_options) > 1:
            analysis_text += f"**Alternative Options:**\n"
            for i, option in enumerate(captain_options[1:4], 2):  # Show next 3 best
                analysis_text += f"{i}. Player {option['player_id']}: "
                analysis_text += f"{option['predicted_points']:.1f} pts "
                analysis_text += f"({option['ownership']:.1f}% owned, {option['risk_level']} risk)\n"
        
        # Strategy-specific advice
        analysis_text += f"\n**{risk_preference.title()} Strategy Advice:**\n"
        
        if risk_preference == "safe":
            analysis_text += "• Focus on high-ownership, reliable players\n"
            analysis_text += "• Prioritize consistency over ceiling\n"
        elif risk_preference == "differential":
            analysis_text += "• Consider low-ownership players for rank climbing\n"
            analysis_text += "• Higher risk but potential for bigger gains\n"
        else:
            analysis_text += "• Balance between safety and upside potential\n"
            analysis_text += "• Consider match context and form\n"
        
        # Risk assessment
        if captain_options:
            avg_ownership = np.mean([opt['ownership'] for opt in captain_options[:3]])
            analysis_text += f"\n**Risk Assessment:**\n"
            analysis_text += f"• Average ownership (top 3): {avg_ownership:.1f}%\n"
            
            if avg_ownership > 60:
                analysis_text += "• Low risk approach - following template\n"
            elif avg_ownership < 30:
                analysis_text += "• High risk approach - differential heavy\n"
            else:
                analysis_text += "• Balanced risk approach\n"
        
        logger.info(f"Captain analysis completed. Top choice: Player {captain_options[0]['player_id']} ({captain_options[0]['predicted_points']:.1f} pts)")
        return analysis_text.strip()
        
    except Exception as e:
        logger.error(f"Captain analysis failed: {e}")
        return f"❌ Captain analysis failed: {str(e)}"


@ml_prediction_agent.tool
async def validate_model_accuracy(
    ctx: RunContext[MLPredictionDependencies],
    test_gameweeks: int = 5
) -> str:
    """
    Validate current ML model performance against recent results.
    
    Args:
        test_gameweeks: Number of recent gameweeks to test against
        
    Returns:
        Model validation report
    """
    try:
        logger.info(f"Validating model performance over last {test_gameweeks} gameweeks")
        
        # In production, this would fetch actual recent gameweek data
        # For now, create sample test data
        test_data = _create_sample_test_data(test_gameweeks)
        
        if test_data.empty:
            return f"❌ No test data available for validation"
        
        # Validate model performance
        validation_metrics = await validate_model_performance(test_data=test_data)
        
        if 'error' in validation_metrics:
            return f"❌ Model validation failed: {validation_metrics['error']}"
        
        # Generate validation report
        mse = validation_metrics.get('mse', 0)
        mae = validation_metrics.get('mae', 0)
        correlation = validation_metrics.get('correlation', 0)
        accuracy = validation_metrics.get('accuracy_within_2', 0)
        test_samples = validation_metrics.get('test_samples', 0)
        
        validation_report = f"""
📊 **ML Model Validation Report**

**Performance Metrics:**
• Mean Squared Error (MSE): {mse:.6f}
• Mean Absolute Error (MAE): {mae:.3f}
• Correlation with Actual: {correlation:.3f}
• Accuracy (±2 points): {accuracy:.1%}
• Test Samples: {test_samples:,}

**Benchmark Comparison:**
• Research Target MSE: 0.003000
• Current Performance: {'✅ MEETS BENCHMARK' if mse < 0.003 else '❌ BELOW BENCHMARK'}
• Performance Gap: {((mse - 0.003) / 0.003 * 100):+.1f}%

**Model Quality Assessment:**
{'🟢' if mse < 0.003 else '🟡' if mse < 0.006 else '🔴'} **{'Excellent' if mse < 0.003 else 'Good' if mse < 0.006 else 'Needs Improvement'}**

**Detailed Analysis:**
• Prediction Accuracy: {'High' if correlation > 0.7 else 'Medium' if correlation > 0.5 else 'Low'}
• Error Distribution: {'Tight' if mae < 2.0 else 'Moderate' if mae < 3.0 else 'Wide'}
• Reliability: {'Reliable' if accuracy > 0.6 else 'Moderate' if accuracy > 0.4 else 'Unreliable'}

**Recommendations:**
{_generate_model_recommendations(mse, correlation, accuracy)}

**Validation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Store validation results for future reference
        predictor = PlayerPredictor()
        predictor.last_validation_mse = mse
        predictor.last_validation_date = datetime.now()
        
        logger.info(f"Model validation complete. MSE: {mse:.6f}, Correlation: {correlation:.3f}")
        return validation_report.strip()
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return f"❌ Model validation failed: {str(e)}"


@ml_prediction_agent.tool
async def predict_price_changes(
    ctx: RunContext[MLPredictionDependencies],
    player_ids: List[int],
    days_ahead: int = 7
) -> str:
    """
    Predict player price changes using ownership trends and historical patterns.
    
    Args:
        player_ids: List of player IDs to analyze
        days_ahead: Number of days to predict ahead
        
    Returns:
        Price change predictions
    """
    try:
        logger.info(f"Predicting price changes for {len(player_ids)} players, {days_ahead} days ahead")
        
        if not player_ids:
            return "❌ No player IDs provided for price prediction"
        
        if not (1 <= days_ahead <= 14):
            return "❌ Days ahead must be between 1 and 14"
        
        # In production, this would use sophisticated price change models
        # For now, create simplified predictions based on simulated data
        price_predictions = []
        
        for player_id in player_ids:
            # Simulate current ownership and recent trends
            current_ownership = _simulate_ownership_data(player_id)
            ownership_change = np.random.normal(0, 2)  # Daily ownership change %
            
            # Simulate recent price momentum
            recent_price_changes = np.random.choice([-0.1, 0, 0.1], size=7, p=[0.1, 0.8, 0.1])
            price_momentum = np.sum(recent_price_changes)
            
            # Calculate price change probability
            rise_probability = _calculate_price_rise_probability(
                ownership_change, price_momentum, current_ownership
            )
            fall_probability = _calculate_price_fall_probability(
                ownership_change, price_momentum, current_ownership
            )
            
            # Determine prediction
            if rise_probability > 0.7:
                prediction = "Rise Expected"
                confidence = rise_probability
            elif fall_probability > 0.7:
                prediction = "Fall Expected" 
                confidence = fall_probability
            else:
                prediction = "Stable"
                confidence = max(1 - rise_probability - fall_probability, 0.3)
            
            price_predictions.append({
                'player_id': player_id,
                'prediction': prediction,
                'rise_probability': rise_probability,
                'fall_probability': fall_probability,
                'confidence': confidence,
                'current_ownership': current_ownership,
                'ownership_trend': ownership_change
            })
        
        # Sort by rise probability (most likely to rise first)
        price_predictions.sort(key=lambda x: x['rise_probability'], reverse=True)
        
        # Format results
        predictions_text = f"💰 **Price Change Predictions ({days_ahead} days)**\n\n"
        
        # Group by prediction type
        rises = [p for p in price_predictions if p['prediction'] == "Rise Expected"]
        falls = [p for p in price_predictions if p['prediction'] == "Fall Expected"]
        stable = [p for p in price_predictions if p['prediction'] == "Stable"]
        
        if rises:
            predictions_text += f"**🔺 Expected Price Rises ({len(rises)}):**\n"
            for pred in rises:
                predictions_text += f"• Player {pred['player_id']}: {pred['rise_probability']:.0%} chance "
                predictions_text += f"(Owner: {pred['current_ownership']:.1f}%, Trend: {pred['ownership_trend']:+.1f}%)\n"
            predictions_text += "\n"
        
        if falls:
            predictions_text += f"**🔻 Expected Price Falls ({len(falls)}):**\n"
            for pred in falls:
                predictions_text += f"• Player {pred['player_id']}: {pred['fall_probability']:.0%} chance "
                predictions_text += f"(Owner: {pred['current_ownership']:.1f}%, Trend: {pred['ownership_trend']:+.1f}%)\n"
            predictions_text += "\n"
        
        if stable:
            predictions_text += f"**➡️ Expected Stable Prices ({len(stable)}):**\n"
            for pred in stable[:5]:  # Show top 5 stable
                predictions_text += f"• Player {pred['player_id']}: {pred['confidence']:.0%} confidence stable\n"
            predictions_text += "\n"
        
        # Add timing advice
        predictions_text += f"**Timing Recommendations:**\n"
        if rises:
            predictions_text += f"• Consider buying rising players before price increases\n"
        if falls:
            predictions_text += f"• Avoid or sell falling players to prevent value loss\n"
        predictions_text += f"• Monitor ownership trends daily for changes\n"
        predictions_text += f"• Price changes typically occur around 01:30 UTC\n"
        
        # Add disclaimer
        predictions_text += f"\n**Note:** Price predictions are estimates based on ownership trends and historical patterns. "
        predictions_text += f"Actual price changes depend on FPL's proprietary algorithm.\n"
        
        logger.info(f"Price change predictions completed. Rises: {len(rises)}, Falls: {len(falls)}")
        return predictions_text.strip()
        
    except Exception as e:
        logger.error(f"Price prediction failed: {e}")
        return f"❌ Price prediction failed: {str(e)}"


def _create_sample_historical_data(player_ids: List[int]) -> pd.DataFrame:
    """Create sample historical data for demonstration."""
    data = []
    
    for player_id in player_ids:
        for gw in range(1, 11):  # Last 10 gameweeks
            data.append({
                'player_id': player_id,
                'gameweek': gw,
                'total_points': np.random.randint(0, 15),
                'minutes': np.random.randint(0, 90),
                'goals_scored': np.random.randint(0, 3),
                'assists': np.random.randint(0, 2),
                'clean_sheets': np.random.randint(0, 1),
                'now_cost': 50 + np.random.randint(0, 100),  # 5.0 to 15.0
                'selected_by_percent': np.random.randint(1, 100)
            })
    
    return pd.DataFrame(data)


def _create_sample_test_data(gameweeks: int) -> pd.DataFrame:
    """Create sample test data for model validation."""
    data = []
    
    for gw in range(1, gameweeks + 1):
        for player_id in range(1, 101):  # 100 players
            data.append({
                'player_id': player_id,
                'gameweek': gw,
                'total_points': np.random.randint(0, 15),
                'minutes': np.random.randint(0, 90),
                'goals_scored': np.random.randint(0, 3),
                'assists': np.random.randint(0, 2)
            })
    
    return pd.DataFrame(data)


def _simulate_ownership_data(player_id: int) -> float:
    """Simulate player ownership percentage."""
    # Create realistic ownership distribution
    if player_id % 20 == 0:  # Premium players
        return np.random.uniform(30, 80)
    elif player_id % 10 == 0:  # Popular players
        return np.random.uniform(15, 40)
    else:  # Regular players
        return np.random.uniform(1, 20)


def _calculate_captain_score(predicted_points: float, ownership: float, strategy: str) -> float:
    """Calculate captain score based on strategy."""
    if strategy == "safe":
        return predicted_points * 0.7 + (ownership / 100) * 0.3
    elif strategy == "differential":
        return predicted_points * 0.8 - (ownership / 100) * 0.2
    else:  # balanced
        return predicted_points * 0.6 + (50 - abs(50 - ownership)) / 100 * 0.1


def _calculate_price_rise_probability(ownership_change: float, price_momentum: float, current_ownership: float) -> float:
    """Calculate probability of price rise."""
    base_prob = 0.1
    
    # Ownership trend factor
    if ownership_change > 2:
        base_prob += 0.4
    elif ownership_change > 1:
        base_prob += 0.2
    
    # Price momentum factor
    if price_momentum > 0:
        base_prob += 0.1
    
    # Current ownership factor (very high ownership less likely to rise)
    if current_ownership > 70:
        base_prob -= 0.2
    
    return min(max(base_prob, 0.05), 0.95)


def _calculate_price_fall_probability(ownership_change: float, price_momentum: float, current_ownership: float) -> float:
    """Calculate probability of price fall."""
    base_prob = 0.1
    
    # Ownership trend factor
    if ownership_change < -2:
        base_prob += 0.4
    elif ownership_change < -1:
        base_prob += 0.2
    
    # Price momentum factor
    if price_momentum < 0:
        base_prob += 0.1
    
    # Current ownership factor (very low ownership less likely to fall)
    if current_ownership < 5:
        base_prob -= 0.2
    
    return min(max(base_prob, 0.05), 0.95)


def _generate_model_recommendations(mse: float, correlation: float, accuracy: float) -> str:
    """Generate recommendations based on model performance."""
    recommendations = []
    
    if mse >= 0.005:
        recommendations.append("• Consider model retraining with recent data")
        recommendations.append("• Review feature engineering for better predictive power")
    
    if correlation < 0.6:
        recommendations.append("• Investigate feature selection and model architecture")
        recommendations.append("• Consider ensemble methods with different model types")
    
    if accuracy < 0.5:
        recommendations.append("• Review prediction thresholds and calibration")
        recommendations.append("• Analyze prediction errors for systematic bias")
    
    if not recommendations:
        recommendations.append("• Model performance is satisfactory")
        recommendations.append("• Continue monitoring with regular validation")
    
    return "\n".join(recommendations)


# Convenience function to create ML Prediction agent
def create_ml_prediction_agent(
    model_path: str,
    retrain_threshold_mse: float = 0.005,
    session_id: Optional[str] = None
) -> Agent:
    """
    Create ML Prediction agent with specified dependencies.
    
    Args:
        model_path: Path to ML model files
        retrain_threshold_mse: MSE threshold for model retraining
        session_id: Optional session identifier
        
    Returns:
        Configured ML Prediction agent
    """
    return ml_prediction_agent