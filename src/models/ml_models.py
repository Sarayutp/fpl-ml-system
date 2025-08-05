"""
ML models for FPL player prediction following 2024 research benchmarks.
Target: MSE < 0.003 for player points prediction (LSTM benchmark from research).
"""

import logging
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from ..config.settings import settings

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Advanced feature engineering for FPL predictions following research patterns.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
    
    def create_features(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set following research patterns.
        
        Args:
            player_data: Raw player data from FPL API
            
        Returns:
            DataFrame with engineered features
        """
        df = player_data.copy()
        
        # Ensure data is sorted by player and gameweek
        df = df.sort_values(['player_id', 'gameweek'])
        
        # CRITICAL: Time series features for 5 gameweeks (research pattern)
        df = self._add_rolling_features(df)
        
        # PATTERN: Per-90 minute statistics from research
        df = self._add_per_90_features(df)
        
        # Expected vs actual performance features
        df = self._add_expected_features(df)
        
        # Fixture difficulty and team strength features
        df = self._add_fixture_features(df)
        
        # Position-specific features
        df = self._add_position_features(df)
        
        # Price and ownership trend features
        df = self._add_market_features(df)
        
        # Fill missing values
        df = self._handle_missing_values(df)
        
        # Store feature columns for later use
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.feature_columns = [col for col in numeric_cols if col not in ['player_id', 'gameweek', 'total_points']]
        
        logger.info(f"Created {len(self.feature_columns)} features for {len(df)} records")
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics features."""
        # 5-game rolling averages (key research pattern)
        for window in [3, 5, 10]:
            df[f'form_last_{window}'] = df.groupby('player_id')['total_points'].rolling(window, min_periods=1).mean()
            df[f'minutes_last_{window}'] = df.groupby('player_id')['minutes'].rolling(window, min_periods=1).mean()
            df[f'goals_last_{window}'] = df.groupby('player_id')['goals_scored'].rolling(window, min_periods=1).sum()
            df[f'assists_last_{window}'] = df.groupby('player_id')['assists'].rolling(window, min_periods=1).sum()
        
        # Rolling standard deviations (consistency metrics)
        df['points_volatility'] = df.groupby('player_id')['total_points'].rolling(5, min_periods=2).std()
        df['minutes_consistency'] = df.groupby('player_id')['minutes'].rolling(5, min_periods=2).std()
        
        return df
    
    def _add_per_90_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add per-90 minute statistics."""
        # Avoid division by zero
        df['minutes_safe'] = df['minutes'].replace(0, 1)
        
        # Per-90 minute stats
        df['goals_per_90'] = (df['goals_scored'] / df['minutes_safe']) * 90
        df['assists_per_90'] = (df['assists'] / df['minutes_safe']) * 90
        df['points_per_90'] = (df['total_points'] / df['minutes_safe']) * 90
        df['clean_sheets_per_90'] = (df['clean_sheets'] / df['minutes_safe']) * 90
        
        # Advanced per-90 stats
        if 'saves' in df.columns:
            df['saves_per_90'] = (df['saves'] / df['minutes_safe']) * 90
        if 'bonus' in df.columns:
            df['bonus_per_90'] = (df['bonus'] / df['minutes_safe']) * 90
        
        return df
    
    def _add_expected_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add expected vs actual performance features."""
        if 'expected_goals' in df.columns:
            df['xg_vs_goals'] = pd.to_numeric(df['expected_goals'], errors='coerce') - df['goals_scored']
            df['xg_overperformance'] = df['goals_scored'] - pd.to_numeric(df['expected_goals'], errors='coerce')
        
        if 'expected_assists' in df.columns:
            df['xa_vs_assists'] = pd.to_numeric(df['expected_assists'], errors='coerce') - df['assists']
            df['xa_overperformance'] = df['assists'] - pd.to_numeric(df['expected_assists'], errors='coerce')
        
        # ICT Index features (if available)
        if all(col in df.columns for col in ['influence', 'creativity', 'threat']):
            df['influence_num'] = pd.to_numeric(df['influence'], errors='coerce')
            df['creativity_num'] = pd.to_numeric(df['creativity'], errors='coerce')
            df['threat_num'] = pd.to_numeric(df['threat'], errors='coerce')
            df['ict_combined'] = df['influence_num'] + df['creativity_num'] + df['threat_num']
        
        return df
    
    def _add_fixture_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fixture difficulty and team strength features."""
        # Placeholder for fixture difficulty (would be joined from fixtures data) 
        df['fixture_difficulty'] = 3  # Default neutral difficulty
        df['is_home'] = 1  # Assume home games for now
        
        # Team strength placeholders (would be joined from teams data)
        df['team_strength'] = 3  # Default neutral strength
        df['opponent_strength'] = 3  # Default neutral opponent
        
        # Upcoming fixtures difficulty (would require future fixture data)
        df['next_5_fixture_difficulty'] = 3.0
        
        return df
    
    def _add_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add position-specific features."""
        # Position encoding
        position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        df['position_name'] = df['element_type'].map(position_map)
        
        # Position-specific features
        df['is_goalkeeper'] = (df['element_type'] == 1).astype(int)
        df['is_defender'] = (df['element_type'] == 2).astype(int)
        df['is_midfielder'] = (df['element_type'] == 3).astype(int)
        df['is_forward'] = (df['element_type'] == 4).astype(int)
        
        # Position-specific performance metrics
        if 'clean_sheets' in df.columns:
            df['defensive_returns'] = df['clean_sheets'] * df['is_defender'] + df['clean_sheets'] * df['is_goalkeeper'] * 2
        
        return df
    
    def _add_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price and ownership trend features."""
        # Price momentum
        df['price_change_momentum'] = df.groupby('player_id')['now_cost'].diff()
        df['price_change_rate'] = df.groupby('player_id')['price_change_momentum'].rolling(3, min_periods=1).mean()
        
        # Ownership trends (if available)
        if 'selected_by_percent' in df.columns:
            df['ownership_change'] = df.groupby('player_id')['selected_by_percent'].diff()
            df['ownership_trend'] = df.groupby('player_id')['ownership_change'].rolling(3, min_periods=1).mean()
        
        # Transfer trends (if available)
        if 'transfers_in_event' in df.columns:
            df['net_transfers'] = df['transfers_in_event'] - df.get('transfers_out_event', 0)
            df['transfer_momentum'] = df.groupby('player_id')['net_transfers'].rolling(3, min_periods=1).mean()
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values appropriately."""
        # Fill numeric columns with 0 or median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['total_points', 'minutes', 'goals_scored', 'assists']:
                df[col] = df[col].fillna(0)  # Performance stats default to 0
            else:
                df[col] = df[col].fillna(df[col].median())  # Other stats use median
        
        # Fill categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')
        
        return df


class PlayerPredictor:
    """
    ML model for predicting player FPL points using XGBoost and LSTM ensemble.
    Target benchmark: MSE < 0.003 (from 2024 research).
    """
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.models = {
            'xgboost': XGBRegressor(
                n_estimators=settings.xgboost_n_estimators,
                max_depth=settings.xgboost_max_depth,
                learning_rate=settings.xgboost_learning_rate,
                random_state=42,
                n_jobs=-1
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        }
        self.lstm_model = None
        self.ensemble_weights = {'xgboost': 0.4, 'random_forest': 0.2, 'lstm': 0.4}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
        self.model_dir = Path(settings.model_cache_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model training/prediction."""
        processed_data = self.feature_engineer.create_features(data)
        self.feature_columns = self.feature_engineer.feature_columns
        return processed_data
    
    def create_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        """
        Create LSTM model architecture following research patterns.
        
        Args:
            input_shape: (sequence_length, n_features)
            
        Returns:
            Compiled LSTM model
        """
        model = Sequential([
            Input(shape=input_shape),
            
            # 1D Convolutional layer for spatial feature extraction
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            
            # LSTM layers for temporal patterns
            LSTM(128, return_sequences=True, dropout=0.2),
            LSTM(64, return_sequences=False, dropout=0.2),
            
            # Dense layers
            Dense(50, activation='relu'),
            Dropout(0.3),
            Dense(25, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear')  # Regression output
        ])
        
        # Compile with Adam optimizer and MSE loss (research standard)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        
        return model
    
    def prepare_lstm_sequences(self, data: pd.DataFrame, sequence_length: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.
        
        Args:
            data: Processed data with features
            sequence_length: Number of gameweeks to look back
            
        Returns:
            X, y arrays for LSTM training
        """
        X_sequences = []
        y_sequences = []
        
        # Group by player and create sequences
        for player_id in data['player_id'].unique():
            player_data = data[data['player_id'] == player_id].sort_values('gameweek')
            
            if len(player_data) < sequence_length + 1:
                continue
            
            # Extract features and target
            features = player_data[self.feature_columns].values
            targets = player_data['total_points'].values
            
            # Create sequences
            for i in range(len(features) - sequence_length):
                X_sequences.append(features[i:i+sequence_length])
                y_sequences.append(targets[i+sequence_length])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train(self, training_data: pd.DataFrame) -> Dict[str, float]:
        """
        Train ensemble model with cross-validation following research patterns.
        
        Args:
            training_data: Historical FPL data with player performance
            
        Returns:
            Dictionary with model performance scores
        """
        logger.info("Starting model training...")
        
        # Prepare features
        processed_data = self.prepare_features(training_data)
        
        if processed_data.empty or 'total_points' not in processed_data.columns:
            raise ValueError("Training data is empty or missing target column")
        
        X = processed_data[self.feature_columns]
        y = processed_data['total_points']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # CRITICAL: Time series cross-validation (research pattern)
        tscv = TimeSeriesSplit(n_splits=5)
        scores = {}
        
        # Train XGBoost and Random Forest
        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            
            cv_scores = cross_val_score(
                model, X_scaled, y, 
                cv=tscv, 
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            mse_scores = -cv_scores
            scores[name] = {
                'mse_mean': mse_scores.mean(),
                'mse_std': mse_scores.std(),
                'mae_mean': None  # Would calculate MAE separately
            }
            
            # Train final model on full data
            model.fit(X_scaled, y)
            
            logger.info(f"{name} MSE: {mse_scores.mean():.6f} ± {mse_scores.std():.6f}")
        
        # Train LSTM model
        logger.info("Training LSTM model...")
        X_lstm, y_lstm = self.prepare_lstm_sequences(processed_data)
        
        if len(X_lstm) > 0:
            # Create and train LSTM
            self.lstm_model = self.create_lstm_model((X_lstm.shape[1], X_lstm.shape[2]))
            
            # Callbacks for training
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            
            # Train LSTM
            history = self.lstm_model.fit(
                X_lstm, y_lstm,
                epochs=100,
                batch_size=64,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            # Get LSTM validation score
            val_loss = min(history.history['val_loss'])
            scores['lstm'] = {
                'mse_mean': val_loss,
                'mse_std': 0.0,
                'mae_mean': min(history.history.get('val_mean_absolute_error', [val_loss]))
            }
            
            logger.info(f"LSTM MSE: {val_loss:.6f}")
        else:
            logger.warning("Not enough data for LSTM training")
            scores['lstm'] = {'mse_mean': float('inf'), 'mse_std': 0.0, 'mae_mean': float('inf')}
        
        self.is_trained = True
        
        # Save models
        self.save_models()
        
        # Log performance summary
        avg_mse = np.mean([score['mse_mean'] for score in scores.values() if score['mse_mean'] != float('inf')])
        logger.info(f"Training complete. Average MSE: {avg_mse:.6f}")
        
        # Check if we meet research benchmark
        if avg_mse < 0.003:
            logger.info("✅ Model meets research benchmark (MSE < 0.003)")
        else:
            logger.warning(f"⚠️ Model MSE {avg_mse:.6f} exceeds research benchmark of 0.003")
        
        return scores
    
    def predict_ensemble(self, features: pd.DataFrame, gameweeks: int = 1) -> np.ndarray:
        """
        Make ensemble predictions combining all models.
        
        Args:
            features: Feature data for prediction
            gameweeks: Number of gameweeks to predict
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        X = features[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        
        # Get predictions from XGBoost and Random Forest
        for name, model in self.models.items():
            try:
                predictions[name] = model.predict(X_scaled)
            except Exception as e:
                logger.error(f"Error predicting with {name}: {e}")
                predictions[name] = np.zeros(len(X))
        
        # Get LSTM predictions if available
        if self.lstm_model is not None and len(features) >= 5:
            try:
                # Prepare LSTM sequences (simplified for single prediction)
                X_lstm, _ = self.prepare_lstm_sequences(features)
                if len(X_lstm) > 0:
                    lstm_pred = self.lstm_model.predict(X_lstm[-1:], verbose=0)
                    predictions['lstm'] = np.full(len(X), lstm_pred[0][0])
                else:
                    predictions['lstm'] = np.zeros(len(X))
            except Exception as e:
                logger.error(f"Error predicting with LSTM: {e}")
                predictions['lstm'] = np.zeros(len(X))
        else:
            predictions['lstm'] = np.zeros(len(X))
        
        # Weighted ensemble
        ensemble_pred = np.zeros(len(X))
        total_weight = 0
        
        for name, weight in self.ensemble_weights.items():
            if name in predictions:
                ensemble_pred += predictions[name] * weight
                total_weight += weight
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return ensemble_pred
    
    def save_models(self) -> None:
        """Save trained models to disk."""
        try:
            # Save XGBoost and Random Forest
            for name, model in self.models.items():
                model_path = self.model_dir / f"{name}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Save LSTM
            if self.lstm_model is not None:
                lstm_path = self.model_dir / "lstm_model"
                self.lstm_model.save(lstm_path)
            
            # Save scaler and feature columns
            scaler_path = self.model_dir / "scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            features_path = self.model_dir / "feature_columns.pkl"
            with open(features_path, 'wb') as f:
                pickle.dump(self.feature_columns, f)
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def load_models(self) -> bool:
        """
        Load trained models from disk.
        
        Returns:
            True if models loaded successfully
        """
        try:
            # Load XGBoost and Random Forest
            for name in self.models.keys():
                model_path = self.model_dir / f"{name}_model.pkl"
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[name] = pickle.load(f)
            
            # Load LSTM
            lstm_path = self.model_dir / "lstm_model"
            if lstm_path.exists():
                self.lstm_model = tf.keras.models.load_model(lstm_path)
            
            # Load scaler and feature columns
            scaler_path = self.model_dir / "scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            features_path = self.model_dir / "feature_columns.pkl"
            if features_path.exists():
                with open(features_path, 'rb') as f:
                    self.feature_columns = pickle.load(f)
            
            self.is_trained = True
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False