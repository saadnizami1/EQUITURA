# models/ensemble_predictor.py
"""
Ensemble Prediction Engine
Combines LSTM Neural Networks with Technical Analysis
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import json
import logging
from datetime import datetime
from lstm_predictor import LSTMStockPredictor

logger = logging.getLogger(__name__)

class EnsemblePredictionEngine:
    """Advanced ensemble combining LSTM, Random Forest, and Technical Analysis"""
    
    def __init__(self):
        self.lstm_predictor = LSTMStockPredictor()
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Ensemble weights (can be optimized)
        self.weights = {
            'lstm': 0.5,        # Neural network prediction
            'random_forest': 0.3, # Tree-based prediction
            'technical': 0.2     # Technical analysis prediction
        }
        
        logger.info("✅ Ensemble Prediction Engine initialized")
    
    def prepare_features_for_rf(self, df, technical_indicators=None):
        """Prepare feature matrix for Random Forest"""
        try:
            features = []
            feature_names = []
            
            # Price-based features
            if len(df) >= 20:
                # Moving averages
                df['SMA_5'] = df['Close'].rolling(5).mean()
                df['SMA_10'] = df['Close'].rolling(10).mean()
                df['SMA_20'] = df['Close'].rolling(20).mean()
                
                # Price ratios
                df['Price_SMA5_Ratio'] = df['Close'] / df['SMA_5']
                df['Price_SMA10_Ratio'] = df['Close'] / df['SMA_10']
                df['Price_SMA20_Ratio'] = df['Close'] / df['SMA_20']
                
                # Volatility
                df['Volatility_5'] = df['Close'].rolling(5).std()
                df['Volatility_10'] = df['Close'].rolling(10).std()
                
                # Volume indicators
                df['Volume_SMA'] = df['Volume'].rolling(10).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
                
                # Price momentum
                df['Price_Change_1'] = df['Close'].pct_change(1)
                df['Price_Change_5'] = df['Close'].pct_change(5)
                
                # High-Low ratios
                df['HL_Ratio'] = (df['High'] - df['Low']) / df['Close']
                df['CO_Ratio'] = (df['Close'] - df['Open']) / df['Open']
                
                feature_cols = [
                    'Price_SMA5_Ratio', 'Price_SMA10_Ratio', 'Price_SMA20_Ratio',
                    'Volatility_5', 'Volatility_10', 'Volume_Ratio',
                    'Price_Change_1', 'Price_Change_5', 'HL_Ratio', 'CO_Ratio'
                ]
                
                # Add technical indicators if provided
                if technical_indicators:
                    for indicator, value in technical_indicators.items():
                        if value is not None and isinstance(value, (int, float)):
                            df[f'Tech_{indicator}'] = value
                            feature_cols.append(f'Tech_{indicator}')
                
                # Drop NaN and return features
                df_clean = df[feature_cols].dropna()
                
                if len(df_clean) > 0:
                    return df_clean.values, feature_cols, df_clean
                
            return None, None, None
            
        except Exception as e:
            logger.error(f"Error preparing RF features: {e}")
            return None, None, None
    
    def train_random_forest(self, symbol):
        """Train Random Forest on technical features"""
        try:
            logger.info(f"🌲 Training Random Forest for {symbol}")
            
            # Load cached data
            df = self.lstm_predictor.load_cached_data(symbol)
            if df is None or len(df) < 50:
                logger.warning(f"Insufficient data for RF training: {len(df) if df is not None else 0} rows")
                return False
            
            # Prepare features
            X, feature_names, df_clean = self.prepare_features_for_rf(df)
            if X is None:
                logger.error("Failed to prepare features for Random Forest")
                return False
            
            # Create target (next day price change)
            y = df_clean['Close'].shift(-1).pct_change().dropna()
            X = X[:-1]  # Remove last row to match y
            
            if len(X) != len(y):
                logger.error(f"Feature-target length mismatch: {len(X)} vs {len(y)}")
                return False
            
            # Train Random Forest
            self.rf_model.fit(X, y)
            
            # Save model
            model_dir = f"saved_models/{symbol}"
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(self.rf_model, f"{model_dir}/rf_model.pkl")
            joblib.dump(feature_names, f"{model_dir}/rf_features.pkl")
            
            # Save feature importance
            importance_data = {
                'features': feature_names,
                'importance': self.rf_model.feature_importances_.tolist(),
                'training_date': datetime.now().isoformat(),
                'training_samples': len(X)
            }
            
            with open(f"{model_dir}/rf_importance.json", 'w') as f:
                json.dump(importance_data, f, indent=2)
            
            logger.info(f"✅ Random Forest trained for {symbol} with {len(X)} samples")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training Random Forest for {symbol}: {e}")
            return False
    
    def predict_with_random_forest(self, symbol, days_ahead=[1, 7, 30]):
        """Generate Random Forest predictions"""
        try:
            # Load model
            model_path = f"saved_models/{symbol}/rf_model.pkl"
            features_path = f"saved_models/{symbol}/rf_features.pkl"
            
            if not os.path.exists(model_path) or not os.path.exists(features_path):
                logger.warning(f"No Random Forest model found for {symbol}")
                return None
            
            rf_model = joblib.load(model_path)
            feature_names = joblib.load(features_path)
            
            # Load recent data
            df = self.lstm_predictor.load_cached_data(symbol)
            if df is None:
                return None
            
            # Prepare features
            X, _, df_clean = self.prepare_features_for_rf(df)
            if X is None:
                return None
            
            # Use last row for prediction
            X_latest = X[-1:] if len(X) > 0 else None
            if X_latest is None:
                return None
            
            # Generate prediction
            predicted_change = rf_model.predict(X_latest)[0]
            current_price = float(df['Close'].iloc[-1])
            
            predictions = {}
            
            for days in days_ahead:
                # Adjust prediction for time horizon
                time_decay = 1.0 / (1 + days * 0.1)
                adjusted_change = predicted_change * time_decay
                predicted_price = current_price * (1 + adjusted_change)
                
                # Calculate confidence
                confidence = max(55, min(80, 70 - abs(adjusted_change * 100) * 2))
                
                predictions[str(days)] = {
                    'predicted_price': round(predicted_price, 2),
                    'predicted_return': round(adjusted_change * 100, 2),
                    'confidence': round(confidence, 1),
                    'direction': 'UP' if adjusted_change > 0 else 'DOWN' if adjusted_change < 0 else 'NEUTRAL',
                    'strength': self._get_strength(adjusted_change * 100),
                    'model_type': 'Random Forest',
                    'timeframe': f'{days} day{"s" if days > 1 else ""}'
                }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in Random Forest prediction for {symbol}: {e}")
            return None
    
    def train_ensemble(self, symbol):
        """Train all ensemble components"""
        try:
            logger.info(f"🎯 Training ensemble models for {symbol}")
            
            # Train LSTM
            lstm_success = self.lstm_predictor.train_model(symbol)
            
            # Train Random Forest
            rf_success = self.train_random_forest(symbol)
            
            success_count = sum([lstm_success, rf_success])
            
            logger.info(f"✅ Ensemble training complete for {symbol}: {success_count}/2 models trained")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ Error training ensemble for {symbol}: {e}")
            return False
    
    def ensemble_predict(self, symbol, technical_data=None, days_ahead=[1, 7, 30]):
        """Generate ensemble predictions combining all models"""
        try:
            logger.info(f"🎯 Generating ensemble predictions for {symbol}")
            
            # Get predictions from all models
            lstm_preds = self.lstm_predictor.predict_prices(symbol, days_ahead)
            rf_preds = self.predict_with_random_forest(symbol, days_ahead)
            tech_preds = self._extract_technical_predictions(technical_data, days_ahead)
            
            # Current price for fallback
            df = self.lstm_predictor.load_cached_data(symbol)
            current_price = float(df['Close'].iloc[-1]) if df is not None else 100
            
            ensemble_predictions = {}
            
            for days in days_ahead:
                day_str = str(days)
                
                # Collect predictions from available models
                predictions = []
                weights = []
                
                if lstm_preds and day_str in lstm_preds:
                    predictions.append(lstm_preds[day_str]['predicted_price'])
                    weights.append(self.weights['lstm'])
                
                if rf_preds and day_str in rf_preds:
                    predictions.append(rf_preds[day_str]['predicted_price'])
                    weights.append(self.weights['random_forest'])
                
                if tech_preds and day_str in tech_preds:
                    predictions.append(tech_preds[day_str]['predicted_price'])
                    weights.append(self.weights['technical'])
                
                # Calculate ensemble prediction
                if predictions:
                    # Normalize weights
                    total_weight = sum(weights)
                    normalized_weights = [w/total_weight for w in weights]
                    
                    # Weighted average
                    ensemble_price = sum(p * w for p, w in zip(predictions, normalized_weights))
                    
                    # Calculate confidence based on agreement
                    price_std = np.std(predictions) if len(predictions) > 1 else 0
                    agreement_bonus = max(0, 20 - price_std)
                    base_confidence = 70 + agreement_bonus
                    
                    # Adjust for number of models
                    model_bonus = len(predictions) * 5
                    ensemble_confidence = min(95, base_confidence + model_bonus)
                    
                else:
                    # Fallback to current price
                    ensemble_price = current_price
                    ensemble_confidence = 50
                
                # Calculate return
                ensemble_return = (ensemble_price - current_price) / current_price * 100
                
                ensemble_predictions[day_str] = {
                    'predicted_price': round(ensemble_price, 2),
                    'predicted_return': round(ensemble_return, 2),
                    'confidence': round(ensemble_confidence, 1),
                    'direction': 'UP' if ensemble_return > 0 else 'DOWN' if ensemble_return < 0 else 'NEUTRAL',
                    'strength': self._get_strength(ensemble_return),
                    'model_type': f'Ensemble ({len(predictions)} models)',
                    'timeframe': f'{days} day{"s" if days > 1 else ""}',
                    'component_predictions': {
                        'lstm': lstm_preds.get(day_str) if lstm_preds else None,
                        'random_forest': rf_preds.get(day_str) if rf_preds else None,
                        'technical': tech_preds.get(day_str) if tech_preds else None
                    },
                    'ensemble_weights': {
                        'lstm': self.weights['lstm'],
                        'random_forest': self.weights['random_forest'],
                        'technical': self.weights['technical']
                    }
                }
            
            logger.info(f"✅ Generated ensemble predictions for {symbol}")
            return ensemble_predictions
            
        except Exception as e:
            logger.error(f"❌ Error in ensemble prediction for {symbol}: {e}")
            return None
    
    def _extract_technical_predictions(self, technical_data, days_ahead):
        """Extract predictions from technical analysis data"""
        if not technical_data or 'signals' not in technical_data:
            return None
        
        try:
            signals = technical_data['signals']
            overall_score = signals.get('overall_score', 0)
            current_price = technical_data.get('current_price', 100)
            
            # Convert signal score to price change
            base_change = (overall_score / 100) * 0.04  # Max 4% change
            
            predictions = {}
            for days in days_ahead:
                time_decay = 1.0 / (1 + days * 0.1)
                predicted_return = base_change * time_decay
                predicted_price = current_price * (1 + predicted_return)
                
                predictions[str(days)] = {
                    'predicted_price': round(predicted_price, 2),
                    'predicted_return': round(predicted_return * 100, 2),
                    'confidence': signals.get('confidence', 60),
                    'direction': 'UP' if predicted_return > 0 else 'DOWN' if predicted_return < 0 else 'NEUTRAL',
                    'strength': self._get_strength(predicted_return * 100),
                    'model_type': 'Technical Analysis',
                    'timeframe': f'{days} day{"s" if days > 1 else ""}'
                }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error extracting technical predictions: {e}")
            return None
    
    def _get_strength(self, return_pct):
        """Determine prediction strength"""
        abs_return = abs(return_pct)
        if abs_return > 5:
            return 'STRONG'
        elif abs_return > 2:
            return 'MODERATE'
        else:
            return 'WEAK'
    
    def get_ensemble_status(self, symbol):
        """Get status of all ensemble components"""
        try:
            status = {
                'symbol': symbol,
                'lstm_available': self.lstm_predictor.is_model_available(symbol),
                'rf_available': os.path.exists(f"saved_models/{symbol}/rf_model.pkl"),
                'lstm_info': self.lstm_predictor.get_model_info(symbol),
                'ensemble_ready': False
            }
            
            status['ensemble_ready'] = status['lstm_available'] or status['rf_available']
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting ensemble status for {symbol}: {e}")
            return None