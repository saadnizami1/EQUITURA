# models/lstm_predictor.py
"""
LSTM Stock Price Predictor for Vestara AI Platform
Real TensorFlow implementation that works with your existing system
"""

import os
import json
import numpy as np
import pandas as pd
import pickle
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

logger = logging.getLogger(__name__)

class LSTMStockPredictor:
    """LSTM Neural Network for Stock Price Prediction"""
    
    def __init__(self):
        self.sequence_length = 60  # Use 60 days to predict next day
        self.models_dir = "saved_models"
        self.ensure_models_directory()
        logger.info("✅ LSTM Stock Predictor initialized")
    
    def ensure_models_directory(self):
        """Create models directory if it doesn't exist"""
        os.makedirs(self.models_dir, exist_ok=True)
    
    def load_cached_data(self, symbol):
        """Load cached stock data and convert to DataFrame"""
        try:
            # Try to load from shared data manager cache
            from shared_data_manager import SharedDataManager
            shared_manager = SharedDataManager()
            stock_data = shared_manager.get_shared_stock_data(symbol)
            
            if not stock_data:
                logger.warning(f"No cached data found for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'Date': pd.to_datetime(stock_data['dates']),
                'Open': stock_data['opens'],
                'High': stock_data['highs'],
                'Low': stock_data['lows'],
                'Close': stock_data['closes'],
                'Volume': stock_data['volumes']
            })
            
            df = df.set_index('Date').sort_index()
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading cached data for {symbol}: {e}")
            return None
    
    def prepare_lstm_data(self, df):
        """Prepare data for LSTM training"""
        try:
            if len(df) < self.sequence_length + 10:
                logger.error(f"Insufficient data: {len(df)} points (need {self.sequence_length + 10})")
                return None, None, None
            
            # Use closing prices for prediction
            prices = df['Close'].values.reshape(-1, 1)
            
            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_prices = scaler.fit_transform(prices)
            
            # Create sequences
            X, y = [], []
            for i in range(self.sequence_length, len(scaled_prices)):
                X.append(scaled_prices[i-self.sequence_length:i, 0])
                y.append(scaled_prices[i, 0])
            
            X, y = np.array(X), np.array(y)
            
            # Reshape for LSTM [samples, time steps, features]
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            logger.info(f"📊 Prepared LSTM data: X shape {X.shape}, y shape {y.shape}")
            return X, y, scaler
            
        except Exception as e:
            logger.error(f"❌ Error preparing LSTM data: {e}")
            return None, None, None
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model architecture"""
        try:
            model = Sequential([
                # First LSTM layer
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                
                # Second LSTM layer
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                
                # Third LSTM layer
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                
                # Dense output layer
                Dense(1)
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae']
            )
            
            logger.info("✅ LSTM model architecture built")
            return model
            
        except Exception as e:
            logger.error(f"❌ Error building LSTM model: {e}")
            return None
    
    def train_model(self, symbol, stock_data=None, epochs=20, batch_size=32):
        """Train LSTM model for a specific symbol"""
        try:
            logger.info(f"🧠 Starting LSTM training for {symbol}")
            
            # Load data
            if stock_data:
                # Convert stock_data dict to DataFrame
                df = pd.DataFrame({
                    'Date': pd.to_datetime(stock_data['dates']),
                    'Open': stock_data['opens'],
                    'High': stock_data['highs'],
                    'Low': stock_data['lows'],
                    'Close': stock_data['closes'],
                    'Volume': stock_data['volumes']
                })
                df = df.set_index('Date').sort_index()
            else:
                df = self.load_cached_data(symbol)
            
            if df is None or len(df) < 90:
                logger.error(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} points")
                return False
            
            # Prepare data for LSTM
            X, y, scaler = self.prepare_lstm_data(df)
            if X is None:
                return False
            
            # Build model
            model = self.build_lstm_model((X.shape[1], 1))
            if model is None:
                return False
            
            # Train the model
            logger.info(f"🚂 Training LSTM model for {symbol} with {epochs} epochs...")
            
            history = model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0,  # Suppress training output
                shuffle=False  # Keep time series order
            )
            
            # Create model directory
            model_dir = os.path.join(self.models_dir, symbol)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(model_dir, "lstm_model.h5")
            model.save(model_path)
            
            # Save scaler
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Save training metadata
            current_price = df['Close'].iloc[-1]
            metadata = {
                'symbol': symbol,
                'training_date': datetime.now().isoformat(),
                'epochs_trained': epochs,
                'sequence_length': self.sequence_length,
                'data_points': len(df),
                'current_price': float(current_price),
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'model_architecture': 'LSTM-50-50-50-Dense',
                'tensorflow_version': tf.__version__,
                'training_time': f"{epochs} epochs",
                'price_range': {
                    'min': float(df['Close'].min()),
                    'max': float(df['Close'].max())
                }
            }
            
            metadata_path = os.path.join(model_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            final_loss = history.history['loss'][-1]
            logger.info(f"✅ LSTM training completed for {symbol}! Loss: {final_loss:.6f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ LSTM training failed for {symbol}: {e}")
            return False
    
    def predict_prices(self, symbol, stock_data=None):
        """Generate LSTM predictions for multiple timeframes"""
        try:
            logger.info(f"🔮 Generating LSTM predictions for {symbol}")
            
            # Check if model exists
            if not self.is_model_available(symbol):
                logger.warning(f"No trained model found for {symbol}")
                return None
            
            # Load model and scaler
            model_dir = os.path.join(self.models_dir, symbol)
            model_path = os.path.join(model_dir, "lstm_model.h5")
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            
            model = load_model(model_path)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            # Load data
            if stock_data:
                df = pd.DataFrame({
                    'Date': pd.to_datetime(stock_data['dates']),
                    'Close': stock_data['closes']
                })
                df = df.set_index('Date').sort_index()
            else:
                df = self.load_cached_data(symbol)
            
            if df is None or len(df) < self.sequence_length:
                logger.error(f"Insufficient data for predictions: {len(df) if df is not None else 0}")
                return None
            
            # Get last sequence for prediction
            last_sequence = df['Close'].tail(self.sequence_length).values.reshape(-1, 1)
            last_sequence_scaled = scaler.transform(last_sequence)
            
            current_price = df['Close'].iloc[-1]
            predictions = {}
            
            # Generate predictions for different timeframes
            for days in [1, 7, 30]:
                try:
                    # Prepare input sequence
                    input_sequence = last_sequence_scaled.reshape(1, self.sequence_length, 1)
                    
                    # Generate prediction
                    prediction_scaled = model.predict(input_sequence, verbose=0)[0][0]
                    predicted_price = scaler.inverse_transform([[prediction_scaled]])[0][0]
                    
                    # Apply time decay for longer predictions
                    time_decay = 0.95 ** (days - 1)  # Decay confidence over time
                    adjusted_prediction = current_price + (predicted_price - current_price) * time_decay
                    
                    # Calculate return percentage
                    predicted_return = ((adjusted_prediction - current_price) / current_price) * 100
                    
                    # Calculate confidence (higher for shorter timeframes)
                    base_confidence = 85
                    time_penalty = (days - 1) * 3  # Reduce confidence for longer predictions
                    confidence = max(60, base_confidence - time_penalty)
                    
                    # Determine direction and strength
                    direction = 'UP' if predicted_return > 0 else 'DOWN' if predicted_return < 0 else 'NEUTRAL'
                    strength = 'STRONG' if abs(predicted_return) > 3 else 'MODERATE' if abs(predicted_return) > 1 else 'WEAK'
                    
                    predictions[str(days)] = {
                        'predicted_price': round(float(adjusted_prediction), 2),
                        'predicted_return': round(float(predicted_return), 2),
                        'confidence': round(confidence, 1),
                        'direction': direction,
                        'strength': strength,
                        'model_type': 'LSTM Neural Network',
                        'timeframe': f'{days} day{"s" if days > 1 else ""}',
                        'current_price_basis': round(float(current_price), 2)
                    }
                    
                except Exception as e:
                    logger.error(f"❌ Error predicting {days}-day for {symbol}: {e}")
                    # Fallback prediction
                    predictions[str(days)] = {
                        'predicted_price': round(float(current_price), 2),
                        'predicted_return': 0.0,
                        'confidence': 50.0,
                        'direction': 'NEUTRAL',
                        'strength': 'WEAK',
                        'model_type': 'LSTM Neural Network (Fallback)',
                        'timeframe': f'{days} day{"s" if days > 1 else ""}',
                        'error': 'Prediction calculation failed'
                    }
            
            logger.info(f"✅ Generated {len(predictions)} LSTM predictions for {symbol}")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ LSTM prediction failed for {symbol}: {e}")
            return None
    
    def is_model_available(self, symbol):
        """Check if a trained model exists for the symbol"""
        try:
            model_dir = os.path.join(self.models_dir, symbol)
            model_path = os.path.join(model_dir, "lstm_model.h5")
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            metadata_path = os.path.join(model_dir, "metadata.json")
            
            return (os.path.exists(model_path) and 
                    os.path.exists(scaler_path) and 
                    os.path.exists(metadata_path))
                    
        except Exception as e:
            logger.error(f"❌ Error checking model availability for {symbol}: {e}")
            return False
    
    def get_model_info(self, symbol):
        """Get information about a trained model"""
        try:
            if not self.is_model_available(symbol):
                return None
            
            metadata_path = os.path.join(self.models_dir, symbol, "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Error getting model info for {symbol}: {e}")
            return None
    
    def get_all_trained_models(self):
        """Get list of all symbols with trained models"""
        try:
            trained_models = []
            
            if not os.path.exists(self.models_dir):
                return trained_models
            
            for item in os.listdir(self.models_dir):
                item_path = os.path.join(self.models_dir, item)
                if os.path.isdir(item_path) and self.is_model_available(item):
                    trained_models.append(item)
            
            return sorted(trained_models)
            
        except Exception as e:
            logger.error(f"❌ Error getting trained models list: {e}")
            return []
    
    def delete_model(self, symbol):
        """Delete a trained model"""
        try:
            model_dir = os.path.join(self.models_dir, symbol)
            if os.path.exists(model_dir):
                import shutil
                shutil.rmtree(model_dir)
                logger.info(f"🗑️ Deleted model for {symbol}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Error deleting model for {symbol}: {e}")
            return False
    
    def get_training_status(self):
        """Get overall training status"""
        try:
            trained_models = self.get_all_trained_models()
            
            status_info = {
                'total_trained_models': len(trained_models),
                'trained_symbols': trained_models,
                'models_directory': self.models_dir,
                'sequence_length': self.sequence_length,
                'tensorflow_version': tf.__version__,
                'last_check': datetime.now().isoformat()
            }
            
            # Get detailed info for each model
            model_details = {}
            for symbol in trained_models:
                model_info = self.get_model_info(symbol)
                if model_info:
                    model_details[symbol] = {
                        'training_date': model_info.get('training_date'),
                        'final_loss': model_info.get('final_loss'),
                        'data_points': model_info.get('data_points'),
                        'current_price': model_info.get('current_price')
                    }
            
            status_info['model_details'] = model_details
            return status_info
            
        except Exception as e:
            logger.error(f"❌ Error getting training status: {e}")
            return {'error': str(e)}