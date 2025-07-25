#!/usr/bin/env python3
"""
Vestara AI Stock Prediction Platform
Train LSTM Models for All Available Stocks
Processes all cache files and creates trained LSTM models
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle
import time

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import configuration
import config

class LSTMModelTrainer:
    """LSTM Model Trainer for Stock Prediction"""
    
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.sequence_length = 60  # Use 60 days to predict next day
        
    def prepare_data(self, stock_data):
        """Prepare data for LSTM training"""
        try:
            # Extract price data
            prices = np.array(stock_data['closes'])
            
            if len(prices) < self.sequence_length + 10:
                print(f"⚠️  Insufficient data: {len(prices)} points (need {self.sequence_length + 10})")
                return None, None, None, None
            
            # Scale the data
            prices_scaled = self.scaler.fit_transform(prices.reshape(-1, 1))
            
            # Create sequences
            X, y = [], []
            for i in range(self.sequence_length, len(prices_scaled)):
                X.append(prices_scaled[i-self.sequence_length:i, 0])
                y.append(prices_scaled[i, 0])
            
            X, y = np.array(X), np.array(y)
            
            # Reshape for LSTM [samples, time steps, features]
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            return X, y, self.scaler, prices
            
        except Exception as e:
            print(f"❌ Data preparation error: {e}")
            return None, None, None, None
    
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        model = tf.keras.Sequential([
            # First LSTM layer with dropout
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.Dropout(0.2),
            
            # Second LSTM layer with dropout
            tf.keras.layers.LSTM(50, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            
            # Third LSTM layer
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dropout(0.2),
            
            # Dense output layer
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
    
    def train_model(self, symbol, stock_data, epochs=20):
        """Train LSTM model for a specific symbol"""
        try:
            print(f"🧠 Training LSTM model for {symbol}...")
            
            # Prepare data
            X, y, scaler, original_prices = self.prepare_data(stock_data)
            if X is None:
                return False, "Insufficient data for training"
            
            print(f"📊 Training data shape: {X.shape}, Target shape: {y.shape}")
            
            # Build model
            self.model = self.build_model((X.shape[1], 1))
            
            # Train the model
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=32,
                verbose=0,  # Suppress training output
                validation_split=0.2,
                shuffle=False  # Keep time series order
            )
            
            # Create model directory
            model_dir = os.path.join("saved_models", symbol)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(model_dir, "lstm_model.h5")
            self.model.save(model_path)
            
            # Save scaler
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Save training metadata
            metadata = {
                'symbol': symbol,
                'training_date': datetime.now().isoformat(),
                'epochs': epochs,
                'sequence_length': self.sequence_length,
                'data_points': len(original_prices),
                'price_range': {
                    'min': float(min(original_prices)),
                    'max': float(max(original_prices))
                },
                'current_price': float(stock_data['current_price']),
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'model_architecture': 'LSTM-50-50-50-Dense',
                'version': '1.0'
            }
            
            metadata_path = os.path.join(model_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            final_loss = history.history['loss'][-1]
            print(f"✅ {symbol} model trained successfully! Final loss: {final_loss:.6f}")
            return True, f"Model trained with loss: {final_loss:.6f}"
            
        except Exception as e:
            print(f"❌ Training error for {symbol}: {e}")
            return False, str(e)

def load_stock_data(symbol):
    """Load stock data from cache file"""
    try:
        cache_file = f"data/cache/finnhub_{symbol}.json"
        if not os.path.exists(cache_file):
            return None
        
        with open(cache_file, 'r') as f:
            return json.load(f)
    except:
        return None

def main():
    """Train LSTM models for all available stocks"""
    print("🚀 VESTARA AI - LSTM TRAINING SYSTEM")
    print("=" * 60)
    print(f"🧠 TensorFlow {tf.__version__}")
    print(f"📊 Training models for all available stocks")
    print("=" * 60)
    
    # Ensure directories exist
    os.makedirs("saved_models", exist_ok=True)
    
    trainer = LSTMModelTrainer()
    successful = 0
    failed = 0
    results = []
    
    # Get all stocks from config
    all_stocks = config.STOCK_SYMBOLS
    
    print(f"\n🔍 Scanning for available stock data...")
    available_stocks = []
    
    for symbol in all_stocks:
        stock_data = load_stock_data(symbol)
        if stock_data:
            available_stocks.append(symbol)
            print(f"✅ {symbol}: {stock_data.get('data_points', 0)} data points")
        else:
            print(f"❌ {symbol}: No cache file found")
    
    print(f"\n📊 Found data for {len(available_stocks)}/{len(all_stocks)} stocks")
    
    if not available_stocks:
        print("❌ No stock data found! Run 'python generate_missing_stock_data.py' first")
        return
    
    print(f"\n🚂 Starting LSTM training for {len(available_stocks)} stocks...")
    print("=" * 60)
    
    for i, symbol in enumerate(available_stocks, 1):
        print(f"\n[{i}/{len(available_stocks)}] Training {symbol}...")
        
        # Load stock data
        stock_data = load_stock_data(symbol)
        if not stock_data:
            print(f"❌ Could not load data for {symbol}")
            failed += 1
            continue
        
        # Train model
        success, message = trainer.train_model(symbol, stock_data)
        
        if success:
            successful += 1
            results.append({
                'symbol': symbol,
                'status': 'success',
                'message': message,
                'data_points': len(stock_data.get('closes', []))
            })
        else:
            failed += 1
            results.append({
                'symbol': symbol,
                'status': 'failed',
                'message': message
            })
        
        # Small delay between trainings
        if i < len(available_stocks):
            time.sleep(1)
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 TRAINING SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully trained: {successful}/{len(available_stocks)} models")
    print(f"❌ Failed: {failed}/{len(available_stocks)} models")
    
    if successful > 0:
        print(f"\n🎉 SUCCESS! {successful} LSTM models ready for predictions")
        print("📂 Models saved in: saved_models/")
        print("\n🌐 Start your platform: python app.py")
        print("🧠 Test predictions: http://localhost:5000/api/stock/AAPL/ml-predictions")
    
    # Show detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for result in results:
        status_icon = "✅" if result['status'] == 'success' else "❌"
        print(f"{status_icon} {result['symbol']}: {result['message']}")
    
    # Save training report
    report = {
        'training_date': datetime.now().isoformat(),
        'tensorflow_version': tf.__version__,
        'total_stocks': len(available_stocks),
        'successful': successful,
        'failed': failed,
        'results': results
    }
    
    with open('training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Training report saved: training_report.json")

if __name__ == "__main__":
    main()