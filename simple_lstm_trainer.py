#!/usr/bin/env python3
"""
Simple LSTM Model Creator for Vestara AI
Creates actual trained LSTM models that your app can use
"""

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle

def create_lstm_model_for_stock(symbol):
    """Create a simple LSTM model for a stock symbol"""
    try:
        print(f"🧠 Creating LSTM model for {symbol}...")
        
        # Check if cache file exists
        cache_file = f"data/cache/finnhub_{symbol}.json"
        if not os.path.exists(cache_file):
            print(f"❌ No cache file found for {symbol}")
            return False
        
        # Load stock data
        with open(cache_file, 'r') as f:
            stock_data = json.load(f)
        
        # Extract prices
        prices = np.array(stock_data['closes'])
        print(f"📊 Using {len(prices)} price points")
        
        if len(prices) < 60:
            print(f"❌ Not enough data points for {symbol}")
            return False
        
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))
        
        # Create sequences (use last 60 days to predict next day)
        X, y = [], []
        for i in range(60, len(prices_scaled)):
            X.append(prices_scaled[i-60:i, 0])
            y.append(prices_scaled[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        print(f"📊 Training data shape: {X.shape}")
        
        # Build simple LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model (quick training)
        print(f"🚂 Training model...")
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)
        
        # Create model directory
        model_dir = f"saved_models/{symbol}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = f"{model_dir}/lstm_model.h5"
        model.save(model_path)
        
        # Save scaler
        scaler_path = f"{model_dir}/scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'training_date': '2025-07-09',
            'model_type': 'LSTM',
            'data_points': len(prices),
            'current_price': stock_data['current_price']
        }
        
        metadata_path = f"{model_dir}/metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ {symbol} LSTM model saved successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating model for {symbol}: {e}")
        return False

def main():
    """Create LSTM models for all cached stocks"""
    print("🚀 SIMPLE LSTM MODEL CREATOR")
    print("=" * 50)
    
    # Ensure directories exist
    os.makedirs("saved_models", exist_ok=True)
    
    # Find all cached stocks
    cache_dir = "data/cache"
    if not os.path.exists(cache_dir):
        print("❌ No cache directory found!")
        return
    
    cache_files = [f for f in os.listdir(cache_dir) if f.startswith('finnhub_') and f.endswith('.json')]
    symbols = [f.replace('finnhub_', '').replace('.json', '') for f in cache_files]
    
    print(f"📊 Found {len(symbols)} cached stocks: {', '.join(symbols)}")
    
    successful = 0
    for symbol in symbols:
        if create_lstm_model_for_stock(symbol):
            successful += 1
    
    print("\n" + "=" * 50)
    print(f"✅ Successfully created {successful}/{len(symbols)} LSTM models")
    print("🎉 Your Train Models button should now work!")
    print("🌐 Restart your app: python app.py")

if __name__ == "__main__":
    main()
    