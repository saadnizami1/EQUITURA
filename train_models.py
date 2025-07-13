# train_models.py
"""
Quick training script for Vestara ML models
"""
import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, 'models')
sys.path.insert(0, models_dir)

print("🧠 Vestara ML Training Script")
print("=" * 50)

try:
    from lstm_predictor import LSTMStockPredictor
    from ensemble_predictor import EnsemblePredictionEngine
    
    print("✅ ML models imported successfully")
    
    # Initialize predictors
    lstm = LSTMStockPredictor()
    ensemble = EnsemblePredictionEngine()
    
    # Test data loading first
    print("\n📊 Testing data loading...")
    data = lstm.load_cached_data('AAPL')
    
    if data is None:
        print("❌ No cached data found for AAPL")
        print("💡 Make sure your app has fetched data first by visiting http://localhost:5000")
        print("💡 Click on AAPL in the dashboard to cache data")
        exit(1)
    
    print(f"✅ Found {len(data)} data points for AAPL")
    print(f"📈 Data available from {data.index.min()} to {data.index.max()}")
    print(f"💰 Current close price: ${data['Close'].iloc[-1]:.2f}")
    
    # Train LSTM
    print("\n🧠 Training LSTM for AAPL...")
    print("⏳ This may take 1-2 minutes...")
    lstm_success = lstm.train_model('AAPL', epochs=15)
    
    if lstm_success:
        print("✅ LSTM training completed successfully!")
    else:
        print("❌ LSTM training failed")
    
    # Train Ensemble
    print("\n🌲 Training Ensemble for AAPL...")
    ensemble_success = ensemble.train_random_forest('AAPL')
    
    if ensemble_success:
        print("✅ Ensemble training completed successfully!")
    else:
        print("❌ Ensemble training failed")
    
    # Test predictions if training succeeded
    if lstm_success:
        print("\n🔮 Testing LSTM predictions...")
        try:
            predictions = lstm.predict_prices('AAPL')
            if predictions.get('status') == 'success':
                print("✅ LSTM predictions working!")
                # Show a sample prediction
                preds = predictions.get('predictions', {})
                if '1' in preds:
                    pred_1day = preds['1']
                    print(f"   📈 1-day prediction: ${pred_1day['predicted_price']} ({pred_1day['predicted_return']:+.2f}%)")
            else:
                print(f"❌ LSTM predictions failed: {predictions.get('message', 'Unknown error')}")
        except Exception as e:
            print(f"❌ LSTM prediction test failed: {e}")
    
    if ensemble_success:
        print("\n🎯 Testing Ensemble predictions...")
        try:
            ensemble_preds = ensemble.ensemble_predict('AAPL')
            if ensemble_preds.get('status') == 'success':
                print("✅ Ensemble predictions working!")
                # Show a sample prediction
                preds = ensemble_preds.get('predictions', {})
                if '1_day' in preds:
                    pred_1day = preds['1_day']
                    print(f"   📊 1-day ensemble: ${pred_1day['predicted_price']} ({pred_1day['predicted_return']:+.2f}%)")
            else:
                print(f"❌ Ensemble predictions failed: {ensemble_preds.get('message', 'Unknown error')}")
        except Exception as e:
            print(f"❌ Ensemble prediction test failed: {e}")
    
    print("\n" + "=" * 50)
    if lstm_success or ensemble_success:
        print("🎉 TRAINING COMPLETED!")
        if lstm_success and ensemble_success:
            print("✅ Both LSTM and Ensemble models are ready")
        elif lstm_success:
            print("✅ LSTM model is ready (Ensemble had issues)")
        else:
            print("✅ Ensemble model is ready (LSTM had issues)")
        print("🚀 Restart your Flask app and ML predictions should work!")
        print("🌐 Visit: http://localhost:5000")
    else:
        print("❌ All training failed. Check errors above.")
        print("💡 Make sure TensorFlow is properly installed")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're in the virtual environment")
except Exception as e:
    print(f"❌ Training error: {e}")
    import traceback
    traceback.print_exc()

print("\n🔍 Debug Info:")
print(f"   Current directory: {os.getcwd()}")
print(f"   Models directory: {models_dir}")
print(f"   Python path includes models: {models_dir in sys.path}")