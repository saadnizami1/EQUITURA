"""
AUTO-RETRAIN SCHEDULER - WEEKLY AUTOMATION
Automatically trains all LSTM models every Sunday at 2 AM
Generates 7-day cached predictions for all stocks
"""

import schedule
import time
import threading
import logging
import os
import json
from datetime import datetime, timedelta
import config

logger = logging.getLogger(__name__)

class AutoRetrainSystem:
    def __init__(self):
        self.is_running = False
        self.last_training_date = None
        self.prediction_cache_duration = 7  # 7 days
        self.training_thread = None
        
        # Create directories
        os.makedirs("data/cache", exist_ok=True)
        os.makedirs("data/ml_predictions", exist_ok=True)
        
        logger.info("🔄 AUTO-RETRAIN SYSTEM: Weekly automation initialized")
    
    def start_scheduler(self):
        """Start the weekly training scheduler"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Schedule weekly training every Sunday at 2 AM
        schedule.every().sunday.at("02:00").do(self.run_weekly_training)
        
        # Check if we need immediate training (if no recent training found)
        if self.should_train_immediately():
            logger.info("🚀 No recent training found - starting immediate training")
            threading.Thread(target=self.run_weekly_training, daemon=True).start()
        
        # Start scheduler thread
        self.training_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.training_thread.start()
        
        logger.info("✅ Weekly auto-training scheduled for Sundays at 2 AM")
    
    def should_train_immediately(self):
        """Check if we need immediate training"""
        try:
            status_file = "data/ml_predictions/training_status.json"
            if not os.path.exists(status_file):
                return True
            
            with open(status_file, 'r') as f:
                status = json.load(f)
            
            last_training = datetime.fromisoformat(status.get('last_training_date', '2020-01-01'))
            days_since_training = (datetime.now() - last_training).days
            
            return days_since_training >= 7
        except:
            return True
    
    def run_weekly_training(self):
        """Run the complete weekly training and prediction generation"""
        try:
            logger.info("🔄 WEEKLY AUTO-TRAINING STARTED")
            logger.info("=" * 50)
            
            # Import ML components
            try:
                from lstm_predictor import LSTMStockPredictor
                from shared_data_manager import SharedDataManager
            except ImportError as e:
                logger.error(f"❌ Failed to import ML components: {e}")
                return
            
            # Initialize components
            shared_manager = SharedDataManager()
            lstm_predictor = LSTMStockPredictor()
            
            successful_predictions = 0
            total_stocks = len(config.STOCK_SYMBOLS)
            
            # Train and generate predictions for all stocks
            for symbol in config.STOCK_SYMBOLS:
                try:
                    logger.info(f"🎯 Processing {symbol}...")
                    
                    # Get stock data
                    stock_data = shared_manager.get_shared_stock_data(symbol)
                    if not stock_data:
                        logger.warning(f"⚠️ No data for {symbol}, skipping")
                        continue
                    
                    # Train LSTM model
                    training_success = lstm_predictor.train_lstm_model(symbol, stock_data)
                    if not training_success:
                        logger.warning(f"⚠️ Training failed for {symbol}")
                        continue
                    
                    # Generate predictions
                    predictions = lstm_predictor.get_lstm_predictions(symbol, stock_data)
                    if predictions:
                        # Cache predictions for 7 days
                        self.cache_predictions(symbol, predictions)
                        successful_predictions += 1
                        logger.info(f"✅ {symbol} - Model trained & predictions cached")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing {symbol}: {e}")
                    continue
            
            # Update training status
            self.update_training_status(successful_predictions, total_stocks)
            
            logger.info("=" * 50)
            logger.info(f"🎉 WEEKLY TRAINING COMPLETED: {successful_predictions}/{total_stocks} stocks")
            logger.info("📅 Next training: Next Sunday at 2 AM")
            
        except Exception as e:
            logger.error(f"❌ Weekly training failed: {e}")
    
    def cache_predictions(self, symbol, predictions):
        """Cache predictions for 7 days"""
        try:
            cache_data = {
                'symbol': symbol,
                'predictions': predictions,
                'cached_date': datetime.now().isoformat(),
                'expires_date': (datetime.now() + timedelta(days=7)).isoformat(),
                'cache_duration_days': 7
            }
            
            cache_file = f"data/ml_predictions/{symbol}_predictions.json"
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"💾 Cached predictions for {symbol} (7 days)")
            
        except Exception as e:
            logger.error(f"❌ Failed to cache predictions for {symbol}: {e}")
    
    def get_cached_predictions(self, symbol):
        """Get cached predictions if still valid"""
        try:
            cache_file = f"data/ml_predictions/{symbol}_predictions.json"
            if not os.path.exists(cache_file):
                return None
            
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is still valid
            expires_date = datetime.fromisoformat(cache_data['expires_date'])
            if datetime.now() > expires_date:
                logger.info(f"⏰ Cache expired for {symbol}")
                return None
            
            logger.info(f"✅ Using cached predictions for {symbol}")
            return cache_data['predictions']
            
        except Exception as e:
            logger.error(f"❌ Error reading cached predictions for {symbol}: {e}")
            return None
    
    def update_training_status(self, successful, total):
        """Update overall training status"""
        try:
            status = {
                'last_training_date': datetime.now().isoformat(),
                'successful_stocks': successful,
                'total_stocks': total,
                'success_rate': f"{(successful/total)*100:.1f}%",
                'next_training_date': self.get_next_sunday_2am().isoformat(),
                'cache_duration_days': 7,
                'system_status': 'active'
            }
            
            with open("data/ml_predictions/training_status.json", 'w') as f:
                json.dump(status, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Failed to update training status: {e}")
    
    def get_next_sunday_2am(self):
        """Calculate next Sunday at 2 AM"""
        now = datetime.now()
        days_ahead = 6 - now.weekday()  # Sunday is 6
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        next_sunday = now + timedelta(days=days_ahead)
        return next_sunday.replace(hour=2, minute=0, second=0, microsecond=0)
    
    def get_training_status(self):
        """Get current training status"""
        try:
            status_file = "data/ml_predictions/training_status.json"
            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    return json.load(f)
            return {'system_status': 'not_initialized'}
        except:
            return {'system_status': 'error'}
    
    def _scheduler_loop(self):
        """Run the scheduler in background"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"❌ Scheduler error: {e}")
                time.sleep(60)
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        self.is_running = False
        if self.training_thread:
            self.training_thread.join(timeout=5)

# Global instance
auto_retrain = AutoRetrainSystem()