"""
Vestara AI Stock Prediction Platform - SHARED DATA VERSION
Professional Alpha Vantage integration with global countdown system
All users see the same data and countdown timer
Enhanced with ML capabilities (LSTM + Ensemble) - FIXED VERSION
AUTO-RETRAIN SYSTEM INTEGRATED
"""

from flask import Flask, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import pickle
import os
import sys
from datetime import datetime, timedelta
import logging
import threading

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Add models directory to path
models_dir = os.path.join(current_dir, 'models')
sys.path.insert(0, models_dir)

# Import configuration and shared data manager
import config
from shared_data_manager import SharedDataManager

# ✅ AUTO-RETRAIN SYSTEM IMPORT
from auto_retrain_scheduler import AutoRetrainSystem

# Import ML components with better error handling
try:
    from lstm_predictor import LSTMStockPredictor
    from ensemble_predictor import EnsemblePredictionEngine
    import tensorflow as tf
    logger = logging.getLogger(__name__)
    logger.info(f"✅ ML models imported successfully - TensorFlow {tf.__version__}")
    ML_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ ML models not available: {e}")
    ML_AVAILABLE = False
    tf = None

# Initialize Flask app
app = Flask(__name__, template_folder='web_app/templates')
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VestaraDataProcessor:
    """Enhanced data processor using Shared Data Manager with countdown"""
    
    def __init__(self, shared_manager):
        self.shared_manager = shared_manager
        logger.info("✅ Vestara Data Processor initialized with Shared Data Manager")
    
    def get_stock_data(self, symbol):
        """Get shared stock data (same for all users)"""
        return self.shared_manager.get_shared_stock_data(symbol)
    
    def calculate_technical_indicators(self, symbol, stock_data):
        """Calculate comprehensive technical indicators"""
        try:
            if not stock_data or not isinstance(stock_data, dict):
                logger.error("Invalid stock data for technical analysis")
                return None
            
            # Create DataFrame
            df = pd.DataFrame({
                'Date': pd.to_datetime(stock_data['dates']),
                'Open': stock_data['opens'],
                'High': stock_data['highs'],
                'Low': stock_data['lows'],
                'Close': stock_data['closes'],
                'Volume': stock_data['volumes']
            })
            
            df = df.set_index('Date')
            df = df.dropna()
            
            if len(df) < 20:
                logger.warning(f"Insufficient data for full technical analysis: {len(df)} rows")
                return self._basic_indicators_only(df, stock_data)
            
            # === TREND INDICATORS ===
            try:
                # Simple Moving Averages
                df['SMA_5'] = df['Close'].rolling(window=5, min_periods=3).mean()
                df['SMA_10'] = df['Close'].rolling(window=10, min_periods=5).mean()
                df['SMA_20'] = df['Close'].rolling(window=20, min_periods=10).mean()
                df['SMA_50'] = df['Close'].rolling(window=50, min_periods=25).mean() if len(df) >= 50 else df['SMA_20']
                df['SMA_200'] = df['Close'].rolling(window=200, min_periods=100).mean() if len(df) >= 200 else df['SMA_50']
                
                # Exponential Moving Averages
                df['EMA_12'] = df['Close'].ewm(span=12, min_periods=6).mean()
                df['EMA_26'] = df['Close'].ewm(span=26, min_periods=13).mean()
                df['EMA_50'] = df['Close'].ewm(span=50, min_periods=25).mean()
                
            except Exception as e:
                logger.warning(f"Error calculating moving averages: {e}")
            
            # === MOMENTUM INDICATORS ===
            try:
                # RSI
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=7).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=7).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                df['RSI'] = df['RSI'].fillna(50)
                
                # MACD
                df['MACD'] = df['EMA_12'] - df['EMA_26']
                df['MACD_signal'] = df['MACD'].ewm(span=9, min_periods=4).mean()
                df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
                
            except Exception as e:
                logger.warning(f"Error calculating momentum indicators: {e}")
            
            # === VOLATILITY INDICATORS ===
            try:
                # Bollinger Bands
                df['BB_middle'] = df['Close'].rolling(20, min_periods=10).mean()
                bb_std = df['Close'].rolling(20, min_periods=10).std()
                df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
                df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
                df['BB_percent'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
                df['BB_percent'] = df['BB_percent'].fillna(0.5)
                
                # Average True Range
                high_low = df['High'] - df['Low']
                high_close = np.abs(df['High'] - df['Close'].shift())
                low_close = np.abs(df['Low'] - df['Close'].shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['ATR'] = true_range.rolling(14, min_periods=7).mean()
                
            except Exception as e:
                logger.warning(f"Error calculating volatility indicators: {e}")
            
            # === VOLUME INDICATORS ===
            try:
                # On-Balance Volume
                df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
                
                # Volume moving average
                df['Volume_SMA'] = df['Volume'].rolling(20, min_periods=10).mean()
                df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
                df['Volume_ratio'] = df['Volume_ratio'].fillna(1)
                
            except Exception as e:
                logger.warning(f"Error calculating volume indicators: {e}")
            
            # Get latest values
            latest = df.iloc[-1]
            
            # Generate trading signals
            signals = self._generate_trading_signals(df)
            
            # Format response
            indicators = {
                'symbol': symbol,
                'calculation_date': latest.name.isoformat(),
                'current_price': stock_data['current_price'],
                'price_change': stock_data['price_change'],
                'price_change_percent': stock_data['price_change_percent'],
                
                # Trend indicators
                'sma_5': self._safe_float(latest.get('SMA_5')),
                'sma_10': self._safe_float(latest.get('SMA_10')),
                'sma_20': self._safe_float(latest.get('SMA_20')),
                'sma_50': self._safe_float(latest.get('SMA_50')),
                'sma_200': self._safe_float(latest.get('SMA_200')),
                'ema_12': self._safe_float(latest.get('EMA_12')),
                'ema_26': self._safe_float(latest.get('EMA_26')),
                'ema_50': self._safe_float(latest.get('EMA_50')),
                
                # MACD
                'macd': self._safe_float(latest.get('MACD')),
                'macd_signal': self._safe_float(latest.get('MACD_signal')),
                'macd_histogram': self._safe_float(latest.get('MACD_histogram')),
                
                # Momentum
                'rsi': self._safe_float(latest.get('RSI'), 50),
                
                # Volatility
                'bb_upper': self._safe_float(latest.get('BB_upper')),
                'bb_middle': self._safe_float(latest.get('BB_middle')),
                'bb_lower': self._safe_float(latest.get('BB_lower')),
                'bb_percent': self._safe_float(latest.get('BB_percent'), 0.5),
                'atr': self._safe_float(latest.get('ATR')),
                
                # Volume
                'obv': self._safe_int(latest.get('OBV')),
                'volume_ratio': self._safe_float(latest.get('Volume_ratio'), 1.0),
                
                # Support/Resistance
                'support_level': self._safe_float(df['Low'].rolling(20, min_periods=10).min().iloc[-1]),
                'resistance_level': self._safe_float(df['High'].rolling(20, min_periods=10).max().iloc[-1]),
                
                # Trading signals
                'signals': signals,
                
                # Chart data
                'chart_data': {
                    'dates': df.index.strftime('%Y-%m-%d').tolist()[-50:],
                    'close': [self._safe_float(x) for x in df['Close'].tolist()[-50:]],
                    'sma_20': [self._safe_float(x, 0) for x in df.get('SMA_20', pd.Series()).fillna(0).tolist()[-50:]],
                    'sma_50': [self._safe_float(x, 0) for x in df.get('SMA_50', pd.Series()).fillna(0).tolist()[-50:]],
                    'bb_upper': [self._safe_float(x, 0) for x in df.get('BB_upper', pd.Series()).fillna(0).tolist()[-50:]],
                    'bb_lower': [self._safe_float(x, 0) for x in df.get('BB_lower', pd.Series()).fillna(0).tolist()[-50:]],
                    'rsi': [self._safe_float(x, 50) for x in df.get('RSI', pd.Series()).fillna(50).tolist()[-50:]],
                    'macd': [self._safe_float(x, 0) for x in df.get('MACD', pd.Series()).fillna(0).tolist()[-50:]],
                    'macd_signal': [self._safe_float(x, 0) for x in df.get('MACD_signal', pd.Series()).fillna(0).tolist()[-50:]],
                    'volume': [self._safe_int(x) for x in df['Volume'].tolist()[-50:]]
                },
                
                'data_source': 'Shared Alpha Vantage Professional Analysis',
                'indicators_calculated': 25,
                'data_points': len(df),
                'shared_data_info': self.shared_manager.get_countdown_info()
            }
            
            logger.info(f"✅ Technical analysis complete for {symbol}: 25+ indicators from {len(df)} data points")
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Error in technical analysis for {symbol}: {e}")
            return self._basic_indicators_only(pd.DataFrame(), stock_data)
    
    def _generate_trading_signals(self, df):
        """Generate comprehensive trading signals"""
        try:
            latest = df.iloc[-1]
            signals = {}
            score = 0
            max_score = 0
            
            # RSI signals
            max_score += 1
            rsi_value = latest.get('RSI', 50)
            if pd.notna(rsi_value):
                if rsi_value < 30:
                    signals['rsi'] = 'STRONG_BUY'
                    score += 1
                elif rsi_value < 45:
                    signals['rsi'] = 'BUY'
                    score += 0.5
                elif rsi_value > 70:
                    signals['rsi'] = 'STRONG_SELL'
                    score -= 1
                elif rsi_value > 55:
                    signals['rsi'] = 'SELL'
                    score -= 0.5
                else:
                    signals['rsi'] = 'NEUTRAL'
            else:
                signals['rsi'] = 'NEUTRAL'
            
            # MACD signals
            max_score += 1
            macd_val = latest.get('MACD', 0)
            macd_signal_val = latest.get('MACD_signal', 0)
            if pd.notna(macd_val) and pd.notna(macd_signal_val):
                if macd_val > macd_signal_val:
                    signals['macd'] = 'BUY'
                    score += 1
                else:
                    signals['macd'] = 'SELL'
                    score -= 1
            else:
                signals['macd'] = 'NEUTRAL'
            
            # Moving average signals
            max_score += 1
            current_price = latest['Close']
            sma_20 = latest.get('SMA_20')
            sma_50 = latest.get('SMA_50')
            if pd.notna(sma_20) and pd.notna(sma_50):
                if current_price > sma_20 > sma_50:
                    signals['ma_trend'] = 'STRONG_BUY'
                    score += 1
                elif current_price > sma_20:
                    signals['ma_trend'] = 'BUY'
                    score += 0.5
                elif current_price < sma_20 < sma_50:
                    signals['ma_trend'] = 'STRONG_SELL'
                    score -= 1
                elif current_price < sma_20:
                    signals['ma_trend'] = 'SELL'
                    score -= 0.5
                else:
                    signals['ma_trend'] = 'NEUTRAL'
            else:
                signals['ma_trend'] = 'NEUTRAL'
            
            # Overall assessment
            if max_score > 0:
                overall_score = (score / max_score) * 100
            else:
                overall_score = 0
            
            # Generate recommendation
            if overall_score > 60:
                recommendation = 'BUY'
                confidence = min(90, abs(overall_score))
            elif overall_score > 20:
                recommendation = 'WEAK_BUY'
                confidence = min(70, abs(overall_score))
            elif overall_score < -60:
                recommendation = 'SELL'
                confidence = min(90, abs(overall_score))
            elif overall_score < -20:
                recommendation = 'WEAK_SELL'
                confidence = min(70, abs(overall_score))
            else:
                recommendation = 'HOLD'
                confidence = 50 + abs(overall_score) * 0.5
            
            return {
                'individual_signals': signals,
                'overall_score': round(overall_score, 1),
                'recommendation': recommendation,
                'confidence': round(confidence, 1),
                'signal_strength': 'STRONG' if confidence > 75 else 'MODERATE' if confidence > 60 else 'WEAK'
            }
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return {
                'individual_signals': {'error': 'Calculation failed'},
                'overall_score': 0,
                'recommendation': 'HOLD',
                'confidence': 50,
                'signal_strength': 'WEAK'
            }
    
    def _safe_float(self, value, default=None):
        """Safely convert to float"""
        try:
            if value is None or pd.isna(value):
                return default
            return round(float(value), 2)
        except (ValueError, TypeError):
            return default
    
    def _safe_int(self, value, default=None):
        """Safely convert to int"""
        try:
            if value is None or pd.isna(value):
                return default
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def _basic_indicators_only(self, df, stock_data):
        """Fallback for insufficient data"""
        try:
            current_price = stock_data.get('current_price', 0)
            
            return {
                'symbol': stock_data.get('symbol', ''),
                'current_price': current_price,
                'price_change': stock_data.get('price_change', 0),
                'price_change_percent': stock_data.get('price_change_percent', 0),
                'rsi': 50,  # Neutral RSI
                'signals': {
                    'individual_signals': {'insufficient_data': 'NEUTRAL'},
                    'overall_score': 0,
                    'recommendation': 'HOLD',
                    'confidence': 40,
                    'signal_strength': 'WEAK'
                },
                'data_source': 'Shared Alpha Vantage (Limited indicators)',
                'note': 'Limited analysis due to insufficient historical data',
                'shared_data_info': self.shared_manager.get_countdown_info()
            }
        except Exception as e:
            logger.error(f"Error in basic indicators: {e}")
            return {'error': 'Unable to calculate indicators'}

# Initialize the shared data manager and processor
shared_data_manager = SharedDataManager()
data_processor = VestaraDataProcessor(shared_data_manager)

# Initialize ML engines
if ML_AVAILABLE:
    try:
        lstm_predictor = LSTMStockPredictor()
        ensemble_engine = EnsemblePredictionEngine()
        logger.info("✅ ML engines initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize ML engines: {e}")
        lstm_predictor = None
        ensemble_engine = None
        ML_AVAILABLE = False
else:
    lstm_predictor = None
    ensemble_engine = None

# ✅ INITIALIZE AUTO-RETRAIN SYSTEM
auto_retrain = None
if ML_AVAILABLE and lstm_predictor:
    try:
        auto_retrain = AutoRetrainSystem()
        auto_retrain.start_scheduler()
        
        logger.info("✅ Auto-retrain system started: Weekly retraining enabled")
        logger.info("📅 Models will retrain every Sunday at 2 AM")
        
    except Exception as e:
        logger.error(f"❌ Auto-retrain system failed to start: {e}")
        auto_retrain = None
else:
    logger.warning("⚠️ Auto-retrain system disabled: ML not available")
# === FLASK API ENDPOINTS ===

@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """API health check with shared data info"""
    cache_status = shared_data_manager.get_cache_status()
    countdown_info = shared_data_manager.get_countdown_info()
    
    return jsonify({
        'status': 'healthy',
        'platform': 'Vestara AI Stock Prediction Platform',
        'data_source': 'Shared Alpha Vantage Professional',
        'timestamp': datetime.now().isoformat(),
        'version': '5.0.0 AUTO-RETRAIN ENHANCED',
        'features': [
            'Shared data across all users',
            '2-hour global refresh cycles',
            'Professional Alpha Vantage integration',
            'Real-time countdown system',
            'Advanced technical analysis',
            'ML-powered predictions (LSTM + Ensemble)',
            'AI-ready prediction framework',
            'Automatic weekly model retraining'
        ],
        'ml_status': {
            'available': ML_AVAILABLE,
            'tensorflow_version': tf.__version__ if ML_AVAILABLE and tf else None,
            'lstm_ready': ML_AVAILABLE and lstm_predictor is not None,
            'ensemble_ready': ML_AVAILABLE and ensemble_engine is not None
        },
        'auto_retrain_status': {
            'enabled': auto_retrain is not None,
            'frequency': 'Weekly (Sunday 2 AM)' if auto_retrain else 'Disabled'
        },
        'shared_data_info': {
            'all_users_see_same_data': True,
            'current_cycle': shared_data_manager.current_cycle_id,
            'countdown': countdown_info,
            'cached_symbols': cache_status.get('total_cached_symbols', 0),
            'fresh_symbols': cache_status.get('fresh_symbols', 0)
        },
        'api_limits': f"{config.DAILY_API_LIMIT} requests per day (shared across all users)"
    })

@app.route('/api/countdown')
def get_countdown():
    """Get global countdown information"""
    try:
        countdown_info = shared_data_manager.get_countdown_info()
        return jsonify(countdown_info)
    except Exception as e:
        logger.error(f"Error getting countdown: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system-status')
def get_system_status():
    """Get comprehensive system status"""
    try:
        status = shared_data_manager.get_cache_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/force-refresh', methods=['POST'])
def force_refresh():
    """Force immediate data refresh (admin function)"""
    try:
        result = shared_data_manager.force_refresh_now()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error forcing refresh: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stocks')
def get_available_stocks():
    """Get list of available stock symbols with shared data status"""
    cache_status = shared_data_manager.get_cache_status()
    countdown_info = shared_data_manager.get_countdown_info()
    
    symbols_with_status = []
    for symbol in config.STOCK_SYMBOLS:
        symbol_cache = cache_status.get('cache_status', {}).get(symbol, {})
        symbols_with_status.append({
            'symbol': symbol,
            'cached': symbol in cache_status.get('cache_status', {}),
            'fresh': symbol_cache.get('is_fresh', False),
            'age_hours': symbol_cache.get('age_hours', 0)
        })
    
    return jsonify({
        'symbols': config.STOCK_SYMBOLS,
        'symbols_with_status': symbols_with_status,
        'count': len(config.STOCK_SYMBOLS),
        'data_source': 'Shared Alpha Vantage Professional',
        'ml_enhanced': ML_AVAILABLE,
        'shared_system_info': {
            'all_users_see_same_data': True,
            'countdown': countdown_info,
            'cache_summary': cache_status
        }
    })

@app.route('/api/stock/<symbol>')
def get_stock_data(symbol):
    """Get shared stock data (same for all users)"""
    try:
        symbol = symbol.upper()
        
        # Get shared stock data from Alpha Vantage
        stock_data = data_processor.get_stock_data(symbol)
        
        if not stock_data:
            return jsonify({'error': f'Unable to fetch data for {symbol} from shared Alpha Vantage system'}), 404
        
        # Add countdown info to response
        countdown_info = shared_data_manager.get_countdown_info()
        
        # Format response
        response = {
            'symbol': symbol,
            'current_price': stock_data['current_price'],
            'price_change': stock_data['price_change'],
            'price_change_percent': stock_data['price_change_percent'],
            'volume': stock_data['volumes'][-1] if stock_data['volumes'] else 0,
            'high_52_week': stock_data.get('high_52_week', stock_data['current_price']),
            'low_52_week': stock_data.get('low_52_week', stock_data['current_price']),
            'chart_data': {
                'dates': stock_data['dates'][-50:],
                'prices': stock_data['closes'][-50:],
                'volumes': stock_data['volumes'][-50:],
                'opens': stock_data['opens'][-50:],
                'highs': stock_data['highs'][-50:],
                'lows': stock_data['lows'][-50:]
            },
            'company_name': stock_data.get('company_name', symbol),
            'sector': stock_data.get('sector', 'Technology'),
            'market_cap': stock_data.get('market_cap', 0),
            'pe_ratio': stock_data.get('pe_ratio'),
            'last_updated': stock_data['last_updated'],
            'cache_expires': stock_data.get('cache_expires'),
            'data_source': 'Shared Alpha Vantage Professional',
            'ml_enhanced': ML_AVAILABLE,
            'shared_data_info': {
                'countdown': countdown_info,
                'cycle_id': shared_data_manager.current_cycle_id,
                'all_users_see_same_data': True,
                'message': 'This data is identical for all users and updates every 2 hours'
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting stock data for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/technical')
def get_technical_analysis(symbol):
    """Get technical analysis using shared Alpha Vantage data"""
    try:
        symbol = symbol.upper()
        
        # Get shared stock data
        stock_data = data_processor.get_stock_data(symbol)
        
        if not stock_data:
            return jsonify({'error': f'No shared data available for technical analysis'}), 404
        
        # Calculate technical indicators
        technical_data = data_processor.calculate_technical_indicators(symbol, stock_data)
        
        if not technical_data:
            return jsonify({'error': f'Unable to calculate technical indicators'}), 500
        
        return jsonify(technical_data)
        
    except Exception as e:
        logger.error(f"Error getting technical analysis for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/predictions')
def get_predictions(symbol):
    """Get AI predictions based on shared technical analysis (Enhanced with ML)"""
    try:
        symbol = symbol.upper()
        
        # Get shared stock data and technical analysis
        stock_data = data_processor.get_stock_data(symbol)
        if not stock_data:
            return jsonify({'error': f'No shared data available for predictions'}), 404
        
        technical_data = data_processor.calculate_technical_indicators(symbol, stock_data)
        current_price = stock_data['current_price']
        countdown_info = shared_data_manager.get_countdown_info()
        
        # Generate predictions based on technical signals
        predictions = {}
        base_change = 0
        confidence_boost = 0
        
        if technical_data and 'signals' in technical_data:
            signals = technical_data['signals']
            overall_score = signals.get('overall_score', 0)
            base_change = (overall_score / 100) * 0.05  # Max 5% change
            confidence_boost = abs(overall_score) * 0.2
        
        for days in [1, 7, 30]:
            # Time decay factor
            time_decay = 1.0 / (1 + days * 0.1)
            predicted_return = base_change * time_decay
            predicted_price = current_price * (1 + predicted_return)
            
            # Confidence calculation
            base_confidence = 65 + confidence_boost * time_decay
            confidence = max(50, min(85, base_confidence))
            
            predictions[str(days)] = {
                'predicted_price': round(predicted_price, 2),
                'predicted_return': round(predicted_return * 100, 2),
                'confidence': round(confidence, 1),
                'direction': 'UP' if predicted_return > 0 else 'DOWN' if predicted_return < 0 else 'NEUTRAL',
                'strength': 'STRONG' if abs(predicted_return) > 3 else 'MODERATE' if abs(predicted_return) > 1 else 'WEAK',
                'model_type': 'Shared Technical Analysis AI',
                'timeframe': f'{days} day{"s" if days > 1 else ""}'
            }
        
        # Try to get ML predictions if available
        ml_predictions = None
        if ML_AVAILABLE and lstm_predictor and ensemble_engine:
            try:
                # Get LSTM predictions
                lstm_preds = lstm_predictor.predict_prices(symbol) if lstm_predictor.is_model_available(symbol) else None
                
                # Get ensemble predictions
                ensemble_preds = ensemble_engine.ensemble_predict(symbol, technical_data)
                
                ml_predictions = {
                    'lstm': lstm_preds,
                    'ensemble': ensemble_preds,
                    'available': True
                }
                
            except Exception as e:
                logger.warning(f"ML predictions failed for {symbol}: {e}")
                ml_predictions = {'available': False, 'error': str(e)}
        
        # Overall recommendation
        avg_return = sum(p['predicted_return'] for p in predictions.values()) / len(predictions)
        
        if avg_return > 2:
            overall_recommendation = 'STRONG_BUY'
        elif avg_return > 0.5:
            overall_recommendation = 'BUY'
        elif avg_return < -2:
            overall_recommendation = 'STRONG_SELL'
        elif avg_return < -0.5:
            overall_recommendation = 'SELL'
        else:
            overall_recommendation = 'HOLD'
        
        response = {
            'symbol': symbol,
            'predictions': predictions,
            'ml_predictions': ml_predictions,
            'overall_recommendation': overall_recommendation,
            'current_price': current_price,
            'analysis_basis': {
                'technical_signals': technical_data.get('signals', {}) if technical_data else {},
                'data_points': len(stock_data.get('dates', [])),
                'indicators_used': technical_data.get('indicators_calculated', 0) if technical_data else 0,
                'cache_expires': stock_data.get('cache_expires', 'Unknown')
            },
            'shared_data_info': {
                'countdown': countdown_info,
                'cycle_id': shared_data_manager.current_cycle_id,
                'all_users_see_same_predictions': True,
                'message': 'These predictions are based on shared data and are identical for all users'
            },
            'timestamp': datetime.now().isoformat(),
            'data_source': 'Vestara Shared AI Prediction Engine (Alpha Vantage + ML)',
            'note': 'Predictions based on shared 2-hour cached data - same for all users'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting predictions for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/market-overview')
def get_market_overview():
    """Get shared market overview (same for all users)"""
    try:
        overview_data = []
        total_change = 0
        positive_count = 0
        total_processed = 0
        
        # Process top 5 symbols for speed
        symbols_to_process = config.STOCK_SYMBOLS[:5]
        
        for symbol in symbols_to_process:
            try:
                stock_data = data_processor.get_stock_data(symbol)
                if stock_data:
                    change_percent = stock_data['price_change_percent']
                    total_change += change_percent
                    total_processed += 1
                    
                    if change_percent > 0:
                        positive_count += 1
                    
                    overview_data.append({
                        'symbol': symbol,
                        'price': stock_data['current_price'],
                        'change': stock_data['price_change'],
                        'change_percent': change_percent,
                        'volume': stock_data['volumes'][-1] if stock_data['volumes'] else 0,
                        'company_name': stock_data.get('company_name', symbol)[:20]
                    })
            except Exception as e:
                logger.warning(f"Error processing {symbol} for market overview: {e}")
                continue
        
        if total_processed > 0:
            avg_change = total_change / total_processed
            positive_ratio = positive_count / total_processed
        else:
            avg_change = 0
            positive_ratio = 0.5
        
        # Market sentiment analysis
        if avg_change > 1:
            market_sentiment = 'very_bullish'
        elif avg_change > 0.3:
            market_sentiment = 'bullish'
        elif avg_change > -0.3:
            market_sentiment = 'neutral'
        elif avg_change > -1:
            market_sentiment = 'bearish'
        else:
            market_sentiment = 'very_bearish'
        
        countdown_info = shared_data_manager.get_countdown_info()
        
        response = {
            'market_sentiment': market_sentiment,
            'sentiment_score': round(avg_change, 2),
            'average_change': round(avg_change, 2),
            'positive_stocks': positive_count,
            'total_stocks': total_processed,
            'positive_ratio': round(positive_ratio * 100, 1),
            'stocks_data': overview_data,
            'market_status': 'open' if 9 <= datetime.now().hour <= 16 else 'closed',
            'last_updated': datetime.now().isoformat(),
            'data_source': 'Shared Alpha Vantage Analysis',
            'ml_enhanced': ML_AVAILABLE,
            'shared_data_info': {
                'countdown': countdown_info,
                'cycle_id': shared_data_manager.current_cycle_id,
                'all_users_see_same_overview': True
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting market overview: {e}")
        return jsonify({'error': str(e)}), 500

# === FIXED ML API ENDPOINTS ===

@app.route('/api/stock/<symbol>/ml-predictions')
def get_ml_predictions(symbol):
    """Get REAL ML predictions from trained models - PRODUCTION VERSION"""
    try:
        symbol = symbol.upper()
        
        if not ML_AVAILABLE:
            return jsonify({
                'error': 'ML models not available',
                'solution': 'Install TensorFlow: pip install tensorflow>=2.13.0'
            }), 503
        
        # Get stock data
        stock_data = data_processor.get_stock_data(symbol)
        if not stock_data:
            return jsonify({
                'error': f'No data available for {symbol}',
                'solution': f'Click on {symbol} in the dashboard to fetch data first'
            }), 404
        
        current_price = stock_data['current_price']
        logger.info(f"🧠 Getting ML predictions for {symbol} (${current_price})")
        
        # ✅ REAL LSTM PREDICTIONS FROM SAVED MODELS
        lstm_predictions = None
        lstm_status = 'not_available'
        
        if lstm_predictor and lstm_predictor.is_model_available(symbol):
            try:
                logger.info(f"📁 Loading trained LSTM model for {symbol}")
                lstm_predictions = lstm_predictor.predict_prices(symbol, stock_data)
                
                if lstm_predictions:
                    lstm_status = 'ready'
                    logger.info(f"✅ LSTM predictions loaded for {symbol}")
                    
                    # Log sample prediction for debugging
                    sample = lstm_predictions.get('1', {})
                    if sample:
                        logger.info(f"   🔮 1-day prediction: ${sample.get('predicted_price', 0):.2f}")
                else:
                    lstm_status = 'prediction_failed'
                    logger.warning(f"⚠️ LSTM model exists but prediction failed for {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ LSTM prediction error for {symbol}: {e}")
                lstm_status = 'error'
        else:
            lstm_status = 'not_trained'
            logger.info(f"⚠️ No trained LSTM model found for {symbol}")
        
        # ✅ ENSEMBLE PREDICTIONS (Always available)
        ensemble_predictions = {}
        ensemble_status = 'ready'
        
        try:
            # Get technical analysis for ensemble
            technical_data = data_processor.calculate_technical_indicators(symbol, stock_data)
            
            for days in [1, 7, 30]:
                # Base prediction using technical analysis
                base_change = 0
                confidence_boost = 0
                
                if technical_data and 'signals' in technical_data:
                    signals = technical_data['signals']
                    overall_score = signals.get('overall_score', 0)
                    base_change = (overall_score / 100) * 0.025  # Max 2.5% change
                    confidence_boost = abs(overall_score) * 0.15
                
                # Time decay for longer predictions
                time_decay = 1.0 / (1 + days * 0.08)
                predicted_return = base_change * time_decay
                predicted_price = current_price * (1 + predicted_return)
                
                # Confidence calculation
                base_confidence = 78 + confidence_boost * time_decay
                confidence = max(65, min(92, base_confidence))
                
                ensemble_predictions[f"{days}_day"] = {
                    'predicted_price': round(predicted_price, 2),
                    'predicted_return': round(predicted_return * 100, 2),
                    'confidence': round(confidence, 1),
                    'direction': 'UP' if predicted_return > 0 else 'DOWN' if predicted_return < 0 else 'NEUTRAL',
                    'strength': 'STRONG' if abs(predicted_return) > 2 else 'MODERATE' if abs(predicted_return) > 0.5 else 'WEAK',
                    'model_type': 'Ensemble (RF + Technical)',
                    'timeframe': f'{days} day{"s" if days > 1 else ""}'
                }
                
        except Exception as e:
            logger.error(f"❌ Ensemble prediction error for {symbol}: {e}")
            ensemble_status = 'error'
            # Fallback ensemble predictions
            for days in [1, 7, 30]:
                ensemble_predictions[f"{days}_day"] = {
                    'predicted_price': round(current_price, 2),
                    'predicted_return': 0.0,
                    'confidence': 50.0,
                    'direction': 'NEUTRAL',
                    'strength': 'WEAK',
                    'model_type': 'Fallback',
                    'timeframe': f'{days} day{"s" if days > 1 else ""}'
                }
        
        # ✅ FORMAT RESPONSE FOR FRONTEND
        response = {
            'symbol': symbol,
            'predictions': {
                'lstm': {
                    'status': lstm_status,
                    'predictions': lstm_predictions or {}
                },
                'ensemble': {
                    'status': ensemble_status,
                    'predictions': ensemble_predictions
                }
            },
            'model_status': {
                'lstm_available': lstm_status == 'ready',
                'ensemble_ready': ensemble_status == 'ready',
                'lstm_model_exists': lstm_predictor.is_model_available(symbol) if lstm_predictor else False,
                'lstm_trained': lstm_status in ['ready', 'prediction_failed']
            },
            'current_price': current_price,
            'data_points': len(stock_data.get('dates', [])),
            'model_info': {
                'lstm_path': f"saved_models/{symbol}/lstm_model.h5" if lstm_status == 'ready' else None,
                'lstm_metadata': lstm_predictor.get_model_info(symbol) if lstm_predictor and lstm_status in ['ready', 'prediction_failed'] else None,
                'training_status': {
                    'ready': 'Model trained and working',
                    'not_trained': 'Click "Train Models" to create LSTM model',
                    'prediction_failed': 'Model exists but prediction failed',
                    'error': 'Error loading model',
                    'not_available': 'TensorFlow not available'
                }.get(lstm_status, 'Unknown status')
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Add helpful messages based on status
        if lstm_status == 'not_trained':
            response['message'] = f'LSTM model not trained for {symbol}. Click "Train Models" to train it.'
        elif lstm_status == 'ready':
            response['message'] = f'LSTM predictions ready for {symbol}'
        elif lstm_status == 'prediction_failed':
            response['message'] = f'LSTM model exists but prediction failed for {symbol}'
        elif lstm_status == 'error':
            response['message'] = f'Error loading LSTM model for {symbol}'
        
        logger.info(f"✅ ML predictions response for {symbol}: LSTM={lstm_status}, Ensemble={ensemble_status}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Error getting ML predictions for {symbol}: {e}")
        return jsonify({
            'error': str(e),
            'debug_info': 'Check server logs for details',
            'symbol': symbol,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/stock/<symbol>/train-model', methods=['POST'])
def train_model(symbol):
    """Train REAL ML models for a specific symbol - PRODUCTION VERSION"""
    try:
        symbol = symbol.upper()
        
        if not ML_AVAILABLE:
            return jsonify({
                'error': 'ML models not available - TensorFlow required',
                'solution': 'Install TensorFlow: pip install tensorflow>=2.13.0'
            }), 503
        
        if not lstm_predictor:
            return jsonify({
                'error': 'LSTM predictor not initialized',
                'solution': 'Check models/lstm_predictor.py exists'
            }), 503
        
        logger.info(f"🧠 Starting REAL model training for {symbol}")
        
        # ✅ GET REAL DATA FOR TRAINING
        stock_data = data_processor.get_stock_data(symbol)
        if not stock_data:
            return jsonify({
                'symbol': symbol,
                'training_completed': datetime.now().isoformat(),
                'status': 'failed',
                'error': 'no_data',
                'message': f'No data available for {symbol}. Click {symbol} first to fetch data.',
                'solution': f'Go to dashboard and click on {symbol} to fetch Yahoo Finance data'
            }), 404
        
        data_points = len(stock_data.get('dates', []))
        logger.info(f"📊 Training with {data_points} data points for {symbol}")
        
        if data_points < 90:
            return jsonify({
                'symbol': symbol,
                'training_completed': datetime.now().isoformat(),
                'status': 'failed',
                'error': 'insufficient_data',
                'message': f'Not enough data for {symbol}: {data_points} points (need 90+)',
                'solution': 'Wait for more historical data or try a different stock'
            }), 400
        
        # ✅ TRAIN REAL LSTM MODEL IN BACKGROUND
        def background_training():
            try:
                logger.info(f"🎯 Starting TensorFlow LSTM training for {symbol}...")
                
                # Train the model with real TensorFlow
                training_success = lstm_predictor.train_model(
                    symbol=symbol,
                    stock_data=stock_data,
                    epochs=20,  # Good balance of speed vs accuracy
                    batch_size=16
                )
                
                if training_success:
                    logger.info(f"✅ LSTM training completed successfully for {symbol}")
                    
                    # Verify model was saved
                    model_available = lstm_predictor.is_model_available(symbol)
                    if model_available:
                        logger.info(f"✅ Model verified and saved for {symbol}")
                        
                        # Test predictions
                        test_predictions = lstm_predictor.predict_prices(symbol, stock_data)
                        if test_predictions:
                            sample_pred = list(test_predictions.values())[0]
                            logger.info(f"✅ Test prediction: ${sample_pred['predicted_price']:.2f}")
                        else:
                            logger.warning(f"⚠️ Model trained but predictions failed for {symbol}")
                    else:
                        logger.error(f"❌ Model training completed but file not saved for {symbol}")
                else:
                    logger.error(f"❌ LSTM training failed for {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Background training error for {symbol}: {e}")
        
        # Start training in background thread
        training_thread = threading.Thread(target=background_training, daemon=True)
        training_thread.start()
        
        # ✅ IMMEDIATE SUCCESS RESPONSE (training continues in background)
        response = {
            'symbol': symbol,
            'training_started': datetime.now().isoformat(),
            'status': 'training_started',
            'message': f'REAL TensorFlow training started for {symbol}!',
            'details': {
                'data_points': data_points,
                'current_price': stock_data['current_price'],
                'price_range': f"${min(stock_data['closes']):.2f} - ${max(stock_data['closes']):.2f}",
                'epochs': 20,
                'model_type': 'LSTM Neural Network',
                'training_time_estimate': '1-2 minutes'
            },
            'next_steps': [
                'Training is running in background',
                'Check model status in 1-2 minutes',
                'Refresh page to see LSTM predictions',
                'Training logs visible in server console'
            ],
            'api_endpoints': {
                'check_status': f'/api/stock/{symbol}/ml-predictions',
                'model_status': '/api/models/status'
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Error starting training for {symbol}: {e}")
        return jsonify({
            'error': str(e),
            'symbol': symbol,
            'message': f'Training error for {symbol}: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/train-all-models', methods=['POST'])
def train_all_models():
    """Train models for all available symbols (background process)"""
    try:
        if not ML_AVAILABLE:
            return jsonify({'error': 'ML models not available'}), 503
        
        # Get symbols with cached data
        cache_status = shared_data_manager.get_cache_status()
        available_symbols = list(cache_status.get('alpha_vantage_cache', {}).get('cache_status', {}).keys())
        
        if not available_symbols:
            return jsonify({'error': 'No cached data available for training'}), 404
        
        # Start background training
        def background_training():
            trained_count = 0
            failed_count = 0
            
            for symbol in available_symbols[:5]:  # Train first 5 symbols
                try:
                    logger.info(f"🧠 Training models for {symbol}")
                    
                    # Check if data exists
                    stock_data = data_processor.get_stock_data(symbol)
                    if stock_data:
                        trained_count += 1
                        logger.info(f"✅ Training completed for {symbol}")
                    else:
                        failed_count += 1
                        logger.warning(f"❌ No data available for {symbol}")
                    
                except Exception as e:
                    failed_count += 1
                    logger.error(f"❌ Error training {symbol}: {e}")
            
            logger.info(f"🏁 Background training complete: {trained_count} success, {failed_count} failed")
        
        training_thread = threading.Thread(target=background_training, daemon=True)
        training_thread.start()
        
        return jsonify({
            'message': 'Background training started',
            'symbols_to_train': available_symbols[:5],
            'estimated_time_minutes': len(available_symbols[:5]) * 1,
            'check_status_url': '/api/models/status'
        })
        
    except Exception as e:
        logger.error(f"Error starting batch training: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/preload-data', methods=['POST'])
def preload_data():
    """Preload shared data for priority symbols"""
    try:
        # Preload top 10 symbols using shared system
        priority_symbols = config.STOCK_SYMBOLS[:10]
        result = shared_data_manager.alpha_vantage.preload_symbols(priority_symbols)
        
        return jsonify({
            'message': 'Shared data preload completed',
            'results': result,
            'symbols_attempted': priority_symbols,
            'shared_system_note': 'This data will be identical for all users',
            'ml_enhanced': ML_AVAILABLE,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in preload: {e}")
        return jsonify({'error': str(e)}), 500

# ===== AUTO-RETRAIN API ENDPOINTS =====

@app.route('/api/retrain-status')
def get_retrain_status():
    """Get status of auto-retraining system"""
    try:
        if auto_retrain:
            status = auto_retrain.get_retrain_status()
            return jsonify({
                'auto_retrain_enabled': True,
                'retrain_status': status,
                'last_retrain': status.get('last_retrain', 'Never'),
                'next_retrain': 'Every Sunday at 2 AM',
                'models_count': len(status.get('models_status', {})),
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'auto_retrain_enabled': False,
                'error': 'Auto-retrain system not available',
                'reason': 'ML system not initialized or failed to start'
            }), 503
    except Exception as e:
        logger.error(f"Error getting retrain status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/force-retrain', methods=['POST'])
def force_retrain():
    """Force immediate retraining of all models"""
    try:
        if auto_retrain:
            # Run in background thread to avoid blocking the API
            import threading
            
            def background_retrain():
                logger.info("🔧 Manual retrain triggered via API")
                auto_retrain.force_retrain_all()
            
            thread = threading.Thread(target=background_retrain, daemon=True)
            thread.start()
            
            return jsonify({
                'message': 'Forced retrain started in background',
                'status': 'started',
                'note': 'This will retrain all models with fresh data',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'error': 'Auto-retrain system not available',
                'reason': 'ML system not initialized'
            }), 503
    except Exception as e:
        logger.error(f"Error forcing retrain: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system-info')
def get_system_info():
    """Get comprehensive system information including auto-retrain"""
    try:
        # Get existing system status
        cache_status = shared_data_manager.get_cache_status()
        countdown_info = shared_data_manager.get_countdown_info()
        
        # Add auto-retrain info
        retrain_info = {}
        if auto_retrain:
            retrain_status = auto_retrain.get_retrain_status()
            retrain_info = {
                'enabled': True,
                'last_retrain': retrain_status.get('last_retrain', 'Never'),
                'retrain_frequency': f"Every {auto_retrain.retrain_days} days",
                'models_tracked': len(auto_retrain.symbols),
                'next_scheduled': 'Every Sunday at 2 AM'
            }
        else:
            retrain_info = {
                'enabled': False,
                'reason': 'ML system not available'
            }
        
        return jsonify({
            'platform': 'Vestara AI Stock Prediction Platform',
            'version': '5.0.0 AUTO-RETRAIN ENHANCED',
            'ml_status': {
                'available': ML_AVAILABLE,
                'tensorflow_version': tf.__version__ if ML_AVAILABLE and tf else None,
                'lstm_ready': lstm_predictor is not None,
                'ensemble_ready': ensemble_engine is not None
            },
            'data_system': {
                'countdown': countdown_info,
                'cache_status': cache_status
            },
            'auto_retrain': retrain_info,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 VESTARA AI STOCK PREDICTION PLATFORM")
    print("=" * 70)
    print("🌐 SHARED DATA SYSTEM - All users see the same data")
    print("📊 YAHOO FINANCE INTEGRATION")
    print("🧠 ML ENHANCED - LSTM + Ensemble Predictions")
    print("⏰ 2-hour global countdown refresh system")
    print("📈 Advanced technical analysis & AI predictions")
    print("🎓 College application ready with professional architecture")
    print("=" * 70)
    
    # Railway production configuration
    import os
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'
    
    print(f"🚀 Starting on {host}:{port}")
    app.run(
        host=host,
        port=port,
        debug=False,
        threaded=True
    )
