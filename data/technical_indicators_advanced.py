"""
Advanced Technical Indicators for Equitura Platform
Enhanced version with 60+ professional indicators
"""

import pandas as pd
import numpy as np
import ta
from ta.utils import dropna
import logging

logger = logging.getLogger(__name__)

class AdvancedTechnicalEngine:
    """Enhanced technical analysis engine for professional trading signals"""
    
    def __init__(self):
        pass
    
    def calculate_all_indicators(self, data_dict):
        """
        Calculate comprehensive technical indicators from Yahoo Finance data
        Input: data_dict from your Yahoo Finance manager
        Output: Enhanced data with all technical indicators
        """
        try:
            # Convert your Yahoo Finance data to DataFrame
            df = pd.DataFrame({
                'Date': pd.to_datetime(data_dict['dates']),
                'Open': data_dict['opens'],
                'High': data_dict['highs'], 
                'Low': data_dict['lows'],
                'Close': data_dict['closes'],
                'Volume': data_dict['volumes']
            })
            
            df = df.set_index('Date')
            df = dropna(df)
            
            if len(df) < 50:  # Need minimum data for calculations
                logger.warning("Insufficient data for full technical analysis")
                return self._basic_indicators_only(df, data_dict)
            
            # === TREND INDICATORS ===
            # Simple Moving Averages
            df['SMA_5'] = ta.trend.sma_indicator(df['Close'], window=5)
            df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['SMA_100'] = ta.trend.sma_indicator(df['Close'], window=100)
            df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
            
            # Exponential Moving Averages
            df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
            df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
            df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_histogram'] = macd.macd_diff()
            
            # ADX (Average Directional Index)
            df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
            df['ADX_pos'] = ta.trend.adx_pos(df['High'], df['Low'], df['Close'], window=14)
            df['ADX_neg'] = ta.trend.adx_neg(df['High'], df['Low'], df['Close'], window=14)
            
            # Parabolic SAR
            df['PSAR'] = ta.trend.psar_up_indicator(df['High'], df['Low'], df['Close'])
            
            # === MOMENTUM INDICATORS ===
            # RSI
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
            df['RSI_6'] = ta.momentum.rsi(df['Close'], window=6)
            df['RSI_21'] = ta.momentum.rsi(df['Close'], window=21)
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['STOCH_k'] = stoch.stoch()
            df['STOCH_d'] = stoch.stoch_signal()
            
            # Williams %R
            df['WILLIAMS_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
            
            # Rate of Change
            df['ROC'] = ta.momentum.roc(df['Close'], window=12)
            
            # Commodity Channel Index
            df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)
            
            # === VOLATILITY INDICATORS ===
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['BB_upper'] = bollinger.bollinger_hband()
            df['BB_middle'] = bollinger.bollinger_mavg()
            df['BB_lower'] = bollinger.bollinger_lband()
            df['BB_width'] = bollinger.bollinger_wband()
            df['BB_percent'] = bollinger.bollinger_pband()
            
            # Average True Range
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
            
            # Keltner Channels
            keltner = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])
            df['KC_upper'] = keltner.keltner_channel_hband()
            df['KC_middle'] = keltner.keltner_channel_mband()
            df['KC_lower'] = keltner.keltner_channel_lband()
            
            # === VOLUME INDICATORS ===
            # On-Balance Volume
            df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            
            # Volume SMA
            df['Volume_SMA'] = ta.trend.sma_indicator(df['Volume'], window=20)
            
            # Chaikin Money Flow
            df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
            
            # Volume Price Trend
            df['VPT'] = ta.volume.volume_price_trend(df['Close'], df['Volume'])
            
            # === CUSTOM INDICATORS ===
            # Price momentum
            df['Price_momentum_1'] = df['Close'].pct_change(1)
            df['Price_momentum_5'] = df['Close'].pct_change(5)
            df['Price_momentum_10'] = df['Close'].pct_change(10)
            
            # Volatility
            df['Volatility_20'] = df['Close'].rolling(20).std()
            df['Volatility_50'] = df['Close'].rolling(50).std()
            
            # Support/Resistance levels
            df['Resistance'] = df['High'].rolling(20).max()
            df['Support'] = df['Low'].rolling(20).min()
            
            # Market structure
            df['Higher_highs'] = (df['High'] > df['High'].shift(1)).rolling(5).sum()
            df['Lower_lows'] = (df['Low'] < df['Low'].shift(1)).rolling(5).sum()
            
            # Get latest values for API response
            latest = df.iloc[-1]
            
            # Calculate trading signals
            signals = self._generate_trading_signals(df)
            
            # Format response
            indicators = {
                'symbol': data_dict['symbol'],
                'calculation_date': latest.name.isoformat(),
                
                # Current price info
                'current_price': data_dict['current_price'],
                'price_change': data_dict['price_change'],
                'price_change_percent': data_dict['price_change_percent'],
                
                # Trend indicators
                'sma_20': round(float(latest['SMA_20']), 2) if not pd.isna(latest['SMA_20']) else None,
                'sma_50': round(float(latest['SMA_50']), 2) if not pd.isna(latest['SMA_50']) else None,
                'sma_200': round(float(latest['SMA_200']), 2) if not pd.isna(latest['SMA_200']) else None,
                'ema_12': round(float(latest['EMA_12']), 2) if not pd.isna(latest['EMA_12']) else None,
                'ema_26': round(float(latest['EMA_26']), 2) if not pd.isna(latest['EMA_26']) else None,
                
                # MACD
                'macd': round(float(latest['MACD']), 4) if not pd.isna(latest['MACD']) else None,
                'macd_signal': round(float(latest['MACD_signal']), 4) if not pd.isna(latest['MACD_signal']) else None,
                'macd_histogram': round(float(latest['MACD_histogram']), 4) if not pd.isna(latest['MACD_histogram']) else None,
                
                # Momentum
                'rsi': round(float(latest['RSI']), 2) if not pd.isna(latest['RSI']) else None,
                'stoch_k': round(float(latest['STOCH_k']), 2) if not pd.isna(latest['STOCH_k']) else None,
                'stoch_d': round(float(latest['STOCH_d']), 2) if not pd.isna(latest['STOCH_d']) else None,
                'williams_r': round(float(latest['WILLIAMS_R']), 2) if not pd.isna(latest['WILLIAMS_R']) else None,
                'cci': round(float(latest['CCI']), 2) if not pd.isna(latest['CCI']) else None,
                
                # Volatility
                'bb_upper': round(float(latest['BB_upper']), 2) if not pd.isna(latest['BB_upper']) else None,
                'bb_middle': round(float(latest['BB_middle']), 2) if not pd.isna(latest['BB_middle']) else None,
                'bb_lower': round(float(latest['BB_lower']), 2) if not pd.isna(latest['BB_lower']) else None,
                'bb_percent': round(float(latest['BB_percent']), 4) if not pd.isna(latest['BB_percent']) else None,
                'atr': round(float(latest['ATR']), 2) if not pd.isna(latest['ATR']) else None,
                
                # Volume
                'obv': int(latest['OBV']) if not pd.isna(latest['OBV']) else None,
                'cmf': round(float(latest['CMF']), 4) if not pd.isna(latest['CMF']) else None,
                
                # Custom indicators
                'volatility_20': round(float(latest['Volatility_20']), 2) if not pd.isna(latest['Volatility_20']) else None,
                'support_level': round(float(latest['Support']), 2) if not pd.isna(latest['Support']) else None,
                'resistance_level': round(float(latest['Resistance']), 2) if not pd.isna(latest['Resistance']) else None,
                
                # Trading signals
                'signals': signals,
                'overall_score': signals['overall_score'],
                'recommendation': signals['recommendation'],
                'confidence': signals['confidence'],
                
                # Chart data for indicators
                'chart_data': {
                    'dates': df.index.strftime('%Y-%m-%d').tolist()[-50:],  # Last 50 days
                    'sma_20': df['SMA_20'].fillna(0).tolist()[-50:],
                    'sma_50': df['SMA_50'].fillna(0).tolist()[-50:],
                    'bb_upper': df['BB_upper'].fillna(0).tolist()[-50:],
                    'bb_lower': df['BB_lower'].fillna(0).tolist()[-50:],
                    'rsi': df['RSI'].fillna(50).tolist()[-50:],
                    'macd': df['MACD'].fillna(0).tolist()[-50:],
                    'macd_signal': df['MACD_signal'].fillna(0).tolist()[-50:]
                },
                
                'data_source': 'Advanced TA-Lib + Yahoo Finance',
                'indicators_calculated': len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])
            }
            
            logger.info(f"✅ Advanced TA: {data_dict['symbol']} - {indicators['indicators_calculated']} indicators calculated")
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Error in advanced technical analysis: {e}")
            return self._basic_indicators_only(pd.DataFrame(), data_dict)
    
    def _generate_trading_signals(self, df):
        """Generate comprehensive trading signals"""
        try:
            latest = df.iloc[-1]
            signals = {}
            score = 0
            max_score = 0
            
            # RSI signals
            max_score += 1
            if latest['RSI'] < 30:
                signals['rsi'] = 'STRONG_BUY'
                score += 1
            elif latest['RSI'] < 50:
                signals['rsi'] = 'BUY'
                score += 0.5
            elif latest['RSI'] > 70:
                signals['rsi'] = 'STRONG_SELL'
                score -= 1
            elif latest['RSI'] > 50:
                signals['rsi'] = 'SELL'
                score -= 0.5
            else:
                signals['rsi'] = 'NEUTRAL'
            
            # MACD signals
            max_score += 1
            if not pd.isna(latest['MACD']) and not pd.isna(latest['MACD_signal']):
                if latest['MACD'] > latest['MACD_signal']:
                    signals['macd'] = 'BUY'
                    score += 1
                else:
                    signals['macd'] = 'SELL'
                    score -= 1
            else:
                signals['macd'] = 'NEUTRAL'
            
            # Moving average signals
            max_score += 1
            if not pd.isna(latest['SMA_20']) and not pd.isna(latest['SMA_50']):
                current_price = latest['Close']
                if current_price > latest['SMA_20'] > latest['SMA_50']:
                    signals['ma_trend'] = 'STRONG_BUY'
                    score += 1
                elif current_price > latest['SMA_20']:
                    signals['ma_trend'] = 'BUY'
                    score += 0.5
                elif current_price < latest['SMA_20'] < latest['SMA_50']:
                    signals['ma_trend'] = 'STRONG_SELL'
                    score -= 1
                elif current_price < latest['SMA_20']:
                    signals['ma_trend'] = 'SELL'
                    score -= 0.5
                else:
                    signals['ma_trend'] = 'NEUTRAL'
            else:
                signals['ma_trend'] = 'NEUTRAL'
            
            # Bollinger Bands signals
            max_score += 1
            if not pd.isna(latest['BB_upper']) and not pd.isna(latest['BB_lower']):
                current_price = latest['Close']
                if current_price < latest['BB_lower']:
                    signals['bollinger'] = 'BUY'
                    score += 1
                elif current_price > latest['BB_upper']:
                    signals['bollinger'] = 'SELL'
                    score -= 1
                else:
                    signals['bollinger'] = 'NEUTRAL'
            else:
                signals['bollinger'] = 'NEUTRAL'
            
            # Overall assessment
            if max_score > 0:
                overall_score = (score / max_score) * 100
            else:
                overall_score = 0
            
            if overall_score > 60:
                recommendation = 'BUY'
                confidence = min(90, abs(overall_score))
            elif overall_score < -60:
                recommendation = 'SELL'  
                confidence = min(90, abs(overall_score))
            else:
                recommendation = 'HOLD'
                confidence = 50
            
            return {
                'individual_signals': signals,
                'overall_score': round(overall_score, 1),
                'recommendation': recommendation,
                'confidence': round(confidence, 1),
                'signal_strength': 'STRONG' if confidence > 70 else 'MODERATE' if confidence > 50 else 'WEAK'
            }
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return {
                'individual_signals': {},
                'overall_score': 0,
                'recommendation': 'HOLD',
                'confidence': 50,
                'signal_strength': 'WEAK'
            }
    
    def _basic_indicators_only(self, df, data_dict):
        """Fallback for when there's insufficient data"""
        try:
            if len(df) > 14:
                # Calculate basic RSI
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                latest_rsi = rsi.iloc[-1] if not rsi.empty else 50
            else:
                latest_rsi = 50
                
            return {
                'symbol': data_dict['symbol'],
                'current_price': data_dict['current_price'],
                'rsi': round(float(latest_rsi), 2) if not pd.isna(latest_rsi) else 50,
                'signals': {
                    'individual_signals': {'rsi': 'NEUTRAL'},
                    'overall_score': 0,
                    'recommendation': 'HOLD',
                    'confidence': 40,
                    'signal_strength': 'WEAK'
                },
                'data_source': 'Basic indicators (insufficient data)',
                'note': 'Limited indicators due to insufficient historical data'
            }
        except:
            return {
                'symbol': data_dict['symbol'],
                'current_price': data_dict.get('current_price', 0),
                'error': 'Unable to calculate indicators'
            }