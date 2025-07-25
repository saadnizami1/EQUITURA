"""
Equitura AI Stock Prediction Platform
Advanced Technical Indicators Module
Calculates 50+ technical indicators from real stock data
"""

import pandas as pd
import numpy as np
import ta
import os
import sys
sys.path.append('..')
import config
from typing import Dict, List, Optional

class TechnicalIndicatorsEngine:
    """
    Advanced technical analysis engine with 50+ indicators
    Uses real stock data from Alpha Vantage
    """
    
    def __init__(self):
        """Initialize the technical indicators engine"""
        self.indicators_calculated = 0
        print("Technical Indicators Engine initialized")
        print(f"Will calculate {len(config.TECHNICAL_INDICATORS)} configured indicators")
    
    def load_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load real stock data from CSV (Alpha Vantage data)
        
        Args:
            symbol (str): Stock symbol
        
        Returns:
            pd.DataFrame: Stock data or None if not found
        """
        try:
            # Try Alpha Vantage file first
            filename = f"{symbol}_REAL_alpha_vantage_data.csv"
            filepath = os.path.join("..", config.DATA_DIR, filename)
            
            if not os.path.exists(filepath):
                # Fallback to other naming convention
                filename = f"{symbol}_historical_data.csv"
                filepath = os.path.join("..", config.DATA_DIR, filename)
            
            if not os.path.exists(filepath):
                print(f"No data file found for {symbol}")
                return None
            
            data = pd.read_csv(filepath)
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values('Date').reset_index(drop=True)
            
            print(f"Loaded {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return None
    
    def calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various moving averages
        """
        # Simple Moving Averages
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        data['EMA_50'] = data['Close'].ewm(span=50).mean()
        
        # Weighted Moving Average
        data['WMA_20'] = data['Close'].rolling(window=20).apply(
            lambda x: (x * np.arange(1, len(x) + 1)).sum() / np.arange(1, len(x) + 1).sum()
        )
        
        return data
    
    def calculate_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum-based indicators
        """
        # RSI (Relative Strength Index)
        data['RSI_14'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        data['RSI_7'] = ta.momentum.RSIIndicator(data['Close'], window=7).rsi()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'])
        data['STOCH_K'] = stoch.stoch()
        data['STOCH_D'] = stoch.stoch_signal()
        
        # Williams %R
        data['WILLIAMS_R'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
        
        # Rate of Change
        data['ROC_10'] = ta.momentum.ROCIndicator(data['Close'], window=10).roc()
        data['ROC_20'] = ta.momentum.ROCIndicator(data['Close'], window=20).roc()
        
        # Money Flow Index
        data['MFI_14'] = ta.volume.MFIIndicator(data['High'], data['Low'], data['Close'], data['Volume']).money_flow_index()
        
        return data
    
    def calculate_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend-following indicators
        """
        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_SIGNAL'] = macd.macd_signal()
        data['MACD_HISTOGRAM'] = macd.macd_diff()
        
        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'])
        data['ADX_14'] = adx.adx()
        data['ADX_POS'] = adx.adx_pos()
        data['ADX_NEG'] = adx.adx_neg()
        
        # Parabolic SAR
        data['PSAR'] = ta.trend.PSARIndicator(data['High'], data['Low'], data['Close']).psar()
        
        # Trix
        data['TRIX'] = ta.trend.TRIXIndicator(data['Close']).trix()
        
        # Mass Index
        data['MASS_INDEX'] = ta.trend.MassIndex(data['High'], data['Low']).mass_index()
        
        # CCI (Commodity Channel Index)
        data['CCI_20'] = ta.trend.CCIIndicator(data['High'], data['Low'], data['Close']).cci()
        
        # DPO (Detrended Price Oscillator)
        data['DPO_20'] = ta.trend.DPOIndicator(data['Close']).dpo()
        
        return data
    
    def calculate_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility indicators
        """
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data['Close'])
        data['BB_HIGH'] = bollinger.bollinger_hband()
        data['BB_LOW'] = bollinger.bollinger_lband()
        data['BB_MID'] = bollinger.bollinger_mavg()
        data['BB_WIDTH'] = bollinger.bollinger_wband()
        data['BB_PERCENT'] = bollinger.bollinger_pband()
        
        # Average True Range
        data['ATR_14'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
        data['ATR_20'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=20).average_true_range()
        
        # Keltner Channels
        keltner = ta.volatility.KeltnerChannel(data['High'], data['Low'], data['Close'])
        data['KELTNER_HIGH'] = keltner.keltner_channel_hband()
        data['KELTNER_LOW'] = keltner.keltner_channel_lband()
        data['KELTNER_MID'] = keltner.keltner_channel_mband()
        
        # Donchian Channels
        donchian = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close'])
        data['DONCHIAN_HIGH'] = donchian.donchian_channel_hband()
        data['DONCHIAN_LOW'] = donchian.donchian_channel_lband()
        data['DONCHIAN_MID'] = donchian.donchian_channel_mband()
        
        return data
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based indicators
        """
        # On Balance Volume
        data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
        
        # Chaikin Money Flow
        data['CMF_20'] = ta.volume.ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], data['Volume']).chaikin_money_flow()
        
        # Accumulation/Distribution Line
        data['AD_LINE'] = ta.volume.AccDistIndexIndicator(data['High'], data['Low'], data['Close'], data['Volume']).acc_dist_index()
        
        # Volume SMA
        data['VOLUME_SMA_20'] = data['Volume'].rolling(window=20).mean()
        data['VOLUME_RATIO'] = data['Volume'] / data['VOLUME_SMA_20']
        
        # Price Volume Trend
        data['PVT'] = ta.volume.VolumePriceTrendIndicator(data['Close'], data['Volume']).volume_price_trend()
        
        # Negative Volume Index
        data['NVI'] = ta.volume.NegativeVolumeIndexIndicator(data['Close'], data['Volume']).negative_volume_index()
        
        return data
    
    def calculate_custom_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate custom indicators for enhanced analysis
        """
        # Price action indicators
        data['HIGH_LOW_RATIO'] = data['High'] / data['Low']
        data['CLOSE_OPEN_RATIO'] = data['Close'] / data['Open']
        
        # Volatility measures
        data['DAILY_RETURN'] = data['Close'].pct_change()
        data['VOLATILITY_10'] = data['DAILY_RETURN'].rolling(window=10).std()
        data['VOLATILITY_30'] = data['DAILY_RETURN'].rolling(window=30).std()
        
        # Price position within range
        data['PRICE_POSITION'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
        
        # Gap analysis
        data['GAP'] = data['Open'] - data['Close'].shift(1)
        data['GAP_PERCENT'] = (data['GAP'] / data['Close'].shift(1)) * 100
        
        # Support and Resistance levels
        data['RESISTANCE_20'] = data['High'].rolling(window=20).max()
        data['SUPPORT_20'] = data['Low'].rolling(window=20).min()
        
        # Trend strength
        data['TREND_STRENGTH'] = (data['Close'] - data['SMA_20']) / data['SMA_20'] * 100
        
        # Volume trend
        data['VOLUME_TREND'] = data['Volume'].rolling(window=5).mean() / data['Volume'].rolling(window=20).mean()
        
        return data
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators
        
        Args:
            data (pd.DataFrame): Stock price data
        
        Returns:
            pd.DataFrame: Data with all technical indicators
        """
        try:
            print("Calculating moving averages...")
            data = self.calculate_moving_averages(data)
            
            print("Calculating momentum indicators...")
            data = self.calculate_momentum_indicators(data)
            
            print("Calculating trend indicators...")
            data = self.calculate_trend_indicators(data)
            
            print("Calculating volatility indicators...")
            data = self.calculate_volatility_indicators(data)
            
            print("Calculating volume indicators...")
            data = self.calculate_volume_indicators(data)
            
            print("Calculating custom indicators...")
            data = self.calculate_custom_indicators(data)
            
            # Count indicators
            indicator_columns = [col for col in data.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']]
            self.indicators_calculated = len(indicator_columns)
            
            print(f"SUCCESS: Calculated {self.indicators_calculated} technical indicators")
            return data
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return data
    
    def generate_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators
        """
        try:
            # RSI signals
            data['RSI_OVERSOLD'] = (data['RSI_14'] < 30).astype(int)
            data['RSI_OVERBOUGHT'] = (data['RSI_14'] > 70).astype(int)
            
            # MACD signals
            data['MACD_BULLISH'] = (data['MACD'] > data['MACD_SIGNAL']).astype(int)
            data['MACD_BEARISH'] = (data['MACD'] < data['MACD_SIGNAL']).astype(int)
            
            # Moving average signals
            data['MA_BULLISH'] = (data['Close'] > data['SMA_20']).astype(int)
            data['MA_BEARISH'] = (data['Close'] < data['SMA_20']).astype(int)
            
            # Bollinger Band signals
            data['BB_SQUEEZE'] = (data['BB_WIDTH'] < data['BB_WIDTH'].rolling(20).mean()).astype(int)
            data['BB_BREAKOUT_UP'] = (data['Close'] > data['BB_HIGH']).astype(int)
            data['BB_BREAKOUT_DOWN'] = (data['Close'] < data['BB_LOW']).astype(int)
            
            # Volume signals
            data['VOLUME_SPIKE'] = (data['VOLUME_RATIO'] > 1.5).astype(int)
            
            # Composite signal
            bullish_signals = ['RSI_OVERSOLD', 'MACD_BULLISH', 'MA_BULLISH', 'BB_BREAKOUT_UP']
            bearish_signals = ['RSI_OVERBOUGHT', 'MACD_BEARISH', 'MA_BEARISH', 'BB_BREAKOUT_DOWN']
            
            data['BULLISH_SCORE'] = data[bullish_signals].sum(axis=1)
            data['BEARISH_SCORE'] = data[bearish_signals].sum(axis=1)
            data['NET_SIGNAL'] = data['BULLISH_SCORE'] - data['BEARISH_SCORE']
            
            print("Generated trading signals")
            return data
            
        except Exception as e:
            print(f"Error generating signals: {e}")
            return data
    
    def save_enhanced_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Save data with technical indicators to CSV
        """
        try:
            filename = f"{symbol}_with_indicators.csv"
            filepath = os.path.join("..", config.DATA_DIR, filename)
            
            data.to_csv(filepath, index=False)
            print(f"Saved enhanced data: {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving enhanced data: {e}")
            return False
    
    def get_latest_analysis(self, symbol: str) -> Optional[Dict]:
        """
        Get latest technical analysis summary
        """
        try:
            data = self.load_stock_data(symbol)
            if data is None:
                return None
            
            data = self.calculate_all_indicators(data)
            data = self.generate_trading_signals(data)
            
            # Get latest values
            latest = data.iloc[-1]
            
            analysis = {
                'symbol': symbol,
                'date': latest['Date'],
                'current_price': latest['Close'],
                'rsi': latest['RSI_14'],
                'macd': latest['MACD'],
                'sma_20': latest['SMA_20'],
                'bb_position': latest['BB_PERCENT'],
                'volume_ratio': latest['VOLUME_RATIO'],
                'bullish_score': latest['BULLISH_SCORE'],
                'bearish_score': latest['BEARISH_SCORE'],
                'net_signal': latest['NET_SIGNAL'],
                'recommendation': 'BUY' if latest['NET_SIGNAL'] > 1 else 'SELL' if latest['NET_SIGNAL'] < -1 else 'HOLD'
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error getting analysis: {e}")
            return None

def main():
    """
    Test the technical indicators engine
    """
    print("EQUITURA TECHNICAL INDICATORS ENGINE - TEST")
    print("=" * 60)
    print("Testing with REAL stock data from Alpha Vantage")
    print()
    
    engine = TechnicalIndicatorsEngine()
    
    # Test with AAPL (should have real Alpha Vantage data)
    test_symbol = "AAPL"
    
    print(f"1. LOADING REAL DATA FOR {test_symbol}")
    print("-" * 40)
    data = engine.load_stock_data(test_symbol)
    
    if data is None:
        print(f"ERROR: No data found for {test_symbol}")
        print("Make sure you've run alpha_vantage_real_data.py first")
        return
    
    print(f"Loaded {len(data)} days of real data")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"Latest price: ${data['Close'].iloc[-1]:.2f}")
    
    print(f"\n2. CALCULATING TECHNICAL INDICATORS")
    print("-" * 40)
    enhanced_data = engine.calculate_all_indicators(data)
    
    print(f"\n3. GENERATING TRADING SIGNALS")
    print("-" * 40)
    enhanced_data = engine.generate_trading_signals(enhanced_data)
    
    print(f"\n4. SAVING ENHANCED DATA")
    print("-" * 40)
    engine.save_enhanced_data(test_symbol, enhanced_data)
    
    print(f"\n5. LATEST TECHNICAL ANALYSIS")
    print("-" * 40)
    analysis = engine.get_latest_analysis(test_symbol)
    
    if analysis:
        print(f"Symbol: {analysis['symbol']}")
        print(f"Current Price: ${analysis['current_price']:.2f}")
        print(f"RSI (14): {analysis['rsi']:.2f}")
        print(f"MACD: {analysis['macd']:.4f}")
        print(f"20-day SMA: ${analysis['sma_20']:.2f}")
        print(f"Bollinger Band Position: {analysis['bb_position']:.2f}%")
        print(f"Volume Ratio: {analysis['volume_ratio']:.2f}")
        print(f"Bullish Signals: {analysis['bullish_score']}")
        print(f"Bearish Signals: {analysis['bearish_score']}")
        print(f"Net Signal: {analysis['net_signal']}")
        print(f"RECOMMENDATION: {analysis['recommendation']}")
    
    print("\n" + "=" * 60)
    print("TECHNICAL INDICATORS TEST COMPLETED!")
    print()
    print(f"✅ Calculated {engine.indicators_calculated} technical indicators")
    print("✅ Generated trading signals")
    print("✅ Saved enhanced data with indicators")
    print("✅ Ready for machine learning models")
    print()
    print("Next: Build ML models using this enhanced technical data")

if __name__ == "__main__":
    main()