"""
Equitura - REAL Stock Data Collection via Alpha Vantage API
This uses your real Alpha Vantage API key to get 100% real stock data
NO MOCK DATA - All data is live from Alpha Vantage
"""

import requests
import pandas as pd
import time
import json
from datetime import datetime
import sys
sys.path.append('..')
import config

class AlphaVantageRealDataCollector:
    """
    Real stock data collector using Alpha Vantage API
    100% real data - no simulations or mock data
    """
    
    def __init__(self):
        self.api_key = config.ALPHA_VANTAGE_API_KEY
        self.base_url = "https://www.alphavantage.co/query"
        print(f"Initialized Alpha Vantage collector with API key: {self.api_key[:8]}...")
    
    def get_real_stock_data(self, symbol, outputsize='compact'):
        """
        Get REAL historical stock data from Alpha Vantage
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            outputsize (str): 'compact' (100 days) or 'full' (20+ years)
        
        Returns:
            pd.DataFrame: Real stock data from Alpha Vantage
        """
        print(f"Fetching REAL data for {symbol} from Alpha Vantage...")
        
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': outputsize,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                print(f"ERROR: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                print(f"API LIMIT: {data['Note']}")
                return None
            
            if 'Time Series (Daily)' not in data:
                print(f"ERROR: Unexpected response format")
                print(f"Response keys: {list(data.keys())}")
                return None
            
            # Parse the real data
            time_series = data['Time Series (Daily)']
            
            # Convert to DataFrame
            df_data = []
            for date_str, values in time_series.items():
                df_data.append({
                    'Date': pd.to_datetime(date_str),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': int(values['5. volume']),
                    'Symbol': symbol
                })
            
            df = pd.DataFrame(df_data)
            df = df.sort_values('Date').reset_index(drop=True)
            
            print(f"SUCCESS: Retrieved {len(df)} days of REAL data for {symbol}")
            print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
            print(f"Latest price: ${df['Close'].iloc[-1]:.2f}")
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
            return None
        except Exception as e:
            print(f"Error processing data: {e}")
            return None
    
    def get_real_current_price(self, symbol):
        """
        Get REAL current/latest price from Alpha Vantage
        """
        print(f"Getting REAL current price for {symbol}...")
        
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'Global Quote' not in data:
                print(f"ERROR: Could not get current price for {symbol}")
                return None
            
            quote = data['Global Quote']
            
            if not quote:
                print(f"ERROR: Empty quote data for {symbol}")
                return None
            
            current_price = float(quote['05. price'])
            change = float(quote['09. change'])
            change_percent = quote['10. change percent'].replace('%', '')
            
            result = {
                'symbol': symbol,
                'current_price': current_price,
                'change': change,
                'change_percent': float(change_percent),
                'volume': int(quote['06. volume']),
                'latest_trading_day': quote['07. latest trading day'],
                'previous_close': float(quote['08. previous close']),
                'timestamp': datetime.now()
            }
            
            print(f"REAL PRICE: {symbol} = ${current_price:.2f} ({change_percent}%)")
            return result
            
        except Exception as e:
            print(f"Error getting current price: {e}")
            return None
    
    def save_real_data(self, symbol, data):
        """
        Save REAL data to CSV
        """
        if data is None or data.empty:
            print(f"No data to save for {symbol}")
            return False
        
        try:
            filename = f"{symbol}_REAL_alpha_vantage_data.csv"
            filepath = f"../data/csv/{filename}"
            
            data.to_csv(filepath, index=False)
            print(f"SAVED REAL DATA: {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving data: {e}")
            return False

def test_real_data_collection():
    """
    Test collecting REAL data from Alpha Vantage
    """
    print("EQUITURA - REAL DATA COLLECTION TEST")
    print("=" * 60)
    print("Using Alpha Vantage API for 100% REAL stock data")
    print("NO MOCK DATA - All data is live from financial markets")
    print()
    
    collector = AlphaVantageRealDataCollector()
    
    # Test with Apple first
    test_symbol = "AAPL"
    
    # Get real historical data
    print(f"1. GETTING REAL HISTORICAL DATA FOR {test_symbol}")
    print("-" * 40)
    real_data = collector.get_real_stock_data(test_symbol, outputsize='compact')
    
    if real_data is not None:
        print(f"✅ SUCCESS: Got {len(real_data)} days of REAL data")
        
        # Save the real data
        collector.save_real_data(test_symbol, real_data)
        
        # Show recent data
        print("\nRECENT REAL PRICE DATA:")
        recent = real_data.tail(5)
        for _, row in recent.iterrows():
            print(f"{row['Date'].strftime('%Y-%m-%d')}: ${row['Close']:.2f}")
        
    else:
        print("❌ FAILED to get real historical data")
        return
    
    print(f"\n2. GETTING REAL CURRENT PRICE FOR {test_symbol}")
    print("-" * 40)
    
    # Wait to respect API limits
    time.sleep(12)  # Alpha Vantage free tier: 5 calls per minute
    
    current_price = collector.get_real_current_price(test_symbol)
    
    if current_price:
        print(f"✅ SUCCESS: Real current price retrieved")
        print(f"Real-time price: ${current_price['current_price']:.2f}")
        print(f"Change today: {current_price['change_percent']:+.2f}%")
        print(f"Volume: {current_price['volume']:,}")
        print(f"Last trading day: {current_price['latest_trading_day']}")
    else:
        print("❌ FAILED to get real current price")
    
    print("\n" + "=" * 60)
    print("REAL DATA COLLECTION TEST COMPLETED")
    print()
    print("KEY POINTS:")
    print("✅ This is 100% REAL data from Alpha Vantage")
    print("✅ No mock data, simulations, or fake numbers")
    print("✅ Direct from financial markets via professional API")
    print("✅ Perfect for college applications - shows real technical skills")
    print()
    print("API LIMITS:")
    print("- Free tier: 5 calls per minute, 500 calls per day")
    print("- Recommended: Wait 12+ seconds between calls")
    print("- This is sufficient for a professional college project")

if __name__ == "__main__":
    test_real_data_collection()