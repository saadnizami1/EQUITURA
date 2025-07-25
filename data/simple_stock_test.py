"""
Simple Stock Data Test - Works offline with sample data
This version creates sample data to test the system when Yahoo Finance is down
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
sys.path.append('..')
import config

def create_sample_stock_data(symbol="AAPL", days=30):
    """
    Create realistic sample stock data for testing
    """
    # Start with a base price
    base_price = 150.0 if symbol == "AAPL" else 100.0
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate realistic price movements
    np.random.seed(42)  # For reproducible results
    
    prices = []
    current_price = base_price
    
    for i in range(len(dates)):
        # Random walk with slight upward bias
        change_percent = np.random.normal(0.001, 0.02)  # Small upward bias, 2% volatility
        current_price = current_price * (1 + change_percent)
        prices.append(current_price)
    
    # Create realistic OHLC data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Create realistic open, high, low based on close
        volatility = close * 0.01  # 1% intraday volatility
        
        open_price = close + np.random.normal(0, volatility * 0.5)
        high = max(open_price, close) + abs(np.random.normal(0, volatility * 0.3))
        low = min(open_price, close) - abs(np.random.normal(0, volatility * 0.3))
        volume = int(np.random.normal(50000000, 10000000))  # Realistic volume
        
        data.append({
            'Date': date,
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': max(volume, 1000000),  # Ensure positive volume
            'Symbol': symbol
        })
    
    return pd.DataFrame(data)

def test_sample_data_creation():
    """
    Test creating sample data for multiple stocks
    """
    print("Equitura Stock Data Test - Sample Data Mode")
    print("=" * 60)
    print("NOTE: Using sample data since Yahoo Finance is unavailable")
    print()
    
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    for symbol in symbols:
        print(f"Creating sample data for {symbol}...")
        
        # Create sample data
        data = create_sample_stock_data(symbol, days=90)
        
        print(f"SUCCESS: Created {len(data)} records for {symbol}")
        print(f"Date range: {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
        print(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        
        # Save to CSV
        filename = f"{symbol}_historical_data.csv"
        filepath = os.path.join("..", config.DATA_DIR, filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data.to_csv(filepath, index=False)
        print(f"Saved to: {filepath}")
        
        # Show current price (last record)
        current = data.iloc[-1]
        previous = data.iloc[-2]
        change = current['Close'] - previous['Close']
        change_percent = (change / previous['Close']) * 100
        
        print(f"Current price: ${current['Close']:.2f}")
        print(f"Change: {change_percent:+.2f}%")
        print(f"Volume: {current['Volume']:,}")
        print()
    
    print("=" * 60)
    print("Sample data created successfully!")
    print("This data can be used to test the technical indicators and ML models")
    print("When Yahoo Finance is working again, we can switch back to real data")
    print()
    print("Next steps:")
    print("1. Test technical indicators with this sample data")
    print("2. Build and train ML models")
    print("3. Create the web interface")
    print("4. Switch back to real data when Yahoo Finance works")

def try_real_yahoo_finance():
    """
    Try to connect to Yahoo Finance with different methods
    """
    print("Testing Yahoo Finance connection...")
    
    try:
        import yfinance as yf
        
        # Method 1: Basic ticker info
        print("Method 1: Getting basic info...")
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        
        if info and 'longName' in info:
            print(f"SUCCESS: {info['longName']}")
            return True
        else:
            print("FAILED: No company info returned")
            
    except Exception as e:
        print(f"FAILED: {str(e)}")
    
    try:
        # Method 2: Try historical data with different parameters
        print("Method 2: Trying historical data...")
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="5d")
        
        if not hist.empty:
            print(f"SUCCESS: Got {len(hist)} days of data")
            return True
        else:
            print("FAILED: No historical data")
            
    except Exception as e:
        print(f"FAILED: {str(e)}")
    
    return False

def main():
    """
    Main test function
    """
    print("Equitura Stock Data Connectivity Test")
    print("=" * 60)
    
    # First try real Yahoo Finance
    if try_real_yahoo_finance():
        print("\nYahoo Finance is working! You can use the original stock_data_collector.py")
        print("Try running: python stock_data_collector.py")
    else:
        print("\nYahoo Finance is not accessible right now.")
        print("This is common - Yahoo Finance often blocks automated access.")
        print("\nCreating sample data for development...")
        print()
        test_sample_data_creation()

if __name__ == "__main__":
    main()