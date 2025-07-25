#!/usr/bin/env python3
"""
Vestara AI Stock Prediction Platform
Generate Missing Stock Data for All 37 Stocks
Creates cache files for the 19 missing stocks using Yahoo Finance
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import time

# Missing stocks that need data
MISSING_STOCKS = [
    "ORCL", "AMD", "INTC", "BAC", "WFC", "AXP", "PYPL", 
    "DIS", "SBUX", "F", "GM", "KO", "PG", "JNJ", "UNH", 
    "CLR", "LMT", "APD", "BABA"
]

def ensure_cache_directory():
    """Create cache directory if it doesn't exist"""
    cache_dir = "data/cache"
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def fetch_yahoo_data(symbol, days=100):
    """Fetch real historical data from Yahoo Finance"""
    try:
        print(f"📥 Fetching {symbol} data from Yahoo Finance...")
        
        # Get stock data for the last 100 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 20)  # Extra buffer
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)
        
        if hist.empty or len(hist) < 50:
            print(f"❌ Insufficient data for {symbol}")
            return None
        
        # Take the last 100 data points
        hist = hist.tail(100)
        
        # Format data to match your existing structure
        stock_data = {
            'symbol': symbol,
            'dates': [date.strftime('%Y-%m-%d') for date in hist.index],
            'opens': hist['Open'].round(2).tolist(),
            'highs': hist['High'].round(2).tolist(),
            'lows': hist['Low'].round(2).tolist(),
            'closes': hist['Close'].round(2).tolist(),
            'volumes': hist['Volume'].tolist(),
            'current_price': float(hist['Close'].iloc[-1]),
            'price_change': float(hist['Close'].iloc[-1] - hist['Close'].iloc[-2]),
            'price_change_percent': float(((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100),
            'last_updated': datetime.now().isoformat(),
            'data_source': 'Yahoo Finance',
            'data_points': len(hist)
        }
        
        print(f"✅ {symbol}: ${stock_data['current_price']:.2f} ({stock_data['price_change_percent']:+.2f}%) - {len(hist)} points")
        return stock_data
        
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return None

def save_cache_file(symbol, stock_data, cache_dir):
    """Save stock data to cache file"""
    try:
        filename = f"finnhub_{symbol}.json"  # Keep same naming convention
        filepath = os.path.join(cache_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(stock_data, f, indent=2)
        
        print(f"💾 Saved {symbol} data to {filepath}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving {symbol}: {e}")
        return False

def main():
    """Generate data for all missing stocks"""
    print("🚀 VESTARA AI - GENERATING MISSING STOCK DATA")
    print("=" * 60)
    print(f"📊 Fetching data for {len(MISSING_STOCKS)} missing stocks")
    print("🌐 Using Yahoo Finance (unlimited free data)")
    print("=" * 60)
    
    cache_dir = ensure_cache_directory()
    successful = 0
    failed = 0
    
    for i, symbol in enumerate(MISSING_STOCKS, 1):
        print(f"\n[{i}/{len(MISSING_STOCKS)}] Processing {symbol}...")
        
        # Fetch data
        stock_data = fetch_yahoo_data(symbol)
        
        if stock_data:
            # Save to cache
            if save_cache_file(symbol, stock_data, cache_dir):
                successful += 1
            else:
                failed += 1
        else:
            failed += 1
        
        # Rate limiting - be nice to Yahoo Finance
        if i < len(MISSING_STOCKS):
            print("⏳ Waiting 2 seconds...")
            time.sleep(2)
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY RESULTS")
    print("=" * 60)
    print(f"✅ Successfully processed: {successful}/{len(MISSING_STOCKS)} stocks")
    print(f"❌ Failed: {failed}/{len(MISSING_STOCKS)} stocks")
    
    if successful > 0:
        print(f"\n🎉 SUCCESS! {successful} stocks now have data for LSTM training")
        print("📂 Cache files saved in: data/cache/")
        print("\n🚀 NEXT STEP: Run 'python train_all_stocks.py' to train LSTM models")
    
    if failed > 0:
        print(f"\n⚠️  {failed} stocks had issues - you can retry them individually")

if __name__ == "__main__":
    main()