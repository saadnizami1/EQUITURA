# fetch_yahoo_finance.py
"""
Alternative data fetcher using Yahoo Finance (yfinance)
No API key needed - more reliable than Finnhub
"""
import yfinance as yf
import json
import os
import sys
import time
from datetime import datetime, timedelta

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import your config
import config

print("🚀 YAHOO FINANCE DATA FETCHER (ALTERNATIVE)")
print("=" * 60)
print("📈 Using yfinance library - no API key needed!")
print("🔄 This will fetch data in Finnhub-compatible format")

# Missing stocks that need data
missing_stocks = [
    "ORCL", "AMD", "INTC", "BAC", "WFC", "AXP", "PYPL", 
    "DIS", "SBUX", "F", "GM", "KO", "PG", "JNJ", "UNH", 
    "CLR", "LMT", "APD", "BABA"
]

print(f"\n📊 Will fetch data for {len(missing_stocks)} missing stocks:")
for i in range(0, len(missing_stocks), 6):
    batch = missing_stocks[i:i+6]
    print(f"   {', '.join(batch)}")

def fetch_yahoo_data(symbol):
    """Fetch stock data from Yahoo Finance and convert to Finnhub format"""
    try:
        print(f"\n🌐 Fetching {symbol} from Yahoo Finance...")
        
        # Create ticker object
        ticker = yf.Ticker(symbol)
        
        # Get 6 months of daily data
        hist_data = ticker.history(period="6mo", interval="1d")
        
        if hist_data.empty:
            print(f"   ❌ {symbol}: No data returned from Yahoo Finance")
            return False
        
        # Get current info
        info = ticker.info
        
        # Convert to Finnhub-compatible format
        formatted_data = {
            'symbol': symbol,
            'current_price': float(hist_data['Close'].iloc[-1]),
            'price_change': float(hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-2]) if len(hist_data) > 1 else 0,
            'price_change_percent': float(((hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-2]) / hist_data['Close'].iloc[-2] * 100)) if len(hist_data) > 1 else 0,
            'dates': [date.strftime('%Y-%m-%d') for date in hist_data.index],
            'opens': hist_data['Open'].tolist(),
            'highs': hist_data['High'].tolist(),
            'lows': hist_data['Low'].tolist(),
            'closes': hist_data['Close'].tolist(),
            'volumes': hist_data['Volume'].tolist(),
            'last_updated': datetime.now().isoformat(),
            'data_source': 'Yahoo Finance via yfinance',
            'company_name': info.get('longName', symbol),
            'sector': info.get('sector', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE'),
            'high_52_week': float(hist_data['High'].max()),
            'low_52_week': float(hist_data['Low'].min())
        }
        
        # Save to cache file (same format as Finnhub)
        cache_file = os.path.join(config.CACHE_DIR, f"finnhub_{symbol}.json")
        with open(cache_file, 'w') as f:
            json.dump(formatted_data, f, indent=2)
        
        current_price = formatted_data['current_price']
        change_pct = formatted_data['price_change_percent']
        data_points = len(formatted_data['closes'])
        
        print(f"   ✅ {symbol}: {data_points} data points, price: ${current_price:.2f} ({change_pct:+.2f}%)")
        print(f"      📊 Company: {formatted_data['company_name']}")
        print(f"      🏢 Sector: {formatted_data['sector']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ {symbol}: Error - {e}")
        return False

# Test yfinance first
print("\n🧪 Testing Yahoo Finance with AAPL...")
test_ticker = yf.Ticker("AAPL")
test_data = test_ticker.history(period="5d")

if test_data.empty:
    print("❌ Yahoo Finance not working! Check internet connection.")
    exit(1)
else:
    print(f"✅ Yahoo Finance working! AAPL: ${test_data['Close'].iloc[-1]:.2f}")

# Fetch data for all missing stocks
successful_fetches = []
failed_fetches = []

for i, symbol in enumerate(missing_stocks, 1):
    print(f"\n[{i}/{len(missing_stocks)}] Processing {symbol}...")
    
    # Check if cache file already exists
    cache_file = os.path.join(config.CACHE_DIR, f"finnhub_{symbol}.json")
    if os.path.exists(cache_file):
        print(f"   ✅ {symbol}: Cache already exists, skipping")
        successful_fetches.append(symbol)
        continue
    
    if fetch_yahoo_data(symbol):
        successful_fetches.append(symbol)
    else:
        failed_fetches.append(symbol)
    
    # Small delay to be respectful
    if i < len(missing_stocks):
        print(f"   ⏳ Waiting 1 second...")
        time.sleep(1)

# Summary
print("\n" + "=" * 60)
print("🏁 YAHOO FINANCE DATA FETCHING COMPLETE!")
print("=" * 60)
print(f"✅ Successfully fetched: {len(successful_fetches)} stocks")
if successful_fetches:
    print(f"   SUCCESS: {', '.join(successful_fetches)}")

print(f"❌ Failed to fetch: {len(failed_fetches)} stocks")
if failed_fetches:
    print(f"   FAILED: {', '.join(failed_fetches)}")

print(f"\n🎯 NEXT STEPS:")
if successful_fetches:
    print("✅ Run training again: python train_all_stocks.py")
    print(f"✅ {len(successful_fetches)} more stocks should now train successfully!")
    print("✅ Data saved in same format as Finnhub - LSTM will work normally")

if failed_fetches:
    print("⚠️ For failed stocks:")
    print("   - Symbol might not exist on Yahoo Finance")
    print("   - Network connection issues")
    print("   - Try running the script again")

print(f"\n📊 Expected Result:")
print(f"   Previous: 18/37 stocks working")
print(f"   After this: {18 + len(successful_fetches)}/37 stocks working")
print(f"   Improvement: +{len(successful_fetches)} stocks! 🚀")

print(f"\n📈 Data Source Info:")
print(f"   Source: Yahoo Finance (via yfinance library)")
print(f"   Format: Finnhub-compatible JSON files")
print(f"   Cache location: {config.CACHE_DIR}")
print(f"   API Key: Not needed! ✅")

if len(successful_fetches) > 0:
    print(f"\n🎉 SUCCESS! Yahoo Finance is working great!")
    print(f"   Consider switching to yfinance permanently")
    print(f"   It's more reliable and doesn't need API keys")