# fetch_missing_data.py
"""
Direct Finnhub data fetcher for missing stocks
Bypasses the web interface and fetches data directly
"""
import requests
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

print("🚀 DIRECT FINNHUB DATA FETCHER")
print("=" * 50)
print(f"🔑 Using API Key: {config.FINNHUB_API_KEY[:10]}...")

# Missing stocks that need data (from your training output)
missing_stocks = [
    "ORCL", "AMD", "INTC", "BAC", "WFC", "AXP", "PYPL", 
    "DIS", "SBUX", "F", "GM", "KO", "PG", "JNJ", "UNH", 
    "CLR", "LMT", "APD", "BABA"
]

print(f"📊 Will fetch data for {len(missing_stocks)} missing stocks:")
print(f"   {', '.join(missing_stocks)}")

def fetch_stock_data(symbol):
    """Fetch stock data directly from Finnhub API"""
    try:
        print(f"\n🌐 Fetching {symbol} from Finnhub...")
        
        # Calculate date range (last 100 trading days ≈ 5 months)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=150)  # Extra buffer for weekends
        
        # Convert to Unix timestamps
        end_timestamp = int(end_date.timestamp())
        start_timestamp = int(start_date.timestamp())
        
        # Finnhub API endpoint for historical data
        url = "https://finnhub.io/api/v1/stock/candle"
        params = {
            'symbol': symbol,
            'resolution': 'D',  # Daily data
            'from': start_timestamp,
            'to': end_timestamp,
            'token': config.FINNHUB_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('s') == 'ok' and 'c' in data:
                # Convert to the format expected by LSTM predictor
                formatted_data = {
                    'symbol': symbol,
                    'current_price': data['c'][-1],  # Last close price
                    'price_change': data['c'][-1] - data['c'][-2] if len(data['c']) > 1 else 0,
                    'price_change_percent': ((data['c'][-1] - data['c'][-2]) / data['c'][-2] * 100) if len(data['c']) > 1 else 0,
                    'dates': [datetime.fromtimestamp(t).strftime('%Y-%m-%d') for t in data['t']],
                    'opens': data['o'],
                    'highs': data['h'],
                    'lows': data['l'],
                    'closes': data['c'],
                    'volumes': data['v'],
                    'last_updated': datetime.now().isoformat(),
                    'data_source': 'Finnhub Professional API'
                }
                
                # Save to cache file
                cache_file = os.path.join(config.CACHE_DIR, f"finnhub_{symbol}.json")
                with open(cache_file, 'w') as f:
                    json.dump(formatted_data, f, indent=2)
                
                print(f"   ✅ {symbol}: {len(data['c'])} data points, price: ${data['c'][-1]:.2f}")
                return True
                
            else:
                print(f"   ❌ {symbol}: No data returned (status: {data.get('s', 'unknown')})")
                return False
                
        else:
            print(f"   ❌ {symbol}: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ {symbol}: Error - {e}")
        return False

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
    
    if fetch_stock_data(symbol):
        successful_fetches.append(symbol)
    else:
        failed_fetches.append(symbol)
    
    # Delay between requests to respect rate limits (60/minute = 1 per second)
    if i < len(missing_stocks):
        print(f"   ⏳ Waiting 2 seconds for rate limit...")
        time.sleep(2)

# Summary
print("\n" + "=" * 50)
print("🏁 DATA FETCHING COMPLETE!")
print("=" * 50)
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

if failed_fetches:
    print("⚠️ For failed stocks:")
    print("   - Check if symbols are correct")
    print("   - Verify Finnhub API key is working")
    print("   - Some stocks might not be available on Finnhub")

print(f"\n📊 Expected Result:")
print(f"   Previous: 18/37 stocks working")
print(f"   After this: {18 + len(successful_fetches)}/37 stocks working")
print(f"   Improvement: +{len(successful_fetches)} stocks! 🚀")

print(f"\n🔑 Finnhub API Status:")
print(f"   Requests made: {len(missing_stocks)}")
print(f"   Rate limit: 60/minute (we used 2 second delays)")
print(f"   API Key: {config.FINNHUB_API_KEY[:10]}...")