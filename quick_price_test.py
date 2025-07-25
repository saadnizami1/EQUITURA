# quick_price_test.py
"""
Quick Stock Price Fetcher - Test Real Data
Fetches current prices for your stocks using yfinance
"""

import yfinance as yf
import time
from datetime import datetime

# Your stock symbols
STOCKS = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "ORCL", "AMD", "INTC", 
    
    # Financial Services
    "JPM", "V", "BAC", "WFC", "AXP", "PYPL",
    
    # Consumer Discretionary
    "DIS", "NKE", "SBUX", "F", "GM", 
    
    # Consumer Staples
    "KO", "PG",
    
    # Healthcare
    "JNJ", "PFE", "UNH",
    
    # Energy
    "XOM", "CLR",
    
    # Industrials
    "BA", "LMT", "APD", "X",
    
    # Communication Services
    "T", "VZ",
    
    # Materials
    "RGLD",
    
    # International/ADR
    "BABA"
]

def get_stock_price(symbol):
    """Get current stock price and info"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get current price
        current_price = info.get('regularMarketPrice', info.get('currentPrice', 0))
        previous_close = info.get('previousClose', current_price)
        
        if current_price == 0:
            return None
        
        # Calculate change
        price_change = current_price - previous_close
        price_change_percent = (price_change / previous_close) * 100 if previous_close else 0
        
        return {
            'symbol': symbol,
            'price': current_price,
            'change': price_change,
            'change_percent': price_change_percent,
            'company': info.get('shortName', symbol)
        }
        
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return None

def main():
    """Fetch and display all stock prices"""
    
    print("🚀 REAL STOCK PRICE FETCHER")
    print("=" * 80)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Fetching {len(STOCKS)} stocks using yfinance...")
    print()
    
    successful = 0
    failed = 0
    
    # Headers
    print(f"{'SYMBOL':<8} {'COMPANY':<25} {'PRICE':<12} {'CHANGE':<12} {'% CHANGE':<10}")
    print("-" * 80)
    
    for symbol in STOCKS:
        try:
            print(f"🔄 {symbol}...", end=" ", flush=True)
            
            data = get_stock_price(symbol)
            
            if data:
                # Format for display
                price_str = f"${data['price']:.2f}"
                change_str = f"${data['change']:+.2f}"
                percent_str = f"{data['change_percent']:+.2f}%"
                
                # Color coding
                if data['change_percent'] >= 0:
                    status = "✅"
                else:
                    status = "🔻"
                
                print(f"\r{status} {data['symbol']:<6} {data['company'][:23]:<25} {price_str:<12} {change_str:<12} {percent_str:<10}")
                successful += 1
            else:
                print(f"\r❌ {symbol:<6} {'Failed':<25} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
                failed += 1
                
        except Exception as e:
            print(f"\r❌ {symbol:<6} {'Error':<25} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
            failed += 1
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    # Summary
    print("\n" + "=" * 80)
    print(f"📊 SUMMARY:")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Total: {len(STOCKS)}")
    print(f"🎯 Success Rate: {(successful/len(STOCKS)*100):.1f}%")
    
    if successful > 0:
        print(f"\n🎉 SUCCESS! yfinance is working and getting REAL stock data!")
        print(f"💡 This proves your platform can use real data instead of Finnhub")
        print(f"🚀 Ready to switch your platform to use yfinance!")
    else:
        print(f"\n⚠️ No stocks fetched successfully")
        print(f"🔧 Check your internet connection or yfinance installation")

if __name__ == "__main__":
    main()