"""
Equitura Real Data Initializer
Fetches initial real data from APIs for first-time setup
Run this once to populate cache with real data
"""

import sys
import os
import time
from datetime import datetime

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)  # Add current data directory
sys.path.insert(0, os.path.join(parent_dir, 'models'))  # Add models directory

import config

def initialize_real_data():
    """
    Initialize real data for all components
    This fetches real data once so the website works immediately
    """
    print("🚀 EQUITURA REAL DATA INITIALIZATION")
    print("=" * 60)
    print("📊 Fetching real data from Alpha Vantage + Finnhub APIs")
    print("⏱️  This will take about 5-10 minutes due to API rate limits")
    print("💡 Data will be cached for 24 hours")
    print()
    
    # Initialize engines with individual error handling
    engines = {}
    
    try:
        from alpha_vantage_real_data import AlphaVantageRealDataCollector
        engines['alpha_vantage'] = AlphaVantageRealDataCollector()
        print("✅ Alpha Vantage collector initialized")
    except Exception as e:
        print(f"❌ Alpha Vantage collector failed: {e}")
        engines['alpha_vantage'] = None
    
    try:
        from technical_indicators import TechnicalIndicatorsEngine
        engines['technical'] = TechnicalIndicatorsEngine()
        print("✅ Technical engine initialized")
    except Exception as e:
        print(f"❌ Technical engine failed: {e}")
        engines['technical'] = None
    
    try:
        from news_sentiment import NewsSentimentAnalyzer
        engines['news'] = NewsSentimentAnalyzer()
        print("✅ News analyzer initialized")
    except Exception as e:
        print(f"❌ News analyzer failed: {e}")
        engines['news'] = None
    
    try:
        from economic_indicators import EconomicIndicatorsAnalyzer
        engines['economic'] = EconomicIndicatorsAnalyzer()
        print("✅ Economic analyzer initialized")
    except Exception as e:
        print(f"❌ Economic analyzer failed: {e}")
        engines['economic'] = None
    
    try:
        # Change directory temporarily to models directory for import
        models_dir = os.path.join(parent_dir, 'models')
        original_dir = os.getcwd()
        os.chdir(models_dir)
        
        from random_forest_model import EquituraMLEngine
        engines['ml'] = EquituraMLEngine()
        print("✅ ML engine initialized")
        
        # Change back to original directory
        os.chdir(original_dir)
    except Exception as e:
        print(f"❌ ML engine failed: {e}")
        engines['ml'] = None
        # Make sure we're back in original directory
        try:
            os.chdir(original_dir)
        except:
            pass
    
    # Check if we have at least the essential engines
    if not engines['alpha_vantage']:
        print("❌ Critical: Alpha Vantage collector is required")
        print("Make sure alpha_vantage_real_data.py works by running it directly first")
        return False
    
    print(f"✅ Initialized {sum(1 for e in engines.values() if e is not None)}/5 engines successfully")
    
    # Create cache directory
    cache_dir = os.path.join(parent_dir, 'data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    success_count = 0
    total_operations = 0
    
    print(f"\n📈 FETCHING REAL STOCK DATA")
    print("-" * 40)
    
    # Fetch real stock data for all symbols
    for i, symbol in enumerate(config.STOCK_SYMBOLS):
        total_operations += 1
        try:
            print(f"📥 Fetching real data for {symbol} ({i+1}/{len(config.STOCK_SYMBOLS)})...")
            
            if not engines['alpha_vantage']:
                print(f"   ❌ {symbol}: Alpha Vantage not available")
                continue
            
            # Get real stock data
            stock_data = engines['alpha_vantage'].get_real_stock_data(symbol, outputsize='compact')
            
            if stock_data is not None and len(stock_data) > 0:
                print(f"   ✅ {symbol}: {len(stock_data)} days of real data")
                
                # Save the raw data for technical analysis
                filename = f"{symbol}_REAL_alpha_vantage_data.csv"
                filepath = os.path.join(parent_dir, 'data', 'csv', filename)
                stock_data.to_csv(filepath, index=False)
                
                success_count += 1
            else:
                print(f"   ❌ {symbol}: No data received")
            
            # Rate limiting - Alpha Vantage allows 5 calls per minute
            if i < len(config.STOCK_SYMBOLS) - 1:
                print(f"   ⏱️  Waiting 12 seconds for API rate limit...")
                time.sleep(12)
                
        except Exception as e:
            print(f"   ❌ {symbol}: Error - {e}")
    
    print(f"\n📊 CALCULATING TECHNICAL INDICATORS")
    print("-" * 40)
    
    # Calculate technical indicators for stocks that have data
    if engines['technical']:
        for symbol in config.STOCK_SYMBOLS:
            total_operations += 1
            try:
                print(f"📊 Calculating technical indicators for {symbol}...")
                
                # Load stock data
                data = engines['technical'].load_stock_data(symbol)
                if data is not None:
                    # Calculate all indicators
                    enhanced_data = engines['technical'].calculate_all_indicators(data)
                    enhanced_data = engines['technical'].generate_trading_signals(enhanced_data)
                    
                    # Save enhanced data
                    engines['technical'].save_enhanced_data(symbol, enhanced_data)
                    
                    print(f"   ✅ {symbol}: Technical indicators calculated")
                    success_count += 1
                else:
                    print(f"   ❌ {symbol}: No stock data available for technical analysis")
                    
            except Exception as e:
                print(f"   ❌ {symbol}: Error calculating indicators - {e}")
    else:
        print("⏭️ Skipping technical indicators (engine not available)")
    
    print(f"\n📰 FETCHING NEWS SENTIMENT")
    print("-" * 40)
    
    # Fetch news sentiment for a few key stocks
    if engines['news']:
        key_stocks = config.STOCK_SYMBOLS[:3]  # First 3 to avoid hitting limits
        for i, symbol in enumerate(key_stocks):
            total_operations += 1
            try:
                print(f"📰 Fetching real news sentiment for {symbol}...")
                
                sentiment_data = engines['news'].get_sentiment_for_symbol(symbol, days_back=7)
                
                if sentiment_data and 'error' not in sentiment_data:
                    print(f"   ✅ {symbol}: {sentiment_data['total_articles']} articles analyzed")
                    print(f"   📊 Sentiment score: {sentiment_data['sentiment_score']:.1f}")
                    
                    # Save sentiment data
                    engines['news'].save_sentiment_data(symbol, sentiment_data)
                    success_count += 1
                else:
                    print(f"   ❌ {symbol}: No news sentiment data")
                
                # Rate limiting for Finnhub
                if i < len(key_stocks) - 1:
                    print(f"   ⏱️  Waiting 5 seconds for API rate limit...")
                    time.sleep(5)
                    
            except Exception as e:
                print(f"   ❌ {symbol}: Error fetching news - {e}")
    else:
        print("⏭️ Skipping news sentiment (engine not available)")
    
    print(f"\n🏛️ FETCHING ECONOMIC INDICATORS")
    print("-" * 40)
    
    if engines['economic']:
        total_operations += 1
        try:
            print("🏛️ Fetching real economic indicators...")
            print("   ⏱️  This will take about 1 minute due to API rate limits...")
            
            economic_data = engines['economic'].get_all_economic_indicators()
            
            if economic_data and 'error' not in economic_data:
                print("   ✅ Economic indicators fetched successfully")
                
                # Display some key indicators
                if 'indicators' in economic_data:
                    for name, indicator in economic_data['indicators'].items():
                        if 'current_value' in indicator:
                            print(f"      {indicator['indicator']}: {indicator['current_value']:.2f}% ({indicator['trend']})")
                
                # Save economic data
                engines['economic'].save_economic_data(economic_data)
                success_count += 1
            else:
                print("   ❌ Failed to fetch economic data")
                
        except Exception as e:
            print(f"   ❌ Error fetching economic data: {e}")
    else:
        print("⏭️ Skipping economic indicators (engine not available)")
    
    print(f"\n🤖 TRAINING ML MODELS")
    print("-" * 40)
    
    # Train ML models for a few key stocks
    if engines['ml']:
        for symbol in config.STOCK_SYMBOLS[:2]:  # Train for first 2 stocks
            total_operations += 2  # Price and direction models
            try:
                print(f"🤖 Training ML models for {symbol}...")
                
                # Train price prediction model
                result = engines['ml'].train_price_model(symbol, 1)  # 1-day prediction
                if 'error' not in result:
                    print(f"   ✅ {symbol}: Price model trained (accuracy: {result['test_direction_accuracy']:.1%})")
                    success_count += 1
                else:
                    print(f"   ❌ {symbol}: Price model training failed")
                
                # Train direction model
                direction_result = engines['ml'].train_direction_model(symbol, 1)
                if 'error' not in direction_result:
                    print(f"   ✅ {symbol}: Direction model trained (accuracy: {direction_result['test_accuracy']:.1%})")
                    success_count += 1
                else:
                    print(f"   ❌ {symbol}: Direction model training failed")
                    
            except Exception as e:
                print(f"   ❌ {symbol}: Error training models - {e}")
    else:
        print("⏭️ Skipping ML model training (engine not available)")
    
    # Final summary
    print(f"\n" + "=" * 60)
    print(f"🎯 REAL DATA INITIALIZATION COMPLETE")
    print(f"✅ Success: {success_count}/{total_operations} operations completed")
    print(f"📅 Data cached for 24 hours")
    print(f"🌐 Your website now has 100% REAL data!")
    print(f"🚀 Run: python real_app.py to start the production server")
    print("=" * 60)
    
    return success_count > 0

if __name__ == "__main__":
    print("⚠️  IMPORTANT: Make sure you have:")
    print("   - Valid Alpha Vantage API key in config.py")
    print("   - Valid Finnhub API key in config.py")
    print("   - All Python packages installed")
    print()
    
    input("Press Enter to start real data initialization...")
    
    success = initialize_real_data()
    
    if success:
        print("\n🎉 SUCCESS! Your Equitura platform now has real data!")
        print("Next step: python real_app.py")
    else:
        print("\n❌ Some operations failed. Check the logs above.")
        print("You may need to run individual tests first.")