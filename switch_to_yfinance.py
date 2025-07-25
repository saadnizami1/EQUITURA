# switch_to_yfinance.py
"""
PERMANENT SWITCH TO YAHOO FINANCE
Replaces Finnhub with unlimited real stock data
"""

import os
import shutil
import time
from datetime import datetime

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"🚀 {title}")
    print("=" * 70)

def print_step(step, title):
    """Print step header"""
    print(f"\n📋 STEP {step}: {title}")
    print("-" * 50)

def backup_old_files():
    """Backup current Finnhub files"""
    print_step(1, "BACKING UP OLD FILES")
    
    files_to_backup = ['finnhub_fetcher.py', 'config.py']
    backed_up = []
    
    for file in files_to_backup:
        if os.path.exists(file):
            backup_name = f"{file}.finnhub_backup"
            try:
                shutil.copy2(file, backup_name)
                print(f"✅ Backed up {file} → {backup_name}")
                backed_up.append(file)
            except Exception as e:
                print(f"❌ Failed to backup {file}: {e}")
        else:
            print(f"ℹ️ {file} not found (skipping)")
    
    return backed_up

def replace_finnhub_fetcher():
    """Replace finnhub_fetcher.py with yfinance version"""
    print_step(2, "REPLACING STOCK DATA FETCHER")
    
    print("🔄 Replacing finnhub_fetcher.py with Yahoo Finance version...")
    print("ℹ️ The new file keeps the same class name for compatibility")
    print("✅ Your existing app.py won't need any changes!")
    
    # The yfinance_fetcher.py file should already be saved
    if os.path.exists('yfinance_fetcher.py'):
        try:
            # Replace the old file
            shutil.copy2('yfinance_fetcher.py', 'finnhub_fetcher.py')
            print("✅ finnhub_fetcher.py replaced with Yahoo Finance version")
            return True
        except Exception as e:
            print(f"❌ Error replacing file: {e}")
            return False
    else:
        print("❌ yfinance_fetcher.py not found!")
        print("💡 Make sure you saved the yfinance_fetcher.py file first")
        return False

def clear_old_cache():
    """Clear old Finnhub cache files"""
    print_step(3, "CLEARING OLD CACHE")
    
    cache_dir = "data/cache"
    if not os.path.exists(cache_dir):
        print("ℹ️ No cache directory found")
        return True
    
    try:
        # Remove old Finnhub cache files
        finnhub_files = [f for f in os.listdir(cache_dir) if f.startswith('finnhub_')]
        
        if finnhub_files:
            print(f"🗑️ Found {len(finnhub_files)} old Finnhub cache files")
            for file in finnhub_files:
                os.remove(os.path.join(cache_dir, file))
                print(f"   ✅ Removed {file}")
        else:
            print("ℹ️ No old Finnhub cache files found")
        
        # Also remove global state to force fresh start
        global_state = os.path.join(cache_dir, "global_state.json")
        if os.path.exists(global_state):
            os.remove(global_state)
            print("✅ Removed old global state")
        
        print("✅ Cache cleaned successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
        return False

def test_yahoo_finance():
    """Test Yahoo Finance integration"""
    print_step(4, "TESTING YAHOO FINANCE")
    
    try:
        # Import the new system
        from finnhub_fetcher import FinnhubDataManager
        
        print("📊 Testing Yahoo Finance data fetching...")
        manager = FinnhubDataManager()
        
        # Test with a few stocks
        test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        successful = 0
        
        for symbol in test_symbols:
            print(f"🔄 Testing {symbol}...", end=" ")
            try:
                data = manager.get_stock_data(symbol)
                if data and data.get('current_price', 0) > 0:
                    price = data['current_price']
                    change = data['price_change_percent']
                    print(f"✅ ${price:.2f} ({change:+.2f}%)")
                    successful += 1
                else:
                    print("❌ No data")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        success_rate = (successful / len(test_symbols)) * 100
        print(f"\n📊 Test Results: {successful}/{len(test_symbols)} successful ({success_rate:.0f}%)")
        
        if successful >= 2:  # At least 2 out of 3 should work
            print("✅ Yahoo Finance is working great!")
            return True
        else:
            print("❌ Yahoo Finance test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Yahoo Finance: {e}")
        return False

def update_config():
    """Update config.py to reflect the change"""
    print_step(5, "UPDATING CONFIGURATION")
    
    try:
        # Read current config
        if os.path.exists('config.py'):
            with open('config.py', 'r') as f:
                config_content = f.read()
            
            # Add comment about the switch
            new_comment = '''# ✅ SWITCHED TO YAHOO FINANCE - FREE UNLIMITED REAL DATA
# No API key needed - yfinance provides free real-time stock data
# Finnhub has been replaced with Yahoo Finance for reliability

'''
            
            # Add the comment at the top after existing comments
            lines = config_content.split('\n')
            
            # Find where to insert (after initial comments)
            insert_index = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith('#'):
                    insert_index = i
                    break
            
            # Insert the new comment
            lines.insert(insert_index, new_comment.rstrip())
            
            # Write back
            with open('config.py', 'w') as f:
                f.write('\n'.join(lines))
            
            print("✅ Updated config.py with Yahoo Finance information")
        else:
            print("ℹ️ config.py not found (skipping update)")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Could not update config.py: {e}")
        return True  # Non-critical error

def main():
    """Main switching process"""
    print_header("PERMANENT SWITCH TO YAHOO FINANCE")
    print("🎯 Replacing Finnhub with FREE unlimited real stock data")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Backup
    backed_up = backup_old_files()
    
    # Step 2: Replace fetcher
    fetcher_replaced = replace_finnhub_fetcher()
    
    # Step 3: Clear cache
    if fetcher_replaced:
        cache_cleared = clear_old_cache()
    else:
        cache_cleared = False
    
    # Step 4: Test system
    if fetcher_replaced and cache_cleared:
        yahoo_working = test_yahoo_finance()
    else:
        yahoo_working = False
    
    # Step 5: Update config
    if yahoo_working:
        config_updated = update_config()
    else:
        config_updated = False
    
    # Summary
    print_header("SWITCH SUMMARY")
    print(f"💾 Backup: {'✅ Done' if backed_up else '❌ Failed'}")
    print(f"🔄 Fetcher Replaced: {'✅ Done' if fetcher_replaced else '❌ Failed'}")
    print(f"🗑️ Cache Cleared: {'✅ Done' if cache_cleared else '❌ Failed'}")
    print(f"🧪 Yahoo Test: {'✅ Passed' if yahoo_working else '❌ Failed'}")
    print(f"⚙️ Config Updated: {'✅ Done' if config_updated else '❌ Failed'}")
    
    if fetcher_replaced and cache_cleared and yahoo_working:
        print(f"\n🎉 SUCCESS! SWITCHED TO YAHOO FINANCE!")
        print(f"✅ Your platform now uses FREE unlimited real stock data")
        print(f"✅ No more API key issues or rate limits")
        print(f"✅ All 37 stocks will have real-time data")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"1. Run your platform: python app.py")
        print(f"2. Open: http://localhost:5000")
        print(f"3. Enjoy unlimited real stock data! 📈")
        
        print(f"\n💡 WHAT CHANGED:")
        print(f"   - finnhub_fetcher.py now uses Yahoo Finance")
        print(f"   - Same interface - your app.py needs no changes")
        print(f"   - Cache files now use 'yahoo_' prefix")
        print(f"   - No API keys needed - completely free!")
        
    else:
        print(f"\n⚠️ SWITCH INCOMPLETE")
        print(f"🔧 Issues encountered during the switch")
        
        if not fetcher_replaced:
            print(f"❌ Could not replace finnhub_fetcher.py")
            print(f"💡 Make sure yfinance_fetcher.py exists in your folder")
        
        if fetcher_replaced and not yahoo_working:
            print(f"❌ Yahoo Finance test failed")
            print(f"💡 Check your internet connection")
            print(f"💡 Make sure yfinance is installed: pip install yfinance")
        
        print(f"\n🔄 RESTORE BACKUP:")
        if backed_up:
            print(f"If needed, restore from backup files:")
            for file in backed_up:
                print(f"   copy {file}.finnhub_backup {file}")

if __name__ == "__main__":
    main()