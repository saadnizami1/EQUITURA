# shared_data_manager.py
"""
Shared Data Management System with 2-Hour Global Countdown
All users see the same data and countdown timer
YFINANCE ONLY VERSION - Using ONLY Yahoo Finance (no API keys needed)
"""

import json
import os
import threading
import time
from datetime import datetime, timedelta
import logging
import yfinance as yf
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class YahooFinanceDataManager:
    """Yahoo Finance data manager - replacement for FinnhubDataManager"""
    
    def __init__(self):
        self.cache_dir = "data/cache"
        self.ensure_cache_directory()
        logger.info("✅ Yahoo Finance Data Manager initialized")
    
    def ensure_cache_directory(self):
        """Ensure cache directory exists"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_stock_data(self, symbol):
        """Get stock data using Yahoo Finance"""
        try:
            # Try cache first
            cached_data = self._load_cached_data(symbol)
            if cached_data and self._is_cache_fresh(cached_data):
                return cached_data
            
            # Fetch fresh data
            fresh_data = self._fetch_from_yahoo(symbol)
            if fresh_data:
                self._cache_data(symbol, fresh_data)
                return fresh_data
            
            # Return cached data if fresh fetch fails
            return cached_data
            
        except Exception as e:
            logger.error(f"❌ Error getting stock data for {symbol}: {e}")
            return None
    
    def _fetch_from_yahoo(self, symbol):
        """Fetch data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data (1 year)
            hist = ticker.history(period="1y")
            if hist.empty:
                return None
            
            # Get current info
            info = ticker.info
            current_price = hist['Close'].iloc[-1]
            previous_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            
            # Format data
            stock_data = {
                'symbol': symbol,
                'current_price': float(current_price),
                'previous_close': float(previous_close),
                'price_change': float(current_price - previous_close),
                'price_change_percent': float(((current_price - previous_close) / previous_close) * 100),
                'dates': [date.strftime('%Y-%m-%d') for date in hist.index],
                'opens': [float(x) for x in hist['Open'].tolist()],
                'highs': [float(x) for x in hist['High'].tolist()],
                'lows': [float(x) for x in hist['Low'].tolist()],
                'closes': [float(x) for x in hist['Close'].tolist()],
                'volumes': [int(x) for x in hist['Volume'].tolist()],
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'Technology'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE'),
                'high_52_week': float(hist['High'].max()),
                'low_52_week': float(hist['Low'].min()),
                'last_updated': datetime.now().isoformat(),
                'cache_expires': (datetime.now() + timedelta(hours=2)).isoformat()
            }
            
            logger.info(f"✅ Fetched data for {symbol} from Yahoo Finance")
            return stock_data
            
        except Exception as e:
            logger.error(f"❌ Error fetching {symbol} from Yahoo Finance: {e}")
            return None
    
    def _load_cached_data(self, symbol):
        """Load cached data from disk"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{symbol}_yahoo.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"❌ Error loading cached data for {symbol}: {e}")
        return None
    
    def _cache_data(self, symbol, data):
        """Cache data to disk"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{symbol}_yahoo.json")
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Error caching data for {symbol}: {e}")
    
    def _is_cache_fresh(self, cached_data):
        """Check if cached data is still fresh (within 2 hours)"""
        try:
            cache_expires = datetime.fromisoformat(cached_data.get('cache_expires', '2000-01-01'))
            return datetime.now() < cache_expires
        except:
            return False
    
    def get_cache_status(self):
        """Get cache status for all symbols"""
        try:
            cache_status = {}
            
            # Import symbols from config
            import config
            symbols = config.STOCK_SYMBOLS[:10]  # First 10 symbols
            
            for symbol in symbols:
                cached_data = self._load_cached_data(symbol)
                if cached_data:
                    is_fresh = self._is_cache_fresh(cached_data)
                    last_updated = datetime.fromisoformat(cached_data.get('last_updated', '2000-01-01'))
                    age_hours = (datetime.now() - last_updated).total_seconds() / 3600
                    
                    cache_status[symbol] = {
                        'cached': True,
                        'is_fresh': is_fresh,
                        'last_updated': cached_data.get('last_updated'),
                        'age_hours': round(age_hours, 1)
                    }
                else:
                    cache_status[symbol] = {
                        'cached': False,
                        'is_fresh': False,
                        'age_hours': 0
                    }
            
            return {
                'cache_status': cache_status,
                'total_symbols': len(symbols),
                'cached_count': sum(1 for status in cache_status.values() if status['cached']),
                'fresh_count': sum(1 for status in cache_status.values() if status['is_fresh'])
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting cache status: {e}")
            return {'error': str(e)}


class SharedDataManager:
    """Manages shared data state for all users with 2-hour countdown"""
    
    def __init__(self):
        # ✅ FIXED: Using Yahoo Finance instead of Finnhub
        self.yahoo_manager = YahooFinanceDataManager()
        self.data_lock = threading.Lock()
        
        # Global state file
        self.state_file = "data/cache/global_state.json"
        
        # ✅ FORCE 2-HOUR SYSTEM
        self.refresh_cycle_hours = 2  # ALWAYS 2 HOURS
        
        # Load or initialize global state
        self._load_global_state()
        
        # Start background refresh thread
        self._start_background_refresher()
        
        logger.info("✅ Shared Data Manager initialized - 2 HOUR SYSTEM with Yahoo Finance")
        logger.info(f"⏰ Next refresh: {self.next_refresh_time}")
    
    def _load_global_state(self):
        """Load global state or create new one - FORCE 2 HOURS"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.last_refresh_time = datetime.fromisoformat(state['last_refresh_time'])
                self.current_cycle_id = state.get('current_cycle_id', 1)
                
                # ✅ FORCE 2-HOUR REFRESH REGARDLESS OF SAVED STATE
                self.refresh_cycle_hours = 2
                self.next_refresh_time = self.last_refresh_time + timedelta(hours=2)
                
                # ✅ SAVE CORRECTED STATE IMMEDIATELY
                self._save_global_state()
                
                logger.info(f"📁 Loaded state: Cycle #{self.current_cycle_id} - FORCED 2h refresh")
            else:
                # Initialize new state
                self._initialize_new_cycle()
                
        except Exception as e:
            logger.error(f"❌ Error loading global state: {e}")
            self._initialize_new_cycle()
    
    def _initialize_new_cycle(self):
        """Initialize a new 2-hour cycle"""
        self.last_refresh_time = datetime.now()
        self.refresh_cycle_hours = 2  # ✅ ALWAYS 2 HOURS
        self.next_refresh_time = self.last_refresh_time + timedelta(hours=2)
        self.current_cycle_id = 1
        
        self._save_global_state()
        logger.info(f"🔄 Initialized new 2-hour cycle: #{self.current_cycle_id}")
    
    def _save_global_state(self):
        """Save global state to disk"""
        try:
            state = {
                'last_refresh_time': self.last_refresh_time.isoformat(),
                'next_refresh_time': self.next_refresh_time.isoformat(),
                'refresh_cycle_hours': 2,  # ✅ ALWAYS SAVE AS 2 HOURS
                'current_cycle_id': self.current_cycle_id,
                'saved_at': datetime.now().isoformat()
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Error saving global state: {e}")
    
    def get_countdown_info(self):
        """Get countdown information for frontend - 2 HOUR SYSTEM"""
        try:
            now = datetime.now()
            
            if now >= self.next_refresh_time:
                # Time for refresh!
                return {
                    'status': 'refresh_needed',
                    'message': 'Data refresh in progress...',
                    'countdown_seconds': 0,
                    'countdown_text': 'Refreshing now...',
                    'current_cycle': self.current_cycle_id,
                    'last_refresh': self.last_refresh_time.isoformat(),
                    'next_refresh': 'In progress...'
                }
            
            # Calculate remaining time
            time_remaining = self.next_refresh_time - now
            total_seconds = int(time_remaining.total_seconds())
            
            # ✅ FORMAT FOR 2-HOUR COUNTDOWN
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            
            # Create countdown text (no days, max 2 hours)
            if hours > 0:
                countdown_text = f"{hours}h {minutes}m {seconds}s"
            else:
                countdown_text = f"{minutes}m {seconds}s"
            
            # Calculate progress (0-100%)
            total_cycle_seconds = 2 * 3600  # 2 hours in seconds
            elapsed_seconds = total_cycle_seconds - total_seconds
            progress_percent = (elapsed_seconds / total_cycle_seconds) * 100
            
            return {
                'status': 'active',
                'message': f'Next data refresh in: {countdown_text}',
                'countdown_seconds': total_seconds,
                'countdown_text': countdown_text,
                'current_cycle': self.current_cycle_id,
                'last_refresh': self.last_refresh_time.isoformat(),
                'next_refresh': self.next_refresh_time.isoformat(),
                'progress_percent': round(progress_percent, 1),
                'refresh_cycle_hours': 2  # ✅ ALWAYS 2 HOURS
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating countdown: {e}")
            return {
                'status': 'error',
                'message': 'Countdown calculation error',
                'countdown_seconds': 0,
                'countdown_text': 'Unknown',
                'current_cycle': self.current_cycle_id,
                'progress_percent': 0,
                'refresh_cycle_hours': 2
            }
    
    def is_refresh_needed(self):
        """Check if 2-hour refresh is needed"""
        return datetime.now() >= self.next_refresh_time
    
    def refresh_all_data(self):
        """Refresh all stock data for the new 2-hour cycle"""
        try:
            with self.data_lock:
                logger.info(f"🔄 Starting 2-hour data refresh for cycle #{self.current_cycle_id + 1}")
                
                # Import symbols from config
                import config
                symbols_to_refresh = config.STOCK_SYMBOLS[:10]  # First 10 symbols
                
                success_count = 0
                failed_count = 0
                
                for symbol in symbols_to_refresh:
                    try:
                        # Force fresh fetch from Yahoo Finance (ignore cache)
                        data = self.yahoo_manager._fetch_from_yahoo(symbol)
                        if data:
                            self.yahoo_manager._cache_data(symbol, data)
                            success_count += 1
                            logger.info(f"✅ Refreshed {symbol}")
                        else:
                            failed_count += 1
                            logger.warning(f"❌ Failed to refresh {symbol}")
                        
                        # Small delay between requests
                        time.sleep(1)
                        
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"❌ Error refreshing {symbol}: {e}")
                
                # ✅ UPDATE TO NEXT 2-HOUR CYCLE
                self.current_cycle_id += 1
                self.last_refresh_time = datetime.now()
                self.next_refresh_time = self.last_refresh_time + timedelta(hours=2)
                
                # Save updated state
                self._save_global_state()
                
                logger.info(f"🏁 2-hour refresh complete: {success_count} success, {failed_count} failed")
                logger.info(f"⏰ Next 2-hour refresh scheduled for: {self.next_refresh_time}")
                
                return {
                    'success': True,
                    'symbols_refreshed': success_count,
                    'symbols_failed': failed_count,
                    'new_cycle_id': self.current_cycle_id,
                    'next_refresh': self.next_refresh_time.isoformat(),
                    'refresh_cycle_hours': 2
                }
                
        except Exception as e:
            logger.error(f"❌ Error in 2-hour refresh: {e}")
            return {
                'success': False,
                'error': str(e),
                'refresh_cycle_hours': 2
            }
    
    def _start_background_refresher(self):
        """Start background thread to handle automatic 2-hour refreshes"""
        def refresh_worker():
            while True:
                try:
                    if self.is_refresh_needed():
                        logger.info("⏰ Automatic 2-hour refresh triggered")
                        self.refresh_all_data()
                    
                    # Check every 5 minutes
                    time.sleep(300)
                    
                except Exception as e:
                    logger.error(f"❌ Error in background refresher: {e}")
                    time.sleep(300)  # Wait 5 minutes before retrying
        
        refresh_thread = threading.Thread(target=refresh_worker, daemon=True)
        refresh_thread.start()
        logger.info("🔄 Background 2-hour refresher started")
    
    def get_shared_stock_data(self, symbol):
        """Get shared stock data (same for all users) - FIXED: Using Yahoo Finance"""
        return self.yahoo_manager.get_stock_data(symbol)
    
    def get_cache_status(self):
        """Get comprehensive cache status"""
        yahoo_status = self.yahoo_manager.get_cache_status()
        countdown_info = self.get_countdown_info()
        
        return {
            'shared_data_info': {
                'current_cycle': self.current_cycle_id,
                'last_refresh': self.last_refresh_time.isoformat(),
                'next_refresh': self.next_refresh_time.isoformat(),
                'refresh_cycle_hours': 2  # ✅ ALWAYS 2 HOURS
            },
            'countdown': countdown_info,
            'yahoo_cache': yahoo_status,
            'system_status': 'All users see the same data - 2 hour refresh cycle - Yahoo Finance'
        }
    
    def force_refresh_now(self):
        """Force immediate refresh (admin function)"""
        logger.info("🔧 Manual 2-hour refresh triggered")
        return self.refresh_all_data()
    
    def reset_to_2_hour_cycle(self):
        """Force reset to 2-hour cycle (emergency function)"""
        try:
            logger.info("🚨 EMERGENCY: Resetting to 2-hour cycle")
            
            # Delete old state file
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
            
            # Force new 2-hour cycle
            self._initialize_new_cycle()
            
            return {
                'success': True,
                'message': 'Reset to 2-hour cycle complete',
                'next_refresh': self.next_refresh_time.isoformat(),
                'cycle_id': self.current_cycle_id
            }
            
        except Exception as e:
            logger.error(f"❌ Error resetting cycle: {e}")
            return {
                'success': False,
                'error': str(e)
            }