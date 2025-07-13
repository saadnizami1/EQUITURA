# shared_data_manager.py
"""
Shared Data Management System with 2-Hour Global Countdown
All users see the same data and countdown timer
FIXED VERSION - Forces 2-hour refresh cycles
"""

import json
import os
import threading
import time
from datetime import datetime, timedelta
import logging
from finnhub_fetcher import FinnhubDataManager

logger = logging.getLogger(__name__)

class SharedDataManager:
    """Manages shared data state for all users with 2-hour countdown"""
    
    def __init__(self):
        self.finnhub_manager = FinnhubDataManager()
        self.data_lock = threading.Lock()
        
        # Global state file
        self.state_file = "data/cache/global_state.json"
        
        # ✅ FORCE 2-HOUR SYSTEM
        self.refresh_cycle_hours = 2  # ALWAYS 2 HOURS
        
        # Load or initialize global state
        self._load_global_state()
        
        # Start background refresh thread
        self._start_background_refresher()
        
        logger.info("✅ Shared Data Manager initialized - 2 HOUR SYSTEM")
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
                        # Force fresh fetch from Finnhub (ignore cache)
                        data = self.finnhub_manager._fetch_from_finnhub(symbol)
                        if data:
                            self.finnhub_manager._cache_data(symbol, data)
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
        """Get shared stock data (same for all users)"""
        return self.finnhub_manager.get_stock_data(symbol)
    
    def get_cache_status(self):
        """Get comprehensive cache status"""
        finnhub_status = self.finnhub_manager.get_cache_status()
        countdown_info = self.get_countdown_info()
        
        return {
            'shared_data_info': {
                'current_cycle': self.current_cycle_id,
                'last_refresh': self.last_refresh_time.isoformat(),
                'next_refresh': self.next_refresh_time.isoformat(),
                'refresh_cycle_hours': 2  # ✅ ALWAYS 2 HOURS
            },
            'countdown': countdown_info,
            'finnhub_cache': finnhub_status,
            'system_status': 'All users see the same data - 2 hour refresh cycle'
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