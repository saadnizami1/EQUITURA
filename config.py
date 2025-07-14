# config.py
"""
EQUITURA AI Stock Prediction Platform
RAILWAY DEPLOYMENT OPTIMIZED CONFIGURATION
Production-ready configuration for Railway.com deployment
"""

import os
from datetime import datetime

# ========== RAILWAY PRODUCTION SETTINGS ==========

# Environment detection - Default to production for Railway
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'production')
DEBUG_MODE = False if ENVIRONMENT == 'production' else True

# Railway-specific server configuration
PORT = int(os.environ.get('PORT', 5000))
HOST = "0.0.0.0"  # Always bind to all addresses for Railway

# ========== DATA SOURCE CONFIGURATION ==========

# Yahoo Finance (Free, no API key required)
YAHOO_FINANCE_ENABLED = True
FINNHUB_ENABLED = False  # Disabled for deployment
ALPHA_VANTAGE_ENABLED = False  # Disabled for deployment

# ========== OPTIMIZED CACHING FOR RAILWAY ==========

# Railway-optimized cache settings
UPDATE_INTERVAL_HOURS = 3  # Reasonable for free tier
CACHE_DURATION_HOURS = 3   # Balance between freshness and performance
DAILY_API_LIMIT = 999999   # Unlimited with Yahoo Finance

# ========== STOCK SYMBOLS ==========

# Core 37 stocks organized by sectors
STOCK_SYMBOLS = [
    # Technology (11 stocks)
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "ORCL", "AMD", "INTC",
    
    # Financial Services (6 stocks)  
    "JPM", "V", "BAC", "WFC", "AXP", "PYPL",
    
    # Consumer Discretionary (5 stocks)
    "DIS", "NKE", "SBUX", "F", "GM",
    
    # Consumer Staples (2 stocks)
    "KO", "PG",
    
    # Healthcare (3 stocks)
    "JNJ", "PFE", "UNH",
    
    # Energy (2 stocks)
    "XOM", "CLR",
    
    # Industrials (4 stocks)
    "BA", "LMT", "APD", "X",
    
    # Communication Services (2 stocks)
    "T", "VZ",
    
    # Materials (1 stock)
    "RGLD",
    
    # International (1 stock)
    "BABA"
]

# Sector mapping for organization
STOCK_SECTORS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "ORCL", "AMD", "INTC"],
    "Financial Services": ["JPM", "V", "BAC", "WFC", "AXP", "PYPL"],
    "Consumer Discretionary": ["DIS", "NKE", "SBUX", "F", "GM"],
    "Consumer Staples": ["KO", "PG"],
    "Healthcare": ["JNJ", "PFE", "UNH"],
    "Energy": ["XOM", "CLR"],
    "Industrials": ["BA", "LMT", "APD", "X"],
    "Communication Services": ["T", "VZ"],
    "Materials": ["RGLD"],
    "International": ["BABA"]
}

# ========== ML CONFIGURATION ==========

# Prediction timeframes
PREDICTION_DAYS_AHEAD = [1, 7, 30]

# ML model settings (optimized for Railway)
MODEL_RETRAIN_INTERVAL_HOURS = 168  # 7 days
AUTO_RETRAIN_ENABLED = False  # Disabled for Railway deployment
AUTO_RETRAIN_SCHEDULE = "disabled"

# Priority stocks for training (most stable/liquid)
PRIORITY_TRAINING_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 
    "NVDA", "META", "NFLX", "JPM", "V"
]

# ========== RAILWAY-SAFE FILE PATHS ==========

# Production-safe directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "csv")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")

def ensure_directories():
    """Create necessary directories if they don't exist (Railway-safe)"""
    directories = [DATA_DIR, MODELS_DIR, CACHE_DIR]
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            # Silent fail for Railway deployment
            pass

# Initialize directories
ensure_directories()

# ========== TECHNICAL INDICATORS ==========

TECHNICAL_INDICATORS = [
    "SMA_20", "SMA_50", "SMA_200",
    "EMA_12", "EMA_26", 
    "RSI_14", "MACD",
    "BOLLINGER_UPPER", "BOLLINGER_LOWER",
    "ATR_14", "OBV"
]

# ========== SYSTEM SETTINGS ==========

# Railway-optimized system configuration
SHARED_SYSTEM_ENABLED = True
GLOBAL_COUNTDOWN_ENABLED = True
BACKGROUND_REFRESH_ENABLED = True

# Logging configuration
LOG_LEVEL = "ERROR" if ENVIRONMENT == 'production' else "DEBUG"

# ========== HELPER FUNCTIONS ==========

def get_sectors():
    """Get list of all available sectors"""
    return list(STOCK_SECTORS.keys())

def get_stocks_by_sector(sector):
    """Get stocks for a specific sector"""
    return STOCK_SECTORS.get(sector, [])

def get_sector_for_stock(symbol):
    """Get sector for a specific stock symbol"""
    for sector, stocks in STOCK_SECTORS.items():
        if symbol in stocks:
            return sector
    return "Technology"  # Default fallback

def is_production():
    """Check if running in production environment"""
    return ENVIRONMENT == 'production'

def get_cache_duration():
    """Get cache duration based on environment"""
    return CACHE_DURATION_HOURS

def get_priority_symbols(count=10):
    """Get priority symbols for training"""
    return PRIORITY_TRAINING_SYMBOLS[:count]

# ========== APPLICATION METADATA ==========

APP_NAME = "EQUITURA AI Stock Prediction Platform"
APP_VERSION = "5.0.0 - Railway Production"
APP_DESCRIPTION = "Professional AI-powered stock prediction platform with real-time data and LSTM neural networks"

# College application readiness
COLLEGE_APPLICATION_READY = True
FEATURES_IMPLEMENTED = [
    "Real-time Yahoo Finance data integration",
    "TensorFlow LSTM neural networks", 
    "Professional trading interface",
    "Advanced technical analysis (25+ indicators)",
    "Production-ready Railway deployment",
    "Comprehensive error handling and logging",
    "Mobile-responsive design"
]

# ========== RAILWAY STARTUP INFO ==========

def print_startup_info():
    """Print startup information (Railway-optimized)"""
    print(f"🚀 {APP_NAME} v{APP_VERSION}")
    print(f"🌍 Environment: {ENVIRONMENT}")
    print(f"🔧 Debug Mode: {DEBUG_MODE}")
    print(f"📈 Tracking {len(STOCK_SYMBOLS)} stocks across {len(STOCK_SECTORS)} sectors")
    print(f"🎯 Priority training: {len(PRIORITY_TRAINING_SYMBOLS)} symbols")
    
    if is_production():
        print("🚀 RAILWAY PRODUCTION MODE - Web deployment ready!")
        print(f"🌐 Server: {HOST}:{PORT}")
        print("📊 Data: Yahoo Finance (free, unlimited)")
        print("⚡ Optimized for Railway.com deployment")
    else:
        print("🛠️ DEVELOPMENT MODE")

# Only print startup info if running as main module
if __name__ != "__main__":
    print_startup_info()
