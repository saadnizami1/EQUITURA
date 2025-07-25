# config.py
"""
Vestara AI Stock Prediction Platform
PRODUCTION CONFIGURATION - Web Deployment Ready
"""

import os
from datetime import datetime

# ========== PRODUCTION SETTINGS ==========

# Environment detection
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'development')
DEBUG_MODE = ENVIRONMENT == 'development'

# Server configuration
PORT = int(os.environ.get('PORT', 5000))  # Heroku/Railway compatible
HOST = "0.0.0.0" if ENVIRONMENT == 'production' else "localhost"

# ========== DATA SOURCE CONFIGURATION ==========

# Yahoo Finance Configuration (NO API KEY NEEDED!)
FINNHUB_API_KEY = "yahoo_finance_free"  # Not used anymore
FINNHUB_ENABLED = False  # Using Yahoo Finance now
YAHOO_FINANCE_ENABLED = True  # NEW: Yahoo Finance is our data source

# ========== CACHING CONFIGURATION ==========

# Production-optimized cache settings
if ENVIRONMENT == 'production':
    UPDATE_INTERVAL_HOURS = 2      # 2 hours in production
    CACHE_DURATION_HOURS = 2       # 2 hours cache
    DAILY_API_LIMIT = 999999       # Unlimited with Yahoo Finance
else:
    UPDATE_INTERVAL_HOURS = 1      # 1 hour in development
    CACHE_DURATION_HOURS = 1       # 1 hour cache
    DAILY_API_LIMIT = 999999       # Unlimited

# ========== STOCK SYMBOLS CONFIGURATION ==========

# All 37 stocks organized by sectors
STOCK_SYMBOLS = [
    # Technology (11 stocks)
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "ORCL",
    "AMD", "INTC", 
    
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
    
    # International/ADR (1 stock)
    "BABA"
]

# Sector categorization mapping
STOCK_SECTORS = {
    # Technology
    "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "ORCL", "AMD", "INTC"],
    
    # Financial Services
    "Financial Services": ["JPM", "V", "BAC", "WFC", "AXP", "PYPL"],
    
    # Consumer Discretionary
    "Consumer Discretionary": ["DIS", "NKE", "SBUX", "F", "GM"],
    
    # Consumer Staples
    "Consumer Staples": ["KO", "PG"],
    
    # Healthcare
    "Healthcare": ["JNJ", "PFE", "UNH"],
    
    # Energy
    "Energy": ["XOM", "CLR"],
    
    # Industrials
    "Industrials": ["BA", "LMT", "APD", "X"],
    
    # Communication Services
    "Communication Services": ["T", "VZ"],
    
    # Materials
    "Materials": ["RGLD"],
    
    # International
    "International": ["BABA"]
}

# ========== ML CONFIGURATION ==========

# Prediction settings
PREDICTION_DAYS_AHEAD = [1, 7, 30]
MODEL_RETRAIN_INTERVAL_HOURS = 168  # 7 days in hours

# Auto-retrain configuration
AUTO_RETRAIN_ENABLED = True
AUTO_RETRAIN_SCHEDULE = "weekly"  # weekly, daily, or disabled

# Priority stocks for initial training (most liquid/popular)
PRIORITY_TRAINING_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 
    "NVDA", "META", "NFLX", "JPM", "V"
]

# ========== FILE PATHS ==========

# Production-safe file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "csv")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")

def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [DATA_DIR, MODELS_DIR, CACHE_DIR]
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create directory {directory}: {e}")

# Initialize on import
ensure_directories()

# ========== TECHNICAL INDICATORS ==========

TECHNICAL_INDICATORS = [
    "SMA_20", "SMA_50", "SMA_200",
    "EMA_12", "EMA_26",
    "RSI_14", "MACD",
    "BOLLINGER_UPPER", "BOLLINGER_LOWER",
    "STOCH_K", "STOCH_D",
    "ATR_14", "OBV"
]

# ========== SYSTEM CONFIGURATION ==========

# Shared system configuration
SHARED_SYSTEM_ENABLED = True
GLOBAL_COUNTDOWN_ENABLED = True
BACKGROUND_REFRESH_ENABLED = True

# Logging configuration
LOG_LEVEL = "INFO" if ENVIRONMENT == 'production' else "DEBUG"

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
    return "Other"

def is_production():
    """Check if running in production environment"""
    return ENVIRONMENT == 'production'

def get_cache_duration():
    """Get cache duration based on environment"""
    return CACHE_DURATION_HOURS

def get_priority_symbols(count=10):
    """Get priority symbols for training"""
    return PRIORITY_TRAINING_SYMBOLS[:count]

# ========== DEPLOYMENT INFO ==========

# Application metadata
APP_NAME = "Vestara AI Stock Prediction Platform"
APP_VERSION = "5.0.0 - Production Ready"
APP_DESCRIPTION = "Professional AI-powered stock prediction platform with LSTM neural networks"

# College application info
COLLEGE_APPLICATION_READY = True
FEATURES_IMPLEMENTED = [
    "Real-time Yahoo Finance data integration",
    "TensorFlow LSTM neural networks",
    "Professional NYSE-inspired interface", 
    "Advanced technical analysis (25+ indicators)",
    "Automated model retraining system",
    "Production-ready deployment configuration",
    "Comprehensive error handling and logging"
]

print(f"📊 {APP_NAME} v{APP_VERSION}")
print(f"🌍 Environment: {ENVIRONMENT}")
print(f"🔧 Debug Mode: {DEBUG_MODE}")
print(f"📈 Tracking {len(STOCK_SYMBOLS)} stocks across {len(STOCK_SECTORS)} sectors")
print(f"🎯 Priority training: {len(PRIORITY_TRAINING_SYMBOLS)} symbols")
if is_production():
    print("🚀 PRODUCTION MODE - Web deployment ready!")
else:
    print("🛠️ DEVELOPMENT MODE")