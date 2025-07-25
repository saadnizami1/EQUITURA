"""
Perfect Equitura Setup Script
One-click setup for the complete platform
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    
    requirements = [
        "yfinance==0.2.18",
        "pandas==2.0.3", 
        "numpy==1.24.3",
        "scikit-learn==1.3.0",
        "flask==2.3.2",
        "flask-cors==4.0.0",
        "plotly==5.15.0",
        "ta==0.11.0",
        "joblib==1.3.1",
        "matplotlib==3.7.1"
    ]
    
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✅ Installed {req}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {req}: {e}")

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "data",
        "models", 
        "saved_models",
        "web_app/templates",
        "web_app/static",
        "data/cache"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created {directory}")

def create_config_file():
    """Create config.py file"""
    print("⚙️ Creating config.py...")
    
    config_content = '''"""
Equitura AI Stock Prediction Platform
Enhanced Configuration
"""

import os
from datetime import datetime

# API Keys (Yahoo Finance is free, no keys needed)
YAHOO_FINANCE_ENABLED = True

# Application settings
DEBUG_MODE = True
PORT = 5000
HOST = "localhost"

# Data collection settings
UPDATE_INTERVAL_SECONDS = 60  # Update frequency
CACHE_DURATION_HOURS = 4  # Cache data for 4 hours

STOCK_SYMBOLS = [
    # Tech Giants
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    
    # AI/Semiconductor  
    "NVDA", "AMD", "INTC", "TSM", "QCOM",
    
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS",
    
    # Consumer
    "TSLA", "NFLX", "DIS", "NKE", "SBUX",
    
    # Healthcare
    "JNJ", "PFE", "UNH", "ABBV", "MRK"
]

# Model settings
PREDICTION_DAYS_AHEAD = [1, 7, 30]  # 1-day, 7-day, 30-day predictions
MODEL_RETRAIN_INTERVAL_HOURS = 24
TECHNICAL_INDICATORS_LOOKBACK = 50

# Machine Learning parameters
RANDOM_FOREST_ESTIMATORS = 100
TRAIN_TEST_SPLIT_RATIO = 0.8
CROSS_VALIDATION_FOLDS = 5

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "csv")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")

def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [DATA_DIR, MODELS_DIR, CACHE_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Initialize on import
ensure_directories()
'''
    
    with open('config.py', 'w') as f:
        f.write(config_content)
    print("✅ Created config.py")

def main():
    print("🎯 PERFECT EQUITURA SETUP")
    print("=" * 40)
    
    # Install requirements
    install_requirements()
    
    # Create directories
    create_directories()
    
    # Create config
    create_config_file()
    
    print("\n🚀 Setup Step 1 Complete!")
    print("=" * 40)
    print("📋 Next steps:")
    print("1. Create your app.py file")
    print("2. Create the enhanced components")
    print("3. Create the web interface")
    print("4. Run: python app.py")

if __name__ == "__main__":
    main()