"""
Equitura AI Stock Prediction Platform
Real-time Stock Data Collection Module
Fetches live stock data from Yahoo Finance API
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sys
sys.path.append('..')
import config

# Configure logging with UTF-8 encoding for Windows compatibility
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.get_log_path('stock_data_collector.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockDataCollector:
    """
    Advanced stock data collection system with real-time updates
    """
    
    def __init__(self):
        """Initialize the stock data collector"""
        self.data_cache = {}
        self.last_update = {}
        self.failed_symbols = set()
        
        # Ensure data directory exists
        os.makedirs(config.DATA_DIR, exist_ok=True)
        
        logger.info("Equitura Stock Data Collector initialized")
        logger.info(f"Tracking {len(config.STOCK_SYMBOLS)} symbols")
    
    def fetch_historical_data(self, symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data for a symbol
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            period (str): Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        
        Returns:
            pd.DataFrame: Historical stock data or None if failed
        """
        try:
            logger.info(f"Fetching historical data for {symbol} (period: {period})")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    hist_data = ticker.history(period=period, interval="1d")
                    if not hist_data.empty:
                        break
                    logger.warning(f"Attempt {attempt + 1}: No data returned for {symbol}")
                    time.sleep(2)  # Wait before retry
                except Exception as retry_error:
                    logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(retry_error)}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        raise retry_error
            
            if hist_data.empty:
                logger.warning(f"No historical data found for {symbol} after {max_retries} attempts")
                # Try with a different period as fallback
                logger.info(f"Trying alternative period '6mo' for {symbol}")
                hist_data = ticker.history(period="6mo", interval="1d")
                
                if hist_data.empty:
                    return None
            
            # Clean and prepare data
            hist_data = self._clean_data(hist_data)
            hist_data['Symbol'] = symbol
            hist_data['Date'] = hist_data.index
            hist_data.reset_index(drop=True, inplace=True)
            
            # Cache the data
            self.data_cache[symbol] = hist_data
            self.last_update[symbol] = datetime.now()
            
            logger.info(f"Successfully fetched {len(hist_data)} days of data for {symbol}")
            return hist_data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            self.failed_symbols.add(symbol)
            return None
    
    def fetch_current_price(self, symbol: str) -> Optional[Dict]:
        """
        Fetch current real-time price and basic info for a symbol
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Current price information or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current price info
            info = ticker.info
            hist = ticker.history(period="1d", interval="1m")
            
            if hist.empty:
                logger.warning(f"No current data available for {symbol}")
                return None
            
            current_price = hist['Close'].iloc[-1]
            previous_close = info.get('previousClose', current_price)
            
            price_info = {
                'symbol': symbol,
                'current_price': current_price,
                'previous_close': previous_close,
                'change': current_price - previous_close,
                'change_percent': ((current_price - previous_close) / previous_close) * 100,
                'volume': hist['Volume'].iloc[-1],
                'timestamp': datetime.now(),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'day_high': info.get('dayHigh', current_price),
                'day_low': info.get('dayLow', current_price),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', current_price),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', current_price)
            }
            
            logger.info(f"{symbol}: ${current_price:.2f} ({price_info['change_percent']:+.2f}%)")
            return price_info
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {str(e)}")
            return None
    
    def fetch_company_info(self, symbol: str) -> Optional[Dict]:
        """
        Fetch detailed company information
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Company information or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            company_info = {
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'country': info.get('country', 'Unknown'),
                'website': info.get('website', ''),
                'business_summary': info.get('longBusinessSummary', ''),
                'full_time_employees': info.get('fullTimeEmployees', 0),
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'peg_ratio': info.get('pegRatio', None),
                'price_to_book': info.get('priceToBook', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'roe': info.get('returnOnEquity', None),
                'roa': info.get('returnOnAssets', None),
                'dividend_yield': info.get('dividendYield', None),
                'beta': info.get('beta', None)
            }
            
            logger.info(f"Company info fetched for {symbol}: {company_info['company_name']}")
            return company_info
            
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {str(e)}")
            return None
    
    def save_data_to_csv(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Save stock data to CSV file
        
        Args:
            symbol (str): Stock symbol
            data (pd.DataFrame): Stock data to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            filename = f"{symbol}_historical_data.csv"
            filepath = config.get_data_path(filename)
            
            data.to_csv(filepath, index=False)
            logger.info(f"Saved {len(data)} records to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {str(e)}")
            return False
    
    def load_data_from_csv(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load stock data from CSV file
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            pd.DataFrame: Loaded stock data or None if failed
        """
        try:
            filename = f"{symbol}_historical_data.csv"
            filepath = config.get_data_path(filename)
            
            if not os.path.exists(filepath):
                logger.warning(f"No saved data found for {symbol}")
                return None
            
            data = pd.read_csv(filepath)
            data['Date'] = pd.to_datetime(data['Date'])
            
            logger.info(f"Loaded {len(data)} records for {symbol} from CSV")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
            return None
    
    def update_all_stocks(self) -> Dict[str, bool]:
        """
        Update data for all configured stock symbols
        
        Returns:
            dict: Success status for each symbol
        """
        logger.info("Starting bulk update for all stocks...")
        results = {}
        
        for symbol in config.STOCK_SYMBOLS:
            try:
                # Fetch historical data
                data = self.fetch_historical_data(symbol)
                
                if data is not None:
                    # Save to CSV
                    saved = self.save_data_to_csv(symbol, data)
                    results[symbol] = saved
                    
                    # Delay to respect rate limits
                    time.sleep(config.YAHOO_FINANCE_DELAY)
                else:
                    results[symbol] = False
                    
            except Exception as e:
                logger.error(f"Error updating {symbol}: {str(e)}")
                results[symbol] = False
        
        successful = sum(results.values())
        total = len(results)
        logger.info(f"Update complete: {successful}/{total} stocks updated successfully")
        
        return results
    
    def get_market_summary(self) -> Dict:
        """
        Get a summary of current market conditions
        
        Returns:
            dict: Market summary information
        """
        try:
            logger.info("Generating market summary...")
            
            current_prices = {}
            total_change = 0
            positive_stocks = 0
            
            for symbol in config.STOCK_SYMBOLS[:5]:  # Sample first 5 for speed
                price_info = self.fetch_current_price(symbol)
                if price_info:
                    current_prices[symbol] = price_info
                    total_change += price_info['change_percent']
                    if price_info['change_percent'] > 0:
                        positive_stocks += 1
                
                time.sleep(0.5)  # Small delay
            
            market_summary = {
                'timestamp': datetime.now(),
                'total_stocks_tracked': len(config.STOCK_SYMBOLS),
                'stocks_sampled': len(current_prices),
                'average_change_percent': total_change / len(current_prices) if current_prices else 0,
                'positive_stocks': positive_stocks,
                'negative_stocks': len(current_prices) - positive_stocks,
                'market_sentiment': 'Bullish' if total_change > 0 else 'Bearish',
                'current_prices': current_prices
            }
            
            logger.info(f"Market Summary: {market_summary['market_sentiment']} "
                       f"({market_summary['average_change_percent']:+.2f}% avg)")
            
            return market_summary
            
        except Exception as e:
            logger.error(f"Error generating market summary: {str(e)}")
            return {}
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate stock data
        
        Args:
            data (pd.DataFrame): Raw stock data
            
        Returns:
            pd.DataFrame: Cleaned stock data
        """
        # Remove any rows with NaN values in critical columns
        data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Ensure positive values
        data = data[data['Close'] > 0]
        data = data[data['Volume'] >= 0]
        
        # Remove outliers (prices that change more than 50% in one day)
        data['price_change'] = data['Close'].pct_change()
        data = data[abs(data['price_change']) <= 0.5]
        data.drop('price_change', axis=1, inplace=True)
        
        return data
    
    def get_data_freshness(self, symbol: str) -> Optional[Dict]:
        """
        Check how fresh the cached data is for a symbol
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Data freshness information
        """
        if symbol not in self.last_update:
            return None
        
        last_update = self.last_update[symbol]
        age_minutes = (datetime.now() - last_update).total_seconds() / 60
        
        return {
            'symbol': symbol,
            'last_update': last_update,
            'age_minutes': age_minutes,
            'is_fresh': age_minutes < 30,  # Consider fresh if updated within 30 minutes
            'needs_update': age_minutes > 60  # Needs update if older than 1 hour
        }

def main():
    """
    Main function for testing the stock data collector
    """
    print("Equitura Stock Data Collector - Testing Mode")
    print("=" * 60)
    
    # Initialize collector
    collector = StockDataCollector()
    
    # Test with a single stock first
    test_symbol = "AAPL"
    print(f"Testing with {test_symbol}...")
    
    # Fetch historical data
    print(f"Fetching historical data for {test_symbol}...")
    hist_data = collector.fetch_historical_data(test_symbol, period="6mo")  # Try 6 months first
    
    if hist_data is not None:
        print(f"SUCCESS: Historical data: {len(hist_data)} records")
        print(f"Date range: {hist_data['Date'].min()} to {hist_data['Date'].max()}")
        print(f"Price range: ${hist_data['Close'].min():.2f} - ${hist_data['Close'].max():.2f}")
        
        # Save to CSV
        collector.save_data_to_csv(test_symbol, hist_data)
    else:
        print("FAILED: Could not fetch historical data")
        print("This might be due to network issues or Yahoo Finance rate limiting")
        print("Trying a simpler test...")
        
        # Try just getting current price instead
        current_price = collector.fetch_current_price(test_symbol)
        if current_price:
            print(f"SUCCESS: Got current price data for {test_symbol}")
        else:
            print("FAILED: Cannot connect to Yahoo Finance")
            print("Check your internet connection and try again")
            return
    
    # Test current price
    print(f"\nFetching current price for {test_symbol}...")
    current_price = collector.fetch_current_price(test_symbol)
    
    if current_price:
        print(f"SUCCESS: Current price: ${current_price['current_price']:.2f}")
        print(f"Change: {current_price['change_percent']:+.2f}%")
        print(f"Volume: {current_price['volume']:,}")
    else:
        print("FAILED: Could not fetch current price")
    
    # Test company info
    print(f"\nFetching company info for {test_symbol}...")
    company_info = collector.fetch_company_info(test_symbol)
    
    if company_info:
        print(f"SUCCESS: Company: {company_info['company_name']}")
        print(f"Sector: {company_info['sector']}")
        print(f"Industry: {company_info['industry']}")
        print(f"Market Cap: ${company_info['market_cap']:,}")
    else:
        print("FAILED: Could not fetch company info")
    
    print("\n" + "=" * 60)
    print("Stock Data Collector test completed!")
    print("Files saved in: data/csv/")
    print("Next: Run 'python technical_indicators.py' to test technical analysis")

if __name__ == "__main__":
    main()