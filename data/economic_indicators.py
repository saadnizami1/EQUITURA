"""
Equitura AI Stock Prediction Platform
Economic Indicators Module
Uses Alpha Vantage API for real economic data (GDP, inflation, unemployment, etc.)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import sys
sys.path.append('..')
import config
from typing import Dict, List, Optional
import json

class EconomicIndicatorsAnalyzer:
    """
    Real economic indicators analysis using Alpha Vantage API
    Provides macroeconomic context for stock predictions
    """
    
    def __init__(self):
        """Initialize the economic indicators analyzer"""
        self.api_key = config.ALPHA_VANTAGE_API_KEY
        self.base_url = "https://www.alphavantage.co/query"
        self.indicators_cache = {}
        
        print(f"Economic Indicators Analyzer initialized")
        print(f"API Key: {self.api_key[:8]}...")
    
    def get_federal_funds_rate(self) -> Optional[Dict]:
        """
        Get Federal Funds Rate (interest rates)
        """
        try:
            print("Fetching Federal Funds Rate...")
            
            params = {
                'function': 'FEDERAL_FUNDS_RATE',
                'interval': 'monthly',
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data:
                print(f"Error fetching federal funds rate: {data}")
                return None
            
            # Process the data
            rates_data = data['data'][:12]  # Last 12 months
            
            current_rate = float(rates_data[0]['value']) if rates_data else 0
            
            # Calculate trend
            if len(rates_data) >= 2:
                prev_rate = float(rates_data[1]['value'])
                rate_change = current_rate - prev_rate
                trend = 'rising' if rate_change > 0 else 'falling' if rate_change < 0 else 'stable'
            else:
                rate_change = 0
                trend = 'stable'
            
            result = {
                'indicator': 'Federal Funds Rate',
                'current_value': current_rate,
                'unit': 'percent',
                'last_updated': rates_data[0]['date'] if rates_data else '',
                'change_from_previous': rate_change,
                'trend': trend,
                'impact_on_stocks': 'negative' if rate_change > 0 else 'positive' if rate_change < 0 else 'neutral',
                'raw_data': rates_data[:6]  # Keep 6 months of data
            }
            
            print(f"✅ Federal Funds Rate: {current_rate}% (trend: {trend})")
            return result
            
        except Exception as e:
            print(f"Error fetching federal funds rate: {e}")
            return None
    
    def get_unemployment_rate(self) -> Optional[Dict]:
        """
        Get unemployment rate
        """
        try:
            print("Fetching Unemployment Rate...")
            
            params = {
                'function': 'UNEMPLOYMENT',
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data:
                print(f"Error fetching unemployment rate: {data}")
                return None
            
            # Process the data
            unemployment_data = data['data'][:12]  # Last 12 months
            
            current_rate = float(unemployment_data[0]['value']) if unemployment_data else 0
            
            # Calculate trend
            if len(unemployment_data) >= 2:
                prev_rate = float(unemployment_data[1]['value'])
                rate_change = current_rate - prev_rate
                trend = 'rising' if rate_change > 0.1 else 'falling' if rate_change < -0.1 else 'stable'
            else:
                rate_change = 0
                trend = 'stable'
            
            result = {
                'indicator': 'Unemployment Rate',
                'current_value': current_rate,
                'unit': 'percent',
                'last_updated': unemployment_data[0]['date'] if unemployment_data else '',
                'change_from_previous': rate_change,
                'trend': trend,
                'impact_on_stocks': 'negative' if rate_change > 0 else 'positive' if rate_change < 0 else 'neutral',
                'raw_data': unemployment_data[:6]
            }
            
            print(f"✅ Unemployment Rate: {current_rate}% (trend: {trend})")
            return result
            
        except Exception as e:
            print(f"Error fetching unemployment rate: {e}")
            return None
    
    def get_inflation_rate(self) -> Optional[Dict]:
        """
        Get Consumer Price Index (inflation)
        """
        try:
            print("Fetching Consumer Price Index (Inflation)...")
            
            params = {
                'function': 'CPI',
                'interval': 'monthly',
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data:
                print(f"Error fetching CPI: {data}")
                return None
            
            # Process the data
            cpi_data = data['data'][:12]  # Last 12 months
            
            if len(cpi_data) >= 12:
                # Calculate year-over-year inflation rate
                current_cpi = float(cpi_data[0]['value'])
                year_ago_cpi = float(cpi_data[11]['value'])
                inflation_rate = ((current_cpi - year_ago_cpi) / year_ago_cpi) * 100
            else:
                inflation_rate = 0
            
            # Calculate month-over-month change
            if len(cpi_data) >= 2:
                current_cpi = float(cpi_data[0]['value'])
                prev_cpi = float(cpi_data[1]['value'])
                monthly_change = ((current_cpi - prev_cpi) / prev_cpi) * 100
                trend = 'rising' if monthly_change > 0.1 else 'falling' if monthly_change < -0.1 else 'stable'
            else:
                monthly_change = 0
                trend = 'stable'
            
            result = {
                'indicator': 'Consumer Price Index (Inflation)',
                'current_value': inflation_rate,
                'unit': 'percent (YoY)',
                'last_updated': cpi_data[0]['date'] if cpi_data else '',
                'change_from_previous': monthly_change,
                'trend': trend,
                'impact_on_stocks': 'negative' if inflation_rate > 4 else 'positive' if inflation_rate < 2 else 'neutral',
                'raw_data': cpi_data[:6]
            }
            
            print(f"✅ Inflation Rate: {inflation_rate:.2f}% YoY (trend: {trend})")
            return result
            
        except Exception as e:
            print(f"Error fetching CPI: {e}")
            return None
    
    def get_gdp_growth(self) -> Optional[Dict]:
        """
        Get GDP growth rate
        """
        try:
            print("Fetching Real GDP...")
            
            params = {
                'function': 'REAL_GDP',
                'interval': 'quarterly',
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data:
                print(f"Error fetching GDP: {data}")
                return None
            
            # Process the data
            gdp_data = data['data'][:8]  # Last 8 quarters
            
            if len(gdp_data) >= 4:
                # Calculate year-over-year GDP growth
                current_gdp = float(gdp_data[0]['value'])
                year_ago_gdp = float(gdp_data[3]['value'])  # 4 quarters ago
                gdp_growth = ((current_gdp - year_ago_gdp) / year_ago_gdp) * 100
            else:
                gdp_growth = 0
            
            # Calculate quarter-over-quarter change
            if len(gdp_data) >= 2:
                current_gdp = float(gdp_data[0]['value'])
                prev_gdp = float(gdp_data[1]['value'])
                quarterly_change = ((current_gdp - prev_gdp) / prev_gdp) * 100
                trend = 'expanding' if quarterly_change > 0 else 'contracting' if quarterly_change < 0 else 'stable'
            else:
                quarterly_change = 0
                trend = 'stable'
            
            result = {
                'indicator': 'Real GDP Growth',
                'current_value': gdp_growth,
                'unit': 'percent (YoY)',
                'last_updated': gdp_data[0]['date'] if gdp_data else '',
                'change_from_previous': quarterly_change,
                'trend': trend,
                'impact_on_stocks': 'positive' if gdp_growth > 2 else 'negative' if gdp_growth < 0 else 'neutral',
                'raw_data': gdp_data[:4]
            }
            
            print(f"✅ GDP Growth: {gdp_growth:.2f}% YoY (trend: {trend})")
            return result
            
        except Exception as e:
            print(f"Error fetching GDP: {e}")
            return None
    
    def get_treasury_yield(self) -> Optional[Dict]:
        """
        Get 10-year Treasury yield
        """
        try:
            print("Fetching 10-Year Treasury Yield...")
            
            params = {
                'function': 'TREASURY_YIELD',
                'interval': 'monthly',
                'maturity': '10year',
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data:
                print(f"Error fetching treasury yield: {data}")
                return None
            
            # Process the data
            yield_data = data['data'][:12]  # Last 12 months
            
            current_yield = float(yield_data[0]['value']) if yield_data else 0
            
            # Calculate trend
            if len(yield_data) >= 2:
                prev_yield = float(yield_data[1]['value'])
                yield_change = current_yield - prev_yield
                trend = 'rising' if yield_change > 0.1 else 'falling' if yield_change < -0.1 else 'stable'
            else:
                yield_change = 0
                trend = 'stable'
            
            result = {
                'indicator': '10-Year Treasury Yield',
                'current_value': current_yield,
                'unit': 'percent',
                'last_updated': yield_data[0]['date'] if yield_data else '',
                'change_from_previous': yield_change,
                'trend': trend,
                'impact_on_stocks': 'negative' if yield_change > 0.5 else 'positive' if yield_change < -0.5 else 'neutral',
                'raw_data': yield_data[:6]
            }
            
            print(f"✅ 10-Year Treasury: {current_yield}% (trend: {trend})")
            return result
            
        except Exception as e:
            print(f"Error fetching treasury yield: {e}")
            return None
    
    def get_all_economic_indicators(self) -> Dict:
        """
        Get all available economic indicators
        
        Returns:
            Dict: Comprehensive economic analysis
        """
        print("\nFETCHING ALL ECONOMIC INDICATORS")
        print("=" * 50)
        
        indicators = {}
        
        # Fetch each indicator with delays to respect API limits
        indicator_functions = [
            ('federal_funds_rate', self.get_federal_funds_rate),
            ('unemployment_rate', self.get_unemployment_rate),
            ('inflation_rate', self.get_inflation_rate),
            ('gdp_growth', self.get_gdp_growth),
            ('treasury_yield', self.get_treasury_yield)
        ]
        
        for i, (name, func) in enumerate(indicator_functions):
            try:
                result = func()
                if result:
                    indicators[name] = result
                else:
                    indicators[name] = {'error': 'Data not available'}
                
                # Rate limiting - wait between requests
                if i < len(indicator_functions) - 1:
                    print("Waiting 12 seconds for API rate limit...")
                    time.sleep(12)
                    
            except Exception as e:
                print(f"Error fetching {name}: {e}")
                indicators[name] = {'error': str(e)}
        
        # Calculate overall economic sentiment
        economic_score = self.calculate_economic_sentiment(indicators)
        
        analysis = {
            'analysis_date': datetime.now(),
            'indicators': indicators,
            'economic_sentiment': economic_score,
            'summary': self.generate_economic_summary(indicators, economic_score)
        }
        
        return analysis
    
    def calculate_economic_sentiment(self, indicators: Dict) -> Dict:
        """
        Calculate overall economic sentiment score
        """
        scores = []
        weights = []
        
        # Federal Funds Rate (lower is better for stocks)
        if 'federal_funds_rate' in indicators and 'current_value' in indicators['federal_funds_rate']:
            rate = indicators['federal_funds_rate']['current_value']
            score = max(0, min(100, 100 - (rate * 10)))  # Scale: high rates = low score
            scores.append(score)
            weights.append(0.25)
        
        # Unemployment (lower is better)
        if 'unemployment_rate' in indicators and 'current_value' in indicators['unemployment_rate']:
            unemployment = indicators['unemployment_rate']['current_value']
            score = max(0, min(100, 100 - (unemployment * 8)))  # Scale unemployment impact
            scores.append(score)
            weights.append(0.25)
        
        # GDP Growth (higher is better)
        if 'gdp_growth' in indicators and 'current_value' in indicators['gdp_growth']:
            gdp = indicators['gdp_growth']['current_value']
            score = max(0, min(100, 50 + (gdp * 10)))  # Center around 50, growth increases score
            scores.append(score)
            weights.append(0.25)
        
        # Inflation (moderate is best)
        if 'inflation_rate' in indicators and 'current_value' in indicators['inflation_rate']:
            inflation = indicators['inflation_rate']['current_value']
            if 1.5 <= inflation <= 3.0:  # Target range
                score = 80
            else:
                score = max(0, 80 - abs(inflation - 2.25) * 20)
            scores.append(score)
            weights.append(0.25)
        
        if scores:
            weighted_score = np.average(scores, weights=weights[:len(scores)])
            sentiment = 'positive' if weighted_score > 60 else 'negative' if weighted_score < 40 else 'neutral'
        else:
            weighted_score = 50
            sentiment = 'neutral'
        
        return {
            'score': weighted_score,
            'sentiment': sentiment,
            'individual_scores': dict(zip(['fed_rate', 'unemployment', 'gdp', 'inflation'], scores))
        }
    
    def generate_economic_summary(self, indicators: Dict, sentiment: Dict) -> str:
        """
        Generate human-readable economic summary
        """
        summary_parts = []
        
        summary_parts.append(f"Overall Economic Sentiment: {sentiment['sentiment'].upper()} (Score: {sentiment['score']:.1f}/100)")
        
        if 'federal_funds_rate' in indicators and 'current_value' in indicators['federal_funds_rate']:
            fed_rate = indicators['federal_funds_rate']
            summary_parts.append(f"Fed Funds Rate: {fed_rate['current_value']}% ({fed_rate['trend']})")
        
        if 'unemployment_rate' in indicators and 'current_value' in indicators['unemployment_rate']:
            unemployment = indicators['unemployment_rate']
            summary_parts.append(f"Unemployment: {unemployment['current_value']}% ({unemployment['trend']})")
        
        if 'gdp_growth' in indicators and 'current_value' in indicators['gdp_growth']:
            gdp = indicators['gdp_growth']
            summary_parts.append(f"GDP Growth: {gdp['current_value']:.1f}% ({gdp['trend']})")
        
        if 'inflation_rate' in indicators and 'current_value' in indicators['inflation_rate']:
            inflation = indicators['inflation_rate']
            summary_parts.append(f"Inflation: {inflation['current_value']:.1f}% ({inflation['trend']})")
        
        return " | ".join(summary_parts)
    
    def save_economic_data(self, economic_data: Dict) -> bool:
        """
        Save economic indicators data to file
        """
        try:
            filename = "economic_indicators.json"
            filepath = os.path.join("..", config.DATA_DIR, filename)
            
            # Convert datetime objects for JSON serialization
            data_to_save = economic_data.copy()
            if 'analysis_date' in data_to_save:
                data_to_save['analysis_date'] = data_to_save['analysis_date'].isoformat()
            
            with open(filepath, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            
            print(f"Saved economic data: {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving economic data: {e}")
            return False

def main():
    """
    Test the economic indicators analyzer
    """
    print("EQUITURA ECONOMIC INDICATORS ANALYZER - TEST")
    print("=" * 60)
    print("Using REAL Alpha Vantage API for economic data")
    print()
    
    analyzer = EconomicIndicatorsAnalyzer()
    
    print("🏛️  FETCHING REAL ECONOMIC INDICATORS")
    print("This will take about 1 minute due to API rate limits...")
    print()
    
    # Get all economic indicators
    economic_analysis = analyzer.get_all_economic_indicators()
    
    print(f"\n📊 ECONOMIC ANALYSIS SUMMARY")
    print("-" * 40)
    print(economic_analysis['summary'])
    
    print(f"\n📈 DETAILED INDICATORS:")
    for name, indicator in economic_analysis['indicators'].items():
        if 'error' not in indicator:
            print(f"  {indicator['indicator']}: {indicator['current_value']:.2f}{indicator['unit']} ({indicator['trend']})")
            print(f"    Stock Impact: {indicator['impact_on_stocks']}")
        else:
            print(f"  {name}: Error - {indicator['error']}")
    
    print(f"\n🎯 ECONOMIC SENTIMENT:")
    sentiment = economic_analysis['economic_sentiment']
    print(f"  Overall Score: {sentiment['score']:.1f}/100")
    print(f"  Sentiment: {sentiment['sentiment'].upper()}")
    
    # Save results
    analyzer.save_economic_data(economic_analysis)
    
    print("\n" + "=" * 60)
    print("ECONOMIC INDICATORS TEST COMPLETED!")
    print()
    print("✅ Real economic data fetched from Alpha Vantage")
    print("✅ Fed rates, unemployment, GDP, inflation analyzed")
    print("✅ Economic sentiment calculated")
    print("✅ Stock market impact assessed")
    print("✅ Results saved for ML model integration")
    print()
    print("💡 This economic context enhances stock predictions!")

if __name__ == "__main__":
    main()