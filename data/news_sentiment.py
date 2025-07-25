"""
Equitura AI Stock Prediction Platform
News Sentiment Analysis Module
Uses Finnhub API for real company news and sentiment analysis
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from textblob import TextBlob
import time
import os
import sys
sys.path.append('..')
import config
from typing import Dict, List, Optional
import json

class NewsSentimentAnalyzer:
    """
    Real-time news sentiment analysis using Finnhub API
    Provides sentiment scores that can enhance stock predictions
    """
    
    def __init__(self):
        """Initialize the news sentiment analyzer"""
        self.api_key = config.FINNHUB_API_KEY
        self.base_url = "https://finnhub.io/api/v1"
        self.session = requests.Session()
        self.session.headers.update({'X-Finnhub-Token': self.api_key})
        
        print(f"News Sentiment Analyzer initialized")
        print(f"API Key: {self.api_key[:8]}...")
    
    def get_company_news(self, symbol: str, days_back: int = 7) -> List[Dict]:
        """
        Get real company news from Finnhub
        
        Args:
            symbol (str): Stock symbol
            days_back (int): Days back to fetch news
        
        Returns:
            List[Dict]: List of news articles
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Format dates for API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            print(f"Fetching news for {symbol} from {start_str} to {end_str}")
            
            # API endpoint
            url = f"{self.base_url}/company-news"
            params = {
                'symbol': symbol,
                'from': start_str,
                'to': end_str
            }
            
            # Make request
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            news_data = response.json()
            
            if not news_data:
                print(f"No news found for {symbol}")
                return []
            
            # Process news articles
            articles = []
            for article in news_data[:20]:  # Limit to 20 most recent
                try:
                    articles.append({
                        'datetime': datetime.fromtimestamp(article.get('datetime', 0)),
                        'headline': article.get('headline', ''),
                        'summary': article.get('summary', ''),
                        'source': article.get('source', ''),
                        'url': article.get('url', ''),
                        'category': article.get('category', ''),
                        'image': article.get('image', ''),
                        'related': article.get('related', symbol)
                    })
                except Exception as e:
                    continue
            
            print(f"Successfully fetched {len(articles)} news articles for {symbol}")
            return articles
            
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching news: {e}")
            return []
        except Exception as e:
            print(f"Error processing news data: {e}")
            return []
    
    def analyze_text_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of text using TextBlob
        
        Args:
            text (str): Text to analyze
        
        Returns:
            Dict: Sentiment analysis results
        """
        try:
            if not text or len(text.strip()) < 10:
                return {
                    'polarity': 0.0,
                    'subjectivity': 0.0,
                    'sentiment': 'neutral',
                    'confidence': 0.0
                }
            
            # Use TextBlob for sentiment analysis
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 (negative) to 1 (positive)
            subjectivity = blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
            
            # Classify sentiment
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            # Calculate confidence based on absolute polarity
            confidence = abs(polarity)
            
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment': sentiment,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'sentiment': 'neutral',
                'confidence': 0.0
            }
    
    def get_sentiment_for_symbol(self, symbol: str, days_back: int = 7) -> Dict:
        """
        Get comprehensive sentiment analysis for a stock symbol
        
        Args:
            symbol (str): Stock symbol
            days_back (int): Days back to analyze
        
        Returns:
            Dict: Comprehensive sentiment analysis
        """
        try:
            print(f"\nAnalyzing sentiment for {symbol}")
            print("-" * 40)
            
            # Get news articles
            articles = self.get_company_news(symbol, days_back)
            
            if not articles:
                return {
                    'symbol': symbol,
                    'total_articles': 0,
                    'average_sentiment': 0.0,
                    'sentiment_trend': 'neutral',
                    'confidence': 0.0,
                    'positive_articles': 0,
                    'negative_articles': 0,
                    'neutral_articles': 0,
                    'recent_headlines': [],
                    'sentiment_score': 0.0
                }
            
            # Analyze sentiment for each article
            article_sentiments = []
            recent_headlines = []
            
            for article in articles:
                # Combine headline and summary for analysis
                text = f"{article['headline']} {article['summary']}"
                
                sentiment = self.analyze_text_sentiment(text)
                sentiment['datetime'] = article['datetime']
                sentiment['headline'] = article['headline']
                sentiment['source'] = article['source']
                
                article_sentiments.append(sentiment)
                
                # Keep recent headlines for display
                if len(recent_headlines) < 5:
                    recent_headlines.append({
                        'headline': article['headline'],
                        'sentiment': sentiment['sentiment'],
                        'polarity': sentiment['polarity'],
                        'datetime': article['datetime'],
                        'source': article['source']
                    })
            
            # Calculate aggregate metrics
            polarities = [s['polarity'] for s in article_sentiments]
            confidences = [s['confidence'] for s in article_sentiments]
            
            avg_sentiment = np.mean(polarities) if polarities else 0.0
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Count sentiment categories
            positive_count = sum(1 for s in article_sentiments if s['sentiment'] == 'positive')
            negative_count = sum(1 for s in article_sentiments if s['sentiment'] == 'negative')
            neutral_count = len(article_sentiments) - positive_count - negative_count
            
            # Determine overall trend
            if avg_sentiment > 0.1:
                trend = 'bullish'
            elif avg_sentiment < -0.1:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            # Calculate weighted sentiment score (recent news weighted more)
            if len(article_sentiments) > 0:
                weights = np.exp(np.linspace(-1, 0, len(article_sentiments)))  # Recent articles get higher weight
                weighted_sentiment = np.average(polarities, weights=weights)
            else:
                weighted_sentiment = 0.0
            
            results = {
                'symbol': symbol,
                'analysis_date': datetime.now(),
                'days_analyzed': days_back,
                'total_articles': len(articles),
                'average_sentiment': avg_sentiment,
                'weighted_sentiment': weighted_sentiment,
                'sentiment_trend': trend,
                'confidence': avg_confidence,
                'positive_articles': positive_count,
                'negative_articles': negative_count,
                'neutral_articles': neutral_count,
                'recent_headlines': recent_headlines,
                'sentiment_score': weighted_sentiment * 100,  # Scale to 0-100
                'sentiment_distribution': {
                    'positive': positive_count / len(articles) * 100,
                    'negative': negative_count / len(articles) * 100,
                    'neutral': neutral_count / len(articles) * 100
                }
            }
            
            print(f"✅ Sentiment Analysis Complete:")
            print(f"   Articles analyzed: {len(articles)}")
            print(f"   Average sentiment: {avg_sentiment:.3f}")
            print(f"   Trend: {trend}")
            print(f"   Distribution: {positive_count}+ / {negative_count}- / {neutral_count}neutral")
            
            return results
            
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'total_articles': 0,
                'sentiment_score': 0.0
            }
    
    def save_sentiment_data(self, symbol: str, sentiment_data: Dict) -> bool:
        """
        Save sentiment analysis results to file
        
        Args:
            symbol (str): Stock symbol
            sentiment_data (Dict): Sentiment analysis results
        
        Returns:
            bool: Success status
        """
        try:
            filename = f"{symbol}_sentiment_analysis.json"
            filepath = os.path.join("..", config.DATA_DIR, filename)
            
            # Convert datetime objects to strings for JSON serialization
            data_to_save = sentiment_data.copy()
            if 'analysis_date' in data_to_save:
                data_to_save['analysis_date'] = data_to_save['analysis_date'].isoformat()
            
            for headline in data_to_save.get('recent_headlines', []):
                if 'datetime' in headline:
                    headline['datetime'] = headline['datetime'].isoformat()
            
            with open(filepath, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            
            print(f"Saved sentiment data: {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving sentiment data: {e}")
            return False
    
    def get_market_sentiment_overview(self, symbols: List[str] = None) -> Dict:
        """
        Get overall market sentiment from multiple stocks
        
        Args:
            symbols (List[str]): List of symbols to analyze
        
        Returns:
            Dict: Market sentiment overview
        """
        if symbols is None:
            symbols = config.STOCK_SYMBOLS[:5]  # Analyze first 5 to respect API limits
        
        print(f"\nGenerating market sentiment overview for {len(symbols)} stocks")
        print("=" * 60)
        
        sentiment_results = {}
        all_sentiments = []
        
        for i, symbol in enumerate(symbols):
            print(f"Analyzing {i+1}/{len(symbols)}: {symbol}")
            
            sentiment = self.get_sentiment_for_symbol(symbol, days_back=3)  # Shorter period for market overview
            sentiment_results[symbol] = sentiment
            
            if sentiment.get('sentiment_score') is not None:
                all_sentiments.append(sentiment['sentiment_score'])
            
            # Rate limiting - wait between requests
            if i < len(symbols) - 1:
                time.sleep(2)  # 2 seconds between requests
        
        # Calculate market-wide metrics
        if all_sentiments:
            market_avg = np.mean(all_sentiments)
            bullish_stocks = sum(1 for s in all_sentiments if s > 10)
            bearish_stocks = sum(1 for s in all_sentiments if s < -10)
            neutral_stocks = len(all_sentiments) - bullish_stocks - bearish_stocks
            
            market_trend = 'bullish' if market_avg > 10 else 'bearish' if market_avg < -10 else 'neutral'
        else:
            market_avg = 0
            bullish_stocks = bearish_stocks = neutral_stocks = 0
            market_trend = 'neutral'
        
        overview = {
            'analysis_date': datetime.now(),
            'symbols_analyzed': symbols,
            'market_sentiment_score': market_avg,
            'market_trend': market_trend,
            'bullish_stocks': bullish_stocks,
            'bearish_stocks': bearish_stocks,
            'neutral_stocks': neutral_stocks,
            'individual_results': sentiment_results
        }
        
        print(f"\n📊 MARKET SENTIMENT OVERVIEW")
        print(f"Market Score: {market_avg:.1f}")
        print(f"Market Trend: {market_trend.upper()}")
        print(f"Stock Distribution: {bullish_stocks} bullish, {bearish_stocks} bearish, {neutral_stocks} neutral")
        
        return overview

def main():
    """
    Test the news sentiment analyzer
    """
    print("EQUITURA NEWS SENTIMENT ANALYZER - TEST")
    print("=" * 60)
    print("Using REAL Finnhub API for news sentiment analysis")
    print()
    
    analyzer = NewsSentimentAnalyzer()
    
    # Test with Apple
    test_symbol = "AAPL"
    
    print(f"1. ANALYZING NEWS SENTIMENT FOR {test_symbol}")
    print("-" * 40)
    
    sentiment_results = analyzer.get_sentiment_for_symbol(test_symbol, days_back=7)
    
    if 'error' not in sentiment_results:
        print(f"\n📊 SENTIMENT SUMMARY FOR {test_symbol}:")
        print(f"Total Articles: {sentiment_results['total_articles']}")
        print(f"Sentiment Score: {sentiment_results['sentiment_score']:.1f}")
        print(f"Trend: {sentiment_results['sentiment_trend'].upper()}")
        print(f"Confidence: {sentiment_results['confidence']:.2f}")
        
        if sentiment_results['recent_headlines']:
            print(f"\n📰 RECENT HEADLINES:")
            for headline in sentiment_results['recent_headlines']:
                emoji = "📈" if headline['sentiment'] == 'positive' else "📉" if headline['sentiment'] == 'negative' else "➡️"
                print(f"  {emoji} {headline['headline'][:80]}...")
                print(f"     Sentiment: {headline['sentiment']} ({headline['polarity']:+.2f})")
        
        # Save results
        analyzer.save_sentiment_data(test_symbol, sentiment_results)
    else:
        print(f"❌ Error: {sentiment_results['error']}")
    
    print(f"\n2. MARKET SENTIMENT OVERVIEW")
    print("-" * 40)
    
    market_overview = analyzer.get_market_sentiment_overview(['AAPL', 'MSFT'])  # Test with 2 stocks
    
    print("\n" + "=" * 60)
    print("NEWS SENTIMENT ANALYSIS TEST COMPLETED!")
    print()
    print("✅ Real news data fetched from Finnhub API")
    print("✅ Sentiment analysis performed on headlines and summaries") 
    print("✅ Market sentiment overview generated")
    print("✅ Results saved for ML model integration")
    print()
    print("💡 This sentiment data can enhance stock predictions!")

if __name__ == "__main__":
    main()