#!/usr/bin/env python3
"""
Vestara AI Model Training Pipeline
Train LSTM and Ensemble models for stock prediction
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Import components
try:
    import config
    from lstm_predictor import LSTMStockPredictor
    from ensemble_predictor import EnsemblePredictionEngine
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the correct directory and all dependencies are installed")
    sys.exit(1)

class ModelTrainingPipeline:
    """Complete training pipeline for Vestara AI models"""
    
    def __init__(self):
        self.lstm_predictor = LSTMStockPredictor()
        self.ensemble_engine = EnsemblePredictionEngine()
        
        # Create results directory
        self.results_dir = "training_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info("✅ Model Training Pipeline initialized")
    
    def check_data_availability(self):
        """Check which symbols have cached data available"""
        available_symbols = []
        missing_symbols = []
        
        cache_dir = "data/cache"
        
        for symbol in config.STOCK_SYMBOLS:
            cache_file = f"{cache_dir}/alphavantage_{symbol}.json"
            
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    
                    # Check if data has enough history
                    if len(data.get('dates', [])) >= 60:
                        available_symbols.append(symbol)
                    else:
                        missing_symbols.append(f"{symbol} (insufficient history)")
                        
                except Exception as e:
                    missing_symbols.append(f"{symbol} (corrupted cache)")
            else:
                missing_symbols.append(f"{symbol} (no cache)")
        
        logger.info(f"📊 Data availability: {len(available_symbols)} available, {len(missing_symbols)} missing")
        
        return available_symbols, missing_symbols
    
    def train_single_symbol(self, symbol, epochs=30):
        """Train all models for a single symbol"""
        try:
            logger.info(f"🎯 Training models for {symbol}")
            
            results = {
                'symbol': symbol,
                'training_date': datetime.now().isoformat(),
                'lstm': {'status': 'not_attempted'},
                'ensemble': {'status': 'not_attempted'}
            }
            
            # Train LSTM
            try:
                logger.info(f"🧠 Training LSTM for {symbol}")
                lstm_success = self.lstm_predictor.train_model(symbol, epochs=epochs)
                
                if lstm_success:
                    lstm_info = self.lstm_predictor.get_model_info(symbol)
                    results['lstm'] = {
                        'status': 'success',
                        'info': lstm_info
                    }
                    logger.info(f"✅ LSTM training completed for {symbol}")
                else:
                    results['lstm'] = {'status': 'failed', 'error': 'Training returned False'}
                    
            except Exception as e:
                results['lstm'] = {'status': 'error', 'error': str(e)}
                logger.error(f"❌ LSTM training error for {symbol}: {e}")
            
            # Train Ensemble (includes Random Forest)
            try:
                logger.info(f"🌲 Training Ensemble for {symbol}")
                ensemble_success = self.ensemble_engine.train_ensemble(symbol)
                
                if ensemble_success:
                    ensemble_status = self.ensemble_engine.get_ensemble_status(symbol)
                    results['ensemble'] = {
                        'status': 'success',
                        'info': ensemble_status
                    }
                    logger.info(f"✅ Ensemble training completed for {symbol}")
                else:
                    results['ensemble'] = {'status': 'failed', 'error': 'Training returned False'}
                    
            except Exception as e:
                results['ensemble'] = {'status': 'error', 'error': str(e)}
                logger.error(f"❌ Ensemble training error for {symbol}: {e}")
            
            # Test predictions
            try:
                if results['lstm']['status'] == 'success':
                    test_pred = self.lstm_predictor.predict_prices(symbol, [1])
                    results['lstm']['test_prediction'] = test_pred
                
                if results['ensemble']['status'] == 'success':
                    test_pred = self.ensemble_engine.ensemble_predict(symbol, None, [1])
                    results['ensemble']['test_prediction'] = test_pred
                    
            except Exception as e:
                logger.warning(f"Test prediction failed for {symbol}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error training {symbol}: {e}")
            return {
                'symbol': symbol,
                'training_date': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def train_multiple_symbols(self, symbols, epochs=30):
        """Train models for multiple symbols"""
        results = []
        
        logger.info(f"🚀 Starting training for {len(symbols)} symbols")
        logger.info(f"📋 Symbols: {', '.join(symbols)}")
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"📈 [{i}/{len(symbols)}] Processing {symbol}")
            
            result = self.train_single_symbol(symbol, epochs)
            results.append(result)
            
            # Save individual result
            with open(f"{self.results_dir}/training_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
                json.dump(result, f, indent=2)
        
        # Save summary
        summary = self.generate_training_summary(results)
        with open(f"{self.results_dir}/training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.print_training_summary(summary)
        
        return results
    
    def generate_training_summary(self, results):
        """Generate training summary statistics"""
        total = len(results)
        lstm_success = sum(1 for r in results if r.get('lstm', {}).get('status') == 'success')
        ensemble_success = sum(1 for r in results if r.get('ensemble', {}).get('status') == 'success')
        
        summary = {
            'training_completed': datetime.now().isoformat(),
            'total_symbols': total,
            'lstm_models': {
                'successful': lstm_success,
                'failed': total - lstm_success,
                'success_rate': f"{(lstm_success/total)*100:.1f}%" if total > 0 else "0%"
            },
            'ensemble_models': {
                'successful': ensemble_success,
                'failed': total - ensemble_success,
                'success_rate': f"{(ensemble_success/total)*100:.1f}%" if total > 0 else "0%"
            },
            'overall_success_rate': f"{(max(lstm_success, ensemble_success)/total)*100:.1f}%" if total > 0 else "0%",
            'detailed_results': results
        }
        
        return summary
    
    def print_training_summary(self, summary):
        """Print formatted training summary"""
        print("\n" + "="*60)
        print("🎯 VESTARA AI MODEL TRAINING SUMMARY")
        print("="*60)
        print(f"📊 Total Symbols: {summary['total_symbols']}")
        print(f"🧠 LSTM Models: {summary['lstm_models']['successful']}/{summary['total_symbols']} ({summary['lstm_models']['success_rate']})")
        print(f"🌲 Ensemble Models: {summary['ensemble_models']['successful']}/{summary['total_symbols']} ({summary['ensemble_models']['success_rate']})")
        print(f"✅ Overall Success: {summary['overall_success_rate']}")
        print("="*60)
        
        # Print individual results
        for result in summary['detailed_results']:
            symbol = result['symbol']
            lstm_status = result.get('lstm', {}).get('status', 'unknown')
            ensemble_status = result.get('ensemble', {}).get('status', 'unknown')
            
            lstm_emoji = "✅" if lstm_status == 'success' else "❌"
            ensemble_emoji = "✅" if ensemble_status == 'success' else "❌"
            
            print(f"{symbol}: LSTM {lstm_emoji} | Ensemble {ensemble_emoji}")
        
        print("="*60)

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Vestara AI Model Training Pipeline')
    parser.add_argument('--symbol', type=str, help='Train specific symbol (e.g., AAPL)')
    parser.add_argument('--symbols', type=str, nargs='+', help='Train multiple symbols')
    parser.add_argument('--all', action='store_true', help='Train all available symbols')
    parser.add_argument('--top', type=int, default=5, help='Train top N symbols (default: 5)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs (default: 30)')
    parser.add_argument('--check-data', action='store_true', help='Check data availability only')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ModelTrainingPipeline()
    
    # Check data availability
    available_symbols, missing_symbols = pipeline.check_data_availability()
    
    if args.check_data:
        print("\n📊 DATA AVAILABILITY CHECK")
        print("="*40)
        print(f"✅ Available symbols ({len(available_symbols)}):")
        for symbol in available_symbols:
            print(f"  - {symbol}")
        
        if missing_symbols:
            print(f"\n❌ Missing/insufficient data ({len(missing_symbols)}):")
            for symbol in missing_symbols:
                print(f"  - {symbol}")
        
        return
    
    # Determine which symbols to train
    if args.symbol:
        # Single symbol
        if args.symbol.upper() in available_symbols:
            symbols_to_train = [args.symbol.upper()]
        else:
            print(f"❌ No data available for {args.symbol}")
            return
            
    elif args.symbols:
        # Multiple specific symbols
        symbols_to_train = [s.upper() for s in args.symbols if s.upper() in available_symbols]
        if not symbols_to_train:
            print("❌ No data available for any specified symbols")
            return
            
    elif args.all:
        # All available symbols
        symbols_to_train = available_symbols
        
    else:
        # Top N symbols (default)
        symbols_to_train = available_symbols[:args.top]
    
    if not symbols_to_train:
        print("❌ No symbols to train")
        return
    
    print(f"\n🚀 Training {len(symbols_to_train)} symbols with {args.epochs} epochs each")
    print(f"📋 Symbols: {', '.join(symbols_to_train)}")
    
    # Start training
    results = pipeline.train_multiple_symbols(symbols_to_train, epochs=args.epochs)
    
    print(f"\n🏁 Training complete! Results saved in '{pipeline.results_dir}' directory")

if __name__ == "__main__":
    main()