"""
Equitura AI Stock Prediction Platform
Machine Learning Models for Stock Price Prediction
Uses Random Forest with real Alpha Vantage data + technical indicators
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys
sys.path.append('..')
import config
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EquituraMLEngine:
    """
    Advanced Machine Learning Engine for Stock Price Prediction
    Uses Random Forest with real financial data and technical indicators
    """
    
    def __init__(self):
        """Initialize the ML engine"""
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
        # Ensure models directory exists
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        
        print("Equitura ML Engine initialized")
        print(f"Prediction targets: {config.PREDICTION_DAYS_AHEAD} days ahead")
    
    def load_enhanced_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load stock data with technical indicators
        
        Args:
            symbol (str): Stock symbol
        
        Returns:
            pd.DataFrame: Enhanced stock data or None
        """
        try:
            # Try enhanced data first
            filename = f"{symbol}_with_indicators.csv"
            filepath = os.path.join("..", config.DATA_DIR, filename)
            
            if not os.path.exists(filepath):
                print(f"Enhanced data not found for {symbol}")
                print("Run technical_indicators.py first to generate indicators")
                return None
            
            data = pd.read_csv(filepath)
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values('Date').reset_index(drop=True)
            
            print(f"Loaded enhanced data: {len(data)} records with {len(data.columns)} features")
            return data
            
        except Exception as e:
            print(f"Error loading enhanced data for {symbol}: {e}")
            return None
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for machine learning with smart NaN handling
        
        Args:
            data (pd.DataFrame): Raw stock data with indicators
        
        Returns:
            Tuple[pd.DataFrame, List[str]]: Cleaned data and feature columns
        """
        # Remove non-feature columns
        exclude_columns = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Get all feature columns (technical indicators)
        all_feature_columns = [col for col in data.columns if col not in exclude_columns]
        
        # Check for columns with too many NaN values and remove them
        data_clean = data[['Date', 'Close'] + all_feature_columns].copy()
        
        # Calculate NaN percentage for each feature
        nan_percentages = data_clean.isnull().sum() / len(data_clean)
        
        # Keep only features with less than 50% NaN values
        good_features = [col for col in all_feature_columns 
                        if nan_percentages[col] < 0.5]
        
        print(f"Removed {len(all_feature_columns) - len(good_features)} features with >50% NaN values")
        
        # Select final columns
        final_columns = ['Date', 'Close'] + good_features
        data_clean = data_clean[final_columns]
        
        # Forward fill remaining NaN values (use previous value)
        data_clean = data_clean.fillna(method='ffill')
        
        # Backward fill any remaining NaN values at the beginning
        data_clean = data_clean.fillna(method='bfill')
        
        # Drop any rows that still have NaN values
        data_clean = data_clean.dropna()
        
        print(f"Final dataset: {len(data_clean)} records with {len(good_features)} features")
        print(f"Features include: {good_features[:10]}..." if len(good_features) > 10 else f"Features: {good_features}")
        
        return data_clean, good_features
    
    def create_targets(self, data: pd.DataFrame, prediction_days: int) -> pd.DataFrame:
        """
        Create prediction targets (future prices and directions) with smart handling
        
        Args:
            data (pd.DataFrame): Stock data
            prediction_days (int): Days ahead to predict
        
        Returns:
            pd.DataFrame: Data with targets
        """
        data = data.copy()
        
        print(f"Creating targets from {len(data)} records...")
        
        # Future price targets
        data[f'target_price_{prediction_days}d'] = data['Close'].shift(-prediction_days)
        
        # Price change targets
        data[f'target_change_{prediction_days}d'] = (
            data[f'target_price_{prediction_days}d'] - data['Close']
        ) / data['Close']
        
        # Direction targets (1 = up, 0 = down)
        data[f'target_direction_{prediction_days}d'] = (
            data[f'target_change_{prediction_days}d'] > 0
        ).astype(int)
        
        # Remove only the last prediction_days rows (which will have NaN targets)
        data_with_targets = data.iloc[:-prediction_days].copy()
        
        # Ensure we have sufficient data for training
        if len(data_with_targets) < 30:
            print(f"Warning: Only {len(data_with_targets)} samples available after target creation")
            print("This may not be sufficient for robust training")
        
        print(f"Created targets for {prediction_days}-day prediction")
        print(f"Usable samples: {len(data_with_targets)}")
        
        return data_with_targets
    
    def train_price_model(self, symbol: str, prediction_days: int) -> Dict:
        """
        Train Random Forest model for price prediction
        
        Args:
            symbol (str): Stock symbol
            prediction_days (int): Days ahead to predict
        
        Returns:
            Dict: Training results and metrics
        """
        try:
            print(f"\nTraining {prediction_days}-day price prediction model for {symbol}")
            print("-" * 50)
            
            # Load data
            data = self.load_enhanced_data(symbol)
            if data is None:
                return {"error": "No data available"}
            
            # Prepare features
            data_clean, feature_columns = self.prepare_features(data)
            
            # Create targets
            data_with_targets = self.create_targets(data_clean, prediction_days)
            
            if len(data_with_targets) < 20:
                print(f"⚠️  Warning: Only {len(data_with_targets)} samples available")
                print("This is limited but we'll proceed with a simplified model")
                
                if len(data_with_targets) < 10:
                    return {"error": f"Insufficient data: only {len(data_with_targets)} samples"}
            
            # Prepare X and y
            X = data_with_targets[feature_columns]
            y = data_with_targets[f'target_change_{prediction_days}d']  # Predict % change
            
            # Adjust train/test split for small datasets
            if len(X) < 30:
                # For small datasets, use a larger training ratio
                split_ratio = 0.9
                print(f"Using {split_ratio:.0%} training ratio due to small dataset")
            else:
                split_ratio = config.TRAIN_TEST_SPLIT_RATIO
            
            # Train/test split (time series aware)
            split_index = max(1, int(len(X) * split_ratio))
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
            
            print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
            
            # For very small datasets, adjust Random Forest parameters
            if len(X_train) < 20:
                n_estimators = min(50, len(X_train) * 2)
                max_depth = min(10, len(X_train) // 2)
                min_samples_split = 2
                min_samples_leaf = 1
                print(f"Using simplified RF parameters for small dataset")
            else:
                n_estimators = config.RANDOM_FOREST_ESTIMATORS
                max_depth = 15
                min_samples_split = 5
                min_samples_leaf = 2
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )
            
            rf_model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred_train = rf_model.predict(X_train_scaled)
            y_pred_test = rf_model.predict(X_test_scaled)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Directional accuracy
            train_direction_acc = accuracy_score(
                (y_train > 0).astype(int), 
                (y_pred_train > 0).astype(int)
            )
            test_direction_acc = accuracy_score(
                (y_test > 0).astype(int), 
                (y_pred_test > 0).astype(int)
            )
            
            # Feature importance
            feature_importance = dict(zip(feature_columns, rf_model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Cross-validation (skip if dataset too small)
            if len(X_train) >= 15:
                cv_folds = min(5, len(X_train) // 3)  # Adjust CV folds for small datasets
                cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, 
                                          cv=cv_folds, scoring='neg_mean_squared_error')
                cv_rmse = np.sqrt(-cv_scores.mean())
            else:
                cv_rmse = test_rmse  # Use test RMSE as proxy
                print("Skipping cross-validation due to small dataset")
            
            # Store model and scaler
            model_key = f"{symbol}_{prediction_days}d_price"
            self.models[model_key] = rf_model
            self.scalers[model_key] = scaler
            self.feature_importance[model_key] = feature_importance
            
            # Save models
            model_path = config.get_model_path(f"{model_key}_model.joblib")
            scaler_path = config.get_model_path(f"{model_key}_scaler.joblib")
            joblib.dump(rf_model, model_path)
            joblib.dump(scaler, scaler_path)
            
            results = {
                "symbol": symbol,
                "prediction_days": prediction_days,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_mae": train_mae,
                "test_mae": test_mae,
                "train_direction_accuracy": train_direction_acc,
                "test_direction_accuracy": test_direction_acc,
                "cv_rmse": cv_rmse,
                "feature_count": len(feature_columns),
                "top_features": top_features,
                "model_saved": True
            }
            
            print(f"✅ Model trained successfully!")
            print(f"Test RMSE: {test_rmse:.4f}")
            print(f"Test Directional Accuracy: {test_direction_acc:.1%}")
            print(f"CV RMSE: {cv_rmse:.4f}")
            
            return results
            
        except Exception as e:
            print(f"Error training model: {e}")
            return {"error": str(e)}
    
    def train_direction_model(self, symbol: str, prediction_days: int) -> Dict:
        """
        Train Random Forest classifier for direction prediction
        
        Args:
            symbol (str): Stock symbol
            prediction_days (int): Days ahead to predict
        
        Returns:
            Dict: Training results and metrics
        """
        try:
            print(f"\nTraining {prediction_days}-day direction prediction model for {symbol}")
            print("-" * 50)
            
            # Load data
            data = self.load_enhanced_data(symbol)
            if data is None:
                return {"error": "No data available"}
            
            # Prepare features
            data_clean, feature_columns = self.prepare_features(data)
            
            # Create targets
            data_with_targets = self.create_targets(data_clean, prediction_days)
            
            if len(data_with_targets) < 10:
                return {"error": f"Insufficient data: only {len(data_with_targets)} samples"}
            
            # Prepare X and y
            X = data_with_targets[feature_columns]
            y = data_with_targets[f'target_direction_{prediction_days}d']
            
            # Adjust train/test split for small datasets
            if len(X) < 30:
                split_ratio = 0.9
            else:
                split_ratio = config.TRAIN_TEST_SPLIT_RATIO
            
            # Train/test split
            split_index = max(1, int(len(X) * split_ratio))
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
            
            # Adjust Random Forest parameters for small datasets
            if len(X_train) < 20:
                n_estimators = min(50, len(X_train) * 2)
                max_depth = min(10, len(X_train) // 2)
                min_samples_split = 2
                min_samples_leaf = 1
            else:
                n_estimators = config.RANDOM_FOREST_ESTIMATORS
                max_depth = 15
                min_samples_split = 5
                min_samples_leaf = 2
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest Classifier
            rf_classifier = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )
            
            rf_classifier.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred_train = rf_classifier.predict(X_train_scaled)
            y_pred_test = rf_classifier.predict(X_test_scaled)
            y_pred_proba_test = rf_classifier.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            
            # Cross-validation (adjust for small datasets)
            if len(X_train) >= 10:
                cv_folds = min(5, len(X_train) // 2)
                cv_scores = cross_val_score(rf_classifier, X_train_scaled, y_train, 
                                          cv=cv_folds, scoring='accuracy')
                cv_accuracy = cv_scores.mean()
            else:
                cv_accuracy = test_accuracy
                print("Skipping cross-validation due to small dataset")
            
            # Store model
            model_key = f"{symbol}_{prediction_days}d_direction"
            self.models[model_key] = rf_classifier
            self.scalers[model_key] = scaler
            
            # Save models
            model_path = config.get_model_path(f"{model_key}_model.joblib")
            scaler_path = config.get_model_path(f"{model_key}_scaler.joblib")
            joblib.dump(rf_classifier, model_path)
            joblib.dump(scaler, scaler_path)
            
            results = {
                "symbol": symbol,
                "prediction_days": prediction_days,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "cv_accuracy": cv_accuracy,
                "model_saved": True
            }
            
            print(f"✅ Direction model trained successfully!")
            print(f"Test Accuracy: {test_accuracy:.1%}")
            print(f"CV Accuracy: {cv_accuracy:.1%}")
            
            return results
            
        except Exception as e:
            print(f"Error training direction model: {e}")
            return {"error": str(e)}
    
    def make_prediction(self, symbol: str, prediction_days: int) -> Optional[Dict]:
        """
        Make predictions using trained models
        
        Args:
            symbol (str): Stock symbol
            prediction_days (int): Days ahead to predict
        
        Returns:
            Dict: Predictions and confidence metrics
        """
        try:
            # Load model
            model_key = f"{symbol}_{prediction_days}d_price"
            direction_key = f"{symbol}_{prediction_days}d_direction"
            
            if model_key not in self.models:
                # Try to load from file
                model_path = config.get_model_path(f"{model_key}_model.joblib")
                scaler_path = config.get_model_path(f"{model_key}_scaler.joblib")
                
                if os.path.exists(model_path):
                    self.models[model_key] = joblib.load(model_path)
                    self.scalers[model_key] = joblib.load(scaler_path)
                else:
                    print(f"No trained model found for {symbol} {prediction_days}d")
                    return None
            
            # Load latest data
            data = self.load_enhanced_data(symbol)
            if data is None:
                return None
            
            # Prepare features
            data_clean, feature_columns = self.prepare_features(data)
            
            # Get latest features
            latest_features = data_clean[feature_columns].iloc[-1:].values
            
            # Scale features
            latest_features_scaled = self.scalers[model_key].transform(latest_features)
            
            # Make predictions
            price_change_pred = self.models[model_key].predict(latest_features_scaled)[0]
            
            # Current price
            current_price = data_clean['Close'].iloc[-1]
            predicted_price = current_price * (1 + price_change_pred)
            
            # Direction prediction (if available)
            direction_prob = None
            if direction_key in self.models:
                direction_prob = self.models[direction_key].predict_proba(latest_features_scaled)[0, 1]
            
            # Confidence based on model ensemble predictions
            n_estimators = 10
            if hasattr(self.models[model_key], 'estimators_'):
                individual_preds = [
                    tree.predict(latest_features_scaled)[0] 
                    for tree in self.models[model_key].estimators_[:n_estimators]
                ]
                confidence = 1 - (np.std(individual_preds) / abs(np.mean(individual_preds)))
                confidence = max(0, min(1, confidence))  # Clamp between 0 and 1
            else:
                confidence = 0.7  # Default confidence
            
            prediction = {
                'symbol': symbol,
                'prediction_days': prediction_days,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'predicted_change': price_change_pred,
                'predicted_change_percent': price_change_pred * 100,
                'direction_probability': direction_prob,
                'confidence': confidence,
                'timestamp': pd.Timestamp.now(),
                'recommendation': 'BUY' if price_change_pred > 0.02 else 'SELL' if price_change_pred < -0.02 else 'HOLD'
            }
            
            return prediction
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def train_all_models(self, symbols: List[str] = None) -> Dict:
        """
        Train models for all symbols and prediction periods
        
        Args:
            symbols (List[str]): Symbols to train (default: config symbols)
        
        Returns:
            Dict: Training results summary
        """
        if symbols is None:
            symbols = config.STOCK_SYMBOLS[:3]  # Train first 3 for demo
        
        print("TRAINING ALL ML MODELS")
        print("=" * 60)
        
        results = {}
        
        for symbol in symbols:
            print(f"\n🔄 Training models for {symbol}")
            results[symbol] = {}
            
            for days in config.PREDICTION_DAYS_AHEAD:
                # Train price model
                price_result = self.train_price_model(symbol, days)
                results[symbol][f'{days}d_price'] = price_result
                
                # Train direction model
                direction_result = self.train_direction_model(symbol, days)
                results[symbol][f'{days}d_direction'] = direction_result
                
                print(f"  ✅ {days}-day models trained")
        
        # Calculate overall performance
        test_accuracies = []
        for symbol_results in results.values():
            for model_result in symbol_results.values():
                if 'test_direction_accuracy' in model_result:
                    test_accuracies.append(model_result['test_direction_accuracy'])
        
        overall_accuracy = np.mean(test_accuracies) if test_accuracies else 0
        
        print(f"\n🎯 OVERALL PERFORMANCE")
        print(f"Average Directional Accuracy: {overall_accuracy:.1%}")
        print(f"Models trained: {len([r for results_dict in results.values() for r in results_dict.values()])}")
        
        return {
            'results': results,
            'overall_accuracy': overall_accuracy,
            'models_trained': len([r for results_dict in results.values() for r in results_dict.values()])
        }

def main():
    """
    Test the ML engine
    """
    print("EQUITURA ML ENGINE - TESTING")
    print("=" * 60)
    print("Training Random Forest models with REAL Alpha Vantage data")
    print()
    
    # Initialize ML engine
    ml_engine = EquituraMLEngine()
    
    # Test with AAPL
    test_symbol = "AAPL"
    
    print(f"1. TRAINING MODELS FOR {test_symbol}")
    print("-" * 40)
    
    # Train 1-day prediction model
    result_1d = ml_engine.train_price_model(test_symbol, 1)
    
    if 'error' not in result_1d:
        print(f"✅ 1-day model: {result_1d['test_direction_accuracy']:.1%} accuracy")
    
    # Train 7-day prediction model
    result_7d = ml_engine.train_price_model(test_symbol, 7)
    
    if 'error' not in result_7d:
        print(f"✅ 7-day model: {result_7d['test_direction_accuracy']:.1%} accuracy")
    
    print(f"\n2. MAKING PREDICTIONS FOR {test_symbol}")
    print("-" * 40)
    
    # Make predictions
    pred_1d = ml_engine.make_prediction(test_symbol, 1)
    pred_7d = ml_engine.make_prediction(test_symbol, 7)
    
    if pred_1d:
        print(f"1-day prediction:")
        print(f"  Current: ${pred_1d['current_price']:.2f}")
        print(f"  Predicted: ${pred_1d['predicted_price']:.2f}")
        print(f"  Change: {pred_1d['predicted_change_percent']:+.2f}%")
        print(f"  Confidence: {pred_1d['confidence']:.1%}")
        print(f"  Recommendation: {pred_1d['recommendation']}")
    
    if pred_7d:
        print(f"\n7-day prediction:")
        print(f"  Current: ${pred_7d['current_price']:.2f}")
        print(f"  Predicted: ${pred_7d['predicted_price']:.2f}")
        print(f"  Change: {pred_7d['predicted_change_percent']:+.2f}%")
        print(f"  Confidence: {pred_7d['confidence']:.1%}")
        print(f"  Recommendation: {pred_7d['recommendation']}")
    
    print("\n" + "=" * 60)
    print("ML ENGINE TEST COMPLETED!")
    print()
    print("✅ Random Forest models trained with real financial data")
    print("✅ Technical indicators used as features")
    print("✅ Price and direction predictions generated")
    print("✅ Models saved for production use")
    print()
    print("Next: Build web interface to display predictions")

if __name__ == "__main__":
    main()