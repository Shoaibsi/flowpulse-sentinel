"""
LSTM Predictor module for FlowPulse Sentinel
Handles price predictions using LSTM neural networks
"""
import os
import logging
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from transformers import AutoModel, AutoModelForSequenceClassification
from sklearn.preprocessing import MinMaxScaler
from flowpulse.config import Config

logger = logging.getLogger("FlowPulseBot.LSTMPredictor")

class LSTMPredictor:
    """LSTM-based price predictor"""
    
    def __init__(self, model_name: str = None, model_path: str = None, window_size: int = None):
        """Initialize the LSTM predictor
        
        Args:
            model_name: Hugging Face model name
            model_path: Local model path
            window_size: Window size for LSTM input
        """
        self.window_size = window_size or Config.WINDOW_SIZE
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        try:
            # First try loading from Hugging Face
            if model_name:
                logger.info(f"Loading LSTM model from Hugging Face: {model_name}")
                try:
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    logger.info("LSTM model loaded successfully from Hugging Face")
                except Exception as e:
                    logger.error(f"Hugging Face load failed: {e}")
                    # Fall back to local model if specified
                    if model_path and os.path.exists(model_path):
                        self._load_local_model(model_path)
            # If no model_name provided or loading failed, try local path
            elif model_path and os.path.exists(model_path):
                self._load_local_model(model_path)
            else:
                logger.warning("No valid model name or path provided")
        except Exception as e:
            logger.error(f"Error initializing LSTM model: {e}")
            self.model = None
    
    def _load_local_model(self, model_path: str):
        """Load model from local path"""
        try:
            logger.info(f"Loading LSTM model from local path: {model_path}")
            self.model = torch.load(model_path, map_location=torch.device('cpu'))
            logger.info("LSTM model loaded successfully from local path")
        except Exception as e:
            logger.error(f"Local model load failed: {e}")
            self.model = None
            
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str):
        """Load model from checkpoint file"""
        try:
            model = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            logger.info(f"Model loaded from checkpoint: {checkpoint_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load from checkpoint: {e}")
            return None
            
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess data for LSTM prediction
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Preprocessed data as numpy array
        """
        if data is None or len(data) < self.window_size:
            logger.error(f"Insufficient data for LSTM prediction. Need at least {self.window_size} data points.")
            return None
            
        try:
            # Extract features (OHLCV)
            features = data[['open', 'high', 'low', 'close', 'volume']].values
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Create sequences
            X = []
            for i in range(len(scaled_features) - self.window_size):
                X.append(scaled_features[i:i + self.window_size])
                
            return np.array(X)
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return None
            
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make price predictions using LSTM
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            logger.error("No model loaded")
            return {
                'success': False,
                'error': 'No model loaded',
                'prediction': None
            }
            
        if data is None or len(data) < self.window_size:
            logger.error(f"Insufficient data: {len(data) if data is not None else 0} rows (need {self.window_size})")
            return {
                'success': False,
                'error': f'Insufficient data: need at least {self.window_size} rows',
                'prediction': None
            }
            
        try:
            # Preprocess data
            X = self.preprocess_data(data)
            if X is None or len(X) == 0:
                return {
                    'success': False,
                    'error': 'Failed to preprocess data',
                    'prediction': None
                }
                
            # Convert to tensor
            X_tensor = torch.tensor(X, dtype=torch.float32)
            
            # Make prediction - handle both custom models and transformers models
            with torch.no_grad():
                self.model.eval()
                
                try:
                    # First try standard PyTorch model approach
                    if hasattr(self.model, 'forward') and callable(getattr(self.model, 'forward')):
                        predictions = self.model(X_tensor)
                    # Then try models with predict method
                    elif hasattr(self.model, 'predict') and callable(getattr(self.model, 'predict')):
                        predictions = self.model.predict(X_tensor)
                    # Finally try transformers model
                    else:
                        outputs = self.model(X_tensor)
                        # Extract predictions from model outputs
                        if hasattr(outputs, 'logits'):
                            predictions = outputs.logits
                        elif isinstance(outputs, dict) and 'logits' in outputs:
                            predictions = outputs['logits']
                        else:
                            predictions = outputs[0] if isinstance(outputs, tuple) else outputs
                except Exception as e:
                    logger.error(f"Failed to get predictions from model: {e}")
                    # Fallback to a simple prediction based on recent trend
                    recent_closes = data['close'].values[-5:]
                    avg_pct_change = np.mean(np.diff(recent_closes) / recent_closes[:-1]) * 100
                    
                    return {
                        'success': True,
                        'warning': f'Model prediction failed, using fallback: {str(e)}',
                        'prediction': {
                            'close': data['close'].values[-1] * (1 + avg_pct_change/100),
                            'pct_change': avg_pct_change,
                            'direction': 'up' if avg_pct_change > 0 else 'down',
                            'confidence': 0.3  # Low confidence for fallback
                        }
                    }
                
            # Convert predictions to numpy if it's a tensor
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()
                
            # Process predictions
            last_window = data[['open', 'high', 'low', 'close', 'volume']].values[-self.window_size:]
            last_window_scaled = self.scaler.fit_transform(last_window)  # Use fit_transform to ensure scaler is updated
            
            # Handle different prediction shapes
            if len(predictions.shape) == 1:
                # Single value prediction - assume it's percentage change
                pct_change = float(predictions[0])
                last_close = data['close'].values[-1]
                predicted_close = last_close * (1 + pct_change/100)
                
                return {
                    'success': True,
                    'prediction': {
                        'close': predicted_close,
                        'pct_change': pct_change,
                        'direction': 'up' if pct_change > 0 else 'down',
                        'confidence': self._calculate_confidence(np.array([[predicted_close]]), last_window)
                    }
                }
            elif predictions.shape[-1] == 1:
                # Single column prediction - assume it's percentage change
                pct_change = float(predictions[0][0])
                last_close = data['close'].values[-1]
                predicted_close = last_close * (1 + pct_change/100)
                
                return {
                    'success': True,
                    'prediction': {
                        'close': predicted_close,
                        'pct_change': pct_change,
                        'direction': 'up' if pct_change > 0 else 'down',
                        'confidence': self._calculate_confidence(np.array([[predicted_close]]), last_window)
                    }
                }
            elif predictions.shape[-1] != 5:  # Not OHLCV format
                # Reshape or pad predictions to match expected format
                logger.warning(f"Prediction shape {predictions.shape} doesn't match expected OHLCV format")
                # Simple approach: use last_window_scaled and just update the close price
                predicted_scaled = last_window_scaled.copy()
                
                # If we have at least 4 values, assume the 4th is close price
                if predictions.shape[-1] >= 4:
                    pct_change = float(predictions[0][3])
                else:
                    # Otherwise use the first value
                    pct_change = float(predictions[0][0]) if predictions.size > 0 else 0
                
                # Calculate predicted close price
                last_close = data['close'].values[-1]
                predicted_close = last_close * (1 + pct_change/100)
                
                return {
                    'success': True,
                    'prediction': {
                        'close': predicted_close,
                        'pct_change': pct_change,
                        'direction': 'up' if pct_change > 0 else 'down',
                        'confidence': self._calculate_confidence(np.array([[predicted_close]]), last_window)
                    }
                }
            
            # Standard flow for full OHLCV predictions
            try:
                predicted_values = self.scaler.inverse_transform(predictions)
                
                # Extract predicted close price
                predicted_close = predicted_values[0][3]  # Index 3 is close price
                
                # Calculate prediction metrics
                last_close = data['close'].values[-1]
                pct_change = (predicted_close - last_close) / last_close * 100
                
                return {
                    'success': True,
                    'prediction': {
                        'close': predicted_close,
                        'pct_change': pct_change,
                        'direction': 'up' if pct_change > 0 else 'down',
                        'confidence': self._calculate_confidence(predicted_values, last_window)
                    }
                }
            except Exception as e:
                logger.error(f"Error in final prediction processing: {e}")
                # Fallback to a simple prediction
                pct_change = float(predictions[0][0]) if predictions.size > 0 else 0
                last_close = data['close'].values[-1]
                predicted_close = last_close * (1 + pct_change/100)
                
                return {
                    'success': True,
                    'warning': f'Final processing failed, using fallback: {str(e)}',
                    'prediction': {
                        'close': predicted_close,
                        'pct_change': pct_change,
                        'direction': 'up' if pct_change > 0 else 'down',
                        'confidence': 0.4  # Moderate confidence for this fallback
                    }
                }
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}',
                'prediction': None
            }
            
    def _calculate_confidence(self, predicted_values, last_window) -> float:
        """Calculate prediction confidence score
        
        Args:
            predicted_values: Predicted values
            last_window: Last window of actual values
            
        Returns:
            Confidence score (0-1)
        """
        try:
            # Simple confidence calculation based on prediction variance
            variance = np.var(predicted_values)
            confidence = 1.0 / (1.0 + variance)
            return min(max(confidence, 0.0), 1.0)  # Clamp between 0 and 1
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5  # Default medium confidence
            
    def benchmark_vs_moving_average(self, ohlc: pd.DataFrame) -> dict:
        """Benchmark vs moving average"""
        try:
            # Calculate moving averages
            ma5 = ohlc['close'].rolling(window=5).mean().iloc[-1]
            ma20 = ohlc['close'].rolling(window=20).mean().iloc[-1]
            
            # Calculate percentage difference
            last_close = ohlc['close'].iloc[-1]
            ma5_diff = (last_close / ma5 - 1) * 100
            ma20_diff = (last_close / ma20 - 1) * 100
            
            return {
                'last_close': last_close,
                'ma5': ma5,
                'ma20': ma20,
                'ma5_diff_pct': ma5_diff,
                'ma20_diff_pct': ma20_diff
            }
        except Exception as e:
            logger.error(f"Error in benchmark calculation: {e}")
            return {}
            
    def detect_prediction_drift(self, new_prediction: float, historical_predictions: list) -> bool:
        """Detect if prediction is drifting from historical patterns"""
        if len(historical_predictions) < 5:
            return False
            
        # Calculate mean and standard deviation of historical predictions
        mean_pred = np.mean(historical_predictions)
        std_pred = np.std(historical_predictions)
        
        # Check if new prediction is more than 2 standard deviations away
        z_score = abs(new_prediction - mean_pred) / (std_pred + 1e-6)  # Add small epsilon to avoid division by zero
        
        return z_score > 2.0  # Drift detected if z-score > 2
