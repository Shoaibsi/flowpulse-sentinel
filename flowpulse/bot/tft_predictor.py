"""
Temporal Fusion Transformer Predictor module for FlowPulse Sentinel
Handles price predictions using TFT neural networks
"""
import os
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger("FlowPulseBot.TFTPredictor")

class TemporalAttention(nn.Module):
    """Multi-head attention for temporal data"""
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_size]
        # For multihead attention: [seq_len, batch_size, hidden_size]
        x = x.permute(1, 0, 2)
        attn_output, _ = self.attention(x, x, x)
        # Return to original shape
        return attn_output.permute(1, 0, 2)

class GatedResidualNetwork(nn.Module):
    """Gated residual network for feature processing"""
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)
        self.gate = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        # Main branch
        h = torch.relu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        
        # Gating mechanism
        g = torch.sigmoid(self.gate(x))
        
        # If input and output dimensions match, add residual connection
        if x.shape[-1] == h.shape[-1]:
            output = self.layer_norm(g * h + (1 - g) * x)
        else:
            output = self.layer_norm(h)
            
        return output

class TFTPredictor(nn.Module):
    """Temporal Fusion Transformer for time series prediction"""
    
    def __init__(self, input_size=5, hidden_size=64, output_size=1, num_layers=3, dropout=0.1, window_size=30):
        """Initialize the TFT predictor
        
        Args:
            input_size: Number of input features (OHLCV = 5)
            hidden_size: Hidden dimension size
            output_size: Output dimension size (typically 1 for price prediction)
            num_layers: Number of transformer layers
            dropout: Dropout rate
            window_size: Window size for input sequence
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.window_size = window_size
        
        # Feature processing
        self.feature_layer = GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout)
        
        # Temporal processing
        self.temporal_layers = nn.ModuleList([
            GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
            for _ in range(num_layers)
        ])
        
        # Self-attention layer
        self.attention = TemporalAttention(hidden_size, num_heads=4)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        # Preprocessing
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def forward(self, x):
        """Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            
        Returns:
            Output tensor of shape [batch_size, output_size]
        """
        # Feature processing
        processed = self.feature_layer(x)
        
        # Temporal processing with residual connections
        for layer in self.temporal_layers:
            processed = layer(processed)
            
        # Apply attention
        attended = self.attention(processed)
        
        # Use the last time step for prediction
        out = self.fc_out(attended[:, -1, :])
        
        return out
    
    @classmethod
    def load_from_pretrained(cls, model_path: str, map_location=None):
        """Load model from pretrained weights
        
        Args:
            model_path: Path to model weights
            map_location: Device to load model on
            
        Returns:
            Loaded model
        """
        try:
            if not map_location and not torch.cuda.is_available():
                map_location = torch.device('cpu')
                
            model = torch.load(model_path, map_location=map_location)
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return None
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[MinMaxScaler]]:
        """Preprocess data for TFT prediction
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Preprocessed data as numpy array and fitted scaler
        """
        if data is None or len(data) < self.window_size:
            logger.error(f"Insufficient data for TFT prediction. Need at least {self.window_size} data points.")
            return None, None
            
        try:
            # Extract features (OHLCV)
            features = data[['open', 'high', 'low', 'close', 'volume']].values
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Create sequences
            X = []
            for i in range(len(scaled_features) - self.window_size + 1):
                X.append(scaled_features[i:i + self.window_size])
                
            return np.array(X), self.scaler
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return None, None
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Make price predictions using TFT
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with prediction results
        """
        if data is None or len(data) < self.window_size:
            logger.error(f"Insufficient data: {len(data) if data is not None else 0} rows (need {self.window_size})")
            return {
                'success': False,
                'error': f'Insufficient data: need at least {self.window_size} rows',
                'prediction': None
            }
            
        try:
            # Preprocess data
            X, scaler = self.preprocess_data(data)
            if X is None or len(X) == 0:
                return {
                    'success': False,
                    'error': 'Failed to preprocess data',
                    'prediction': None
                }
                
            # Convert to tensor
            X_tensor = torch.tensor(X, dtype=torch.float32)
            
            # Make prediction
            with torch.no_grad():
                self.eval()
                predictions = self(X_tensor)
                
            # Convert predictions to numpy
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()
                
            # Get the last prediction
            last_pred = predictions[-1][0]
            
            # Calculate percentage change
            last_close = data['close'].values[-1]
            pct_change = ((last_pred - last_close) / last_close) * 100
            
            # Calculate confidence based on prediction variance
            if len(predictions) > 1:
                pred_std = np.std(predictions.flatten())
                confidence = max(0.0, min(1.0, 1.0 - (pred_std / last_close)))
            else:
                confidence = 0.5  # Default confidence
                
            return {
                'success': True,
                'prediction': {
                    'close': float(last_pred),
                    'pct_change': float(pct_change),
                    'direction': 'up' if pct_change > 0 else 'down',
                    'confidence': float(confidence)
                }
            }
                
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            
            # Fallback to a simple prediction based on recent trend
            try:
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
            except Exception as fallback_error:
                logger.error(f"Fallback prediction also failed: {fallback_error}")
                return {
                    'success': False,
                    'error': f'Prediction failed: {str(e)}',
                    'prediction': None
                }

def create_pretrained_model(save_path: str = "models/tft.pt"):
    """Create and save a pretrained TFT model
    
    Args:
        save_path: Path to save the model
        
    Returns:
        Created model
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create model
        model = TFTPredictor(
            input_size=5,  # OHLCV
            hidden_size=64,
            output_size=1,  # Price prediction
            num_layers=3,
            window_size=30  # Reduced from 60
        )
        
        # Save model
        torch.save(model, save_path)
        logger.info(f"Created pretrained TFT model and saved to {save_path}")
        
        return model
    except Exception as e:
        logger.error(f"Failed to create pretrained model: {e}")
        return None
