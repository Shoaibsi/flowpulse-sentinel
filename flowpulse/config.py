"""
Configuration settings for FlowPulse Sentinel
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # API Keys
    POLYGON_KEY = os.environ.get("POLYGON_KEY", "your_polygon_api_key")
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "your_telegram_bot_token")
    HF_TOKEN = os.environ.get("HF_TOKEN", "your_huggingface_token")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    TASTYTRADE_USERNAME = os.environ.get("TASTYTRADE_USERNAME", "")
    TASTYTRADE_PASSWORD = os.environ.get("TASTYTRADE_PASSWORD", "")
    TASTYTRADE_ACCOUNT_ID = os.environ.get("TASTYTRADE_ACCOUNT_ID", "")
    
    # Data Settings
    DATA_CACHE_TTL = 300  # 5 minutes in seconds
    PREDICTION_CACHE_TTL = 3600  # 1 hour in seconds
    DATA_CACHE_SIZE = 100  # Maximum number of items in data cache
    PREDICTION_CACHE_SIZE = 50  # Maximum number of items in prediction cache
    POLYGON_MAX_CALLS = 5  # Maximum calls per minute for Polygon API
    MAX_RETRIES = 3  # Maximum number of retries for API calls
    REQUEST_DELAY = 2.0  # Delay between retries in seconds
    
    # Model Settings
    LSTM_MODEL_PATH = "models/lstm.pt"
    TFT_MODEL_PATH = "models/tft.pt"
    GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"  # Updated to use flash model
    GEMINI_SAFETY_SETTINGS = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
    }
    GEMINI_GENERATION_CONFIG = {
        "temperature": 0.7,
        "max_output_tokens": 1000
    }
    GEMINI_USAGE_LIMIT = 1000  # Requests per minute
    GEMINI_COST_PER_REQUEST = 0.0005  # USD
    
    # Alert Settings
    MAX_ALERTS_PER_HOUR = 10
    ALERT_PRIORITY_THRESHOLD = 0.8  # 0-1 scale
    
    # Market Settings
    MARKET_OPEN_TIME = "09:30"  # Eastern Time
    MARKET_TIMEZONE = "America/New_York"
    
    # Tickers to monitor
    DEFAULT_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA", "NVDA", "AMD", "AMZN", "META", "GOOGL"]
    
    # Technical Analysis Settings
    WINDOW_SIZE = 30  # 30-day window for TFT (reduced from 60)
    IV_PERCENTILE_DAYS = 30  # 30-day window for IV percentile
    
    # Unusual Flow Detection
    VOLUME_OI_RATIO_THRESHOLD = 2.5  # Volume/OI ratio threshold
    IV_RANK_THRESHOLD = 0.7  # IV rank threshold (0-1)
    
    # Database Settings
    DB_PATH = os.environ.get("DB_PATH", "flowpulse.db")  # Persistent DB file
