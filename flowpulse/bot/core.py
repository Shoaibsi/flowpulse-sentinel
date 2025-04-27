"""
FlowPulse Sentinel - Core Bot Implementation
"""
import time
import logging
from logging.handlers import TimedRotatingFileHandler
import pandas as pd
from flowpulse.config import Config
import torch
import duckdb
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Union
from sklearn.preprocessing import MinMaxScaler
import os
import json
from collections import OrderedDict
import traceback
import sys
import numpy as np
import threading
import pandas_ta as ta
import asyncio
from flowpulse.utils.cache import CacheManager  # Import CacheManager

# Import the TFTPredictor class
from flowpulse.bot.tft_predictor import TFTPredictor

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# --- Refactored Logging Setup ---
log_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
log_file = os.path.join(log_dir, "flowpulse.log")

# File Handler with daily rotation (midnight), keeping 7 backups
file_handler = TimedRotatingFileHandler(
    log_file,
    when='midnight',
    interval=1,
    backupCount=7,
    encoding='utf-8'
)
file_handler.setFormatter(log_formatter)

# Console Handler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)

# Get root logger and add handlers
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(stream_handler)

logger = logging.getLogger("FlowPulseBot") # Keep specific logger if needed
# --- End Refactored Logging Setup ---


# Performance metrics logger (unchanged)
perf_logger = logging.getLogger("FlowPulseBot.Performance")
perf_logger.setLevel(logging.INFO)
perf_log_file = os.path.join(log_dir, "performance.log")
perf_handler = logging.FileHandler(perf_log_file)
perf_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
perf_logger.addHandler(perf_handler)

def log_performance(func):
    """Decorator to log function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        perf_logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

class LRUCache:
    """Least Recently Used (LRU) cache with size limit and optional disk persistence"""
    
    def __init__(self, max_size: int = 100, persist_path: Optional[str] = None):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.persist_path = persist_path
        self.lock = threading.Lock()
        
        if persist_path and os.path.exists(persist_path):
            self._load_from_disk()

    def _load_from_disk(self):
        """Thread-safe cache loading"""
        with self.lock:
            try:
                with open(self.persist_path, 'r') as f:
                    data = json.load(f)
                    for key, (timestamp_str, value) in data.items():
                        try:
                            # Convert timestamp string
                            timestamp = datetime.fromisoformat(timestamp_str)
                            
                            # Handle special types
                            if isinstance(value, dict):
                                if '__pd_dataframe__' in value:
                                    value = pd.DataFrame(value['data'])
                                elif '__np_array__' in value:
                                    value = np.array(value['data'])
                            
                            self.cache[key] = (timestamp, value)
                        except Exception as e:
                            logger.warning(f"Failed to load cache item {key}: {e}")
                logger.info(f"Loaded {len(self.cache)} items from cache file {self.persist_path}")
            except Exception as e:
                logger.error(f"Cache load error: {e}")
                self.cache = OrderedDict()
    
    def _persist(self):
        """Handle timestamp serialization properly"""
        if not self.persist_path:
            return
            
        try:
            serializable = {}
            for key, (timestamp, value) in self.cache.items():
                # Convert timestamp to ISO format string
                timestamp_str = timestamp.isoformat()
                
                # Handle pandas/numpy types
                if isinstance(value, pd.DataFrame):
                    value = {'__pd_dataframe__': True, 'data': value.to_dict(orient='records')}
                elif isinstance(value, np.ndarray):
                    value = {'__np_array__': True, 'data': value.tolist()}
                
                # Skip non-serializable objects
                try:
                    json.dumps({'t': timestamp_str, 'v': value})
                    serializable[key] = (timestamp_str, value)
                except TypeError as e:
                    logger.warning(f"Skipping non-serializable cache item {key}: {e}")
                    continue
                    
            # Atomic write
            temp_path = f"{self.persist_path}.tmp"
            with open(temp_path, 'w') as f:
                json.dump(serializable, f, indent=2)
            
            os.replace(temp_path, self.persist_path)
        except Exception as e:
            logger.error(f"Cache persistence failed: {e}")
            # Clean up corrupted cache
            if os.path.exists(self.persist_path):
                os.remove(self.persist_path)
    
    def get(self, key: str) -> Tuple[Optional[datetime], Any]:
        """Get item from cache, return (timestamp, value) or (None, None) if not found"""
        with self.lock:
            if key not in self.cache:
                return None, None
                
            # Move to end (most recently used)
            timestamp, value = self.cache.pop(key)
            self.cache[key] = (timestamp, value)
            return timestamp, value
    
    def put(self, key: str, value: Any) -> None:
        """Add item to cache with current timestamp"""
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                self.cache.pop(key)
                
            # Check if we need to remove oldest item
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)  # Remove oldest
                
            # Add new item
            self.cache[key] = (datetime.now(), value)
            
            # Persist to disk
            self._persist()
    
    def remove(self, key: str) -> bool:
        """Remove item from cache, return True if found, False otherwise"""
        with self.lock:
            if key in self.cache:
                self.cache.pop(key)
                self._persist()
                return True
            return False
    
    def clear(self) -> None:
        """Clear all items from cache"""
        with self.lock:
            self.cache.clear()
            self._persist()
    
    def contains(self, key: str) -> bool:
        """Check if key exists in cache"""
        with self.lock:
            return key in self.cache
    
    def size(self) -> int:
        """Get current cache size"""
        with self.lock:
            return len(self.cache)
    
    def keys(self) -> List[str]:
        """Get list of all keys in cache"""
        with self.lock:
            return list(self.cache.keys())

class FlowPulseBot:
    def __init__(self):
        logger.info("Initializing FlowPulse Sentinel...")
        
        # Phase 1: Core Setup
        self._init_config()
        
        # Initialize with safe defaults
        self.data_fetcher = None
        self.alert_manager = None
        self.tft_predictor = None
        self.gemini = None
        self.db = None
        
        # Initialize caches
        logger.debug("Initializing CacheManager...")
        self.cache_manager = CacheManager("./cache", self.config.DATA_CACHE_TTL) 
        # logger.info(f"CacheManager initialized: {self.cache_manager}") # No longer needed
        
        logger.debug("Initializing AlertManager...")
        self.alert_manager = AlertManager(self.config)
        
        logger.info("Core initialization complete")
        
    def _validate_tickers(self, tickers: List[str]) -> List[str]:
        """Filter out invalid tickers"""
        valid = []
        for t in tickers:
            if isinstance(t, str) and 1 <= len(t) <= 5 and t.isalpha():
                valid.append(t.upper())
            else:
                logger.warning(f"Removing invalid ticker: {t}")
        return valid or ['SPY']  # Default fallback
        
    def _init_config(self):
        """Step 4: Config Setup"""
        from flowpulse.config import Config
        self.config = Config() # Instantiate the Config class
        logger.info("Configuration loaded")
        
    async def initialize(self):
        """Initializes asynchronous components like the data fetcher."""
        logger.info("Initializing FlowPulseBot...")
        self.data_fetcher = await self._init_data_fetcher()
        if self.data_fetcher is None:
            logger.critical("Data fetcher failed to initialize! Bot cannot continue.")
            raise RuntimeError("Failed to initialize TastytradeDataFetcher")
        else:
            logger.info("Data fetcher initialized successfully.")

    async def _init_data_fetcher(self):
        """Initialize data fetcher with TastyTrade focus and ensure session is ready."""
        logger.info("Initializing TastytradeDataFetcher instance...")
        try:
            # Get TastyTrade credentials from environment
            username = os.getenv('TASTYTRADE_USERNAME')
            password = os.getenv('TASTYTRADE_PASSWORD')
            
            if not username or not password:
                raise ValueError("TastyTrade credentials not found in environment variables")
            
            # Initialize TastyTrade data fetcher instance
            from flowpulse.bot.data_fetcher import TastytradeDataFetcher
            data_fetcher = TastytradeDataFetcher(username, password)
            
            # Explicitly initialize the session and wait for it
            logger.info("Attempting to establish TastyTrade session...")
            init_success = await data_fetcher._initialize_session()
            
            if init_success:
                logger.info("Successfully initialized and connected TastyTrade data fetcher")
                return data_fetcher
            else:
                logger.error("TastyTrade session initialization failed. Fetcher not ready.")
                return None # Return None if session failed
            
        except Exception as e:
            logger.error(f"Failed during TastyTrade data fetcher setup: {e}")
            logger.debug(traceback.format_exc())
            return None # Return None on any exception during setup

    def _init_alert_manager(self):
        """Initialize alert manager with validation"""
        try:
            from flowpulse.bot.alert_manager import TelegramAlertManager
            
            # Check if we have the required credentials
            if not self.config.TELEGRAM_BOT_TOKEN:
                logger.warning("No Telegram bot token provided, alerts will be disabled")
                return None
                
            chat_id = os.environ.get("TELEGRAM_CHAT_ID")
            if not chat_id:
                logger.warning("No Telegram chat ID provided, alerts will be disabled")
                return None
                
            # Initialize the alert manager
            manager = TelegramAlertManager(
                token=self.config.TELEGRAM_BOT_TOKEN,
                chat_id=chat_id
            )
            
            # Test connection
            if not manager.send_test_message("FlowPulse initialized"):
                raise ValueError("Telegram test message failed")
                
            logger.info("Telegram alert manager initialized successfully")
            return manager
            
        except Exception as e:
            logger.error(f"Alert system initialization failed: {e}")
            return None
        
    def _init_db(self):
        """Step 7: DuckDB Initialization"""
        try:
            # Try to connect to the database
            self.db = duckdb.connect(self.config.DB_PATH)
            
            # Create tables if they don't exist
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS options_flows (
                    ticker STRING,
                    expiry DATE,
                    strike FLOAT,
                    option_type STRING,
                    volume INTEGER,
                    open_interest INTEGER,
                    iv FLOAT,
                    ts TIMESTAMP
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    ticker STRING,
                    prediction FLOAT,
                    confidence FLOAT,
                    ts TIMESTAMP
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    ticker STRING,
                    expiry DATE,
                    strike FLOAT,
                    option_type STRING,
                    analysis STRING,
                    ts TIMESTAMP
                )
            """)
            
            logger.info(f"Connected to database: {self.config.DB_PATH}")
            
        except duckdb.IOException as e:
            if "File is already open" in str(e):
                logger.warning(f"Database file is locked: {e}")
                
                # Try to connect in read-only mode as fallback
                try:
                    logger.info("Attempting to connect in read-only mode...")
                    self.db = duckdb.connect(self.config.DB_PATH, read_only=True)
                    logger.info("Connected to database in read-only mode")
                except Exception as inner_e:
                    logger.error(f"Failed to connect to database in read-only mode: {inner_e}")
                    
                    # Try to create a new database with a timestamp suffix
                    new_db_path = f"{self.config.DB_PATH}.{int(time.time())}"
                    logger.warning(f"Creating new database at: {new_db_path}")
                    try:
                        self.db = duckdb.connect(new_db_path)
                        logger.info(f"Created new database: {new_db_path}")
                    except Exception as create_e:
                        logger.error(f"Failed to create new database: {create_e}")
                        self.db = None
            else:
                logger.error(f"Database connection error: {e}")
                self.db = None

    def _init_models(self):
        """Initialize TFT and Gemini models"""
        # Initialize TFT model
        try:
            from flowpulse.bot.tft_predictor import TFTPredictor, create_pretrained_model
            local_path = getattr(self.config, 'TFT_MODEL_PATH', "models/tft.pt")
            
            if os.path.exists(local_path):
                self.tft = TFTPredictor.load_from_pretrained(local_path)
                if self.tft is None:
                    logger.warning("Failed to load TFT model, creating new")
                    self.tft = create_pretrained_model(local_path)
            else:
                self.tft = create_pretrained_model(local_path)
            logger.info("TFT model initialization complete")
        except Exception as e:
            logger.error(f"TFT initialization error: {e}")
            self.tft = None

        # Initialize Gemini
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.config.GEMINI_API_KEY)
            
            # Use safety settings from config
            self.gemini = genai.GenerativeModel(
                model_name=self.config.GEMINI_MODEL_NAME,
                safety_settings=self.config.GEMINI_SAFETY_SETTINGS,
                generation_config=self.config.GEMINI_GENERATION_CONFIG
            )
            
            # Test connection
            self.gemini.start_chat(history=[]).send_message("Test")
            logger.info("Gemini model initialized successfully")
            
        except ImportError:
            logger.error("google-generativeai package not installed")
            self.gemini = None
        except Exception as e:
            logger.error(f"Gemini initialization failed: {str(e)[:200]}")
            self.gemini = None

    @log_performance
    async def run(self):
        """Main event loop (Step 50)"""
        await self.initialize()
        logger.info("Starting FlowPulse Sentinel main loop")
        while True:
            try:
                if self._is_market_open():
                    logger.info("Market is open, analyzing tickers")
                    await self._analyze_tickers(self.config.DEFAULT_TICKERS)
                else:
                    logger.info("Market is closed, waiting...")
                
                # Sleep for 5 minutes
                await asyncio.sleep(self.config.DATA_CACHE_TTL)
            except Exception as e:
                self._handle_error(e)

    def _is_market_open(self) -> bool:
        """Check if the market is open"""
        # Always return True - we're removing market hours check
        return True

    async def _analyze_tickers(self, tickers: List[str]):
        """Robust ticker analysis with fallbacks"""
        for ticker in tickers:
            try:
                logger.info(f"Processing {ticker}")
                
                # Get data with fallback
                ohlc = await self._get_ohlc_with_fallback(ticker)
                options_df = await self._get_options_with_fallback(ticker) 
                
                # If options data is missing, skip (ignore OHLC for now)
                if options_df is None or not options_df:
                    logger.warning(f"Skipping {ticker} - insufficient options data (OHLC: {'OK' if ohlc is not None else 'FAIL - Ignored'}, Options: FAIL)")
                    continue
                    
                # Prediction with fallback
                prediction = None

                # Analyze for unusual flows (expects DataFrame, returns List[Dict])
                unusual_flows = await self._find_unusual_flows(ticker, options_df)
                
                # Combine data and generate alerts
                if unusual_flows:
                    for idx, flow_details in enumerate(unusual_flows): # flow_details is now a dict
                        try:
                            # Generate analysis using prediction dict and flow dict
                            analysis = await self._generate_analysis(ticker, prediction, flow_details) 
                            
                            if analysis:  # Only proceed if analysis was successful
                                # Call the correct alert manager method
                                success = await self.alert_manager.send_alert(
                                    ticker=ticker, 
                                    flow=flow_details, 
                                    tft_pred=prediction, # Pass the whole prediction dict (or adjust if needed)
                                    analysis=analysis
                                )
                                # self.alert_manager.send_alert_message(alert_message) # Old incorrect call
                                if success:
                                    logger.info(f"Successfully processed and sent alert for {ticker} flow {idx+1}")
                                else:
                                    logger.warning(f"Failed to send alert for {ticker} flow {idx+1}")
                            
                            # Store alert details (adjust as needed for DB schema)
                            await self._store_alert(ticker, flow_details.get('description', 'N/A'), analysis) 
                            
                        except Exception as e:
                            # Log specific error during analysis/alerting for this flow
                            logger.error(f"Alert/Analysis processing failed for one flow in {ticker}: {e}", exc_info=True) 
                             
            except Exception as e:
                # Catch errors processing the entire ticker
                logger.error(f"Critical error processing ticker {ticker}: {e}", exc_info=True) 
                continue # Move to the next ticker
                
    async def _find_unusual_flows(self, ticker: str, options_data: Dict) -> List[Dict]:
        """Analyzes options data to find contracts with unusual activity."""
        unusual_flows = []
        min_vol_oi_ratio = 5.0
        min_premium = 50000 # Minimum total premium (Volume * Price) to consider

        if not options_data or 'expirations' not in options_data:
            logger.warning(f"[_find_unusual_flows] No valid options data or expirations for {ticker}. Cannot analyze.")
            return []

        logger.info(f"[_find_unusual_flows] Analyzing options data for {ticker}...")

        try:
            for expiration_data in options_data.get('expirations', []):
                expiry_date = expiration_data.get('expiration-date')
                for strike_data in expiration_data.get('strikes', []):
                    strike_price = strike_data.get('strike-price')
                    
                    # Analyze Calls
                    call_quote = strike_data.get('call_quote')
                    if call_quote:
                        vol = self._safe_float(call_quote.get('volume'))
                        oi = self._safe_float(call_quote.get('open_interest'))
                        last_price = self._safe_float(call_quote.get('last_price'))
                        bid = self._safe_float(call_quote.get('bid'))
                        ask = self._safe_float(call_quote.get('ask'))
                        
                        # Use last price if available, otherwise estimate with mid/ask/bid
                        price_to_use = last_price if last_price is not None and last_price > 0 else call_quote.get('mid')
                        if price_to_use is None or price_to_use <= 0:
                            price_to_use = ask if ask is not None and ask > 0 else bid
                            
                        if vol is not None and oi is not None and price_to_use is not None and price_to_use > 0:
                            if oi > 0 and (vol / oi) >= min_vol_oi_ratio:
                                premium = vol * price_to_use
                                if premium >= min_premium:
                                    flow = {
                                        'symbol': strike_data.get('call'),
                                        'type': 'call',
                                        'strike': strike_price,
                                        'expiry': expiry_date,
                                        'volume': vol,
                                        'open_interest': oi,
                                        'vol_oi_ratio': round(vol / oi, 2),
                                        'price': price_to_use,
                                        'premium': round(premium, 2),
                                        'bid': bid,
                                        'ask': ask,
                                        'description': f"CALL Vol/OI: {round(vol / oi, 2)} Prem: ${premium:,.0f}"
                                    }
                                    unusual_flows.append(flow)
                                    logger.debug(f"Unusual CALL detected: {flow}")

                    # Analyze Puts
                    put_quote = strike_data.get('put_quote')
                    if put_quote:
                        vol = self._safe_float(put_quote.get('volume'))
                        oi = self._safe_float(put_quote.get('open_interest'))
                        last_price = self._safe_float(put_quote.get('last_price'))
                        bid = self._safe_float(put_quote.get('bid'))
                        ask = self._safe_float(put_quote.get('ask'))
                        
                        price_to_use = last_price if last_price is not None and last_price > 0 else put_quote.get('mid')
                        if price_to_use is None or price_to_use <= 0:
                             price_to_use = ask if ask is not None and ask > 0 else bid
                             
                        if vol is not None and oi is not None and price_to_use is not None and price_to_use > 0:
                            if oi > 0 and (vol / oi) >= min_vol_oi_ratio:
                                premium = vol * price_to_use
                                if premium >= min_premium:
                                    flow = {
                                        'symbol': strike_data.get('put'),
                                        'type': 'put',
                                        'strike': strike_price,
                                        'expiry': expiry_date,
                                        'volume': vol,
                                        'open_interest': oi,
                                        'vol_oi_ratio': round(vol / oi, 2),
                                        'price': price_to_use,
                                        'premium': round(premium, 2),
                                        'bid': bid,
                                        'ask': ask,
                                        'description': f"PUT Vol/OI: {round(vol / oi, 2)} Prem: ${premium:,.0f}"
                                    }
                                    unusual_flows.append(flow)
                                    logger.debug(f"Unusual PUT detected: {flow}")

        except Exception as e:
            logger.error(f"[_find_unusual_flows] Error analyzing {ticker}: {e}", exc_info=True)

        logger.info(f"[_find_unusual_flows] Analysis complete for {ticker}. Found {len(unusual_flows)} unusual flows.")
        return unusual_flows
        
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert a value to float, returning None on failure."""
        if value is None: 
            return None
        try:
            # Handle potential string representations of numbers
            if isinstance(value, str):
                # Remove commas if present (e.g., "1,234.56")
                value = value.replace(',', '')
            return float(value)
        except (ValueError, TypeError):
            return None

    async def _get_ohlc_with_fallback(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get OHLC data with multiple fallback mechanisms"""
        # Attempt to retrieve from cache first
        cache_key = f"ohlc_{ticker}"
        cache_result = self._data_cache.get(cache_key)

        # Check if cache_result is not None before unpacking
        if cache_result is not None:
            timestamp, cached_data = cache_result
            if timestamp and (datetime.now() - timestamp) < timedelta(minutes=self.config.CACHE_EXPIRY_MINUTES):
                logger.info(f"Using cached OHLC data for {ticker}")
                perf_logger.info(f"Cache hit for OHLC: {ticker}")
                return cached_data
            else:
                logger.info(f"Cached OHLC data for {ticker} expired or invalid timestamp.")
        else:
            logger.info(f"No valid cache entry for OHLC: {ticker}")

        # --- Main Fetching Logic --- 
        try: 
            fetched_data = None
            error_messages = []

            # Prioritize Tastytrade
            if self.data_fetcher: # Check if fetcher exists
                try:
                    logger.info(f"Attempting to fetch OHLC for {ticker} via Tastytrade...")
                    # Make sure the get_ohlc method is awaited if it's async
                    # If get_ohlc is defined async def:
                    fetched_data = await self.data_fetcher.get_ohlc(ticker)
                    if fetched_data is not None and not fetched_data.empty:
                        logger.info(f"Successfully fetched OHLC for {ticker} via Tastytrade.")
                        perf_logger.info(f"Data fetched for OHLC: {ticker} (Tastytrade)")
                        self._data_cache[cache_key] = fetched_data # Use dict assignment
                        return fetched_data # Return DataFrame
                    else:
                        logger.warning(f"Tastytrade fetch returned empty/None for {ticker}")
                        error_messages.append("Tastytrade: No data")
                except Exception as e:
                    logger.error(f"Error fetching OHLC from Tastytrade for {ticker}: {e}")
                    error_messages.append(f"Tastytrade: {e}")
            else:
                logger.error("No data fetcher initialized.")
                error_messages.append("Core: No data fetcher")

            # --- Fallback Logic (Polygon, yfinance, etc.) ---
            # Consider simplifying or removing this if Tastytrade is sufficient and stable
            # Placeholder for existing fallback logic if needed...
            # ... (rest of the fallback attempts using Polygon, yfinance would go here)


            # If all attempts fail, try generating synthetic data or return None
            if fetched_data is None or fetched_data.empty:
                logger.warning(f"All data sources failed for OHLC {ticker}. Errors: {error_messages}")
                # Optionally generate synthetic data
                # fetched_data = self._generate_synthetic_ohlc(ticker)
                # if fetched_data is not None:
                #     logger.info(f"Generated synthetic OHLC data for {ticker}")
                #     return fetched_data
                    
                logger.error(f"Failed to retrieve OHLC data for {ticker} from any source.")
                return None # Explicitly return None if all fail
                    
        except Exception as e: 
            logger.exception(f"Unexpected error in _get_ohlc_with_fallback for {ticker}: {e}")
            return None

    async def _get_options_with_fallback(self, ticker: str) -> Optional[Dict]:
        """Get options data with multiple fallback mechanisms and enrich with quotes."""
        cache_key = f"options_{ticker}"
        
        # --- Debug Logging Start ---
        # logger.debug(f"[_get_options_with_fallback] Accessing cache for key: {cache_key}")
        # logger.debug(f"[_get_options_with_fallback] Type of self: {type(self)}")
        # has_cache_manager = hasattr(self, 'cache_manager')
        # logger.debug(f"[_get_options_with_fallback] Does self have cache_manager? {has_cache_manager}")
        # if has_cache_manager:
        #     logger.debug(f"[_get_options_with_fallback] Type of self.cache_manager: {type(self.cache_manager)}")
        # else:
        #     logger.warning("[_get_options_with_fallback] self.cache_manager attribute NOT FOUND!")
        # --- Debug Logging End ---
        
        cached_data = self.cache_manager.get(cache_key)
        if cached_data:
            logger.info(f"Using cached Options data for {ticker}")

        fetched_data = None
        source_used = "None"
        errors = []

        # 1. Try Tastytrade first
        if self.tasty_fetcher and self.tasty_fetcher.session:
            logger.info(f"Attempting to fetch Options for {ticker} via Tastytrade...")
            try:
                # Fetch the basic chain structure first
                chain_data = await self.tasty_fetcher.get_option_chain(ticker)
                if chain_data and chain_data.get('expirations'):
                    logger.info(f"Successfully fetched base options chain for {ticker} via Tastytrade.")
                    
                    # Extract all option symbols from the chain data
                    all_symbols = [] 
                    for exp_data in chain_data.get('expirations', []):
                        for strike_data in exp_data.get('strikes', []):
                            if strike_data.get('call'):
                                all_symbols.append(strike_data['call'])
                            if strike_data.get('put'):
                                all_symbols.append(strike_data['put'])
                    
                    if all_symbols:
                        # Fetch the quotes for these symbols
                        quotes_data = await self.tasty_fetcher.get_options_quotes(all_symbols)
                        
                        if quotes_data:
                            logger.info(f"Successfully fetched {len(quotes_data)} quotes for {ticker} options.")
                            # Enrich the original chain_data with quotes
                            for exp_data in chain_data.get('expirations', []):
                                for strike_data in exp_data.get('strikes', []):
                                    call_symbol = strike_data.get('call')
                                    put_symbol = strike_data.get('put')
                                    if call_symbol and call_symbol in quotes_data:
                                        strike_data['call_quote'] = quotes_data[call_symbol]
                                    if put_symbol and put_symbol in quotes_data:
                                        strike_data['put_quote'] = quotes_data[put_symbol]
                            
                            fetched_data = chain_data # Now enriched
                            source_used = "Tastytrade"
                        else:
                            logger.warning(f"Fetched base chain for {ticker} but failed to get quotes.")
                            # Decide: proceed with only chain structure or consider it a failure?
                            # For now, let's treat it as a partial success but log warning.
                            fetched_data = chain_data # Return structure only
                            source_used = "Tastytrade (Structure Only)"
                            errors.append("Tastytrade: Failed to fetch quotes")
                    else:
                        logger.warning(f"No option symbols found in the chain data for {ticker}.")
                        errors.append("Tastytrade: No symbols in chain")
                else:
                    logger.warning(f"Tastytrade options chain fetch returned empty/None for {ticker}")
                    errors.append("Tastytrade: Empty chain")
            except Exception as e:
                logger.error(f"Error fetching Tastytrade options for {ticker}: {e}", exc_info=True)
                errors.append(f"Tastytrade: {e}")
        else:
             errors.append("Tastytrade: Not initialized or no session")
             

        # TODO: Add fallback logic for other providers (e.g., Polygon, YFinance) here if needed
        # Example placeholder:
        # if not fetched_data and self.polygon_client:
        #    logger.info(f"Attempting to fetch Options for {ticker} via Polygon...")
        #    try: ...

        if fetched_data:
            logger.info(f"Successfully fetched Options for {ticker} via {source_used}.")
            performance_logger.info(f"Data fetched for Options: {ticker} ({source_used})")
            self.cache_manager.set(cache_key, fetched_data, self.options_cache_ttl)
        else:
            logger.error(f"Failed to retrieve Options data for {ticker} from any source. Errors: {errors}")
            performance_logger.warning(f"Data fetch FAILED for Options: {ticker}")


        return fetched_data

    async def _fetch_and_prepare_data(self, ticker: str) -> Tuple[Optional[Union[pd.DataFrame, Dict]], Optional[Dict]]:
        pass
