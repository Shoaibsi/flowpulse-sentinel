"""
Data Fetcher module for FlowPulse Sentinel
Handles fetching data from Polygon and YFinance
"""
import logging
import os
import time
import json
import uuid
import traceback
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict
import numpy as np
from functools import wraps
import asyncio
from cachetools import TTLCache
import aiohttp
import asyncio
import json
from datetime import datetime, timedelta
from tastytrade.session import Session
from tastytrade.streamer import DXLinkStreamer
from tastytrade.watchlists import Watchlist
from tastytrade.instruments import get_option_chain
from tastytrade.account import Account

logger = logging.getLogger(__name__)

# DataFrame serialization helpers
def dataframe_to_json(df):
    """Convert DataFrame to JSON-serializable format"""
    if df is None:
        return None
    return json.loads(df.to_json(orient='records'))

def json_to_dataframe(json_str):
    """Convert JSON back to DataFrame"""
    if json_str is None:
        return None
    return pd.read_json(StringIO(json.dumps(json_str)))

# Rate limiting decorator
def rate_limit(calls_per_minute=5):
    """Decorator to implement rate limiting for API calls"""
    min_interval = 60.0 / calls_per_minute
    last_call_time = {}
    locks = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create unique key for each function
            key = f"{func.__name__}_{id(self)}"
            
            # Initialize lock if needed
            if key not in locks:
                locks[key] = threading.Lock()
                
            with locks[key]:
                # Check if we need to wait
                current_time = time.time()
                if key in last_call_time:
                    elapsed = current_time - last_call_time[key]
                    if elapsed < min_interval:
                        sleep_time = min_interval - elapsed
                        logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                        time.sleep(sleep_time)
                
                # Update last call time
                last_call_time[key] = current_time
                
                # Call the original function
                return func(self, *args, **kwargs)
        return wrapper
    return decorator

class PolygonDataFetcher:
    """Step 11: Polygon WebSocket client"""
    def __init__(self, api_key: str, calls_per_minute: int = 5):
        self.client = RESTClient(api_key)
        self.calls_per_minute = calls_per_minute
        logger.info(f"Polygon data fetcher initialized with {calls_per_minute} calls/minute rate limit")
        
    @rate_limit(calls_per_minute=5)
    def get_ohlc(self, ticker: str, days: int = 60) -> Optional[pd.DataFrame]:
        """Get OHLCV data for a ticker"""
        try:
            # Calculate start and end dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates for Polygon API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Fetch data from Polygon
            aggs = self.client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start_str,
                to=end_str
            )
            
            if not aggs:
                logger.warning(f"No data returned from Polygon for {ticker}")
                return None
                
            # Convert to DataFrame
            data = []
            for agg in aggs:
                data.append({
                    'date': pd.to_datetime(agg.timestamp, unit='ms'),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume
                })
                
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLC data from Polygon for {ticker}: {e}")
            return None
            
    @rate_limit(calls_per_minute=5)
    def get_options_chain(self, ticker: str) -> List[Dict]:
        """Get options chain for a ticker using the free tier endpoints"""
        # Check if client is initialized
        if self.client is None:
            logger.error("Polygon client not initialized, cannot fetch options chain")
            return []
            
        try:
            # Get current date
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Get options contracts for the ticker
            options_contracts = []
            
            # For free tier, we can use the reference/options/contracts endpoint
            try:
                # Get options expirations for the underlying ticker
                expirations = self.get_options_expirations(ticker)
                
                if not expirations:
                    logger.warning(f"No options expirations found for {ticker} from Polygon")
                    return []
                
                # Process each expiration
                for expiry in expirations[:3]:  # Limit to 3 expirations
                    try:
                        # Get options contracts for this expiration
                        contracts = self.client.list_options_contracts(
                            underlying_ticker=ticker,
                            expiration_date=expiry,
                            limit=100  # Limit to 100 contracts
                        )
                        
                        # Process each contract
                        for contract in contracts:
                            try:
                                # Extract contract details
                                contract_ticker = contract.ticker if hasattr(contract, 'ticker') else None
                                if not contract_ticker:
                                    continue
                                    
                                # Extract contract type (call/put)
                                option_type = 'call' if 'C' in contract_ticker else 'put'
                                
                                # Extract strike price
                                strike = contract.strike_price if hasattr(contract, 'strike_price') else None
                                
                                if not strike:
                                    continue
                                
                                # Add to options contracts
                                options_contracts.append({
                                    'ticker': ticker,
                                    'expiry': expiry,
                                    'strike': float(strike),
                                    'option_type': option_type,
                                    'volume': 0,  # Not available in free tier
                                    'open_interest': 0,  # Not available in free tier
                                    'iv': 0,  # Not available in free tier
                                    'last_price': 0  # Not available in free tier
                                })
                            except Exception as e:
                                logger.warning(f"Error processing contract: {e}")
                                continue
                    except Exception as e:
                        logger.warning(f"Error fetching contracts for expiry {expiry}: {e}")
                        continue
                
            except Exception as e:
                logger.warning(f"Error fetching options contracts for {ticker} from Polygon: {e}")
            
            # If we got contracts, return them
            if options_contracts:
                logger.info(f"Successfully fetched {len(options_contracts)} options contracts for {ticker} from Polygon")
                return options_contracts
            else:
                logger.warning(f"No options contracts found for {ticker} from Polygon, will try YFinance")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching options chain from Polygon for {ticker}: {e}")
            return []
    
    @rate_limit(calls_per_minute=5)
    def get_options_expirations(self, ticker: str) -> List[str]:
        """Get options expirations for a ticker"""
        try:
            # Get current date
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Use the newer Polygon API syntax
            contracts = self.client.list_options_contracts(
                underlying_ticker=ticker,
                contract_type='call',  # Just use calls to get expirations
                expiration_date_gte=today,
                limit=1000
            )
            
            # Extract unique expiration dates
            expirations = set()
            for contract in contracts:
                if hasattr(contract, 'expiration_date'):
                    expirations.add(contract.expiration_date)
            
            return sorted(list(expirations))
        except Exception as e:
            logger.error(f"Error fetching options expirations from Polygon for {ticker}: {e}")
            return []

class YFinanceDataFetcher:
    """Data fetcher for YFinance API"""
    
    def __init__(self, request_delay=2.0):
        """Initialize YFinance data fetcher"""
        self.request_delay = request_delay
        self.last_request = 0
        
        # Set up a requests session with headers to avoid rate limiting
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        })
        
        # Get cookies from Yahoo Finance
        try:
            response = self.session.get('https://finance.yahoo.com')
            if response.status_code == 200:
                self.session.cookies.update(response.cookies)
        except Exception as e:
            logger.warning(f"Failed to get Yahoo Finance cookies: {e}")
        
        logger.info("YFinance data fetcher initialized")
        
    def _rate_limit(self):
        now = time.time()
        elapsed = now - self.last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request = time.time()
        
    def test_connection(self) -> bool:
        """Test connection to YFinance API"""
        try:
            # Try fetching AAPL data as a test
            import yfinance as yf
            test_ticker = yf.Ticker("AAPL", session=self.session)
            test_data = test_ticker.history(period="1d")
            return len(test_data) > 0
        except Exception as e:
            logger.error(f"Error testing YFinance connection: {e}")
            return False
    
    def get_ohlc(self, ticker: str, period: str = "80d") -> pd.DataFrame:
        """Alias for get_ohlc_data for backwards compatibility"""
        return self.get_ohlc_data(ticker, period)
    
    def get_ohlc_data(self, ticker: str, period: str = "80d") -> pd.DataFrame:
        """
        Get OHLC data for a ticker
        
        Args:
            ticker: Ticker symbol
            period: Time period for data (default: 80d)
            
        Returns:
            DataFrame with OHLC data
        """
        try:
            import yfinance as yf
            
            self._rate_limit()
            
            max_attempts = 3
            attempt = 1
            while attempt <= max_attempts:
                logger.info(f"Fetching OHLC data from YFinance for {ticker}, attempt {attempt}/{max_attempts}")
                
                try:
                    # Create a new Ticker instance each time with our session
                    stock = yf.Ticker(ticker, session=self.session)
                    
                    # Try direct URL access first
                    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=80d"
                    response = self.session.get(url)
                    if response.status_code == 200:
                        chart_data = response.json()
                        if 'chart' in chart_data and 'result' in chart_data['chart']:
                            result = chart_data['chart']['result'][0]
                            timestamps = result['timestamp']
                            quote = result['indicators']['quote'][0]
                            
                            data = pd.DataFrame({
                                'timestamp': pd.to_datetime(timestamps, unit='s'),
                                'open': quote.get('open', []),
                                'high': quote.get('high', []),
                                'low': quote.get('low', []),
                                'close': quote.get('close', []),
                                'volume': quote.get('volume', [])
                            })
                            
                            # Add symbol column
                            data['symbol'] = ticker
                            
                            return data
                    
                    # Fallback to yfinance
                    data = stock.history(period=period)
                    
                    if len(data) > 0:
                        # Add symbol column
                        data['symbol'] = ticker
                        
                        # Reset index to make date a column
                        data = data.reset_index()
                        
                        # Rename columns to match our format
                        data = data.rename(columns={
                            'Date': 'timestamp',
                            'Open': 'open',
                            'High': 'high',
                            'Low': 'low',
                            'Close': 'close',
                            'Volume': 'volume'
                        })
                        
                        # Ensure timestamp is datetime
                        data['timestamp'] = pd.to_datetime(data['timestamp'])
                        
                        return data
                        
                    logger.warning(f"No data returned from YFinance for {ticker}")
                    
                except Exception as e:
                    logger.warning(f"YFinance attempt {attempt}/{max_attempts} failed: {e}")
                    
                if attempt < max_attempts:
                    logger.info(f"Retrying ({attempt}/{max_attempts})...")
                    time.sleep(5 * attempt)  # Longer backoff
                    
                attempt += 1
                
            logger.error(f"All YFinance OHLC attempts failed for {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting OHLC data from YFinance: {e}")
            return None
    
    def get_options_chain(self, ticker: str) -> pd.DataFrame:
        """
        Get options chain for a ticker
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            DataFrame with options chain data
        """
        try:
            import yfinance as yf
            
            self._rate_limit()
            
            max_attempts = 3
            attempt = 1
            while attempt <= max_attempts:
                logger.info(f"Fetching options data from YFinance for {ticker}, attempt {attempt}/{max_attempts}")
                
                try:
                    # Create a new Ticker instance each time with our session
                    stock = yf.Ticker(ticker, session=self.session)
                    
                    # Try direct URL access first
                    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}/options"
                    response = self.session.get(url)
                    if response.status_code == 200:
                        options_data = response.json()
                        if 'optionChain' in options_data and 'result' in options_data['optionChain']:
                            result = options_data['optionChain']['result'][0]
                            if 'options' in result and len(result['options']) > 0:
                                all_options = []
                                for option in result['options']:
                                    if 'calls' in option:
                                        calls = pd.DataFrame(option['calls'])
                                        calls['option_type'] = 'call'
                                        all_options.append(calls)
                                    if 'puts' in option:
                                        puts = pd.DataFrame(option['puts'])
                                        puts['option_type'] = 'put'
                                        all_options.append(puts)
                                
                                if all_options:
                                    df = pd.concat(all_options, ignore_index=True)
                                    
                                    # Rename columns to match our format
                                    df = df.rename(columns={
                                        'strike': 'strike',
                                        'lastPrice': 'last',
                                        'bid': 'bid',
                                        'ask': 'ask',
                                        'volume': 'volume',
                                        'openInterest': 'open_interest',
                                        'impliedVolatility': 'iv',
                                        'contractSymbol': 'option_symbol'
                                    })
                                    
                                    # Calculate missing Greeks if not present
                                    if 'delta' not in df.columns:
                                        df['delta'] = None
                                    if 'gamma' not in df.columns:
                                        df['gamma'] = None
                                    if 'theta' not in df.columns:
                                        df['theta'] = None
                                    if 'vega' not in df.columns:
                                        df['vega'] = None
                                    if 'rho' not in df.columns:
                                        df['rho'] = None
                                    
                                    # Convert expiration to datetime
                                    df['expiration'] = pd.to_datetime(df['expiration'])
                                    
                                    # Set proper data types
                                    numeric_cols = ['strike', 'bid', 'ask', 'last', 'volume', 'open_interest', 
                                                'iv', 'delta', 'gamma', 'theta', 'vega', 'rho']
                                    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                                    
                                    return df
                    
                    # Fallback to yfinance
                    expirations = stock.options
                    
                    if not expirations:
                        logger.warning(f"No options expirations found for {ticker}")
                        return None
                    
                    # Get options data for each expiration
                    all_options = []
                    for date in expirations:
                        try:
                            # Get option chain for this expiration
                            chain = stock.option_chain(date)
                            
                            # Process calls
                            calls = chain.calls.copy()
                            calls['option_type'] = 'call'
                            calls['expiration'] = date
                            
                            # Process puts
                            puts = chain.puts.copy()
                            puts['option_type'] = 'put'
                            puts['expiration'] = date
                            
                            # Combine calls and puts
                            options = pd.concat([calls, puts])
                            
                            # Add symbol
                            options['symbol'] = ticker
                            
                            all_options.append(options)
                            
                        except Exception as e:
                            logger.warning(f"Error getting options for {ticker} expiration {date}: {e}")
                            continue
                    
                    if all_options:
                        # Combine all expirations
                        df = pd.concat(all_options, ignore_index=True)
                        
                        # Rename columns to match our format
                        df = df.rename(columns={
                            'strike': 'strike',
                            'lastPrice': 'last',
                            'bid': 'bid',
                            'ask': 'ask',
                            'volume': 'volume',
                            'openInterest': 'open_interest',
                            'impliedVolatility': 'iv',
                            'contractSymbol': 'option_symbol'
                        })
                        
                        # Calculate missing Greeks if not present
                        if 'delta' not in df.columns:
                            df['delta'] = None
                        if 'gamma' not in df.columns:
                            df['gamma'] = None
                        if 'theta' not in df.columns:
                            df['theta'] = None
                        if 'vega' not in df.columns:
                            df['vega'] = None
                        if 'rho' not in df.columns:
                            df['rho'] = None
                        
                        # Convert expiration to datetime
                        df['expiration'] = pd.to_datetime(df['expiration'])
                        
                        # Set proper data types
                        numeric_cols = ['strike', 'bid', 'ask', 'last', 'volume', 'open_interest', 
                                     'iv', 'delta', 'gamma', 'theta', 'vega', 'rho']
                        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                        
                        return df
                        
                except Exception as e:
                    logger.warning(f"YFinance attempt {attempt}/{max_attempts} failed: {e}")
                    
                if attempt < max_attempts:
                    logger.info(f"Retrying ({attempt}/{max_attempts})...")
                    time.sleep(8 * attempt)  # Longer backoff
                    
                attempt += 1
                
            logger.error(f"All YFinance options attempts failed for {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting options chain from YFinance: {e}")
            return None

class TastytradeDataFetcher:
    def __init__(self, username: str, password: str):
        """Initialize TastyTrade data fetcher. Session is initialized separately."""
        self.username = username
        self.password = password
        self.session: Optional[Session] = None
        self.streamer: Optional[DXLinkStreamer] = None
        self.running = False
        self.logger = logging.getLogger(__name__)

    async def _initialize_session(self):
        """Initialize TastyTrade session using the Session class."""
        self.logger.info("Attempting to initialize TastyTrade session...")
        try:
            # Use the Session class for session handling
            self.session = Session(self.username, self.password) # Instantiation likely handles login implicitly in v9.11
            self.logger.info("TastyTrade Session object created (login likely implicit).")
            return True
        except AttributeError as ae:
            # This specific error is now handled by removing the call
            self.logger.error(f"Unexpected AttributeError during session initialization: {ae}")
            return False
        except Exception as e:
            self.logger.exception(f"Failed to initialize TastyTrade session: {e}")
            return False

    async def _initialize_streamer(self):
        """Initializes the DXLinkStreamer for real-time data."""
        if not self.session:
            self.logger.error("Session not initialized, cannot initialize streamer.")
            return False
        
        try:
            token_response = await self.session.get_streamer_token()
            token = token_response['data']['token']
            dxlink_url = token_response['data']['dxlink-url']
            # Assuming DXFeedStreamer takes Session object directly or adjust as needed
            self.streamer = DXLinkStreamer(self.session, token, dxlink_url) 
            self.logger.info("TastyTrade DXFeed streamer ready.")
            return True
        except AttributeError:
             self.logger.warning("Session object does not have 'get_streamer_token'. Streamer not initialized.")
             self.streamer = None
             return False
        except Exception as streamer_ex:
             self.logger.error(f"Error initializing streamer: {streamer_ex}", exc_info=True)
             self.streamer = None
             return False

    async def get_ohlc(self, ticker: str, interval: str = 'day', period: str = '1y') -> Optional[pd.DataFrame]:
        """Fetch OHLC data for a ticker. NOTE: Historical data query via direct API call is not supported in tastytrade v9.11 SDK as initially assumed. Use a streamer or alternative library."""
        if not self.session:
            self.logger.error("Session not initialized. Cannot fetch OHLC data.")
            return None

        self.logger.warning(f"Direct historical OHLC fetch for {ticker} not implemented for tastytrade v9.11 SDK. Returning None.")
        return None

    async def get_option_chain(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get the options chain for a ticker."""
        if not self.session:
            self.logger.error("Session not initialized. Cannot fetch options chain.")
            return None

        try:
            self.logger.info(f"Fetching options chain for {ticker}...")
            # get_option_chain in v9.11 is NOT async, remove await
            chain_data = get_option_chain(self.session, ticker)

            if not chain_data:
                self.logger.warning(f"No options chain data returned for {ticker}.")
                return None

            # Basic processing/validation (structure might vary based on actual output)
            processed_chain = {
                'symbol': ticker,
                'expirations': [],
                'chain': {}
            }
            
            # Example processing if chain_data is a defaultdict(list) of options
            if isinstance(chain_data, defaultdict):
                expirations = sorted(list(chain_data.keys()))
                processed_chain['expirations'] = expirations
                processed_chain['chain'] = dict(chain_data) # Convert defaultdict back to dict for easier handling/JSON
            else:
                # Handle other potential structures or log a warning
                self.logger.warning(f"Unexpected structure for options chain data for {ticker}: {type(chain_data)}")
                # Attempt to return raw data if structure is unknown
                return chain_data # Return raw data as fallback

            self.logger.info(f"Successfully fetched options chain for {ticker} with {len(processed_chain['expirations'])} expirations.")
            return processed_chain

        except TypeError as te:
             # This error (awaiting non-awaitable) should now be resolved
             self.logger.exception(f"TypeError fetching/processing options chain for {ticker}: {te}") # Keep logging for unexpected issues
             return None
        except Exception as e:
            self.logger.exception(f"Error fetching/processing options chain for {ticker}: {e}")
            return None

    async def get_market_metrics(self, symbols: List[str]) -> Optional[Dict[str, Any]]:
        """Fetch market metrics for a list of symbols."""
        if not self.session:
            self.logger.error("Session not initialized. Cannot fetch market metrics.")
            return None
        
        try:
            # Assuming there's a function in market_data or similar for metrics
            # Replace 'get_market_metrics_function' with the actual function name from v9.11
            # metrics_data = await get_market_metrics_function(self.session, symbols)
            # self.logger.info(f"Successfully fetched market metrics for {len(symbols)} symbols.")
            # return metrics_data
            self.logger.warning("get_market_metrics is not implemented for v9.11 yet.")
            return None # Placeholder
        except Exception as e:
            self.logger.exception(f"Error fetching market metrics: {e}")
            return None

    async def get_options_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch market metrics (quotes) for a list of option symbols."""
        if not self.session:
            self.logger.error("Session not initialized. Cannot fetch options quotes.")
            return {}
        if not symbols:
            self.logger.warning("get_options_quotes called with empty symbol list.")
            return {}

        # Tastytrade API has limits on URL length, chunk the requests if necessary
        # A safe limit might be ~100 symbols per request based on typical symbol lengths
        chunk_size = 100 
        all_quotes = {}
        
        self.logger.info(f"Fetching quotes for {len(symbols)} option symbols...")

        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            symbols_str = ",".join(chunk)
            url = f"{self.api_url}/market-metrics?symbols={symbols_str}"

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, headers=self.session.get_request_headers())
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                    
                response_data = response.json()
                metrics_items = response_data.get('data', {}).get('items', [])

                if not metrics_items:
                    self.logger.warning(f"No market metrics returned for symbols chunk: {symbols_str[:50]}...")
                    continue

                for item in metrics_items:
                    symbol = item.get('symbol')
                    if symbol:
                        # Extract relevant quote fields
                        quote = {
                            'volume': item.get('volume'),
                            'open_interest': item.get('open-interest'),
                            'last_price': item.get('price'), # 'price' usually means last trade price
                            'bid': item.get('bid-price'),
                            'ask': item.get('ask-price'),
                            'mid': self._calculate_mid_price(item.get('bid-price'), item.get('ask-price')),
                            'implied_volatility': item.get('implied-volatility')
                            # Add other fields like greeks if needed
                        }
                        all_quotes[symbol] = quote
                
                self.logger.debug(f"Fetched quotes for chunk {i//chunk_size + 1} ({len(metrics_items)} items)")
                await asyncio.sleep(0.1) # Small delay to avoid hitting rate limits if many chunks

            except httpx.HTTPStatusError as e:
                self.logger.error(f"HTTP error fetching options quotes for chunk {symbols_str[:50]}...: {e.response.status_code} - {e.response.text}")
            except httpx.RequestError as e:
                self.logger.error(f"Request error fetching options quotes for chunk {symbols_str[:50]}...: {e}")
            except json.JSONDecodeError:
                self.logger.error(f"Error decoding JSON response for options quotes chunk {symbols_str[:50]}...")
            except Exception as e:
                self.logger.error(f"Unexpected error fetching options quotes chunk {symbols_str[:50]}...: {e}", exc_info=True)

        self.logger.info(f"Finished fetching quotes. Got data for {len(all_quotes)} out of {len(symbols)} symbols.")
        return all_quotes

    def _calculate_mid_price(self, bid_str: Optional[str], ask_str: Optional[str]) -> Optional[float]:
        """Safely calculate the mid-price from string inputs."""
        try:
            bid = float(bid_str) if bid_str else None
            ask = float(ask_str) if ask_str else None
            if bid is not None and ask is not None and bid > 0 and ask > 0:
                 # Basic validation: ask should ideally be >= bid
                 if ask < bid:
                     self.logger.debug(f"Ask price {ask} is less than bid price {bid}. Using ask as mid.")
                     return ask # Or handle this case differently? Maybe return None?
                 return round((bid + ask) / 2.0, 4) # Round to typical price precision
            elif ask is not None and ask > 0:
                return ask # Return ask if bid is invalid/zero
            elif bid is not None and bid > 0:
                 return bid # Return bid if ask is invalid/zero
            else:
                return None
        except (ValueError, TypeError) as e:
            self.logger.debug(f"Could not calculate mid price from bid='{bid_str}', ask='{ask_str}': {e}")
            return None

    async def close(self):
        """Cleanly close the session and streamer."""
        if self.streamer:
            await self.streamer.disconnect()
            self.logger.info("Streamer disconnected.")
        # No explicit session close/logout seems available or needed in v9.11 based on current findings
        self.logger.info("TastytradeDataFetcher closed.")
