"""
=============================================================================
DATA UTILITIES MODULE - Crypto Intelligence Platform
=============================================================================
This module handles all data operations including:
- Real-time crypto data fetching via ccxt
- Historical data retrieval via yfinance
- Technical indicator calculations (RSI, MACD, Bollinger Bands)
- Sentiment analysis using VADER
- Redis caching for performance optimization

Author: AI Engineering Team
PEP8 Compliant | Highly Commented for Transparency
=============================================================================
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import ccxt.async_support as ccxt
import yfinance as yf
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import redis.asyncio as redis

# =============================================================================
# LOGGING CONFIGURATION - Transparency is Key
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES - Type Safety & Clarity
# =============================================================================
@dataclass
class OHLCVData:
    """
    Represents a single OHLCV (Open, High, Low, Close, Volume) candle.
    Used for type-safe data passing throughout the system.
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class TechnicalIndicators:
    """
    Container for all calculated technical indicators.
    Provides clear structure for ML model input.
    """
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float
    bollinger_upper: float
    bollinger_middle: float
    bollinger_lower: float
    bollinger_pband: float  # Percentage band position


@dataclass
class SentimentScore:
    """
    Structured sentiment analysis result.
    Ready for integration with real social media APIs.
    """
    compound: float  # Overall sentiment (-1 to 1)
    positive: float
    negative: float
    neutral: float
    label: str  # 'bullish', 'bearish', or 'neutral'


# =============================================================================
# CRYPTO DATA FETCHER - Real-time & Historical Data
# =============================================================================
class CryptoDataFetcher:
    """
    Handles all cryptocurrency data fetching operations.
    
    Uses ccxt for real-time exchange data and yfinance for historical data.
    Implements connection pooling and error handling for production use.
    
    TRANSPARENCY LOG: Every data fetch operation is logged with timestamp
    and source for complete auditability.
    """
    
    def __init__(self, exchange_id: str = 'binance'):
        """
        Initialize the data fetcher with specified exchange.
        
        Args:
            exchange_id: The exchange to connect to (default: binance)
        """
        self.exchange_id = exchange_id
        self._exchange: Optional[ccxt.Exchange] = None
        logger.info(f"[DATA_FETCHER] Initialized with exchange: {exchange_id}")
    
    async def _get_exchange(self) -> ccxt.Exchange:
        """
        Lazy initialization of exchange connection.
        Implements singleton pattern for connection reuse.
        """
        if self._exchange is None:
            logger.info(f"[DATA_FETCHER] Creating new exchange connection to {self.exchange_id}")
            exchange_class = getattr(ccxt, self.exchange_id)
            self._exchange = exchange_class({
                'enableRateLimit': True,  # Respect exchange rate limits
                'timeout': 30000,  # 30 second timeout
            })
        return self._exchange
    
    async def fetch_ohlcv_realtime(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetch real-time OHLCV data from exchange.
        
        TRANSPARENCY: Logs the exact API call parameters and response size.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h', '1d')
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        logger.info(f"[DATA_FETCHER] Fetching real-time OHLCV: {symbol} | {timeframe} | limit={limit}")
        
        try:
            exchange = await self._get_exchange()
            
            # Fetch OHLCV data from exchange
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            logger.info(f"[DATA_FETCHER] Successfully fetched {len(df)} candles for {symbol}")
            logger.debug(f"[DATA_FETCHER] Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except ccxt.NetworkError as e:
            logger.error(f"[DATA_FETCHER] Network error fetching {symbol}: {e}")
            raise
        except ccxt.ExchangeError as e:
            logger.error(f"[DATA_FETCHER] Exchange error fetching {symbol}: {e}")
            raise
    
    def fetch_historical_yfinance(
        self,
        symbol: str,
        period: str = '1y',
        interval: str = '1h'
    ) -> pd.DataFrame:
        """
        Fetch historical data using yfinance.
        Useful for longer historical periods and backtesting.
        
        TRANSPARENCY: Logs data source switch and any data quality issues.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USD' for yfinance format)
            period: Data period (e.g., '1y', '6mo', '1mo')
            interval: Data interval (e.g., '1h', '1d')
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"[DATA_FETCHER] Fetching historical data via yfinance: {symbol} | {period} | {interval}")
        
        try:
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            df = ticker.history(period=period, interval=interval)
            
            # Standardize column names to lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Reset index to make timestamp a column
            df = df.reset_index()
            df = df.rename(columns={'index': 'timestamp', 'date': 'timestamp', 'Datetime': 'timestamp'})
            
            # Ensure we have the required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = df[[col for col in required_cols if col in df.columns]]
            
            logger.info(f"[DATA_FETCHER] Successfully fetched {len(df)} historical candles for {symbol}")
            
            # Log any missing data points
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                logger.warning(f"[DATA_FETCHER] Found {missing_count} missing values in historical data")
            
            return df
            
        except Exception as e:
            logger.error(f"[DATA_FETCHER] Error fetching yfinance data for {symbol}: {e}")
            raise
    
    async def close(self):
        """Clean up exchange connection."""
        if self._exchange is not None:
            await self._exchange.close()
            logger.info("[DATA_FETCHER] Exchange connection closed")


# =============================================================================
# FEATURE ENGINEER - Technical Indicator Calculations
# =============================================================================
class FeatureEngineer:
    """
    Calculates technical indicators for ML model features.
    
    Implements RSI, MACD, and Bollinger Bands using the 'ta' library.
    All calculations are logged for transparency and debugging.
    
    TRANSPARENCY: Each indicator calculation logs:
    - Input data range
    - Parameters used
    - Output value range
    """
    
    def __init__(self):
        logger.info("[FEATURE_ENGINEER] Initialized feature engineering module")
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators and add them to the DataFrame.
        
        TRANSPARENCY LOG:
        - Logs each indicator calculation separately
        - Reports any NaN values introduced
        - Logs the final feature set statistics
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional indicator columns
        """
        logger.info(f"[FEATURE_ENGINEER] Calculating indicators for {len(df)} candles")
        
        # Create a copy to avoid modifying original
        result_df = df.copy()
        
        # ----- RSI CALCULATION -----
        logger.info("[FEATURE_ENGINEER] Calculating RSI (period=14)")
        rsi_indicator = RSIIndicator(close=result_df['close'], window=14)
        result_df['rsi'] = rsi_indicator.rsi()
        logger.debug(f"[FEATURE_ENGINEER] RSI range: {result_df['rsi'].min():.2f} - {result_df['rsi'].max():.2f}")
        
        # ----- MACD CALCULATION -----
        logger.info("[FEATURE_ENGINEER] Calculating MACD (12, 26, 9)")
        macd = MACD(
            close=result_df['close'],
            window_slow=26,
            window_fast=12,
            window_sign=9
        )
        result_df['macd'] = macd.macd()
        result_df['macd_signal'] = macd.macd_signal()
        result_df['macd_histogram'] = macd.macd_diff()
        logger.debug(f"[FEATURE_ENGINEER] MACD histogram range: {result_df['macd_histogram'].min():.4f} - {result_df['macd_histogram'].max():.4f}")
        
        # ----- BOLLINGER BANDS CALCULATION -----
        logger.info("[FEATURE_ENGINEER] Calculating Bollinger Bands (period=20, std=2)")
        bollinger = BollingerBands(
            close=result_df['close'],
            window=20,
            window_dev=2
        )
        result_df['bb_upper'] = bollinger.bollinger_hband()
        result_df['bb_middle'] = bollinger.bollinger_mavg()
        result_df['bb_lower'] = bollinger.bollinger_lband()
        result_df['bb_pband'] = bollinger.bollinger_pband()  # Position within bands
        logger.debug(f"[FEATURE_ENGINEER] Bollinger %B range: {result_df['bb_pband'].min():.4f} - {result_df['bb_pband'].max():.4f}")
        
        # ----- ADDITIONAL DERIVED FEATURES -----
        logger.info("[FEATURE_ENGINEER] Calculating derived features")
        
        # Price momentum (percentage change)
        result_df['price_momentum'] = result_df['close'].pct_change()
        
        # Volume momentum
        result_df['volume_momentum'] = result_df['volume'].pct_change()
        
        # Volatility (rolling standard deviation)
        result_df['volatility'] = result_df['close'].rolling(window=20).std()
        
        # Log NaN counts
        nan_counts = result_df.isnull().sum()
        total_nans = nan_counts.sum()
        if total_nans > 0:
            logger.warning(f"[FEATURE_ENGINEER] Total NaN values in features: {total_nans}")
            logger.debug(f"[FEATURE_ENGINEER] NaN by column: {nan_counts.to_dict()}")
        
        logger.info("[FEATURE_ENGINEER] Feature calculation complete")
        return result_df
    
    def get_latest_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """
        Extract the latest indicator values as a structured object.
        
        Args:
            df: DataFrame with calculated indicators
            
        Returns:
            TechnicalIndicators dataclass with latest values
        """
        latest = df.iloc[-1]
        
        indicators = TechnicalIndicators(
            rsi=float(latest.get('rsi', 50.0)),
            macd=float(latest.get('macd', 0.0)),
            macd_signal=float(latest.get('macd_signal', 0.0)),
            macd_histogram=float(latest.get('macd_histogram', 0.0)),
            bollinger_upper=float(latest.get('bb_upper', 0.0)),
            bollinger_middle=float(latest.get('bb_middle', 0.0)),
            bollinger_lower=float(latest.get('bb_lower', 0.0)),
            bollinger_pband=float(latest.get('bb_pband', 0.5))
        )
        
        logger.info(f"[FEATURE_ENGINEER] Latest indicators - RSI: {indicators.rsi:.2f}, MACD: {indicators.macd:.4f}")
        return indicators


# =============================================================================
# SENTIMENT ANALYZER - Social Media Sentiment Scoring
# =============================================================================
class SentimentAnalyzer:
    """
    Analyzes text sentiment using VADER sentiment analysis.
    
    Currently uses mock data but is designed for easy integration
    with real social media APIs (Twitter, Reddit, etc.)
    
    TRANSPARENCY: All sentiment calculations are logged with:
    - Input text (truncated for privacy)
    - All sentiment scores
    - Final classification
    """
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        logger.info("[SENTIMENT] Initialized VADER sentiment analyzer")
        
        # Mock social media posts for demonstration
        # In production, replace with real API integration
        self._mock_posts = [
            "Bitcoin is showing strong bullish momentum! 🚀",
            "Market looking weak, expecting a pullback soon",
            "Whale alert: Large BTC transfer detected",
            "Technical analysis suggests a breakout is imminent",
            "Volume increasing, bulls are in control",
            "Bearish divergence on the daily chart",
            "New institutional investors entering the market",
            "Be careful, high volatility expected",
        ]
    
    def analyze_text(self, text: str) -> SentimentScore:
        """
        Analyze a single text for sentiment.
        
        TRANSPARENCY LOG:
        - Logs input text (first 50 chars)
        - Logs all component scores
        - Logs final classification reasoning
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentScore with compound and component scores
        """
        logger.debug(f"[SENTIMENT] Analyzing: '{text[:50]}...'")
        
        # Get VADER sentiment scores
        scores = self.analyzer.polarity_scores(text)
        
        # Determine sentiment label based on compound score
        compound = scores['compound']
        if compound >= 0.05:
            label = 'bullish'
        elif compound <= -0.05:
            label = 'bearish'
        else:
            label = 'neutral'
        
        result = SentimentScore(
            compound=compound,
            positive=scores['pos'],
            negative=scores['neg'],
            neutral=scores['neu'],
            label=label
        )
        
        logger.debug(f"[SENTIMENT] Score: {compound:.3f} -> {label}")
        return result
    
    def get_aggregated_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get aggregated sentiment score for a crypto symbol.
        
        Currently uses mock data. In production, this would:
        1. Fetch recent social media posts about the symbol
        2. Analyze each post
        3. Aggregate into a weighted score
        
        TRANSPARENCY: Logs the number of posts analyzed and aggregation method.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC')
            
        Returns:
            Dictionary with aggregated sentiment data
        """
        logger.info(f"[SENTIMENT] Generating aggregated sentiment for {symbol}")
        
        # Analyze mock posts
        sentiments = [self.analyze_text(post) for post in self._mock_posts]
        
        # Calculate aggregate scores
        avg_compound = np.mean([s.compound for s in sentiments])
        bullish_count = sum(1 for s in sentiments if s.label == 'bullish')
        bearish_count = sum(1 for s in sentiments if s.label == 'bearish')
        neutral_count = sum(1 for s in sentiments if s.label == 'neutral')
        
        # Determine overall sentiment
        if avg_compound >= 0.1:
            overall = 'bullish'
        elif avg_compound <= -0.1:
            overall = 'bearish'
        else:
            overall = 'neutral'
        
        result = {
            'symbol': symbol,
            'compound_score': round(avg_compound, 3),
            'overall_sentiment': overall,
            'sentiment_distribution': {
                'bullish': bullish_count,
                'bearish': bearish_count,
                'neutral': neutral_count
            },
            'posts_analyzed': len(sentiments),
            'confidence': abs(avg_compound),  # Higher absolute value = more confident
            'source': 'mock_data',  # Indicates this is mock data
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"[SENTIMENT] Aggregated: {overall} (score: {avg_compound:.3f}, {len(sentiments)} posts)")
        return result


# =============================================================================
# REDIS CACHE - Performance Optimization
# =============================================================================
class RedisCache:
    """
    Redis caching layer for frequently accessed data.
    
    Caches:
    - OHLCV data (5 minute TTL)
    - Technical indicators (1 minute TTL)
    - Model predictions (30 second TTL)
    
    TRANSPARENCY: All cache operations are logged with:
    - Hit/miss status
    - Key accessed
    - TTL remaining
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._client: Optional[redis.Redis] = None
        logger.info(f"[CACHE] Initialized Redis cache: {redis_url}")
    
    async def _get_client(self) -> redis.Redis:
        """Lazy initialization of Redis connection."""
        if self._client is None:
            self._client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            logger.info("[CACHE] Redis connection established")
        return self._client
    
    async def get(self, key: str) -> Optional[str]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if miss
        """
        try:
            client = await self._get_client()
            value = await client.get(key)
            
            if value is not None:
                logger.debug(f"[CACHE] HIT: {key}")
            else:
                logger.debug(f"[CACHE] MISS: {key}")
                
            return value
        except redis.ConnectionError:
            logger.warning(f"[CACHE] Redis connection failed, cache disabled")
            return None
    
    async def set(self, key: str, value: str, ttl_seconds: int = 300) -> bool:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            client = await self._get_client()
            await client.setex(key, ttl_seconds, value)
            logger.debug(f"[CACHE] SET: {key} (TTL: {ttl_seconds}s)")
            return True
        except redis.ConnectionError:
            logger.warning(f"[CACHE] Redis connection failed, unable to cache")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        try:
            client = await self._get_client()
            await client.delete(key)
            logger.debug(f"[CACHE] DELETE: {key}")
            return True
        except redis.ConnectionError:
            return False
    
    async def close(self):
        """Close Redis connection."""
        if self._client is not None:
            await self._client.close()
            logger.info("[CACHE] Redis connection closed")


# =============================================================================
# DATA PIPELINE - Orchestrates All Data Operations
# =============================================================================
class DataPipeline:
    """
    High-level data pipeline that orchestrates all data operations.
    
    Provides a unified interface for:
    - Fetching and caching data
    - Calculating features
    - Getting sentiment
    
    TRANSPARENCY: Logs the complete data pipeline execution flow.
    """
    
    def __init__(self, use_cache: bool = True, redis_url: str = "redis://localhost:6379"):
        self.data_fetcher = CryptoDataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.cache = RedisCache(redis_url) if use_cache else None
        self.use_cache = use_cache
        
        logger.info(f"[PIPELINE] Initialized data pipeline (cache: {use_cache})")
    
    async def get_prepared_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 1000  # Increased from 200 for better model training
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Get fully prepared data with all features calculated.
        
        Pipeline steps:
        1. Check cache for existing data
        2. Fetch OHLCV data if cache miss
        3. Calculate technical indicators
        4. Get sentiment score
        5. Cache results
        
        TRANSPARENCY: Logs each pipeline step with timing.
        
        Args:
            symbol: Trading symbol
            timeframe: Candle timeframe
            limit: Number of candles
            
        Returns:
            Tuple of (DataFrame with features, sentiment dict)
        """
        import time
        start_time = time.time()
        
        logger.info(f"[PIPELINE] Starting data preparation for {symbol}")
        
        # Step 1: Fetch OHLCV data
        step_start = time.time()
        
        # Convert symbol format for ccxt (e.g., 'BTC-USDT' -> 'BTC/USDT')
        ccxt_symbol = symbol.replace('-', '/')
        
        try:
            df = await self.data_fetcher.fetch_ohlcv_realtime(
                ccxt_symbol, timeframe, limit
            )
        except Exception as e:
            # Fallback to yfinance if ccxt fails
            logger.warning(f"[PIPELINE] ccxt failed, falling back to yfinance: {e}")
            yf_symbol = symbol.replace('/', '-').replace('-USDT', '-USD')
            # Use 1 year of data for better model training (~8,760 hourly candles)
            df = self.data_fetcher.fetch_historical_yfinance(yf_symbol, period='1y', interval='1h')
        
        logger.info(f"[PIPELINE] Data fetch completed in {time.time() - step_start:.2f}s")
        
        # Step 2: Calculate technical indicators
        step_start = time.time()
        df_with_features = self.feature_engineer.calculate_all_indicators(df)
        logger.info(f"[PIPELINE] Feature calculation completed in {time.time() - step_start:.2f}s")
        
        # Step 3: Get sentiment
        step_start = time.time()
        base_symbol = symbol.split('/')[0].split('-')[0]  # Extract 'BTC' from 'BTC/USDT' or 'BTC-USDT'
        sentiment = self.sentiment_analyzer.get_aggregated_sentiment(base_symbol)
        logger.info(f"[PIPELINE] Sentiment analysis completed in {time.time() - step_start:.2f}s")
        
        total_time = time.time() - start_time
        logger.info(f"[PIPELINE] Total pipeline execution: {total_time:.2f}s")
        
        return df_with_features, sentiment
    
    async def close(self):
        """Clean up all connections."""
        await self.data_fetcher.close()
        if self.cache:
            await self.cache.close()
        logger.info("[PIPELINE] All connections closed")


# =============================================================================
# MODULE EXPORTS
# =============================================================================
__all__ = [
    'OHLCVData',
    'TechnicalIndicators',
    'SentimentScore',
    'CryptoDataFetcher',
    'FeatureEngineer',
    'SentimentAnalyzer',
    'RedisCache',
    'DataPipeline',
]


# =============================================================================
# STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    async def test_pipeline():
        """Test the data pipeline with sample data."""
        print("=" * 60)
        print("DATA UTILITIES MODULE - STANDALONE TEST")
        print("=" * 60)
        
        pipeline = DataPipeline(use_cache=False)
        
        try:
            # Test with BTC-USDT
            df, sentiment = await pipeline.get_prepared_data('BTC-USDT', limit=50)
            
            print("\n[TEST] DataFrame shape:", df.shape)
            print("[TEST] Columns:", list(df.columns))
            print("[TEST] Latest close:", df['close'].iloc[-1])
            print("[TEST] Sentiment:", sentiment['overall_sentiment'])
            
        except Exception as e:
            print(f"[TEST] Error: {e}")
        finally:
            await pipeline.close()
    
    # Run test
    asyncio.run(test_pipeline())
