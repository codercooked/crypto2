"""
=============================================================================
MAIN API MODULE - Crypto Intelligence Platform
=============================================================================
FastAPI application providing:
- REST endpoints for crypto predictions and explainable AI insights
- WebSocket endpoint for live streaming predictions
- Health checks and model status endpoints

API Endpoints:
- GET /api/v1/predict/{symbol} - Full prediction from all 3 models
- GET /api/v1/decision-tree/logic-path/{symbol} - D3.js decision tree JSON
- GET /api/v1/random-forest/feature-impact/{symbol} - Ranked feature importance
- GET /api/v1/lstm/forecast/{symbol} - 24h forecast with confidence intervals
- GET /api/v1/conflict-analysis/{symbol} - Conflict score and explanation
- WS /ws/signals - Live prediction streaming

Author: AI Engineering Team
PEP8 Compliant | Highly Commented for Transparency
=============================================================================
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import os

# Local imports
from data_utils import DataPipeline, CryptoDataFetcher, FeatureEngineer, SentimentAnalyzer
from ml_engine import (
    ModelManager, 
    DecisionTreeModel, 
    RandomForestModel, 
    LSTMModel,
    ConflictEngine,
    ModelPrediction,
    ConflictAnalysis
)

# =============================================================================
# LOGGING CONFIGURATION - Maximum Transparency
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS - Request/Response Schemas
# =============================================================================
class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    timestamp: str
    models: Dict[str, bool]
    version: str


class PredictionResponse(BaseModel):
    """Full prediction response schema."""
    symbol: str
    timestamp: str
    predictions: Dict[str, Any]
    conflict_analysis: Dict[str, Any]
    consensus: Dict[str, Any]
    sentiment: Optional[Dict[str, Any]]


class LogicPathResponse(BaseModel):
    """Decision tree logic path response schema."""
    symbol: str
    model: str
    tree: Dict[str, Any]
    tree_stats: Dict[str, Any]
    generated_at: str


class FeatureImpactResponse(BaseModel):
    """Random forest feature impact response schema."""
    symbol: str
    model: str
    ranked_features: List[Dict[str, Any]]
    top_3_summary: str
    model_stats: Dict[str, Any]


class LSTMForecastResponse(BaseModel):
    """LSTM forecast response schema."""
    symbol: str
    current_price: float
    forecast_horizon: str
    forecasts: List[Dict[str, Any]]
    summary: Dict[str, Any]
    confidence_interval_method: str


class ConflictAnalysisResponse(BaseModel):
    """Conflict analysis response schema."""
    symbol: str
    conflict_score: float
    conflict_alert: bool
    consensus_signal: str
    model_signals: Dict[str, str]
    explanation: str
    recommendations: List[str]


class MarketVitalsResponse(BaseModel):
    """Market vitals and technical indicators."""
    symbol: str
    rsi: float
    macd: float
    volatility: float
    momentum: float
    volume_momentum: float
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: str
    timestamp: str


# =============================================================================
# WEBSOCKET CONNECTION MANAGER
# =============================================================================
class ConnectionManager:
    """
    Manages WebSocket connections for live signal streaming.
    
    Features:
    - Active connection tracking
    - Broadcast capability for sending to all clients
    - Per-symbol subscription support
    
    TRANSPARENCY: Logs all connection/disconnection events.
    """
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.symbol_subscriptions: Dict[str, Set[WebSocket]] = {}
        logger.info("[WEBSOCKET] Connection manager initialized")
    
    async def connect(self, websocket: WebSocket, symbols: List[str] = None):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        
        # Subscribe to symbols if specified
        if symbols:
            for symbol in symbols:
                if symbol not in self.symbol_subscriptions:
                    self.symbol_subscriptions[symbol] = set()
                self.symbol_subscriptions[symbol].add(websocket)
        
        logger.info(f"[WEBSOCKET] New connection. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection."""
        self.active_connections.discard(websocket)
        
        # Remove from all symbol subscriptions
        for subscribers in self.symbol_subscriptions.values():
            subscribers.discard(websocket)
        
        logger.info(f"[WEBSOCKET] Connection closed. Remaining: {len(self.active_connections)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return
        
        message_json = json.dumps(message)
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.warning(f"[WEBSOCKET] Failed to send to client: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    async def broadcast_to_symbol(self, symbol: str, message: Dict[str, Any]):
        """Broadcast message to clients subscribed to a specific symbol."""
        subscribers = self.symbol_subscriptions.get(symbol, set())
        
        if not subscribers:
            return
        
        message_json = json.dumps(message)
        disconnected = set()
        
        for connection in subscribers:
            try:
                await connection.send_text(message_json)
            except Exception:
                disconnected.add(connection)
        
        for conn in disconnected:
            self.disconnect(conn)


# =============================================================================
# GLOBAL STATE - Application Context
# =============================================================================
# Initialize managers
connection_manager = ConnectionManager()
model_manager: Optional[ModelManager] = None
data_pipeline: Optional[DataPipeline] = None

# Cached data for quick access
cached_predictions: Dict[str, Dict[str, Any]] = {}
last_training_time: Optional[datetime] = None


# =============================================================================
# APPLICATION LIFECYCLE
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    
    Startup:
    - Initialize data pipeline
    - Initialize ML models
    - Perform initial training
    
    Shutdown:
    - Clean up connections
    - Close data pipeline
    """
    global model_manager, data_pipeline, last_training_time
    
    logger.info("[APP] Starting Crypto Intelligence Platform...")
    
    # Initialize components
    data_pipeline = DataPipeline(use_cache=False)  # Disable Redis cache for simplicity
    model_manager = ModelManager()
    
    # Initial model training with sample data
    logger.info("[APP] Performing initial model training...")
    try:
        # Fetch initial data for training (1000 candles = ~41 days of hourly data)
        df, sentiment = await data_pipeline.get_prepared_data('BTC-USDT', limit=1000)
        model_manager.train_all(df)
        last_training_time = datetime.utcnow()
        logger.info("[APP] Initial training complete")
    except Exception as e:
        logger.warning(f"[APP] Initial training failed, models will train on first request: {e}")
    
    logger.info("[APP] Crypto Intelligence Platform is ready!")
    
    yield
    
    # Shutdown
    logger.info("[APP] Shutting down...")
    if data_pipeline:
        await data_pipeline.close()
    logger.info("[APP] Shutdown complete")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================
app = FastAPI(
    title="Crypto Intelligence Platform",
    description="High-performance FastAPI backend providing Explainable AI signals "
                "using Decision Tree, Random Forest, and LSTM models.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for serving the frontend
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    logger.info(f"[APP] Static files mounted from {STATIC_DIR}")


# =============================================================================
# HEALTH CHECK ENDPOINTS
# =============================================================================
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and model availability.
    
    TRANSPARENCY: Shows which models are trained and ready.
    """
    logger.info("[API] Health check requested")
    
    models_status = {
        "decision_tree": model_manager.decision_tree.is_trained if model_manager else False,
        "random_forest": model_manager.random_forest.is_trained if model_manager else False,
        "lstm": model_manager.lstm.is_trained if model_manager else False,
    }
    
    return HealthResponse(
        status="healthy" if all(models_status.values()) else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        models=models_status,
        version="1.0.0"
    )


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - serves dashboard or API info."""
    # Check if frontend exists
    index_path = os.path.join(STATIC_DIR, "index.html") if os.path.exists(STATIC_DIR) else None
    if index_path and os.path.exists(index_path):
        return FileResponse(index_path)
    
    return {
        "name": "Crypto Intelligence Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "dashboard": "/static/index.html",
        "health": "/health",
        "description": "Explainable AI signals using 3 distinct ML models"
    }


@app.get("/dashboard", tags=["Health"])
async def dashboard():
    """Redirect to dashboard frontend."""
    return RedirectResponse(url="/static/index.html")


# =============================================================================
# LIVE PRICES ENDPOINT - Using Binance via data pipeline
# =============================================================================
@app.get("/api/v1/prices", tags=["Prices"])
async def get_live_prices():
    """
    Get live prices for major crypto assets.
    
    Returns current price and 24h change for BTC, ETH, and SOL.
    Uses the same data source (Binance) as the prediction models.
    """
    logger.info("[API] Fetching live prices")
    
    try:
        symbols = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT']
        prices = {}
        
        for symbol in symbols:
            try:
                # Use the data fetcher for reliable prices
                df, _ = await data_pipeline.get_prepared_data(symbol, '1h', 25)
                
                if df is not None and len(df) > 0:
                    current_price = float(df['close'].iloc[-1])
                    prev_price = float(df['close'].iloc[0])
                    change_percent = ((current_price - prev_price) / prev_price) * 100
                    
                    key = symbol.split('-')[0]  # BTC, ETH, SOL
                    prices[key] = {
                        'price': round(current_price, 2),
                        'change_24h': round(change_percent, 2),
                        'symbol': symbol
                    }
                else:
                    key = symbol.split('-')[0]
                    prices[key] = {'price': 0, 'change_24h': 0, 'symbol': symbol}
                    
            except Exception as e:
                logger.warning(f"[API] Failed to fetch {symbol}: {e}")
                key = symbol.split('-')[0]
                prices[key] = {'price': 0, 'change_24h': 0, 'symbol': symbol}
        
        logger.info(f"[API] Prices fetched: BTC=${prices.get('BTC', {}).get('price', 0):.2f}")
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'prices': prices
        }
        
    except Exception as e:
        logger.error(f"[API] Price fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PREDICTION ENDPOINTS
# =============================================================================
@app.get(
    "/api/v1/predict/{symbol}",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Full Prediction",
    description="Get comprehensive prediction from all 3 models with conflict analysis."
)
async def get_prediction(
    symbol: str,
    timeframe: str = Query(default="1h", description="Candle timeframe"),
    limit: int = Query(default=1000, ge=50, le=2000, description="Number of candles to analyze")
):
    """
    Get full prediction from all three models.
    
    TRANSPARENCY LOG:
    - Logs the symbol and parameters
    - Logs each model's prediction
    - Logs conflict detection results
    
    Args:
        symbol: Trading pair (e.g., 'BTC-USDT')
        timeframe: Candle timeframe (e.g., '1h', '4h', '1d')
        limit: Number of historical candles to fetch
        
    Returns:
        Full prediction with all model outputs and conflict analysis
    """
    logger.info(f"[API] Prediction request: {symbol} | {timeframe} | limit={limit}")
    
    try:
        # Fetch and prepare data
        df, sentiment = await data_pipeline.get_prepared_data(symbol, timeframe, limit)
        
        # Ensure models are trained
        if not model_manager.decision_tree.is_trained:
            logger.info("[API] Training models on fetched data")
            model_manager.train_all(df)
        
        # Get full prediction
        prediction = model_manager.get_full_prediction(df, sentiment)
        
        # Cache the prediction
        cached_predictions[symbol] = prediction
        
        logger.info(f"[API] Prediction complete for {symbol}: {prediction['consensus']['signal']}")
        
        return PredictionResponse(
            symbol=symbol,
            timestamp=prediction['timestamp'],
            predictions=prediction['predictions'],
            conflict_analysis=prediction['conflict_analysis'],
            consensus=prediction['consensus'],
            sentiment=sentiment
        )
        
    except Exception as e:
        logger.error(f"[API] Prediction error for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate prediction: {str(e)}"
        )


# =============================================================================
# DECISION TREE ENDPOINTS - XAI Logic Path
# =============================================================================
@app.get(
    "/api/v1/decision-tree/logic-path/{symbol}",
    response_model=LogicPathResponse,
    tags=["Decision Tree"],
    summary="D3.js Logic Path",
    description="Get D3.js-compatible nested JSON showing the complete decision tree structure."
)
async def get_decision_tree_logic_path(
    symbol: str,
    timeframe: str = Query(default="1h", description="Candle timeframe"),
    limit: int = Query(default=1000, ge=50, le=2000, description="Number of candles")
):
    """
    Extract the decision tree logic as D3.js-compatible JSON.
    
    This endpoint returns a nested JSON structure that can be directly
    used with D3.js tree visualization to show exactly which thresholds
    led to each prediction.
    
    TRANSPARENCY: Every node includes:
    - Feature name and threshold (e.g., "RSI ≤ 30.5")
    - Sample counts at each node
    - Class distribution (bullish/bearish/neutral)
    - Gini impurity score
    
    Example node:
    {
        "name": "RSI",
        "threshold": 30.5,
        "condition": "RSI ≤ 30.5",
        "samples": 100,
        "children": [...]
    }
    """
    logger.info(f"[API] Decision Tree logic path request: {symbol}")
    
    try:
        # Ensure model is trained
        if not model_manager.decision_tree.is_trained:
            df, _ = await data_pipeline.get_prepared_data(symbol, timeframe, limit)
            model_manager.decision_tree.train(df)
        
        # Get logic path
        logic_path = model_manager.decision_tree.get_logic_path()
        
        logger.info(f"[API] Logic path generated: depth={logic_path['tree_stats']['depth']}")
        
        return LogicPathResponse(
            symbol=symbol,
            model="DecisionTree",
            tree=logic_path['tree'],
            tree_stats=logic_path['tree_stats'],
            generated_at=logic_path['generated_at']
        )
        
    except Exception as e:
        logger.error(f"[API] Logic path error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate logic path: {str(e)}"
        )


# =============================================================================
# MARKET VITALS ENDPOINT
# =============================================================================
@app.get(
    "/api/v1/market-vitals/{symbol}",
    response_model=MarketVitalsResponse,
    tags=["Technical Analysis"],
    summary="Technical Indicators",
    description="Get raw technical indicators and market vitals."
)
async def get_market_vitals(
    symbol: str,
    timeframe: str = Query(default="1h", description="Candle timeframe"),
    limit: int = Query(default=100, ge=50, le=500, description="Number of candles")
):
    """
    Get latest market vitals and technical indicators.
    """
    logger.info(f"[API] Market vitals request: {symbol}")
    
    try:
        # Fetch data
        df, _ = await data_pipeline.get_prepared_data(symbol, timeframe, limit)
        
        # Extract latest row
        latest = df.iloc[-1]
        
        return MarketVitalsResponse(
            symbol=symbol,
            rsi=round(float(latest.get('rsi', 50.0)), 2),
            macd=round(float(latest.get('macd', 0.0)), 4),
            volatility=round(float(latest.get('volatility', 0.0)), 4),
            momentum=round(float(latest.get('price_momentum', 0.0)) * 100, 2),
            volume_momentum=round(float(latest.get('volume_momentum', 0.0)) * 100, 2),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"[API] Market vitals error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# RANDOM FOREST ENDPOINTS - Feature Impact
# =============================================================================
@app.get(
    "/api/v1/random-forest/feature-impact/{symbol}",
    response_model=FeatureImpactResponse,
    tags=["Random Forest"],
    summary="Feature Impact Ranking",
    description="Get ranked list of feature importances from the Random Forest model."
)
async def get_feature_impact(
    symbol: str,
    timeframe: str = Query(default="1h", description="Candle timeframe"),
    limit: int = Query(default=1000, ge=50, le=2000, description="Number of candles")
):
    """
    Get ranked feature importance from Random Forest.
    
    Returns features ranked by their contribution to the model's
    predictions, expressed as percentages.
    
    TRANSPARENCY: Each feature includes:
    - Human-readable name (e.g., "RSI (Momentum)")
    - Technical name used in the model
    - Importance score (0.0 - 1.0)
    - Percentage representation
    - Rank position
    
    Example output:
    [
        {"feature": "RSI (Momentum)", "percentage": "35.2%", "rank": 1},
        {"feature": "MACD (Trend)", "percentage": "28.1%", "rank": 2},
        ...
    ]
    """
    logger.info(f"[API] Feature impact request: {symbol}")
    
    try:
        # Ensure model is trained
        if not model_manager.random_forest.is_trained:
            df, _ = await data_pipeline.get_prepared_data(symbol, timeframe, limit)
            model_manager.random_forest.train(df)
        
        # Get feature impact
        feature_impact = model_manager.random_forest.get_feature_impact()
        
        logger.info(f"[API] Feature impact: {feature_impact['top_3_summary']}")
        
        return FeatureImpactResponse(
            symbol=symbol,
            model="RandomForest",
            ranked_features=feature_impact['ranked_features'],
            top_3_summary=feature_impact['top_3_summary'],
            model_stats=feature_impact['model_stats']
        )
        
    except Exception as e:
        logger.error(f"[API] Feature impact error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get feature impact: {str(e)}"
        )


# =============================================================================
# LSTM ENDPOINTS - 24-Hour Forecast
# =============================================================================
@app.get(
    "/api/v1/lstm/forecast/{symbol}",
    response_model=LSTMForecastResponse,
    tags=["LSTM"],
    summary="24-Hour Forecast",
    description="Get 24-hour price forecast with confidence intervals."
)
async def get_lstm_forecast(
    symbol: str,
    timeframe: str = Query(default="1h", description="Candle timeframe"),
    limit: int = Query(default=1000, ge=60, le=2000, description="Number of candles")
):
    """
    Get LSTM 24-hour price forecast with confidence intervals.
    
    The forecast includes:
    - Predicted price for each of the next 24 hours
    - 95% confidence intervals (lower and upper bounds)
    - Uncertainty measure that grows with forecast horizon
    - Overall trend assessment (bullish/bearish/neutral)
    - Expected price change percentage
    
    TRANSPARENCY: The confidence interval method is explained
    in the response, showing how uncertainty is calculated.
    
    Confidence Interval Method:
    - Based on historical prediction errors (std deviation)
    - Scaled by √(forecast_hour) to account for growing uncertainty
    - Multiplied by 1.96 for 95% confidence level
    """
    logger.info(f"[API] LSTM forecast request: {symbol}")
    
    try:
        # Fetch data
        df, _ = await data_pipeline.get_prepared_data(symbol, timeframe, limit)
        
        # Ensure model is trained
        if not model_manager.lstm.is_trained:
            model_manager.lstm.train(df)
        
        # Get forecast
        forecast = model_manager.lstm.predict_with_confidence(df)
        
        logger.info(f"[API] Forecast: {forecast['summary']['trend']} "
                   f"({forecast['summary']['expected_change_percent']:+.2f}%)")
        
        return LSTMForecastResponse(
            symbol=symbol,
            current_price=forecast['current_price'],
            forecast_horizon=forecast['forecast_horizon'],
            forecasts=forecast['forecasts'],
            summary=forecast['summary'],
            confidence_interval_method=forecast.get(
                'confidence_interval_method',
                'Historical error std * sqrt(horizon) * 1.96 for 95% CI'
            )
        )
        
    except Exception as e:
        logger.error(f"[API] LSTM forecast error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate forecast: {str(e)}"
        )


# =============================================================================
# CONFLICT ENGINE ENDPOINTS
# =============================================================================
@app.get(
    "/api/v1/conflict-analysis/{symbol}",
    response_model=ConflictAnalysisResponse,
    tags=["Conflict Engine"],
    summary="Conflict Analysis",
    description="Analyze conflicts between model predictions and get explanations."
)
async def get_conflict_analysis(
    symbol: str,
    timeframe: str = Query(default="1h", description="Candle timeframe"),
    limit: int = Query(default=1000, ge=50, le=2000, description="Number of candles")
):
    """
    Analyze conflicts between model predictions.
    
    The Conflict Engine:
    1. Aggregates predictions from all 3 models
    2. Calculates a Conflict Score (0-100)
    3. Generates natural language explanation for disagreements
    4. Provides actionable recommendations
    
    TRANSPARENCY: The explanation details exactly WHY models disagree:
    - If LSTM is bullish but Random Forest is bearish:
      "Price trend is up, but underlying volume metrics are weakening"
    - If Decision Tree conflicts with ensemble:
      "Simple rules triggered, but broader analysis shows different pattern"
    
    Conflict Score Interpretation:
    - 0-30: Low conflict, models generally agree
    - 30-50: Moderate conflict, some disagreement
    - 50-70: High conflict, significant disagreement
    - 70-100: Severe conflict, strong opposing signals
    """
    logger.info(f"[API] Conflict analysis request: {symbol}")
    
    try:
        # Fetch data
        df, sentiment = await data_pipeline.get_prepared_data(symbol, timeframe, limit)
        
        # Ensure models are trained
        if not model_manager.decision_tree.is_trained:
            model_manager.train_all(df)
        
        # Get individual predictions
        dt_pred = model_manager.decision_tree.predict(df)
        rf_pred = model_manager.random_forest.predict(df)
        lstm_pred = model_manager.lstm.get_prediction_as_model_prediction(df)
        
        # Run conflict analysis
        conflict = model_manager.conflict_engine.analyze(
            dt_pred, rf_pred, lstm_pred, sentiment
        )
        
        logger.info(f"[API] Conflict score: {conflict.conflict_score}, "
                   f"Alert: {conflict.conflict_alert}")
        
        return ConflictAnalysisResponse(
            symbol=symbol,
            conflict_score=conflict.conflict_score,
            conflict_alert=conflict.conflict_alert,
            consensus_signal=conflict.consensus_signal,
            model_signals=conflict.model_signals,
            explanation=conflict.explanation,
            recommendations=conflict.recommendations
        )
        
    except Exception as e:
        logger.error(f"[API] Conflict analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze conflicts: {str(e)}"
        )


# =============================================================================
# WEBSOCKET ENDPOINT - Live Signal Streaming
# =============================================================================
@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    """
    WebSocket endpoint for live signal streaming.
    
    Streams predictions as soon as new candles close.
    
    Connection Protocol:
    1. Client connects to ws://host/ws/signals
    2. Client sends subscription message: {"symbols": ["BTC-USDT", "ETH-USDT"]}
    3. Server streams predictions for subscribed symbols
    4. Client can unsubscribe: {"action": "unsubscribe", "symbols": ["BTC-USDT"]}
    
    Message Format (Server -> Client):
    {
        "type": "prediction",
        "symbol": "BTC-USDT",
        "consensus": "bullish",
        "conflict_score": 25.3,
        "conflict_alert": false,
        "timestamp": "2024-01-15T12:00:00Z"
    }
    
    TRANSPARENCY: All WebSocket events are logged for debugging.
    """
    logger.info("[WEBSOCKET] New connection attempt")
    
    # Accept connection with default symbols
    await connection_manager.connect(websocket, symbols=["BTC-USDT"])
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to Crypto Intelligence Platform",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        while True:
            # Wait for client messages
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                
                # Handle subscription requests
                if "symbols" in data:
                    symbols = data["symbols"]
                    for symbol in symbols:
                        if symbol not in connection_manager.symbol_subscriptions:
                            connection_manager.symbol_subscriptions[symbol] = set()
                        connection_manager.symbol_subscriptions[symbol].add(websocket)
                    
                    await websocket.send_json({
                        "type": "subscribed",
                        "symbols": symbols,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    logger.info(f"[WEBSOCKET] Client subscribed to: {symbols}")
                
                # Handle prediction requests
                elif data.get("action") == "get_prediction":
                    symbol = data.get("symbol", "BTC-USDT")
                    
                    try:
                        df, sentiment = await data_pipeline.get_prepared_data(symbol, limit=100)
                        prediction = model_manager.get_full_prediction(df, sentiment)
                        
                        await websocket.send_json({
                            "type": "prediction",
                            "symbol": symbol,
                            "consensus": prediction['consensus']['signal'],
                            "conflict_score": prediction['consensus']['conflict_score'],
                            "conflict_alert": prediction['consensus']['conflict_alert'],
                            "timestamp": prediction['timestamp']
                        })
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "message": str(e),
                            "timestamp": datetime.utcnow().isoformat()
                        })
                
                # Handle unsubscribe
                elif data.get("action") == "unsubscribe":
                    symbols = data.get("symbols", [])
                    for symbol in symbols:
                        if symbol in connection_manager.symbol_subscriptions:
                            connection_manager.symbol_subscriptions[symbol].discard(websocket)
                    
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "symbols": symbols,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
            except asyncio.TimeoutError:
                # Send heartbeat on timeout
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info("[WEBSOCKET] Client disconnected")
    except Exception as e:
        logger.error(f"[WEBSOCKET] Error: {e}")
    finally:
        connection_manager.disconnect(websocket)


# =============================================================================
# BACKGROUND TASKS - Periodic Prediction Updates
# =============================================================================
async def periodic_prediction_task():
    """
    Background task that generates predictions periodically.
    
    Runs every minute to simulate new candle closes.
    Broadcasts predictions to all subscribed WebSocket clients.
    """
    logger.info("[BACKGROUND] Starting periodic prediction task")
    
    default_symbols = ["BTC-USDT", "ETH-USDT"]
    
    while True:
        await asyncio.sleep(60)  # Wait 1 minute
        
        for symbol in default_symbols:
            try:
                df, sentiment = await data_pipeline.get_prepared_data(symbol, limit=100)
                prediction = model_manager.get_full_prediction(df, sentiment)
                
                # Broadcast to WebSocket clients
                message = {
                    "type": "prediction",
                    "symbol": symbol,
                    "consensus": prediction['consensus']['signal'],
                    "conflict_score": prediction['consensus']['conflict_score'],
                    "conflict_alert": prediction['consensus']['conflict_alert'],
                    "timestamp": prediction['timestamp']
                }
                
                await connection_manager.broadcast_to_symbol(symbol, message)
                logger.debug(f"[BACKGROUND] Broadcast prediction for {symbol}")
                
            except Exception as e:
                logger.error(f"[BACKGROUND] Prediction error for {symbol}: {e}")


# =============================================================================
# ERROR HANDLERS
# =============================================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler with detailed logging."""
    logger.error(f"[API] HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "Request failed",
            "detail": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler."""
    logger.exception(f"[API] Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("CRYPTO INTELLIGENCE PLATFORM")
    print("Explainable AI Signals | 3 ML Models | Conflict Detection")
    print("=" * 60)
    print("\nStarting server...")
    print("API Docs: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("\nEndpoints:")
    print("  GET  /api/v1/predict/{symbol}")
    print("  GET  /api/v1/decision-tree/logic-path/{symbol}")
    print("  GET  /api/v1/random-forest/feature-impact/{symbol}")
    print("  GET  /api/v1/lstm/forecast/{symbol}")
    print("  GET  /api/v1/conflict-analysis/{symbol}")
    print("  WS   /ws/signals")
    print("\n  Dashboard: http://localhost:8000/")
    print("=" * 60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
