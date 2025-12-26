
"""
=============================================================================
ML ENGINE MODULE - Crypto Intelligence Platform
=============================================================================
This module contains all machine learning models and the Conflict Engine:

1. DecisionTreeModel - With D3.js-compatible logic path extraction
2. RandomForestModel - With feature importance ranking
3. LSTMModel - With 24-hour forecast and confidence intervals
4. ConflictEngine - Aggregates models and detects disagreements

Author: AI Engineering Team
PEP8 Compliant | Highly Commented for Transparency
=============================================================================
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd

try:
    from sklearn.tree import DecisionTreeClassifier, _tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. ML models will be disabled.")

import warnings

# Suppress TensorFlow warnings for cleaner logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not available. LSTM model will be disabled.")

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================
class Signal(Enum):
    """
    Trading signal enumeration.
    Used for model prediction outputs.
    """
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class ModelPrediction:
    """
    Standardized prediction output from any model.
    Ensures consistent interface across all model types.
    """
    model_name: str
    signal: str  # 'bullish', 'bearish', or 'neutral'
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Human-readable explanation
    timestamp: str
    metadata: Dict[str, Any]  # Model-specific additional data


@dataclass
class ConflictAnalysis:
    """
    Output from the Conflict Engine.
    Contains aggregated analysis and disagreement detection.
    """
    conflict_score: float  # 0-100
    conflict_alert: bool
    consensus_signal: str  # Overall signal after resolving conflicts
    model_signals: Dict[str, str]  # Signal from each model
    explanation: str  # Natural language explanation
    recommendations: List[str]


# =============================================================================
# FEATURE COLUMNS - Shared Configuration
# =============================================================================
FEATURE_COLUMNS = [
    'rsi',
    'macd',
    'macd_histogram',
    'bb_pband',
    'price_momentum',
    'volume_momentum',
    'volatility'
]


# =============================================================================
# DECISION TREE MODEL - With D3.js Logic Path Extraction
# =============================================================================
class DecisionTreeModel:
    """
    Decision Tree classifier for crypto signals with explainable logic paths.
    
    KEY FEATURE: Extracts the complete decision tree structure as a nested
    JSON object compatible with D3.js visualization.
    
    TRANSPARENCY: Every decision node threshold and path is logged and
    can be traced to understand exactly why a prediction was made.
    """
    
    def __init__(self, max_depth: int = 5, min_samples_split: int = 10):
        """
        Initialize the Decision Tree model.
        
        Args:
            max_depth: Maximum depth of the tree (limits complexity)
            min_samples_split: Minimum samples required to split a node
        """
        if SKLEARN_AVAILABLE:
            self.model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                class_weight='balanced'  # Handle imbalanced signals
            )
            self.scaler = MinMaxScaler()
        else:
            self.model = None
            self.scaler = None

        self.feature_names = FEATURE_COLUMNS
        self.is_trained = False
        
        logger.info(f"[DECISION_TREE] Initialized with max_depth={max_depth}")
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the decision tree on prepared data.
        
        TRANSPARENCY LOG:
        - Logs training data shape
        - Logs class distribution
        - Logs tree complexity after training
        
        Args:
            df: DataFrame with features and price data
            
        Returns:
            Training metrics dictionary
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("[DECISION_TREE] Sklearn not available, skipping training")
            return {}

        logger.info("[DECISION_TREE] Starting training...")
        
        # Prepare features and labels
        df_clean = df.dropna()
        
        if len(df_clean) < 50:
            logger.warning("[DECISION_TREE] Insufficient data for training, using synthetic data")
            df_clean = self._generate_synthetic_data(1000)  # Increased from 200
        
        # Create labels based on future price movement
        df_clean = df_clean.copy()
        df_clean['future_return'] = df_clean['close'].shift(-1) / df_clean['close'] - 1
        df_clean['label'] = pd.cut(
            df_clean['future_return'],
            bins=[-np.inf, -0.001, 0.001, np.inf],
            labels=[0, 1, 2]  # 0=bearish, 1=neutral, 2=bullish
        )
        
        # Remove last row (no future data) and any remaining NaN
        df_clean = df_clean.dropna()
        
        # Check if we have valid features
        available_features = [f for f in self.feature_names if f in df_clean.columns]
        if not available_features:
            logger.warning("[DECISION_TREE] No features available, generating synthetic features")
            df_clean = self._generate_synthetic_data(1000)  # Increased from 200
            available_features = self.feature_names
        
        self.feature_names = available_features
        
        X = df_clean[self.feature_names].values
        y = df_clean['label'].astype(int).values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"[DECISION_TREE] Training on {len(X_train)} samples")
        logger.info(f"[DECISION_TREE] Class distribution: {np.bincount(y_train)}")
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate metrics
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        metrics = {
            'train_accuracy': round(train_accuracy, 4),
            'test_accuracy': round(test_accuracy, 4),
            'tree_depth': self.model.get_depth(),
            'n_leaves': self.model.get_n_leaves(),
            'n_features': len(self.feature_names),
            'n_samples': len(X_train)
        }
        
        logger.info(f"[DECISION_TREE] Training complete - Accuracy: {test_accuracy:.2%}")
        logger.info(f"[DECISION_TREE] Tree depth: {metrics['tree_depth']}, Leaves: {metrics['n_leaves']}")
        
        return metrics
    
    def _generate_synthetic_data(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data for initial model training."""
        np.random.seed(42)
        
        data = {
            'close': np.random.uniform(40000, 45000, n_samples),
            'rsi': np.random.uniform(20, 80, n_samples),
            'macd': np.random.uniform(-100, 100, n_samples),
            'macd_histogram': np.random.uniform(-50, 50, n_samples),
            'bb_pband': np.random.uniform(0, 1, n_samples),
            'price_momentum': np.random.uniform(-0.05, 0.05, n_samples),
            'volume_momentum': np.random.uniform(-0.5, 0.5, n_samples),
            'volatility': np.random.uniform(100, 500, n_samples),
        }
        
        return pd.DataFrame(data)
    
    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        """
        Make a prediction with full explanation.
        
        TRANSPARENCY: Logs the exact path taken through the tree
        and which thresholds were triggered.
        
        Args:
            features: DataFrame with feature values
            
        Returns:
            ModelPrediction with signal and reasoning
        """
        if not SKLEARN_AVAILABLE:
             return ModelPrediction(
                model_name="DecisionTree",
                signal=Signal.NEUTRAL.value,
                confidence=0.0,
                reasoning="Model disabled (missing dependencies)",
                timestamp=datetime.utcnow().isoformat(),
                metadata={}
            )

        if not self.is_trained:
            logger.warning("[DECISION_TREE] Model not trained, training with synthetic data")
            self.train(self._generate_synthetic_data(1000))  # Increased from 200
        
        # Prepare features
        available_features = [f for f in self.feature_names if f in features.columns]
        X = features[available_features].iloc[-1:].values
        X_scaled = self.scaler.transform(X)
        
        # Get prediction and probabilities
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Map to signal
        signal_map = {0: Signal.BEARISH, 1: Signal.NEUTRAL, 2: Signal.BULLISH}
        signal = signal_map.get(prediction, Signal.NEUTRAL)
        
        # Get decision path for explanation
        decision_path = self._get_decision_path_text(X_scaled[0])
        
        confidence = float(max(probabilities))
        
        logger.info(f"[DECISION_TREE] Prediction: {signal.value} (confidence: {confidence:.2%})")
        logger.debug(f"[DECISION_TREE] Decision path: {decision_path}")
        
        return ModelPrediction(
            model_name="DecisionTree",
            signal=signal.value,
            confidence=confidence,
            reasoning=decision_path,
            timestamp=datetime.utcnow().isoformat(),
            metadata={
                'probabilities': {
                    'bearish': float(probabilities[0]),
                    'neutral': float(probabilities[1]),
                    'bullish': float(probabilities[2])
                },
                'tree_depth_used': self.model.get_depth()
            }
        )
    
    def _get_decision_path_text(self, sample: np.ndarray) -> str:
        """
        Generate human-readable decision path for a sample.
        
        Args:
            sample: Feature vector for single sample
            
        Returns:
            String describing the decision path
        """
        tree = self.model.tree_
        feature = tree.feature
        threshold = tree.threshold
        
        node_indicator = self.model.decision_path([sample])
        node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
        
        path_parts = []
        for node_id in node_index:
            if tree.feature[node_id] != _tree.TREE_UNDEFINED:
                feature_name = self.feature_names[feature[node_id]]
                thresh = threshold[node_id]
                
                if sample[feature[node_id]] <= thresh:
                    path_parts.append(f"{feature_name} ≤ {thresh:.4f}")
                else:
                    path_parts.append(f"{feature_name} > {thresh:.4f}")
        
        return " → ".join(path_parts) if path_parts else "Direct classification"
    
    def get_logic_path(self, features: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Convert the decision tree to a D3.js-compatible nested JSON structure.
        
        This is the KEY XAI FEATURE for the Decision Tree model.
        The output can be directly used with D3.js tree visualization.
        
        TRANSPARENCY: Every node includes:
        - Feature name and threshold
        - Number of samples
        - Class distribution
        - Gini impurity
        
        Args:
            features: Optional features to highlight the decision path
            
        Returns:
            Nested dictionary structure compatible with D3.js
        """
        if not SKLEARN_AVAILABLE:
            return {'tree': {}, 'tree_stats': {}}

        if not self.is_trained:
            logger.warning("[DECISION_TREE] Model not trained, training first")
            self.train(self._generate_synthetic_data(1000))  # Increased from 200
        
        logger.info("[DECISION_TREE] Generating D3.js-compatible logic path")
        
        tree = self.model.tree_
        
        def build_tree(node_id: int) -> Dict[str, Any]:
            """Recursively build the tree structure."""
            
            # Get node information
            n_samples = int(tree.n_node_samples[node_id])
            gini = float(tree.impurity[node_id])
            values = tree.value[node_id][0].tolist()
            
            # Determine dominant class
            class_names = ['bearish', 'neutral', 'bullish']
            dominant_class = class_names[int(np.argmax(values))]
            
            # Check if leaf node
            if tree.feature[node_id] == _tree.TREE_UNDEFINED:
                return {
                    'name': f"Predict: {dominant_class.upper()}",
                    'type': 'leaf',
                    'samples': n_samples,
                    'gini': round(gini, 4),
                    'class_distribution': {
                        'bearish': int(values[0]),
                        'neutral': int(values[1]),
                        'bullish': int(values[2])
                    },
                    'dominant_class': dominant_class,
                    'confidence': round(max(values) / sum(values), 4) if sum(values) > 0 else 0
                }
            
            # Internal node - has feature split
            feature_name = self.feature_names[tree.feature[node_id]]
            threshold = float(tree.threshold[node_id])
            
            node = {
                'name': f"{feature_name}",
                'type': 'decision',
                'threshold': round(threshold, 4),
                'condition': f"{feature_name} ≤ {threshold:.4f}",
                'samples': n_samples,
                'gini': round(gini, 4),
                'class_distribution': {
                    'bearish': int(values[0]),
                    'neutral': int(values[1]),
                    'bullish': int(values[2])
                },
                'children': [
                    {
                        'branch': 'true',
                        'label': f"≤ {threshold:.4f}",
                        **build_tree(tree.children_left[node_id])
                    },
                    {
                        'branch': 'false',
                        'label': f"> {threshold:.4f}",
                        **build_tree(tree.children_right[node_id])
                    }
                ]
            }
            
            return node
        
        # Build the tree starting from root
        logic_path = build_tree(0)
        
        # Add metadata
        result = {
            'model': 'DecisionTree',
            'version': '1.0',
            'generated_at': datetime.utcnow().isoformat(),
            'tree_stats': {
                'depth': int(self.model.get_depth()),
                'n_leaves': int(self.model.get_n_leaves()),
                'features': self.feature_names
            },
            'tree': logic_path
        }
        
        logger.info(f"[DECISION_TREE] Logic path generated with {self.model.get_n_leaves()} leaves")
        return result


# =============================================================================
# RANDOM FOREST MODEL - With Feature Importance Ranking
# =============================================================================
class RandomForestModel:
    """
    Random Forest classifier for crypto signals with feature importance analysis.
    
    KEY FEATURE: Extracts and ranks feature importances to explain
    which indicators have the most impact on predictions.
    
    TRANSPARENCY: Logs importance of each feature and explains
    how the ensemble reached its decision.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        """
        Initialize the Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of each tree
        """
        if SKLEARN_AVAILABLE:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1  # Use all CPU cores
            )
            self.scaler = MinMaxScaler()
        else:
            self.model = None
            self.scaler = None
            
        self.feature_names = FEATURE_COLUMNS
        self.is_trained = False
        
        logger.info(f"[RANDOM_FOREST] Initialized with {n_estimators} estimators")
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the random forest on prepared data.
        
        TRANSPARENCY LOG:
        - Logs training data statistics
        - Logs out-of-bag score
        - Logs feature importance ranking
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("[RANDOM_FOREST] Sklearn not available, skipping training")
            return {}

        logger.info("[RANDOM_FOREST] Starting training...")
        
        # Prepare features and labels
        df_clean = df.dropna()
        
        if len(df_clean) < 50:
            logger.warning("[RANDOM_FOREST] Insufficient data, using synthetic data")
            df_clean = self._generate_synthetic_data(1000)  # Increased from 200
        
        # Create labels
        df_clean = df_clean.copy()
        df_clean['future_return'] = df_clean['close'].shift(-1) / df_clean['close'] - 1
        df_clean['label'] = pd.cut(
            df_clean['future_return'],
            bins=[-np.inf, -0.001, 0.001, np.inf],
            labels=[0, 1, 2]
        )
        
        df_clean = df_clean.dropna()
        
        # Get available features
        available_features = [f for f in self.feature_names if f in df_clean.columns]
        if not available_features:
            df_clean = self._generate_synthetic_data(1000)  # Increased from 200
            available_features = self.feature_names
        
        self.feature_names = available_features
        
        X = df_clean[self.feature_names].values
        y = df_clean['label'].astype(int).values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"[RANDOM_FOREST] Training on {len(X_train)} samples with {len(self.feature_names)} features")
        
        # Enable OOB scoring
        self.model.set_params(oob_score=True)
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate metrics
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        # Log feature importances
        importances = self.model.feature_importances_
        for name, imp in sorted(zip(self.feature_names, importances), key=lambda x: -x[1]):
            logger.info(f"[RANDOM_FOREST] Feature '{name}': {imp:.4f} ({imp*100:.1f}%)")
        
        metrics = {
            'train_accuracy': round(train_accuracy, 4),
            'test_accuracy': round(test_accuracy, 4),
            'oob_score': round(self.model.oob_score_, 4),
            'n_estimators': self.model.n_estimators,
            'n_features': len(self.feature_names)
        }
        
        logger.info(f"[RANDOM_FOREST] Training complete - Accuracy: {test_accuracy:.2%}, OOB: {self.model.oob_score_:.2%}")
        
        return metrics
    
    def _generate_synthetic_data(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic data for initial model training."""
        np.random.seed(42)
        
        data = {
            'close': np.random.uniform(40000, 45000, n_samples),
            'rsi': np.random.uniform(20, 80, n_samples),
            'macd': np.random.uniform(-100, 100, n_samples),
            'macd_histogram': np.random.uniform(-50, 50, n_samples),
            'bb_pband': np.random.uniform(0, 1, n_samples),
            'price_momentum': np.random.uniform(-0.05, 0.05, n_samples),
            'volume_momentum': np.random.uniform(-0.5, 0.5, n_samples),
            'volatility': np.random.uniform(100, 500, n_samples),
        }
        
        return pd.DataFrame(data)
    
    def predict(self, features: pd.DataFrame) -> ModelPrediction:
        """
        Make a prediction with feature importance explanation.
        
        TRANSPARENCY: Logs contribution of each feature to the prediction.
        """
        if not SKLEARN_AVAILABLE:
             return ModelPrediction(
                model_name="RandomForest",
                signal=Signal.NEUTRAL.value,
                confidence=0.0,
                reasoning="Model disabled (missing dependencies)",
                timestamp=datetime.utcnow().isoformat(),
                metadata={}
            )

        if not self.is_trained:
            logger.warning("[RANDOM_FOREST] Model not trained, training first")
            self.train(self._generate_synthetic_data(1000))  # Increased from 200
        
        # Prepare features
        available_features = [f for f in self.feature_names if f in features.columns]
        X = features[available_features].iloc[-1:].values
        X_scaled = self.scaler.transform(X)
        
        # Get prediction
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Map to signal
        signal_map = {0: Signal.BEARISH, 1: Signal.NEUTRAL, 2: Signal.BULLISH}
        signal = signal_map.get(prediction, Signal.NEUTRAL)
        
        confidence = float(max(probabilities))
        
        # Get feature importance summary for reasoning
        feature_impact = self.get_feature_impact()
        top_features = feature_impact['ranked_features'][:3]
        reasoning = f"Top influencing factors: {', '.join([f['feature'] for f in top_features])}"
        
        logger.info(f"[RANDOM_FOREST] Prediction: {signal.value} (confidence: {confidence:.2%})")
        
        return ModelPrediction(
            model_name="RandomForest",
            signal=signal.value,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.utcnow().isoformat(),
            metadata={
                'probabilities': {
                    'bearish': float(probabilities[0]),
                    'neutral': float(probabilities[1]),
                    'bullish': float(probabilities[2])
                },
                'n_estimators': self.model.n_estimators,
                'feature_importances': feature_impact['ranked_features'][:5]
            }
        )
    
    def get_feature_impact(self) -> Dict[str, Any]:
        """
        Get ranked feature importance with detailed breakdown.
        
        This is the KEY XAI FEATURE for the Random Forest model.
        Returns importances as percentages with human-readable names.
        
        TRANSPARENCY: Each feature's contribution is explained
        with both raw importance and percentage.
        
        Returns:
            Dictionary with ranked features and statistics
        """
        if not SKLEARN_AVAILABLE:
            return {'ranked_features': [], 'top_3_summary': "Model disabled"}

        if not self.is_trained:
            logger.warning("[RANDOM_FOREST] Model not trained, training first")
            self.train(self._generate_synthetic_data(1000))  # Increased from 200
        
        logger.info("[RANDOM_FOREST] Calculating feature impact ranking")
        
        importances = self.model.feature_importances_
        
        # Map feature names to human-readable descriptions
        feature_display_names = {
            'rsi': 'RSI (Momentum)',
            'macd': 'MACD (Trend)',
            'macd_histogram': 'MACD Histogram',
            'bb_pband': 'Bollinger %B (Volatility)',
            'price_momentum': 'Price Momentum',
            'volume_momentum': 'Volume Momentum',
            'volatility': 'Volatility',
        }
        
        # Create ranked list
        ranked_features = []
        for name, importance in sorted(zip(self.feature_names, importances), key=lambda x: -x[1]):
            display_name = feature_display_names.get(name, name)
            ranked_features.append({
                'feature': display_name,
                'technical_name': name,
                'importance': round(float(importance), 4),
                'percentage': f"{importance * 100:.1f}%",
                'rank': len(ranked_features) + 1
            })
        
        result = {
            'model': 'RandomForest',
            'generated_at': datetime.utcnow().isoformat(),
            'total_features': len(self.feature_names),
            'ranked_features': ranked_features,
            'top_3_summary': ", ".join([
                f"{f['feature']}: {f['percentage']}"
                for f in ranked_features[:3]
            ]),
            'model_stats': {
                'n_estimators': self.model.n_estimators,
                'max_depth': self.model.max_depth,
                'oob_score': round(self.model.oob_score_, 4) if hasattr(self.model, 'oob_score_') else None
            }
        }
        
        logger.info(f"[RANDOM_FOREST] Top 3 features: {result['top_3_summary']}")
        
        return result


# =============================================================================
# LSTM MODEL - With 24-Hour Forecast and Confidence Intervals
# =============================================================================
class LSTMModel:
    """
    LSTM neural network for time-series price forecasting.
    
    KEY FEATURES:
    - 24-hour ahead price prediction
    - Confidence intervals based on prediction uncertainty
    - Trend classification (bullish/bearish/neutral)
    
    TRANSPARENCY: Logs model architecture, training progress,
    and confidence interval calculations.
    """
    
    def __init__(self, lookback: int = 60, forecast_hours: int = 24):
        """
        Initialize the LSTM model.
        
        Args:
            lookback: Number of past hours to use for prediction
            forecast_hours: Number of hours to forecast ahead
        """
        self.lookback = lookback
        self.forecast_hours = forecast_hours
        self.model: Optional[Any] = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        self.historical_errors: List[float] = []  # For confidence interval calculation
        
        logger.info(f"[LSTM] Initialized with lookback={lookback}, forecast={forecast_hours}h")
        
        if not TENSORFLOW_AVAILABLE:
            logger.warning("[LSTM] TensorFlow not available - model will use fallback predictions")
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Any:
        """
        Build the LSTM neural network architecture.
        
        Architecture:
        - 2 LSTM layers with dropout for regularization
        - Dense output layer for multi-step prediction
        
        TRANSPARENCY: Logs complete model summary.
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("[LSTM] TensorFlow not available, cannot build model")
            return None
        
        logger.info(f"[LSTM] Building model with input shape: {input_shape}")
        
        model = Sequential([
            # First LSTM layer with return sequences
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers
            Dense(25, activation='relu'),
            Dense(self.forecast_hours)  # Output: predictions for each hour
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        logger.info("[LSTM] Model architecture built:")
        model.summary(print_fn=lambda x: logger.debug(x))
        
        return model
    
    def train(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train the LSTM model on prepared data.
        
        TRANSPARENCY LOG:
        - Logs training progress (loss per epoch)
        - Logs validation metrics
        - Calculates historical errors for confidence intervals
        """
        logger.info("[LSTM] Starting training...")
        
        if not TENSORFLOW_AVAILABLE:
            logger.warning("[LSTM] TensorFlow not available, using fallback")
            self.is_trained = True
            return {'status': 'fallback_mode', 'reason': 'TensorFlow not available'}
        
        # Prepare data
        if 'close' not in df.columns or len(df) < self.lookback + self.forecast_hours:
            logger.warning("[LSTM] Insufficient data, using synthetic data")
            df = self._generate_synthetic_data(2000)  # Increased from 500 for better LSTM training
        
        prices = df['close'].values.reshape(-1, 1)
        
        # Scale data
        scaled_prices = self.scaler.fit_transform(prices)
        
        # Create sequences
        X, y = self._create_sequences(scaled_prices)
        
        if len(X) < 50:
            logger.warning("[LSTM] Not enough sequences, generating more data")
            df = self._generate_synthetic_data(2000)  # Increased from 500
            prices = df['close'].values.reshape(-1, 1)
            scaled_prices = self.scaler.fit_transform(prices)
            X, y = self._create_sequences(scaled_prices)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"[LSTM] Training on {len(X_train)} sequences, testing on {len(X_test)}")
        
        # Build model
        self.model = self._build_model((X_train.shape[1], X_train.shape[2]))
        
        # Train with early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0
        )
        
        self.is_trained = True
        
        # Calculate historical errors for confidence intervals
        predictions = self.model.predict(X_test, verbose=0)
        errors = np.abs(predictions - y_test)
        self.historical_errors = errors.flatten().tolist()
        
        # Metrics
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        metrics = {
            'train_loss': round(final_train_loss, 6),
            'val_loss': round(final_val_loss, 6),
            'epochs_trained': len(history.history['loss']),
            'mean_absolute_error': round(float(np.mean(errors)), 6),
            'std_error': round(float(np.std(errors)), 6)
        }
        
        logger.info(f"[LSTM] Training complete - Val Loss: {final_val_loss:.6f}")
        
        return metrics
    
    def _generate_synthetic_data(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic price data for training."""
        np.random.seed(42)
        
        # Generate random walk with trend
        prices = [40000]
        for _ in range(n_samples - 1):
            change = np.random.randn() * 100 + 5  # Slight upward bias
            prices.append(max(30000, prices[-1] + change))  # Floor at 30k
        
        return pd.DataFrame({'close': prices})
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and targets for LSTM."""
        X, y = [], []
        
        for i in range(len(data) - self.lookback - self.forecast_hours):
            X.append(data[i:(i + self.lookback)])
            y.append(data[(i + self.lookback):(i + self.lookback + self.forecast_hours)].flatten())
        
        return np.array(X), np.array(y)
    
    def predict_with_confidence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate 24-hour forecast with confidence intervals.
        
        This is the KEY XAI FEATURE for the LSTM model.
        
        TRANSPARENCY:
        - Logs prediction methodology
        - Explains confidence interval calculation
        - Shows uncertainty at each forecast point
        
        Args:
            df: DataFrame with historical price data
            
        Returns:
            Dictionary with forecasts and confidence intervals
        """
        logger.info("[LSTM] Generating 24-hour forecast with confidence intervals")
        
        # Get current price for reference
        current_price = float(df['close'].iloc[-1]) if 'close' in df.columns else 40000.0
        current_time = datetime.utcnow()
        
        if not TENSORFLOW_AVAILABLE or not self.is_trained or self.model is None:
            logger.warning("[LSTM] Using fallback prediction (model not available)")
            return self._fallback_prediction(current_price, current_time)
        
        # Prepare input sequence
        prices = df['close'].values[-self.lookback:].reshape(-1, 1)
        
        if len(prices) < self.lookback:
            # Pad with repeated values if not enough data
            padding = np.repeat(prices[0], self.lookback - len(prices)).reshape(-1, 1)
            prices = np.vstack([padding, prices])
        
        scaled_prices = self.scaler.transform(prices)
        X = scaled_prices.reshape(1, self.lookback, 1)
        
        # Get prediction
        scaled_predictions = self.model.predict(X, verbose=0)[0]
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(
            scaled_predictions.reshape(-1, 1)
        ).flatten()
        
        # Calculate confidence intervals
        # Using historical errors to estimate uncertainty
        if self.historical_errors:
            std_error = np.std(self.historical_errors)
            # Scale std error back to price space
            std_error_price = std_error * (self.scaler.data_max_[0] - self.scaler.data_min_[0])
        else:
            # Default uncertainty (1% of current price)
            std_error_price = current_price * 0.01
        
        # Generate hourly forecasts with confidence intervals
        # Uncertainty grows with forecast horizon
        forecasts = []
        for hour in range(self.forecast_hours):
            predicted_price = float(predictions[hour])
            
            # Uncertainty grows with time (√time scaling)
            uncertainty_multiplier = np.sqrt(hour + 1)
            uncertainty = std_error_price * uncertainty_multiplier
            
            forecast = {
                'hour': hour + 1,
                'timestamp': (current_time + timedelta(hours=hour + 1)).isoformat(),
                'predicted_price': round(predicted_price, 2),
                'lower_bound': round(predicted_price - 1.96 * uncertainty, 2),  # 95% CI
                'upper_bound': round(predicted_price + 1.96 * uncertainty, 2),
                'uncertainty': round(uncertainty, 2),
                'confidence_level': 0.95
            }
            forecasts.append(forecast)
        
        # Calculate overall trend
        price_change = predictions[-1] - current_price
        percent_change = (price_change / current_price) * 100
        
        if percent_change > 1:
            signal = Signal.BULLISH
            trend = "upward"
        elif percent_change < -1:
            signal = Signal.BEARISH
            trend = "downward"
        else:
            signal = Signal.NEUTRAL
            trend = "sideways"
        
        result = {
            'model': 'LSTM',
            'generated_at': datetime.utcnow().isoformat(),
            'current_price': current_price,
            'forecast_horizon': f"{self.forecast_hours} hours",
            'forecasts': forecasts,
            'summary': {
                'final_predicted_price': round(float(predictions[-1]), 2),
                'expected_change': round(price_change, 2),
                'expected_change_percent': round(percent_change, 2),
                'trend': trend,
                'signal': signal.value,
                'average_uncertainty': round(float(np.mean([f['uncertainty'] for f in forecasts])), 2)
            },
            'confidence_interval_method': 'Historical prediction error std * sqrt(horizon) * 1.96 for 95% CI'
        }
        
        logger.info(f"[LSTM] Forecast: {trend} ({percent_change:+.2f}%), Signal: {signal.value}")
        
        return result
    
    def _fallback_prediction(self, current_price: float, current_time: datetime) -> Dict[str, Any]:
        """Generate fallback prediction when model is not available."""
        logger.info("[LSTM] Using fallback prediction (random walk with slight trend)")
        
        # Simple random walk with mean reversion
        forecasts = []
        price = current_price
        
        for hour in range(self.forecast_hours):
            # Random walk with slight upward bias
            change = np.random.randn() * current_price * 0.005  # 0.5% std dev per hour
            price = price + change
            
            uncertainty = current_price * 0.01 * np.sqrt(hour + 1)
            
            forecasts.append({
                'hour': hour + 1,
                'timestamp': (current_time + timedelta(hours=hour + 1)).isoformat(),
                'predicted_price': round(price, 2),
                'lower_bound': round(price - 1.96 * uncertainty, 2),
                'upper_bound': round(price + 1.96 * uncertainty, 2),
                'uncertainty': round(uncertainty, 2),
                'confidence_level': 0.95
            })
        
        final_price = forecasts[-1]['predicted_price']
        percent_change = ((final_price - current_price) / current_price) * 100
        
        if percent_change > 1:
            signal = "bullish"
            trend = "upward"
        elif percent_change < -1:
            signal = "bearish"
            trend = "downward"
        else:
            signal = "neutral"
            trend = "sideways"
        
        return {
            'model': 'LSTM (Fallback)',
            'generated_at': datetime.utcnow().isoformat(),
            'current_price': current_price,
            'forecast_horizon': f"{self.forecast_hours} hours",
            'forecasts': forecasts,
            'summary': {
                'final_predicted_price': final_price,
                'expected_change': round(final_price - current_price, 2),
                'expected_change_percent': round(percent_change, 2),
                'trend': trend,
                'signal': signal,
                'average_uncertainty': round(float(np.mean([f['uncertainty'] for f in forecasts])), 2)
            },
            'note': 'Using fallback random walk model (TensorFlow not available or model not trained)'
        }
    
    def get_prediction_as_model_prediction(self, df: pd.DataFrame) -> ModelPrediction:
        """Convert LSTM forecast to standard ModelPrediction format."""
        forecast = self.predict_with_confidence(df)
        summary = forecast['summary']
        
        return ModelPrediction(
            model_name="LSTM",
            signal=summary['signal'],
            confidence=min(0.95, 1 - (summary['average_uncertainty'] / forecast['current_price'])),
            reasoning=f"24h forecast shows {summary['trend']} trend ({summary['expected_change_percent']:+.2f}%)",
            timestamp=forecast['generated_at'],
            metadata={
                'final_price': summary['final_predicted_price'],
                'current_price': forecast['current_price'],
                'trend': summary['trend']
            }
        )


# =============================================================================
# CONFLICT ENGINE - Model Disagreement Detection & Natural Language Explanation
# =============================================================================
class ConflictEngine:
    """
    Aggregates outputs from all models and detects conflicts.
    
    KEY FEATURES:
    - Calculates conflict score (0-100) based on signal divergence
    - Generates natural language explanations for disagreements
    - Provides consensus signal with confidence weighting
    
    TRANSPARENCY: Every conflict detection is logged with:
    - Individual model signals
    - Conflict score breakdown
    - Reasoning for the explanation generated
    """
    
    def __init__(self):
        logger.info("[CONFLICT_ENGINE] Initialized conflict detection engine")
        
        # Map signals to numeric values for divergence calculation
        self._signal_values = {
            'bullish': 1,
            'neutral': 0,
            'bearish': -1
        }
    
    def analyze(
        self,
        dt_prediction: ModelPrediction,
        rf_prediction: ModelPrediction,
        lstm_prediction: ModelPrediction,
        sentiment_data: Optional[Dict[str, Any]] = None
    ) -> ConflictAnalysis:
        """
        Analyze predictions from all models for conflicts.
        
        TRANSPARENCY LOG:
        - Logs each model's signal
        - Logs conflict score calculation
        - Explains the reasoning behind the natural language explanation
        
        Args:
            dt_prediction: Decision Tree prediction
            rf_prediction: Random Forest prediction
            lstm_prediction: LSTM prediction
            sentiment_data: Optional sentiment analysis data
            
        Returns:
            ConflictAnalysis with score, alert status, and explanation
        """
        logger.info("[CONFLICT_ENGINE] Analyzing model predictions for conflicts")
        
        # Extract signals
        signals = {
            'decision_tree': dt_prediction.signal,
            'random_forest': rf_prediction.signal,
            'lstm': lstm_prediction.signal
        }
        
        logger.info(f"[CONFLICT_ENGINE] Signals - DT: {signals['decision_tree']}, "
                   f"RF: {signals['random_forest']}, LSTM: {signals['lstm']}")
        
        # Calculate conflict score
        conflict_score = self._calculate_conflict_score(signals)
        logger.info(f"[CONFLICT_ENGINE] Conflict score: {conflict_score}")
        
        # Determine if alert is needed (score > 50 indicates significant conflict)
        conflict_alert = conflict_score > 50
        
        # Calculate consensus signal using confidence-weighted voting
        confidence_weights = {
            'decision_tree': dt_prediction.confidence,
            'random_forest': rf_prediction.confidence,
            'lstm': lstm_prediction.confidence
        }
        consensus_signal = self._calculate_consensus(signals, confidence_weights)
        
        # Generate natural language explanation
        explanation = self._generate_explanation(
            signals, 
            conflict_score, 
            consensus_signal,
            dt_prediction,
            rf_prediction,
            lstm_prediction,
            sentiment_data
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            conflict_score, 
            signals, 
            consensus_signal
        )
        
        result = ConflictAnalysis(
            conflict_score=conflict_score,
            conflict_alert=conflict_alert,
            consensus_signal=consensus_signal,
            model_signals=signals,
            explanation=explanation,
            recommendations=recommendations
        )
        
        if conflict_alert:
            logger.warning(f"[CONFLICT_ENGINE] CONFLICT ALERT: {explanation}")
        else:
            logger.info(f"[CONFLICT_ENGINE] No significant conflict detected")
        
        return result
    
    def _calculate_conflict_score(self, signals: Dict[str, str]) -> float:
        """
        Calculate a conflict score from 0-100.
        
        Methodology:
        - 0 = Perfect agreement (all same signal)
        - 50 = Moderate disagreement (mix of signals)
        - 100 = Maximum conflict (bullish vs bearish with no neutral)
        
        TRANSPARENCY: Logs the calculation breakdown.
        """
        # Convert signals to numeric values
        values = [self._signal_values[s] for s in signals.values()]
        
        # Calculate variance (0 if all same, higher with more disagreement)
        variance = np.var(values)
        
        # Maximum possible variance (bullish=1, bearish=-1) is 1.33
        # Normalize to 0-100 scale
        max_variance = 1.33
        normalized_variance = min(variance / max_variance, 1.0)
        
        # Count conflicting pairs
        bullish_count = sum(1 for v in values if v == 1)
        bearish_count = sum(1 for v in values if v == -1)
        
        # Direct conflict penalty (bullish AND bearish present)
        direct_conflict = 1 if (bullish_count > 0 and bearish_count > 0) else 0
        
        # Final score: weighted combination
        base_score = normalized_variance * 60
        conflict_penalty = direct_conflict * 40
        
        final_score = min(100, base_score + conflict_penalty)
        
        logger.debug(f"[CONFLICT_ENGINE] Score breakdown - Variance: {variance:.4f}, "
                    f"Direct conflict: {direct_conflict}, Final: {final_score:.1f}")
        
        return round(final_score, 1)
    
    def _calculate_consensus(
        self, 
        signals: Dict[str, str], 
        weights: Dict[str, float]
    ) -> str:
        """
        Calculate weighted consensus signal.
        
        Uses confidence values as weights to prioritize more confident models.
        """
        weighted_sum = 0
        total_weight = 0
        
        for model, signal in signals.items():
            value = self._signal_values[signal]
            weight = weights[model]
            weighted_sum += value * weight
            total_weight += weight
        
        if total_weight == 0:
            return 'neutral'
        
        avg_value = weighted_sum / total_weight
        
        if avg_value > 0.3:
            return 'bullish'
        elif avg_value < -0.3:
            return 'bearish'
        else:
            return 'neutral'
    
    def _generate_explanation(
        self,
        signals: Dict[str, str],
        conflict_score: float,
        consensus: str,
        dt_pred: ModelPrediction,
        rf_pred: ModelPrediction,
        lstm_pred: ModelPrediction,
        sentiment_data: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generate natural language explanation for the conflict analysis.
        
        Creates human-readable text explaining WHY models disagree.
        
        TRANSPARENCY: Explains the specific technical indicators and 
        conditions that led each model to its conclusion.
        """
        logger.debug("[CONFLICT_ENGINE] Generating natural language explanation")
        
        # Identify the conflicting models
        bullish_models = [m for m, s in signals.items() if s == 'bullish']
        bearish_models = [m for m, s in signals.items() if s == 'bearish']
        neutral_models = [m for m, s in signals.items() if s == 'neutral']
        
        # No conflict case
        if len(set(signals.values())) == 1:
            return f"All models agree on a {consensus} signal with high confidence."
        
        # Build explanation based on conflict type
        explanations = []
        
        # LSTM vs others (price trend vs indicators)
        if 'lstm' in bullish_models and any(m in bearish_models for m in ['decision_tree', 'random_forest']):
            explanations.append(
                "Price trend analysis (LSTM) shows upward momentum, but underlying "
                "technical indicators (Decision Tree/Random Forest) suggest weakening "
                "conditions. This divergence often precedes a trend reversal."
            )
        elif 'lstm' in bearish_models and any(m in bullish_models for m in ['decision_tree', 'random_forest']):
            explanations.append(
                "Price trend analysis (LSTM) shows downward pressure, but technical "
                "indicators suggest strengthening fundamentals. This may indicate "
                "a buying opportunity if the downtrend stabilizes."
            )
        
        # Decision Tree vs Random Forest (rule-based vs ensemble perspective)
        if 'decision_tree' in bullish_models and 'random_forest' in bearish_models:
            explanations.append(
                "Simple decision rules indicate bullish conditions, but the ensemble "
                "model sees more nuanced bearish signals. The market may be at a "
                "decision point with mixed signals."
            )
        elif 'decision_tree' in bearish_models and 'random_forest' in bullish_models:
            explanations.append(
                "Primary indicators show bearish patterns, but the broader feature "
                "analysis is bullish. Individual thresholds may be triggered while "
                "the overall picture remains positive."
            )
        
        # Add sentiment context if available
        if sentiment_data and 'overall_sentiment' in sentiment_data:
            sent = sentiment_data['overall_sentiment']
            if sent != consensus:
                explanations.append(
                    f"Note: Social sentiment is {sent}, which contrasts with the "
                    f"technical consensus of {consensus}."
                )
        
        # Combine explanations
        if explanations:
            return " ".join(explanations)
        else:
            # Generic explanation
            return (
                f"Models show {conflict_score:.0f}% disagreement. "
                f"Bullish: {', '.join(bullish_models) or 'none'}. "
                f"Bearish: {', '.join(bearish_models) or 'none'}. "
                f"Neutral: {', '.join(neutral_models) or 'none'}. "
                f"Weighted consensus: {consensus}."
            )
    
    def _generate_recommendations(
        self,
        conflict_score: float,
        signals: Dict[str, str],
        consensus: str
    ) -> List[str]:
        """
        Generate actionable recommendations based on conflict analysis.
        """
        recommendations = []
        
        if conflict_score > 70:
            recommendations.append("⚠️ HIGH CONFLICT: Consider waiting for clearer signals before taking action")
            recommendations.append("Review individual model reasoning to understand the disagreement")
        elif conflict_score > 40:
            recommendations.append("⚡ MODERATE CONFLICT: Reduce position size due to uncertainty")
            recommendations.append("Set tighter stop-losses to manage risk")
        else:
            recommendations.append("✅ LOW CONFLICT: Models show reasonable agreement")
        
        # Signal-specific recommendations
        if consensus == 'bullish':
            recommendations.append(f"Consensus is bullish - consider long positions with appropriate risk management")
        elif consensus == 'bearish':
            recommendations.append(f"Consensus is bearish - consider defensive positions or hedging")
        else:
            recommendations.append("Neutral consensus - market may be ranging, consider range-trading strategies")
        
        return recommendations


# =============================================================================
# MODEL MANAGER - Orchestrates All Models
# =============================================================================
class ModelManager:
    """
    High-level manager that orchestrates all ML models.
    
    Provides unified interface for:
    - Training all models
    - Getting predictions from all models
    - Running conflict analysis
    """
    
    def __init__(self):
        self.decision_tree = DecisionTreeModel()
        self.random_forest = RandomForestModel()
        self.lstm = LSTMModel()
        self.conflict_engine = ConflictEngine()
        
        logger.info("[MODEL_MANAGER] Initialized with all models")
    
    def train_all(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all models on the provided data."""
        logger.info("[MODEL_MANAGER] Training all models...")
        
        results = {
            'decision_tree': self.decision_tree.train(df),
            'random_forest': self.random_forest.train(df),
            'lstm': self.lstm.train(df)
        }
        
        logger.info("[MODEL_MANAGER] All models trained successfully")
        return results
    
    def get_full_prediction(
        self, 
        df: pd.DataFrame, 
        sentiment: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get predictions from all models and conflict analysis.
        
        Returns comprehensive prediction with all model outputs
        and conflict detection.
        """
        logger.info("[MODEL_MANAGER] Generating full prediction from all models")
        
        # Get individual predictions
        dt_pred = self.decision_tree.predict(df)
        rf_pred = self.random_forest.predict(df)
        lstm_pred = self.lstm.get_prediction_as_model_prediction(df)
        
        # Run conflict analysis
        conflict = self.conflict_engine.analyze(
            dt_pred, rf_pred, lstm_pred, sentiment
        )
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'predictions': {
                'decision_tree': asdict(dt_pred),
                'random_forest': asdict(rf_pred),
                'lstm': asdict(lstm_pred)
            },
            'conflict_analysis': asdict(conflict),
            'consensus': {
                'signal': conflict.consensus_signal,
                'conflict_alert': conflict.conflict_alert,
                'conflict_score': conflict.conflict_score
            }
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================
__all__ = [
    'Signal',
    'ModelPrediction',
    'ConflictAnalysis',
    'DecisionTreeModel',
    'RandomForestModel',
    'LSTMModel',
    'ConflictEngine',
    'ModelManager',
    'FEATURE_COLUMNS',
]


# =============================================================================
# STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ML ENGINE MODULE - STANDALONE TEST")
    print("=" * 60)
    
    # Generate test data
    np.random.seed(42)
    n_samples = 300
    
    test_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='H'),
        'open': np.random.uniform(40000, 45000, n_samples),
        'high': np.random.uniform(41000, 46000, n_samples),
        'low': np.random.uniform(39000, 44000, n_samples),
        'close': np.random.uniform(40000, 45000, n_samples),
        'volume': np.random.uniform(1000, 10000, n_samples),
        'rsi': np.random.uniform(20, 80, n_samples),
        'macd': np.random.uniform(-100, 100, n_samples),
        'macd_histogram': np.random.uniform(-50, 50, n_samples),
        'bb_pband': np.random.uniform(0, 1, n_samples),
        'price_momentum': np.random.uniform(-0.05, 0.05, n_samples),
        'volume_momentum': np.random.uniform(-0.5, 0.5, n_samples),
        'volatility': np.random.uniform(100, 500, n_samples),
    })
    
    # Test model manager
    manager = ModelManager()
    
    # Train all models
    print("\n[TEST] Training models...")
    train_results = manager.train_all(test_data)
    print(f"[TEST] Decision Tree accuracy: {train_results['decision_tree']['test_accuracy']:.2%}")
    print(f"[TEST] Random Forest accuracy: {train_results['random_forest']['test_accuracy']:.2%}")
    
    # Get full prediction
    print("\n[TEST] Getting predictions...")
    prediction = manager.get_full_prediction(test_data)
    print(f"[TEST] Consensus: {prediction['consensus']['signal']}")
    print(f"[TEST] Conflict Score: {prediction['conflict_analysis']['conflict_score']}")
    print(f"[TEST] Alert: {prediction['conflict_analysis']['conflict_alert']}")
    
    # Test decision tree logic path
    print("\n[TEST] Decision Tree Logic Path...")
    logic_path = manager.decision_tree.get_logic_path()
    print(f"[TEST] Tree depth: {logic_path['tree_stats']['depth']}")
    print(f"[TEST] Number of leaves: {logic_path['tree_stats']['n_leaves']}")
    
    # Test random forest feature impact
    print("\n[TEST] Random Forest Feature Impact...")
    feature_impact = manager.random_forest.get_feature_impact()
    print(f"[TEST] Top 3: {feature_impact['top_3_summary']}")
    
    # Test LSTM forecast
    print("\n[TEST] LSTM Forecast...")
    forecast = manager.lstm.predict_with_confidence(test_data)
    print(f"[TEST] Current price: ${forecast['current_price']:,.2f}")
    print(f"[TEST] 24h forecast: ${forecast['summary']['final_predicted_price']:,.2f}")
    print(f"[TEST] Expected change: {forecast['summary']['expected_change_percent']:+.2f}%")
    
    print("\n" + "=" * 60)
    print("ML ENGINE TEST COMPLETE")
    print("=" * 60)
