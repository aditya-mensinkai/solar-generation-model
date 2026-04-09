"""
SolarSense ML Model
-------------------
XGBoost correction layer.

Current status: SCAFFOLD — physics-first architecture.
The model is trained on real-world metered generation data vs PVGIS estimates.
Until that labelled dataset exists, this module:

  1. Provides the full train/predict/save/load interface.
  2. Returns None from load_model() when no saved model is found.
  3. predictor.py gracefully falls back to physics-only when load_model() → None.

Feature vector (7 features):
    [GHI, TEMP, DNI, DHI, HUMIDITY, WIND, AREA]

Training data schema expected by train_model():
    X: array-like of shape (n_samples, 7)  — feature matrix
    y: array-like of shape (n_samples,)    — actual monthly kWh (ground truth)
"""

import os
import pickle
from typing import Any, Dict, List, Optional

from solar_predictor import config
from solar_predictor.utils import get_logger

logger = get_logger(__name__)

# Feature ordering must stay consistent between training and inference
FEATURE_ORDER: List[str] = ["GHI", "TEMP", "DNI", "DHI", "HUMIDITY", "WIND", "AREA"]


def build_feature_vector(monthly_features: Dict[str, float], area: float) -> List[float]:
    """
    Construct an ordered feature vector for a single month.

    Args:
        monthly_features: Single-month dict from preprocessing output,
                          e.g. {"GHI": 5.2, "TEMP": 28.1, ...}.
        area:             Rooftop area in m².

    Returns:
        List of 7 floats in FEATURE_ORDER.
    """
    return [
        monthly_features.get("GHI", 0.0),
        monthly_features.get("TEMP", 0.0),
        monthly_features.get("DNI", 0.0),
        monthly_features.get("DHI", 0.0),
        monthly_features.get("HUMIDITY", 0.0),
        monthly_features.get("WIND", 0.0),
        area,
    ]


def train_model(X: Any, y: Any) -> Any:
    """
    Train an XGBoost regressor as a correction layer over physics estimates.

    Args:
        X: Feature matrix, shape (n_samples, 7). Each row is one month's
           feature vector (see FEATURE_ORDER).
        y: Target vector, shape (n_samples,). Actual monthly kWh from metered data.

    Returns:
        Trained XGBRegressor instance.

    Raises:
        ImportError: If xgboost is not installed.
        RuntimeError: If training fails.
    """
    try:
        from xgboost import XGBRegressor  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "xgboost is not installed. Run: pip install xgboost"
        ) from exc

    logger.info("Training XGBoost correction model on %d samples…", len(y))

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    try:
        model.fit(X, y)
        logger.info("XGBoost training complete.")
        return model
    except Exception as exc:
        raise RuntimeError(f"XGBoost training failed: {exc}") from exc


def predict_energy(model: Any, features: List[float]) -> float:
    """
    Run inference for a single monthly feature vector.

    Args:
        model:    Trained XGBRegressor instance.
        features: Feature vector of length 7 (see FEATURE_ORDER).

    Returns:
        Predicted monthly kWh as a float.
    """
    import numpy as np  # type: ignore

    x = np.array([features], dtype=float)
    prediction: float = float(model.predict(x)[0])
    logger.debug("ML prediction: %.2f kWh", prediction)
    return max(prediction, 0.0)   # clamp negatives


def save_model(model: Any) -> None:
    """
    Persist the trained model to disk using pickle.

    Args:
        model: Trained XGBRegressor to save.
    """
    os.makedirs(os.path.dirname(config.ML_MODEL_PATH), exist_ok=True)
    with open(config.ML_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", config.ML_MODEL_PATH)


def load_model() -> Optional[Any]:
    """
    Load a previously saved XGBoost model.

    Returns:
        Trained model if found, or None if the file does not exist.
        Callers should treat None as 'no ML model available'.
    """
    if not os.path.exists(config.ML_MODEL_PATH):
        logger.info(
            "No trained ML model found at %s — using physics-only mode.",
            config.ML_MODEL_PATH,
        )
        return None
    with open(config.ML_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("ML model loaded from %s", config.ML_MODEL_PATH)
    return model
