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

# Legacy feature ordering (kept for backwards compatibility)
FEATURE_ORDER: List[str] = ["GHI", "TEMP", "DNI", "DHI", "HUMIDITY", "WIND", "AREA"]

# Module-level cache to avoid reloading model on every call
_model_cache = None


def build_feature_vector(
    monthly_features: Dict[str, float],
    area: float,
    lat: float = 0.0,
    lon: float = 0.0,
    month: int = 1,
    tilt: float = 20.0,
    azimuth: float = 180.0,
    panel_efficiency: float = 0.20,
    performance_ratio: float = 0.80,
) -> List[float]:
    """
    Construct an ordered feature vector for a single month.

    Supports both legacy 7-feature format and new 12-feature format
    required by the trained model.

    Args:
        monthly_features: Single-month dict from preprocessing output,
                          e.g. {"GHI": 5.2, "TEMP": 28.1, ...}.
        area:             Rooftop area in m².
        lat:              Latitude (for new model format).
        lon:              Longitude (for new model format).
        month:            Month number 1-12 (for new model format).
        tilt:             Panel tilt angle in degrees (for new model format).
        azimuth:          Panel azimuth angle (for new model format).
        panel_efficiency: Panel efficiency factor (for new model format).
        performance_ratio: Performance ratio (for new model format).

    Returns:
        List of features matching the model's expected format.
    """
    # Handle HUMIDITY which may be None (not available from PVGIS seriescalc)
    humidity = monthly_features.get("HUMIDITY")
    if humidity is None:
        humidity = -1.0

    # Build feature dict with new model column names
    feature_dict = {
        "latitude": lat,
        "longitude": lon,
        "month": month,
        "GHI": monthly_features.get("GHI", 0.0),
        "temperature": monthly_features.get("TEMP", 0.0),
        "humidity": humidity,
        "wind_speed": monthly_features.get("WIND", 0.0),
        "rooftop_area": area,
        "panel_efficiency": panel_efficiency,
        "performance_ratio": performance_ratio,
        "tilt_angle": tilt,
        "orientation": azimuth,
    }

    return feature_dict


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


def predict_energy(model: Any, features: Dict[str, float]) -> float:
    """
    Run inference for a single monthly feature vector.

    Supports both legacy XGBoost models and new sklearn Pipeline models
    stored as dicts with 'pipeline', 'feature_cols', and 'target_col' keys.

    Args:
        model:    Trained model (XGBRegressor or dict with pipeline).
        features: Feature dict with keys matching model's expected columns.

    Returns:
        Predicted monthly kWh as a float.
    """
    import numpy as np  # type: ignore

    # Check if model is the new dict format
    if isinstance(model, dict) and "pipeline" in model:
        pipeline = model["pipeline"]
        feature_cols = model.get("feature_cols", [])

        # Extract features in the order expected by the model
        feature_list = [features.get(col, 0.0) for col in feature_cols]
        x = np.array([feature_list], dtype=float)
        prediction: float = float(pipeline.predict(x)[0])
    else:
        # Legacy format: direct XGBRegressor
        # Convert dict to list in FEATURE_ORDER
        feature_list = [
            features.get("GHI", 0.0),
            features.get("TEMP", 0.0),
            features.get("DNI", 0.0),
            features.get("DHI", 0.0),
            features.get("HUMIDITY", 0.0),
            features.get("WIND", 0.0),
            features.get("rooftop_area", 0.0),
        ]
        x = np.array([feature_list], dtype=float)
        prediction = float(model.predict(x)[0])

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
    Load a previously saved XGBoost model with caching and error handling.

    Supports both standard pickle files and zlib-compressed pickles.

    Returns:
        Trained model if found, or None if the file does not exist or is corrupted.
        Callers should treat None as 'no ML model available'.
    """
    global _model_cache

    # Return cached model if available
    if _model_cache is not None:
        return _model_cache

    if not os.path.exists(config.ML_MODEL_PATH):
        logger.warning(
            "ML model not found at %s — using physics fallback.",
            config.ML_MODEL_PATH,
        )
        return None

    try:
        with open(config.ML_MODEL_PATH, "rb") as f:
            raw_data = f.read()

        # Try standard pickle first
        try:
            _model_cache = pickle.loads(raw_data)
        except pickle.UnpicklingError:
            # Try zlib decompression (for compressed models)
            import zlib
            try:
                decompressed = zlib.decompress(raw_data)
                _model_cache = pickle.loads(decompressed)
                logger.debug("Model loaded with zlib decompression")
            except Exception:
                # Not zlib either
                raise

        logger.info("ML model loaded from %s", config.ML_MODEL_PATH)
        return _model_cache
    except Exception as e:
        logger.error("Failed to load ML model: %s", e)
        return None
