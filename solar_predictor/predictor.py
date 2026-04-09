"""
SolarSense Predictor
--------------------
Main prediction pipeline.

Flow:
    1. Validate inputs
    2. Fetch PVGIS data (with cache + retry)
    3. Preprocess → monthly feature dict
    4. Physics model → monthly_energy() + annual_energy()
    5. ML model (if trained + mode allows) → monthly corrections
    6. Hybrid blend (if both available)
    7. Compute avg daily & seasonal trend
    8. Return structured JSON

Mode behaviour (controlled by config.PREDICTION_MODE):
    "physics" → always use physics model only
    "ml"      → use ML if available, else fall back to physics (with warning)
    "hybrid"  → blend ML + physics if ML available, else fall back to physics
"""

from typing import Any, Dict

from solar_predictor import config
from solar_predictor.data_fetcher import fetch_pvgis_data
from solar_predictor.ml_model import (
    FEATURE_ORDER,
    build_feature_vector,
    load_model,
    predict_energy,
)
from solar_predictor.physics_model import adjusted_energy  # noqa: F401  (used implicitly via monthly_energy)
from solar_predictor.physics_model import annual_energy, monthly_energy
from solar_predictor.preprocessing import preprocess_pvgis
from solar_predictor.utils import (
    MonthlyEnergy,
    build_seasonal_trend,
    get_logger,
    safe_round,
    validate_inputs,
)

logger = get_logger(__name__)


def predict_solar(
    area: float,
    lat: float,
    lon: float,
    mode: str = config.PREDICTION_MODE,
) -> Dict[str, Any]:
    """
    Predict monthly and annual solar generation for a rooftop PV system.

    Args:
        area: Rooftop / panel area in square metres.
        lat:  Latitude in decimal degrees.
        lon:  Longitude in decimal degrees.
        mode: Override for prediction mode ("physics" | "ml" | "hybrid").
              Defaults to config.PREDICTION_MODE.

    Returns:
        Dict conforming to the SolarSense output schema:
        {
            "monthly_generation":  {"01": float, ..., "12": float},
            "annual_generation":   float,
            "avg_daily_generation": float,
            "seasonal_trend":      {"Winter": float, ...},
            "metadata": {
                "model_used":  "physics" | "ml" | "hybrid",
                "error_margin": "±10%",
                "data_source":  "PVGIS v5.2 (EU JRC)",
                "note":        str,
            }
        }

    Raises:
        ValueError: On invalid inputs.
        RuntimeError: If PVGIS data cannot be fetched after retries.
    """
    # ── Step 1: Validate ──────────────────────────────────────────────────────
    validate_inputs(area, lat, lon)
    logger.info("predict_solar called: area=%.1f m², lat=%.4f, lon=%.4f, mode=%s",
                area, lat, lon, mode)

    # ── Step 2: Fetch PVGIS data ──────────────────────────────────────────────
    raw_pvgis = fetch_pvgis_data(lat, lon)

    # ── Step 3: Preprocess → monthly features ─────────────────────────────────
    monthly_features = preprocess_pvgis(raw_pvgis)

    # ── Step 4: Physics model (always computed — used as base or fallback) ────
    phys_monthly: MonthlyEnergy = monthly_energy(area, monthly_features)
    phys_annual: float = annual_energy(phys_monthly)

    # ── Step 5 & 6: ML and/or hybrid ─────────────────────────────────────────
    model_used: str
    final_monthly: MonthlyEnergy

    if mode == "physics":
        final_monthly = phys_monthly
        model_used = "physics"

    elif mode in ("ml", "hybrid"):
        ml_model = load_model()

        if ml_model is None:
            logger.warning(
                "Mode '%s' requested but no trained ML model found. "
                "Falling back to physics-only.",
                mode,
            )
            final_monthly = phys_monthly
            model_used = "physics"
        else:
            # Run ML predictions for every month
            ml_monthly: MonthlyEnergy = {}
            for month, features in monthly_features.items():
                fvec = build_feature_vector(features, area)
                ml_monthly[month] = predict_energy(ml_model, fvec)

            if mode == "ml":
                final_monthly = ml_monthly
                model_used = "ml"
            else:  # "hybrid"
                final_monthly = _blend(phys_monthly, ml_monthly)
                model_used = "hybrid"
    else:
        logger.warning("Unknown mode '%s'; defaulting to physics.", mode)
        final_monthly = phys_monthly
        model_used = "physics"

    # ── Step 7: Derived statistics ────────────────────────────────────────────
    final_annual: float = annual_energy(final_monthly)
    avg_daily: float = safe_round(final_annual / 365.0)
    seasonal: Dict[str, float] = build_seasonal_trend(final_monthly)

    # ── Step 8: Assemble output ───────────────────────────────────────────────
    result: Dict[str, Any] = {
        "monthly_generation": {m: safe_round(v) for m, v in final_monthly.items()},
        "annual_generation": safe_round(final_annual),
        "avg_daily_generation": avg_daily,
        "seasonal_trend": seasonal,
        "metadata": {
            "model_used": model_used,
            "error_margin": config.ERROR_MARGIN,
            "data_source": "PVGIS v5.2 (EU JRC)",
            "note": config.DATA_SOURCE_NOTE,
            "inputs": {
                "area_m2": area,
                "lat": lat,
                "lon": lon,
            },
        },
    }

    logger.info(
        "Prediction complete — annual=%.1f kWh, model=%s",
        final_annual, model_used,
    )
    return result


# ── Private helpers ────────────────────────────────────────────────────────────

def _blend(
    phys: MonthlyEnergy,
    ml: MonthlyEnergy,
) -> MonthlyEnergy:
    """
    Produce a weighted blend of physics and ML monthly energy estimates.

    Weights come from config (ML_WEIGHT + PHYSICS_WEIGHT = 1.0).

    Args:
        phys: Physics monthly kWh dict.
        ml:   ML monthly kWh dict.

    Returns:
        Blended MonthlyEnergy dict.
    """
    blended: MonthlyEnergy = {}
    for month in phys:
        p = phys.get(month, 0.0)
        m = ml.get(month, 0.0)
        blended[month] = round(
            config.ML_WEIGHT * m + config.PHYSICS_WEIGHT * p, 3
        )
    return blended
