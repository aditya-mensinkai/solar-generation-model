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
    tilt: float = config.DEFAULT_TILT,
    azimuth: float = config.DEFAULT_AZIMUTH,
    shading_factor: float = config.DEFAULT_SHADING,
    inverter_type: str = "string",
    inverter_capacity_kw: float = None,
    years_of_operation: int = 0,
) -> Dict[str, Any]:
    """
    Predict monthly and annual solar generation for a rooftop PV system.

    Args:
        area:                 Rooftop / panel area in square metres.
        lat:                  Latitude in decimal degrees.
        lon:                  Longitude in decimal degrees.
        mode:                 Prediction mode ("physics" | "ml" | "hybrid").
        tilt:                 Panel tilt angle in degrees (0–90).
        azimuth:              Panel azimuth angle in degrees (0–360, 180=south).
        shading_factor:       Shading factor 0–1 (1.0 = no shading).
        inverter_type:        "string" or "micro" inverter type.
        inverter_capacity_kw: Optional inverter capacity for clipping.
        years_of_operation:   Years of panel operation (for degradation).

    Returns:
        Dict conforming to the SolarSense output schema with advanced factors.

    Raises:
        ValueError: On invalid inputs.
        RuntimeError: If PVGIS data cannot be fetched after retries.
    """
    # ── Step 1: Validate ──────────────────────────────────────────────────────
    validate_inputs(area, lat, lon)

    # Validate advanced parameters
    if not (0 <= tilt <= 90):
        raise ValueError(f"Tilt must be 0–90°; got {tilt}")
    if not (0 <= azimuth <= 360):
        raise ValueError(f"Azimuth must be 0–360°; got {azimuth}")
    if not (0 <= shading_factor <= 1):
        raise ValueError(f"Shading factor must be 0–1; got {shading_factor}")
    if inverter_type not in ("string", "micro"):
        raise ValueError(f"Inverter type must be 'string' or 'micro'; got {inverter_type}")

    logger.info("predict_solar called: area=%.1f m², lat=%.4f, lon=%.4f, mode=%s, "
                "tilt=%.1f°, azimuth=%.1f°",
                area, lat, lon, mode, tilt, azimuth)

    # ── Step 2: Fetch PVGIS data ──────────────────────────────────────────────
    raw_pvgis = fetch_pvgis_data(lat, lon)

    # ── Step 3: Preprocess → monthly features ─────────────────────────────────
    monthly_features = preprocess_pvgis(raw_pvgis)

    # ── Step 4: Physics model (always computed — used as base or fallback) ────
    phys_monthly: MonthlyEnergy = monthly_energy(
        area, monthly_features,
        tilt=tilt, azimuth=azimuth, shading_factor=shading_factor,
        inverter_type=inverter_type, inverter_capacity_kw=inverter_capacity_kw,
        years_of_operation=years_of_operation
    )
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
            # Run ML predictions for every month with safety clamps
            ml_monthly: MonthlyEnergy = {}
            month_num_map = {
                "01": 1, "02": 2, "03": 3, "04": 4, "05": 5, "06": 6,
                "07": 7, "08": 8, "09": 9, "10": 10, "11": 11, "12": 12
            }

            for month, features in monthly_features.items():
                phys_val = phys_monthly.get(month, 0.0)
                try:
                    # Build feature vector with location and system parameters
                    fvec = build_feature_vector(
                        monthly_features=features,
                        area=area,
                        lat=lat,
                        lon=lon,
                        month=month_num_map.get(month, 1),
                        tilt=tilt,
                        azimuth=azimuth,
                    )
                    pred = predict_energy(ml_model, fvec)

                    # SAFETY CLAMPS: Prevent unrealistic ML predictions
                    # Clamp ML prediction to 0.5x - 1.5x of physics estimate
                    lower_bound = phys_val * 0.5
                    upper_bound = phys_val * 1.5

                    if pred < 0:
                        pred = 0.0
                    if pred > upper_bound:
                        pred = upper_bound
                    if pred < lower_bound:
                        pred = lower_bound

                    ml_monthly[month] = pred
                    logger.debug("%s: Physics=%.2f, ML=%.2f (clamped)", month, phys_val, pred)

                except Exception as e:
                    logger.warning("ML failed for month %s: %s — using physics fallback", month, e)
                    ml_monthly[month] = phys_val

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
            "advanced_factors": {
                "tilt": tilt,
                "azimuth": azimuth,
                "shading_factor": shading_factor,
                "inverter_type": inverter_type,
                "inverter_capacity_kw": inverter_capacity_kw,
                "years_of_operation": years_of_operation,
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
    Produce a weighted average blend of physics and ML monthly energy estimates.

    Uses configured weights (ML_WEIGHT + PHYSICS_WEIGHT = 1.0) to combine
    physics-based and ML-based predictions. This provides a stable hybrid
    that leverages ML corrections while maintaining physics explainability.

    Formula: blended = ML_WEIGHT × ml + PHYSICS_WEIGHT × physics

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
        # Weighted average blend (60% ML, 40% physics by default)
        blended_val = config.ML_WEIGHT * m + config.PHYSICS_WEIGHT * p
        blended[month] = round(blended_val, 3)
    return blended
