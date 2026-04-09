"""
SolarSense Configuration
------------------------
All tunable constants live here. No magic numbers elsewhere.
"""

# ── Data Source ────────────────────────────────────────────────────────────────
PVGIS_BASE_URL = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
PVGIS_TIMEOUT_SECONDS = 30
PVGIS_MAX_RETRIES = 3
PVGIS_RETRY_BACKOFF = 2.0          # seconds; doubles on each retry

# ── Physics Model ──────────────────────────────────────────────────────────────
PANEL_EFFICIENCY = 0.20            # standard monocrystalline panel
# NOTE: PVGIS "loss" parameter already accounts for system losses (inverter + wiring)
# so we set PERFORMANCE_RATIO = 1.0 to avoid double-counting
PERFORMANCE_RATIO = 1.0            # PVGIS already accounts for system losses
TEMP_LOSS_THRESHOLD_C = 30         # above this, efficiency drops
TEMP_LOSS_FACTOR = 0.90            # multiplier when above threshold
DEFAULT_DUST_FACTOR = 0.95         # 5 % dust/soiling loss
DEFAULT_SHADING = 1.0              # no shading (1.0 = full sun)
SOILING_FACTOR = 0.90              # 10% soiling loss for dusty regions (India)
# Actual days per month for accurate calculations
# February fixed at 28 days (non-leap approximation)
# Impact negligible for annual estimation
DAYS_PER_MONTH = {
    "01": 31, "02": 28, "03": 31, "04": 30,
    "05": 31, "06": 30, "07": 31, "08": 31,
    "09": 30, "10": 31, "11": 30, "12": 31
}

# Peak Sun Hours (PSH): effective hours of peak sunlight per day
# Solar panels only generate energy during ~5–6 hours of peak sunlight
# Using 24 hours would cause ~5× overestimation
PEAK_SUN_HOURS = 5.5  # Average for India (configurable by location)

# Advanced Physics Parameters
TEMP_COEFF = 0.004          # 0.4% efficiency loss per °C above 25°C
DEGRADATION_RATE = 0.005    # 0.5% annual panel degradation
DEFAULT_TILT = 20.0         # default panel tilt (degrees)
DEFAULT_AZIMUTH = 180.0     # default azimuth (180° = south-facing for India)

# ── Hybrid Blending Weights ────────────────────────────────────────────────────
ML_WEIGHT = 0.60
PHYSICS_WEIGHT = 0.40              # ML_WEIGHT + PHYSICS_WEIGHT must equal 1.0

# ── Model Persistence ─────────────────────────────────────────────────────────
ML_MODEL_PATH = "solar_predictor/models/xgboost_correction.pkl"

# ── Prediction Mode ───────────────────────────────────────────────────────────
# Options: "physics" | "ml" | "hybrid"
# "ml" and "hybrid" fall back to "physics" when no trained model exists.
PREDICTION_MODE = "physics"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "WARNING"  # Reduce verbosity in production; change to INFO for debugging
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# ── Metadata ──────────────────────────────────────────────────────────────────
DATA_SOURCE_NOTE = "PVGIS v5.2 (EU JRC) — spatial resolution ~5 km; results may vary ±10%"
ERROR_MARGIN = "±10%"
