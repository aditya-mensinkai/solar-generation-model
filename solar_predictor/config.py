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
# Actual days per month for accurate calculations (non-leap year)
DAYS_PER_MONTH = {
    "01": 31, "02": 28, "03": 31, "04": 30,
    "05": 31, "06": 30, "07": 31, "08": 31,
    "09": 30, "10": 31, "11": 30, "12": 31
}

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
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# ── Metadata ──────────────────────────────────────────────────────────────────
DATA_SOURCE_NOTE = "PVGIS v5.2 (EU JRC) — spatial resolution ~5 km; results may vary ±10%"
ERROR_MARGIN = "±10%"
