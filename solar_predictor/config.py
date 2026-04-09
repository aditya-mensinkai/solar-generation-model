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
PERFORMANCE_RATIO = 0.75           # accounts for inverter + wiring losses
TEMP_LOSS_THRESHOLD_C = 30         # above this, efficiency drops
TEMP_LOSS_FACTOR = 0.90            # multiplier when above threshold
DEFAULT_DUST_FACTOR = 0.95         # 5 % dust/soiling loss
DEFAULT_SHADING = 1.0              # no shading (1.0 = full sun)
DAYS_PER_MONTH = 30                # uniform approximation; kept here so it's easy to change

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
