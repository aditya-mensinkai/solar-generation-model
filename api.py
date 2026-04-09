"""
SolarSense FastAPI
------------------
POST /solar-predict  →  predict_solar()

Run locally:
    uvicorn api:app --reload --port 8000

Then test with:
    curl -X POST http://localhost:8000/solar-predict \\
         -H "Content-Type: application/json" \\
         -d '{"area": 20.0, "lat": 12.97, "lon": 77.59}'
"""

from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from solar_predictor import config
from solar_predictor.predictor import predict_solar
from solar_predictor.utils import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="SolarSense API",
    description="Physics-first rooftop solar generation predictor backed by PVGIS data.",
    version="1.0.0",
)


# ── Request / Response schemas ────────────────────────────────────────────────

class SolarPredictRequest(BaseModel):
    area: float = Field(..., gt=0, description="Rooftop area in square metres")
    lat: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    lon: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    mode: Optional[Literal["physics", "ml", "hybrid"]] = Field(
        default=None,
        description="Prediction mode override. Defaults to server config.",
    )

    @field_validator("area")
    @classmethod
    def area_must_be_realistic(cls, v: float) -> float:
        if v > 10_000:
            raise ValueError("area exceeds 10,000 m² — this endpoint is for rooftop systems.")
        return v


class MetadataSchema(BaseModel):
    model_used: str
    error_margin: str
    data_source: str
    note: str
    inputs: Dict[str, Any]


class SolarPredictResponse(BaseModel):
    monthly_generation: Dict[str, float]
    annual_generation: float
    avg_daily_generation: float
    seasonal_trend: Dict[str, float]
    metadata: MetadataSchema


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check() -> Dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "service": "SolarSense"}


@app.post("/solar-predict", response_model=SolarPredictResponse)
def solar_predict(request: SolarPredictRequest) -> Dict[str, Any]:
    """
    Predict monthly and annual solar generation for a rooftop PV system.

    Inputs:
    - **area**: Panel/rooftop area in m²
    - **lat**: Latitude in decimal degrees
    - **lon**: Longitude in decimal degrees
    - **mode** *(optional)*: `physics` | `ml` | `hybrid`

    Returns structured generation forecast with seasonal trend and metadata.
    """
    mode = request.mode or config.PREDICTION_MODE
    logger.info(
        "POST /solar-predict — area=%.1f m², lat=%.4f, lon=%.4f, mode=%s",
        request.area, request.lat, request.lon, mode,
    )

    try:
        result = predict_solar(
            area=request.area,
            lat=request.lat,
            lon=request.lon,
            mode=mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Upstream data source unavailable: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during prediction.")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc

    return result
