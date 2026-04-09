"""
SolarSense Data Fetcher
-----------------------
Fetches hourly/daily irradiance and weather data from the PVGIS v5.2 API
(European Commission Joint Research Centre).

PVGIS returns hourly time-series data which we aggregate to monthly averages
in preprocessing.py. Caching uses functools.lru_cache for bounded memory.
"""

import time
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import requests

from solar_predictor import config
from solar_predictor.utils import get_logger

logger = get_logger(__name__)


def _cache_key(lat: float, lon: float) -> Tuple[float, float]:
    """Round coordinates to 3 dp for higher precision cache keys (~100m)."""
    return (round(lat, 3), round(lon, 3))


@lru_cache(maxsize=100)
def _cached_fetch(cache_key: Tuple[float, float], lat: float, lon: float) -> Dict[str, Any]:
    """
    Internal cached fetch function.
    cache_key is only used for cache hashing; lat/lon are the actual coordinates.
    """
    params = {
        "lat": lat,
        "lon": lon,
        "usehorizon": 1,
        "peakpower": 1,          # normalise to 1 kWp so we can scale by real area later
        "pvtechchoice": "crystSi",
        "mountingplace": "building",
        # IMPORTANT: loss reduced to 10% because dust_factor is applied in physics model
        # PVGIS 14% includes ~4% dust; we apply custom 5% dust_factor separately
        "loss": 10,              # reduced system losses (wiring, inverter only)
        "outputformat": "json",
        # PVGIS seriescalc uses a typical meteorological year (TMY),
        # so multi-year ranges do not provide true averaging.
        "startyear": 2020,
        "endyear": 2020,
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, config.PVGIS_MAX_RETRIES + 1):
        try:
            logger.info(
                "PVGIS request attempt %d/%d (lat=%.4f, lon=%.4f)",
                attempt, config.PVGIS_MAX_RETRIES, lat, lon,
            )
            response = requests.get(
                config.PVGIS_BASE_URL,
                params=params,
                timeout=config.PVGIS_TIMEOUT_SECONDS,
            )
            # Log error details before raising
            if response.status_code != 200:
                logger.error("PVGIS ERROR RESPONSE: %s", response.text)
                try:
                    error_json = response.json()
                    logger.error("PVGIS error JSON: %s", error_json)
                except Exception:
                    pass
            response.raise_for_status()
            data: Dict[str, Any] = response.json()
            logger.info("PVGIS fetch successful.")
            return data

        except requests.exceptions.HTTPError as exc:
            logger.warning("HTTP error on attempt %d: %s", attempt, exc)
            last_error = exc
        except requests.exceptions.ConnectionError as exc:
            logger.warning("Connection error on attempt %d: %s", attempt, exc)
            last_error = exc
        except requests.exceptions.Timeout as exc:
            logger.warning("Timeout on attempt %d: %s", attempt, exc)
            last_error = exc
        except requests.exceptions.RequestException as exc:
            logger.warning("Request error on attempt %d: %s", attempt, exc)
            last_error = exc

        if attempt < config.PVGIS_MAX_RETRIES:
            wait = config.PVGIS_RETRY_BACKOFF * attempt
            logger.info("Retrying in %.1f s…", wait)
            time.sleep(wait)

    raise RuntimeError(
        f"PVGIS API unavailable after {config.PVGIS_MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


def fetch_pvgis_data(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetch one year of hourly PV generation and meteorological data from PVGIS.

    Uses the ``seriescalc`` endpoint which returns hourly time-series for a
    reference meteorological year. Key parameters returned:

    - ``P``   – PV system output power (W)  [used to derive GHI proxy]
    - ``G(i)``– In-plane irradiance (W/m²)   → GHI
    - ``T2m`` – Ambient temperature at 2 m (°C)
    - ``WS10m``– Wind speed at 10 m (m/s)
    - ``Int`` – Solar radiation reconstruction flag

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.

    Returns:
        Raw PVGIS JSON response as a Python dict.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    key = _cache_key(lat, lon)
    logger.info("Fetching PVGIS data for (lat=%.2f, lon=%.2f)", lat, lon)
    return _cached_fetch(key, lat, lon)


def clear_cache() -> None:
    """Purge the in-process PVGIS cache (useful for testing)."""
    _cached_fetch.cache_clear()
    logger.debug("PVGIS cache cleared.")
