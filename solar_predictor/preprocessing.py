"""
SolarSense Preprocessing
------------------------
Converts raw PVGIS hourly time-series JSON into clean monthly-average feature
dicts used by both the physics model and the ML model.

Output schema:
{
    "01": { "GHI": float, "TEMP": float, "DNI": float,
            "DHI": float, "HUMIDITY": float|None, "WIND": float },
    ...
    "12": { ... }
}

PVGIS does not return separate DNI/DHI/Humidity fields from the seriescalc
endpoint. We use in-plane irradiance G(i) as GHI proxy, derive a reasonable
DNI split, and set humidity to None (not available from seriescalc).
"""

import math
from collections import defaultdict
from typing import Any, Dict, List

from solar_predictor.utils import MonthlyData, get_logger

logger = get_logger(__name__)

# PVGIS JSON keys we care about
_FIELD_MAP = {
    "G(i)": "GHI",    # in-plane irradiance W/m²
    "T2m":  "TEMP",   # ambient temperature °C
    "WS10m": "WIND",  # wind speed m/s
}

# Months that are present in any given year
_ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]


def preprocess_pvgis(raw: Dict[str, Any]) -> MonthlyData:
    """
    Transform a raw PVGIS seriescalc JSON response into monthly feature averages.

    Args:
        raw: Full PVGIS API JSON response (as returned by data_fetcher.fetch_pvgis_data).

    Returns:
        Dict of 12 monthly feature dicts (see module docstring for schema).

    Raises:
        KeyError: If the expected PVGIS response structure is not found.
        ValueError: If no valid hourly records are present after cleaning.
    """
    try:
        hourly_records: List[Dict[str, Any]] = raw["outputs"]["hourly"]
    except KeyError as exc:
        raise KeyError(
            "Unexpected PVGIS response structure; missing 'outputs.hourly'."
        ) from exc

    if not hourly_records:
        raise ValueError("PVGIS returned an empty hourly dataset.")

    # Accumulate raw values per month
    # NOTE: HUMIDITY is not available from PVGIS seriescalc endpoint
    buckets: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {k: [] for k in ("GHI", "TEMP", "WIND", "DNI", "DHI")}
    )

    skipped = 0
    for record in hourly_records:
        month_str = _extract_month(record.get("time", ""))
        if month_str is None:
            skipped += 1
            continue

        ghi = _safe_float(record.get("G(i)"))
        temp = _safe_float(record.get("T2m"))
        wind = _safe_float(record.get("WS10m"))

        if ghi is None or temp is None:
            skipped += 1
            continue

        # Derive DNI/DHI via simple erbs decomposition (fraction of diffuse)
        dni, dhi = _erbs_split(ghi)

        buckets[month_str]["GHI"].append(ghi)
        buckets[month_str]["TEMP"].append(temp)
        buckets[month_str]["WIND"].append(wind if wind is not None else 0.0)
        buckets[month_str]["DNI"].append(dni)
        buckets[month_str]["DHI"].append(dhi)
        # HUMIDITY not accumulated - not available from PVGIS seriescalc

    if skipped:
        logger.warning("Skipped %d malformed PVGIS records during preprocessing.", skipped)

    monthly: MonthlyData = {}
    for month in _ALL_MONTHS:
        data = buckets.get(month, {})
        # Validate that each month has data - raise error if empty
        monthly[month] = {
            "GHI":      _avg(data.get("GHI", []), month),
            "TEMP":     _avg(data.get("TEMP", []), month),
            "DNI":      _avg(data.get("DNI", []), month),
            "DHI":      _avg(data.get("DHI", []), month),
            "HUMIDITY": None,  # PVGIS seriescalc does not provide humidity
            "WIND":     _avg(data.get("WIND", []), month),
        }

    _validate_monthly(monthly)
    logger.info("Preprocessing complete — %d months populated.", len(monthly))
    return monthly


# ── Private helpers ────────────────────────────────────────────────────────────

def _extract_month(time_str: str) -> str | None:
    """
    Parse the month from a PVGIS time string like '20200101:0000'.

    Returns:
        Zero-padded month string ("01"–"12") or None on parse failure.
    """
    try:
        # format: YYYYMMDD:HHMM
        return time_str[4:6]
    except (IndexError, TypeError):
        return None


def _safe_float(value: Any) -> float | None:
    """Convert *value* to float; return None if invalid or NaN."""
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _avg(values: List[float], month: str = "") -> float:
    """Return the mean of *values*, or raise ValueError for an empty list."""
    if not values:
        raise ValueError(f"No valid data for month {month}")
    return round(sum(values) / len(values), 4)


def _erbs_split(ghi: float) -> tuple[float, float]:
    """
    Estimate DNI and DHI from GHI using the Erbs decomposition model.

    Reference: Erbs, D.G., Klein, S.A., Duffie, J.A. (1982).

    Args:
        ghi: Global horizontal irradiance in W/m².

    Returns:
        (dni, dhi) both in W/m².
    """
    if ghi <= 0:
        return 0.0, 0.0
    # Simplified clearness index → diffuse fraction
    kt = min(ghi / 1000.0, 1.0)
    if kt <= 0.22:
        df = 1.0 - 0.09 * kt
    elif kt <= 0.80:
        df = 0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4
    else:
        df = 0.165
    dhi = df * ghi
    dni = (ghi - dhi) / max(kt, 0.001)
    return round(dni, 4), round(dhi, 4)


def _validate_monthly(monthly: MonthlyData) -> None:
    """Warn if any month has zero GHI (likely missing data)."""
    zero_months = [m for m, d in monthly.items() if d["GHI"] == 0.0]
    if zero_months:
        logger.warning(
            "Months with zero GHI (possible missing data): %s",
            ", ".join(zero_months),
        )
