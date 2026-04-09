"""
SolarSense Utilities
--------------------
Shared helpers: logging factory, validation, and lightweight type aliases.
"""

import logging
import sys
from typing import Dict, Any

from solar_predictor import config


# ── Logging ───────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a consistently configured logger for *name*."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    return logger


# ── Type Aliases ──────────────────────────────────────────────────────────────

MonthlyData = Dict[str, Dict[str, float]]   # e.g. {"01": {"GHI": 5.2, ...}}
MonthlyEnergy = Dict[str, float]             # e.g. {"01": 123.4, ...}


# ── Validation Helpers ────────────────────────────────────────────────────────

def validate_inputs(area: float, lat: float, lon: float) -> None:
    """
    Raise ValueError for obviously invalid prediction inputs.

    Args:
        area: Rooftop area in square metres (must be > 0).
        lat:  Latitude in decimal degrees (−90 to 90).
        lon:  Longitude in decimal degrees (−180 to 180).
    """
    if area <= 0:
        raise ValueError(f"Rooftop area must be positive; got {area}")
    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitude must be in [−90, 90]; got {lat}")
    if not (-180 <= lon <= 180):
        raise ValueError(f"Longitude must be in [−180, 180]; got {lon}")


def safe_round(value: float, decimals: int = 2) -> float:
    """Round *value* to *decimals* places, returning 0.0 for non-finite results."""
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return 0.0


def seasonal_label(month_str: str) -> str:
    """
    Return the meteorological season name for a zero-padded month string.

    Args:
        month_str: Two-digit month, e.g. "01".

    Returns:
        One of "Winter", "Spring", "Summer", "Autumn".
    """
    month = int(month_str)
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def build_seasonal_trend(monthly_energy: MonthlyEnergy) -> Dict[str, float]:
    """
    Aggregate monthly kWh figures into seasonal totals.

    Args:
        monthly_energy: Mapping of zero-padded month strings to kWh values.

    Returns:
        Dict with keys "Winter", "Spring", "Summer", "Autumn" and summed kWh.
    """
    seasons: Dict[str, float] = {"Winter": 0.0, "Spring": 0.0, "Summer": 0.0, "Autumn": 0.0}
    for month, kwh in monthly_energy.items():
        seasons[seasonal_label(month)] += kwh
    return {k: safe_round(v) for k, v in seasons.items()}
