"""
SolarSense Physics Model
------------------------
Deterministic, explainable energy yield calculations.

Three functions are MANDATED by the project spec and used by predictor.py:
    1. adjusted_energy(area, ghi, temp, shading, dust_factor)
    2. monthly_energy(area, monthly_ghi)
    3. annual_energy(monthly_energy_dict)
"""

from solar_predictor import config
from solar_predictor.utils import MonthlyData, MonthlyEnergy, get_logger

logger = get_logger(__name__)


# ── Core calculation ───────────────────────────────────────────────────────────

def adjusted_energy(
    area: float,
    ghi: float,
    temp: float,
    month: str,
    shading: float = config.DEFAULT_SHADING,
    dust_factor: float = config.DEFAULT_DUST_FACTOR,
) -> float:
    """
    Estimate monthly energy output (kWh) for one month using a physics model.

    The formula follows a standard PV yield approach:
        E = Area × Efficiency × GHI × Hours × PR × Days

    CRITICAL: GHI from PVGIS is average POWER (W/m²), not energy.
    We must multiply by 24 hours to convert to daily energy (Wh/m²/day).

    Losses applied:
        - Temperature derating  (if temp > threshold)
        - Dust / soiling loss   (dust_factor, default 0.95)
        - Shading factor        (applied to performance ratio)

    Args:
        area:        Panel / rooftop area in m².
        ghi:         Mean daily global horizontal irradiance for the month (W/m²).
        temp:        Mean ambient temperature for the month (°C).
        month:       Month string ("01"-"12") for days lookup.
        shading:     Shading factor 0–1 (1.0 = no shading). Default 1.0.
        dust_factor: Fractional loss from soiling. Default 0.95 (5 % loss).

    Returns:
        Estimated monthly energy in kWh.
    """
    efficiency: float = config.PANEL_EFFICIENCY

    # Temperature derating: panels lose ~0.4 %/°C above STC (25 °C);
    # simplified here to a single step at the configured threshold.
    if temp > config.TEMP_LOSS_THRESHOLD_C:
        efficiency *= config.TEMP_LOSS_FACTOR
        logger.debug("Temperature derating applied (temp=%.1f °C).", temp)

    # Soiling / dust loss
    efficiency *= dust_factor

    # Performance ratio adjusted for shading
    # NOTE: PVGIS already accounts for system losses via its "loss" parameter,
    # so PERFORMANCE_RATIO should be 1.0 to avoid double-counting.
    pr: float = config.PERFORMANCE_RATIO * shading

    # CRITICAL FIX: GHI is power (W/m²), not energy.
    # Convert to energy: W/m² × 24 hours = Wh/m²/day
    # Then: area × efficiency × daily energy × PR × days = monthly kWh
    hours_per_day: float = 24.0
    days_in_month: int = config.DAYS_PER_MONTH.get(month, 30)

    # Formula: area (m²) × efficiency × (GHI × 24 / 1000) × PR × days
    energy_kwh: float = (
        area * efficiency * (ghi * hours_per_day / 1000.0) * pr * days_in_month
    )

    logger.debug(
        "adjusted_energy: area=%.1f m², ghi=%.1f W/m², temp=%.1f °C, "
        "month=%s, days=%d → %.2f kWh",
        area, ghi, temp, month, days_in_month, energy_kwh,
    )
    return round(energy_kwh, 3)


# ── Monthly aggregation ────────────────────────────────────────────────────────

def monthly_energy(
    area: float,
    monthly_ghi: MonthlyData,
) -> MonthlyEnergy:
    """
    Compute physics-based energy output for every month.

    Args:
        area:        Rooftop / panel area in m².
        monthly_ghi: Preprocessed monthly feature dict (output of preprocessing.py).

    Returns:
        Dict mapping zero-padded month strings to kWh values,
        e.g. {"01": 85.3, "02": 102.7, ...}.
    """
    energy_output: MonthlyEnergy = {}

    for month, data in monthly_ghi.items():
        ghi: float = data["GHI"]
        temp: float = data["TEMP"]
        # Pass month to adjusted_energy for correct days per month lookup
        energy_output[month] = adjusted_energy(area, ghi, temp, month)

    logger.info("Physics monthly energy computed for %d months.", len(energy_output))
    return energy_output


# ── Annual aggregation ─────────────────────────────────────────────────────────

def annual_energy(monthly_energy_dict: MonthlyEnergy) -> float:
    """
    Sum monthly kWh values to obtain annual generation.

    Args:
        monthly_energy_dict: Output of monthly_energy().

    Returns:
        Total annual generation in kWh.
    """
    total: float = sum(monthly_energy_dict.values())
    logger.info("Annual energy total: %.2f kWh", total)
    return round(total, 2)
