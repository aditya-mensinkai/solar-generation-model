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


# ── Advanced loss models ───────────────────────────────────────────────────────

def shading_loss(shading_factor: float, inverter_type: str) -> float:
    """
    Non-linear shading loss based on inverter type.
    String inverters suffer more from partial shading than microinverters.

    Args:
        shading_factor: Shading factor 0–1 (1.0 = no shading).
        inverter_type: "string" or "micro" inverter type.

    Returns:
        Efficiency factor (0–1) accounting for shading losses.
    """
    if inverter_type == "string":
        # String inverters: strong non-linear loss from partial shading
        return shading_factor ** 2
    # Microinverters: each panel operates independently, linear loss
    return shading_factor


def temperature_loss(temp: float) -> float:
    """
    Linear temperature coefficient model.
    Panels lose ~0.4% efficiency per °C above 25°C (STC).

    Args:
        temp: Ambient temperature in °C.

    Returns:
        Efficiency factor (0–1) accounting for temperature losses.
    """
    return 1.0 - (config.TEMP_COEFF * max(0, temp - 25))


def inverter_clipping_loss(area: float, inverter_capacity_kw: float) -> float:
    """
    Approximate inverter clipping loss as a percentage.

    In reality, clipping only occurs during peak production hours,
    resulting in 5-15% loss, not complete truncation of daily energy.

    Args:
        area: Panel area in m².
        inverter_capacity_kw: Inverter AC rating in kW.

    Returns:
        Efficiency factor (0–1) accounting for inverter clipping.
    """
    if inverter_capacity_kw is None or inverter_capacity_kw <= 0:
        return 1.0

    # Approximate DC system size (kW)
    system_kw = area * 0.2  # 200W/m² assumption

    if system_kw <= inverter_capacity_kw:
        return 1.0  # No clipping

    # DC/AC ratio
    ratio = system_kw / inverter_capacity_kw

    # Realistic clipping loss curve
    if ratio <= 1.0:
        return 1.0
    elif ratio <= 1.2:
        return 0.98   # 2% loss
    elif ratio <= 1.5:
        return 0.92   # 8% loss
    elif ratio <= 2.0:
        return 0.85   # 15% loss
    else:
        return 0.75   # 25% loss


# ── Core calculation ───────────────────────────────────────────────────────────

def adjusted_energy(
    area: float,
    monthly_energy_per_kwp: float,
    temp: float,
    month: str,
    tilt: float = config.DEFAULT_TILT,
    azimuth: float = config.DEFAULT_AZIMUTH,
    shading_factor: float = config.DEFAULT_SHADING,
    inverter_type: str = "string",
    inverter_capacity_kw: float = None,
    years_of_operation: int = 0,
    dust_factor: float = config.DEFAULT_DUST_FACTOR,
) -> float:
    """
    Estimate monthly energy output (kWh) for one month using PVGIS power data.

    Uses PVGIS "P" values (hourly power output for 1 kWp system) which are
    pre-summed to monthly energy values (kWh/kWp). This avoids the PSH
    conversion error and double-counting of losses.

    Formula:
        E = Monthly_Energy_Per_kWp × System_kW × Efficiency

    Where:
        - Monthly_Energy_Per_kWp = sum of hourly "P" values / 1000 (kWh/kWp)
        - System_kW = Area × 0.2 (200W/m² for modern panels)
        - Efficiency = product of optional correction factors

    CRITICAL: PVGIS "P" already includes:
        - Panel efficiency
        - System losses (inverter, wiring)
        - Tilt and azimuth orientation
        - Temperature effects (partial)
        - Soiling effects (partial via loss parameter)

    ONLY apply these optional corrections:
        - Temperature coefficient (fine-tuning)
        - Shading losses (user-specified)
        - Dust/soiling adjustment (user-specified)
        - Inverter clipping (if specified)
        - Panel degradation (if specified)

    Args:
        area:                 Panel / rooftop area in m².
        monthly_energy_per_kwp: Monthly energy per kWp from PVGIS (kWh/kWp).
        temp:                 Mean ambient temperature for the month (°C).
        month:                Month string ("01"-"12") for logging.
        tilt:                 Panel tilt angle in degrees (0 = flat).
        azimuth:              Panel azimuth angle (180° = south-facing).
        shading_factor:       Shading factor 0–1 (1.0 = no shading).
        inverter_type:        "string" or "micro" inverter type.
        inverter_capacity_kw: Optional inverter capacity for clipping.
        years_of_operation:   Years of panel operation (for degradation).
        dust_factor:          Fractional loss from soiling.

    Returns:
        Estimated monthly energy in kWh.
    """
    # System size scaling: 200W/m² assumption for modern panels
    # 20 m² × 0.2 kW/m² = 4 kW system
    system_kw: float = area * 0.2

    # Base energy from PVGIS (already for 1 kWp system)
    # PVGIS "P" summed over month = kWh/kWp
    # Scale by actual system size
    base_monthly_kwh: float = monthly_energy_per_kwp * system_kw

    # Apply ONLY optional corrections (not already in PVGIS)
    efficiency: float = 1.0

    # Temperature coefficient loss (fine-tuning)
    # PVGIS accounts for some temperature effects, but we apply user correction
    efficiency *= temperature_loss(temp)
    logger.debug("Temperature loss applied: factor=%.3f", temperature_loss(temp))

    # PVGIS "P" already includes tilt/azimuth, system losses, and base efficiency
    # Do NOT apply these again

    # Shading losses (user-specified, not in PVGIS)
    efficiency *= shading_loss(shading_factor, inverter_type)
    logger.debug("Shading loss applied: factor=%.3f, inverter=%s", shading_factor, inverter_type)

    # Dust/soiling loss (user-specified override)
    # PVGIS has some soiling via loss parameter, but allow user adjustment
    efficiency *= dust_factor

    # Calculate energy with corrections
    energy_kwh: float = base_monthly_kwh * efficiency

    # Inverter clipping: apply loss factor instead of hard truncation
    # In reality, clipping only occurs during peak hours (5-15% loss)
    clip_factor = inverter_clipping_loss(area, inverter_capacity_kw)
    if clip_factor < 1.0:
        logger.debug("Inverter clipping applied: factor=%.3f", clip_factor)
        energy_kwh *= clip_factor

    # Panel degradation over time (0.5% per year)
    if years_of_operation > 0:
        degradation = config.DEGRADATION_RATE * years_of_operation
        energy_kwh *= (1 - degradation)
        logger.debug("Degradation applied: %.1f%% after %d years", degradation * 100, years_of_operation)

    logger.debug(
        "adjusted_energy: area=%.1f m², monthly_energy_per_kwp=%.1f kWh/kWp, temp=%.1f °C, "
        "month=%s, tilt=%.1f°, azimuth=%.1f° → %.2f kWh",
        area, monthly_energy_per_kwp, temp, month, tilt, azimuth, energy_kwh,
    )
    return round(energy_kwh, 3)


# ── Monthly aggregation ────────────────────────────────────────────────────────

def monthly_energy(
    area: float,
    monthly_ghi: MonthlyData,
    tilt: float = config.DEFAULT_TILT,
    azimuth: float = config.DEFAULT_AZIMUTH,
    shading_factor: float = config.DEFAULT_SHADING,
    inverter_type: str = "string",
    inverter_capacity_kw: float = None,
    years_of_operation: int = 0,
) -> MonthlyEnergy:
    """
    Compute physics-based energy output for every month.

    Args:
        area:                 Rooftop / panel area in m².
        monthly_ghi:          Preprocessed monthly feature dict.
        tilt:                 Panel tilt angle in degrees.
        azimuth:              Panel azimuth angle (180° = south).
        shading_factor:       Shading factor 0–1.
        inverter_type:        "string" or "micro" inverter.
        inverter_capacity_kw: Optional inverter capacity for clipping.
        years_of_operation:   Years of panel operation.

    Returns:
        Dict mapping month strings to kWh values.
    """
    energy_output: MonthlyEnergy = {}

    for month, data in monthly_ghi.items():
        # Use PVGIS direct energy values (kWh/kWp) - most accurate
        monthly_energy_per_kwp: float = data.get("ENERGY_KWH_KWP", 0.0)
        temp: float = data["TEMP"]
        energy_output[month] = adjusted_energy(
            area, monthly_energy_per_kwp, temp, month,
            tilt=tilt, azimuth=azimuth, shading_factor=shading_factor,
            inverter_type=inverter_type, inverter_capacity_kw=inverter_capacity_kw,
            years_of_operation=years_of_operation
        )

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
