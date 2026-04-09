"""
SolarSense — Example Usage
--------------------------
Run from the project root (same directory that contains solar_predictor/):

    python example_usage.py

This calls predict_solar() directly (no server required).
"""

import json
import sys

# Adjust path if running outside the project root
# sys.path.insert(0, "/path/to/project")

from solar_predictor.predictor import predict_solar


def main() -> None:
    # ── Example: Bengaluru rooftop, 20 m² panel area ──────────────────────────
    area = 20.0   # m²
    lat  = 12.97  # Bengaluru
    lon  = 77.59

    print(f"\n{'='*60}")
    print(f"  SolarSense Prediction")
    print(f"  Area: {area} m²  |  Lat: {lat}  |  Lon: {lon}")
    print(f"{'='*60}\n")

    try:
        result = predict_solar(area=area, lat=lat, lon=lon, mode="hybrid")
        print(json.dumps(result, indent=2))
    except RuntimeError as exc:
        print(f"[ERROR] Could not fetch PVGIS data: {exc}", file=sys.stderr)
        print("Make sure you have an internet connection.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Annual generation: {result['annual_generation']} kWh")
    print(f"  Daily average:     {result['avg_daily_generation']} kWh/day")
    print(f"  Model used:        {result['metadata']['model_used']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
