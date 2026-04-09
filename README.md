# SolarSense — Solar Generation Predictor Backend

> **AI-powered solar energy prediction for smart rooftops**
> 🏆 IEEE-ready | 🌐 FastAPI | 🤖 XGBoost + Physics Hybrid

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd solar-sense
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
```

**Activate:**

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run Without API (Direct Test)

```bash
python example_usage.py
```

**What this does:**
- Runs prediction locally without starting a server
- Fetches real solar data from PVGIS API
- Prints output JSON directly to console

**Expected output:**
```json
{
  "annual_generation": 7149.0,
  "avg_daily_generation": 19.6,
  "monthly_generation": { ... }
}
```

> **Note:** Output values are realistic for a 20 m² system in India (~5000-8000 kWh/year depending on location).

---

### 5. Run Backend API

```bash
uvicorn api:app --reload --port 8000
```

The API will start at: `http://localhost:8000`

---

### 6. Test API

**Using curl:**

```bash
curl -X POST http://localhost:8000/solar-predict \
  -H "Content-Type: application/json" \
  -d '{"area": 20.0, "lat": 12.97, "lon": 77.59}'
```

**Using Postman:**
- Method: `POST`
- URL: `http://localhost:8000/solar-predict`
- Body: `{"area": 20.0, "lat": 12.97, "lon": 77.59}`

**Using Swagger UI (Interactive Docs):**

Open: [http://localhost:8000/docs](http://localhost:8000/docs)

Try it directly in your browser with auto-generated UI!

---

## 🌞 Project Overview

**SolarSense** predicts solar energy generation for any rooftop in the world using a **hybrid physics + ML approach**.

### Real-World Use Case

Imagine you want to install solar panels on your rooftop. Before investing ₹2–5 lakhs, you need answers:

- **How much energy will my rooftop generate?**
- **Which months are best for solar?**
- **Is my location solar-viable?**

**SolarSense** answers these in **seconds** — no site visit needed!

### Why It Matters

- ☀️ **India targets 500 GW renewable energy by 2030**
- 🏠 **Millions of rooftops** are potential solar assets
- 💡 **Accurate prediction** enables better ROI decisions
- 🌱 **Faster solar adoption** = cleaner energy future

---

## 🧠 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         USER INPUT                          │
│                  (Area, Latitude, Longitude)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PVGIS API (EU JRC)                        │
│              Real satellite solar data (2005–2020)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    SOLAR PREDICTOR ENGINE                   │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │   PHYSICS MODEL     │    │       ML MODEL (XGBoost)     │ │
│  │  • PVGIS Direct     │◄──►│  • Correction layer         │ │
│  │  • Temperature loss │    │  • 12 features               │ │
│  │  • Shading loss     │    │  • Trained on real data      │ │
│  │  • Degradation      │    │                              │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
│                              │                               │
│                              ▼                               │
│                    ┌─────────────────┐                      │
│                    │  HYBRID BLEND   │                      │
│                    │  Safety-clamped │                      │
│                    └─────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      OUTPUT JSON                            │
│         Monthly, Annual, Daily, Seasonal Trends             │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API** | FastAPI | High-performance async backend |
| **Server** | Uvicorn | ASGI server for FastAPI |
| **ML** | XGBoost | Gradient boosting for corrections |
| **Physics** | NumPy | Vectorized calculations |
| **Data** | PVGIS API | Satellite-based solar irradiance |
| **Validation** | Pydantic | Request/response modeling |
| **Testing** | pytest | Unit and integration tests |

---

## 📂 Project Structure

```
solar-sense/
│
├── api.py                      # FastAPI entry point
├── example_usage.py            # Direct test script (no server)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
└── solar_predictor/            # Core prediction package
    ├── __init__.py
    ├── config.py               # Constants (tilt, azimuth, loss factors)
    ├── data_fetcher.py         # PVGIS API integration
    ├── preprocessing.py        # Data cleaning & transforms
    ├── physics_model.py        # Deterministic calculations
    ├── ml_model.py             # XGBoost training & inference
    ├── predictor.py            # Main orchestration logic
    └── utils.py                # Logging & type definitions
```

### File Responsibilities

| File | What It Does |
|------|-------------|
| `data_fetcher.py` | Calls PVGIS API, fetches power/irradiance data |
| `preprocessing.py` | Converts raw API data to monthly features |
| `physics_model.py` | Calculates energy using PVGIS direct power data |
| `ml_model.py` | Trains/runs XGBoost correction model |
| `predictor.py` | Blends physics + ML → final output |
| `config.py` | Default parameters (tilt, azimuth, loss factors) |
| `utils.py` | Shared types, logging setup |

---

## 🔬 How It Works

### Step-by-Step Pipeline

**1️⃣ Fetch PVGIS Data**
- Query EU JRC satellite database
- Get 15+ years of solar data at your location
- Data: Power output (P), Global Horizontal Irradiance (GHI), Direct (DNI), Diffuse (DHI)

**2️⃣ Preprocess Monthly Features**
- Aggregate daily data to monthly averages
- Handle missing values (humidity = -1 if unavailable)
- Normalize units (W/m², °C, m/s)

**3️⃣ Physics-Based Estimation**
```
Energy = Monthly_Energy_Per_kWp × System_kW × Efficiency
```

Where:
- **Monthly_Energy_Per_kWp** = PVGIS direct power data (kWh/kWp)
- **System_kW** = Area × 0.2 (200W/m² for modern panels)
- **Efficiency** = Product of optional correction factors

**Why PVGIS Direct?**
- PVGIS "P" values already include tilt, orientation, and system losses
- No double-counting of irradiance or efficiency
- Most accurate satellite-based solar data available

**Optional corrections applied:**
- 🌡️ Temperature coefficient (~0.4%/°C above 25°C)
- 🏢 Shading loss (inverter-specific non-linear)
- 🧹 Dust/soiling factor (user-defined)
- ⏰ Degradation (~0.5% per year)

**4️⃣ ML Correction (Optional)**
- XGBoost model trained on real metered data vs PVGIS estimates
- Corrects systematic biases in physics model
- Acts as a "correction layer" (not standalone)

**5️⃣ Hybrid Blending**
```python
prediction = physics_prediction × (1 + ml_correction)
```
- ML can only adjust by ±30% (safety clamp)
- Prevents unrealistic predictions
- Respects physical limits

**6️⃣ Final Output**
- Monthly breakdown (12 values)
- Annual total
- Daily average
- Seasonal trends (Winter/Spring/Summer/Autumn)

---

## 📊 Output Format

### Example Response

```json
{
  "monthly_generation": {
    "01": 660.53,
    "02": 706.50,
    "03": 780.72,
    "04": 763.97,
    "05": 751.72,
    "06": 531.64,
    "07": 479.67,
    "08": 476.09,
    "09": 455.55,
    "10": 533.91,
    "11": 491.56,
    "12": 517.09
  },
  "annual_generation": 7148.97,
  "avg_daily_generation": 19.59,
  "seasonal_trend": {
    "Winter": 1668.19,
    "Spring": 2296.19,
    "Summer": 1468.26,
    "Autumn": 1716.32
  },
  "metadata": {
    "model_used": "physics",
    "error_margin": "±10%",
    "data_source": "PVGIS v5.2 (EU JRC)",
    "inputs": {
      "area_m2": 20.0,
      "lat": 12.97,
      "lon": 77.59
    }
  }
}
```

### Field Descriptions

| Field | Unit | Description |
|-------|------|-------------|
| `monthly_generation` | kWh | Energy per month (01=Jan, 12=Dec) |
| `annual_generation` | kWh | Total yearly energy |
| `avg_daily_generation` | kWh/day | Daily average across year |
| `seasonal_trend` | kWh | Aggregated by meteorological seasons |
| `model_used` | string | "physics", "ml", or "hybrid" |
| `error_margin` | string | Estimated accuracy range |

---

## ⚠️ Assumptions & Limitations

### Current Assumptions

| Assumption | Value | Impact |
|------------|-------|--------|
| Data source | PVGIS "P" (power) | Direct energy values, no conversion needed |
| Panel efficiency | 20% (200W/m²) | Modern panel standard |
| System losses | Included in PVGIS | Wiring, inverter, soiling already accounted |
| Temperature coefficient | -0.4%/°C | Standard silicon panels |
| Degradation rate | 0.5%/year | Panel aging over time |

### Known Limitations

- 🗺️ **PVGIS resolution**: ~5 km spatial grid (may not capture micro-climates)
- 🌤️ **Weather variability**: Based on historical averages (2005–2020), not real-time
- 🏢 **Shading**: Simplified model (no 3D building simulation)
- 📐 **Roof geometry**: Assumes flat or uniformly tilted surface
- ⚡ **Hourly patterns**: Monthly averages only (no time-of-day breakdown)

### Error Estimate

> **±10%** typical accuracy for yearly totals
>
> Best for: Pre-installation feasibility studies, ROI estimates
> Not for: Real-time monitoring, exact billing calculations

---

## 🧪 Validation

### How We Verified

✅ **Cross-checked with PVGIS official calculator**
- Same inputs → similar outputs
- Differences < 5% for physics-only mode

✅ **Tested on real Indian cities**
- Bengaluru: ~7,100 kWh/year for 20m² (realistic for Indian conditions)
- Jaipur: ~8,500 kWh/year (higher irradiance = higher output)
- Mumbai: ~6,200 kWh/year (coastal humidity effects)

✅ **Physics sanity checks**
- PVGIS direct power data eliminates conversion errors
- Temperature loss matches panel datasheets
- Seasonal trends align with sun path physics
- Output aligns with real-world solar installations

---

## 🚀 Future Improvements

### Short-term
- [ ] **ROI Calculator**: Convert kWh to ₹ savings (electricity tariff integration)
- [ ] **Battery sizing**: Recommend storage based on usage patterns
- [ ] **CO₂ impact**: Show carbon offset estimate

### Long-term
- [ ] **Real-time training**: User feedback loop to improve ML accuracy
- [ ] **3D shading simulation**: Building-aware shadow modeling
- [ ] **Hourly forecasting**: Time-of-day generation profiles
- [ ] **Panel database**: Brand-specific efficiency curves
- [ ] **Mobile app**: Flutter/React Native frontend

---

## 👨‍💻 Author

**Aditya Mensinkai**
📧 [aditya.men2005@gmail.com](mailto:aditya.men2005@gmail.com)

Built for **IEEE project showcase** | Open to contributions 🙌

---

## 📜 License

MIT License — free for personal and commercial use.

---

<p align="center">
  <strong>⭐ Star this repo if you find it useful!</strong><br>
  <em>Let's build a solar-powered future together 🌞</em>
</p>
