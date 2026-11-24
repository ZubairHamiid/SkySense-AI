import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "storage.db"
MODELS_DIR = BASE_DIR / "models"
MODEL_REGISTRY_DIR = MODELS_DIR / "registry"

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, MODEL_REGISTRY_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Default Target City & Multi-City Geolocation Registry
DEFAULT_CITY = os.getenv("TARGET_CITY", "Lahore")

# Comprehensive Pakistan Meteorological & AQI Monitoring Basin Registry
PAKISTAN_CITIES = {
    "Lahore": {
        "lat": 31.5204,
        "lon": 74.3587,
        "timezone": "Asia/Karachi",
        "province": "Punjab",
        "climate": "Semi-Arid / Severe Winter Smog",
        "elevation_m": 217
    },
    "Karachi": {
        "lat": 24.8607,
        "lon": 67.0011,
        "timezone": "Asia/Karachi",
        "province": "Sindh",
        "climate": "Maritime Coastal / High Humidity",
        "elevation_m": 8
    },
    "Islamabad": {
        "lat": 33.6844,
        "lon": 73.0479,
        "timezone": "Asia/Karachi",
        "province": "Federal Capital",
        "climate": "Subtropical Foothills / High Rain",
        "elevation_m": 507
    },
    "Peshawar": {
        "lat": 34.0151,
        "lon": 71.5249,
        "timezone": "Asia/Karachi",
        "province": "Khyber Pakhtunkhwa",
        "climate": "Continental Valley / Dust & Inversion",
        "elevation_m": 359
    },
    "Quetta": {
        "lat": 30.1798,
        "lon": 66.9750,
        "timezone": "Asia/Karachi",
        "province": "Balochistan",
        "climate": "High-Altitude Arid / Cold Winters",
        "elevation_m": 1680
    },
    "Multan": {
        "lat": 30.1575,
        "lon": 71.5249,
        "timezone": "Asia/Karachi",
        "province": "Punjab",
        "climate": "Hyper-Thermal Desert / Low Humidity",
        "elevation_m": 122
    },
    "Faisalabad": {
        "lat": 31.4504,
        "lon": 73.1350,
        "timezone": "Asia/Karachi",
        "province": "Punjab",
        "climate": "Industrial Agricultural / Particulate Heavy",
        "elevation_m": 184
    },
    "Sargodha": {
        "lat": 32.0836,
        "lon": 72.6711,
        "timezone": "Asia/Karachi",
        "province": "Punjab",
        "climate": "Agricultural Canal Basin / Smog-Prone",
        "elevation_m": 190
    },
    "Rawalpindi": {
        "lat": 33.5651,
        "lon": 73.0169,
        "timezone": "Asia/Karachi",
        "province": "Punjab",
        "climate": "Potohar Plateau / Moderate Humidity",
        "elevation_m": 508
    }
}

CITY_LATITUDE = float(os.getenv("CITY_LATITUDE", str(PAKISTAN_CITIES["Lahore"]["lat"])))
CITY_LONGITUDE = float(os.getenv("CITY_LONGITUDE", str(PAKISTAN_CITIES["Lahore"]["lon"])))
CITY_TIMEZONE = os.getenv("CITY_TIMEZONE", "Asia/Karachi")

# The 6 metrics required by 10Pearls AgentWeather specification:
METRICS = [
    "aqi",                        # Air Quality Index (US AQI / PM2.5 calculation)
    "temperature_c",              # Temperature in °C
    "precipitation_mm",           # Precipitation in mm (includes rain probability)
    "humidity_pct",               # Relative Humidity (%)
    "uv_index",                   # UV Index (0 - 12+)
    "wind_speed_kmh"              # Wind Speed in km/h (includes wind direction)
]

# Forecast horizon: 24 hours ahead
FORECAST_HORIZON_HOURS = 24

# API Keys (optional; project has full zero-cost Open-Meteo & rule-based fallbacks)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
AQICN_API_KEY = os.getenv("AQICN_API_KEY", "")

# Hopsworks Feature Store Configuration
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "AgentWeather")

# Drift thresholds
DATA_DRIFT_PVALUE_THRESHOLD = float(os.getenv("DATA_DRIFT_PVALUE_THRESHOLD", "0.05"))
MODEL_DRIFT_ERROR_INCREASE_PCT = float(os.getenv("MODEL_DRIFT_ERROR_INCREASE_PCT", "0.25"))

