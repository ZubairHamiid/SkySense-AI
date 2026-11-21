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

# Default Target City: Lahore (Famous for dynamic weather & AQI monitoring)
DEFAULT_CITY = os.getenv("TARGET_CITY", "Lahore")
CITY_LATITUDE = float(os.getenv("CITY_LATITUDE", "31.5204"))
CITY_LONGITUDE = float(os.getenv("CITY_LONGITUDE", "74.3587"))
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

