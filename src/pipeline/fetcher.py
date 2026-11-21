import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple
import logging

from config.settings import (
    CITY_LATITUDE,
    CITY_LONGITUDE,
    CITY_TIMEZONE,
    DEFAULT_CITY,
    OPENWEATHER_API_KEY,
    AQICN_API_KEY
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataFetcher")


def calculate_us_aqi_from_pm25(pm25: float) -> float:
    """
    Standard EPA AQI Breakpoints for PM2.5 (24-hour / hourly equivalent in ug/m3).
    Formula: I = ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low
    """
    if pd.isna(pm25) or pm25 < 0:
        return np.nan
    
    breakpoints = [
        (0.0, 12.0, 0, 50),        # Good
        (12.1, 35.4, 51, 100),     # Moderate
        (35.5, 55.4, 101, 150),    # Unhealthy for Sensitive Groups
        (55.5, 150.4, 151, 200),   # Unhealthy
        (150.5, 250.4, 201, 300),  # Very Unhealthy
        (250.5, 350.4, 301, 400),  # Hazardous
        (350.5, 500.4, 401, 500),  # Hazardous
    ]
    
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25 <= c_high:
            return round(((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low, 1)
            
    if pm25 > 500.4:
        return 500.0
    return 0.0


class DataFetcher:
    """
    Fetches weather and pollutant data from multiple sources (Open-Meteo, OpenWeather, AQICN),
    fuses & time-aligns them, and applies a documented imputation strategy.
    """
    
    def __init__(self, lat: float = CITY_LATITUDE, lon: float = CITY_LONGITUDE, city: str = DEFAULT_CITY):
        self.lat = lat
        self.lon = lon
        self.city = city
        
    def fetch_open_meteo_historical(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical weather and air pollution from Open-Meteo Archive API.
        start_date, end_date format: 'YYYY-MM-DD'
        """
        logger.info(f"Fetching historical Open-Meteo data for {self.city} ({start_date} to {end_date})...")
        
        # 1. Historical Weather Archive
        weather_url = "https://archive-api.open-meteo.com/v1/archive"
        weather_params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "wind_direction_10m"
            ],
            "timezone": CITY_TIMEZONE
        }
        
        # 2. Historical Air Quality Archive
        aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        aq_params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": [
                "pm2_5",
                "pm10",
                "carbon_monoxide",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "ozone",
                "uv_index",
                "us_aqi"
            ],
            "timezone": CITY_TIMEZONE
        }
        
        res_w = requests.get(weather_url, params=weather_params, timeout=30)
        res_w.raise_for_status()
        w_data = res_w.json().get("hourly", {})
        df_weather = pd.DataFrame(w_data)
        
        res_aq = requests.get(aq_url, params=aq_params, timeout=30)
        res_aq.raise_for_status()
        aq_data = res_aq.json().get("hourly", {})
        df_aq = pd.DataFrame(aq_data)
        
        # Merge on timestamp
        df_weather["time"] = pd.to_datetime(df_weather["time"])
        df_aq["time"] = pd.to_datetime(df_aq["time"])
        
        merged = pd.merge(df_weather, df_aq, on="time", how="outer").sort_values("time").reset_index(drop=True)
        
        # Standardize column names
        merged = merged.rename(columns={
            "temperature_2m": "temperature_c",
            "relative_humidity_2m": "humidity_pct",
            "precipitation": "precipitation_mm",
            "wind_speed_10m": "wind_speed_kmh",
            "wind_direction_10m": "wind_direction_deg"
        })
        
        # If us_aqi is missing or 0 but pm2_5 is available, calculate standard US AQI
        if "us_aqi" in merged.columns and "pm2_5" in merged.columns:
            calculated_aqi = merged["pm2_5"].apply(calculate_us_aqi_from_pm25)
            merged["aqi"] = merged["us_aqi"].combine_first(calculated_aqi)
        elif "pm2_5" in merged.columns:
            merged["aqi"] = merged["pm2_5"].apply(calculate_us_aqi_from_pm25)
        else:
            merged["aqi"] = np.nan
            
        # Precipitation probability estimation for historical records
        if "precipitation_probability_pct" not in merged.columns:
            merged["precipitation_probability_pct"] = np.where(
                merged["precipitation_mm"] > 0.1,
                np.clip(merged["precipitation_mm"] * 25.0, 30.0, 100.0),
                0.0
            )
            
        merged["source"] = "open_meteo_archive"
        merged["city"] = self.city
        return self.impute_missing_values(merged)
        
    def fetch_live_recent(self, past_hours: int = 48, forecast_hours: int = 24) -> pd.DataFrame:
        """
        Fetches live recent observations (past 48 hours) plus next 24 hours forecast from Open-Meteo.
        """
        logger.info(f"Fetching recent and forecast data for {self.city}...")
        
        # Real-time forecast & recent observations
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "past_hours": past_hours,
            "forecast_hours": forecast_hours,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "precipitation_probability",
                "wind_speed_10m",
                "wind_direction_10m",
                "uv_index"
            ],
            "timezone": CITY_TIMEZONE
        }
        
        aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        aq_params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "past_hours": past_hours,
            "forecast_hours": forecast_hours,
            "hourly": [
                "pm2_5",
                "pm10",
                "carbon_monoxide",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "ozone",
                "uv_index",
                "us_aqi"
            ],
            "timezone": CITY_TIMEZONE
        }
        
        res_w = requests.get(weather_url, params=weather_params, timeout=20)
        res_w.raise_for_status()
        w_df = pd.DataFrame(res_w.json().get("hourly", {}))
        
        res_aq = requests.get(aq_url, params=aq_params, timeout=20)
        res_aq.raise_for_status()
        aq_df = pd.DataFrame(res_aq.json().get("hourly", {}))
        
        w_df["time"] = pd.to_datetime(w_df["time"])
        aq_df["time"] = pd.to_datetime(aq_df["time"])
        
        merged = pd.merge(w_df, aq_df, on="time", how="outer", suffixes=("", "_aq")).sort_values("time").reset_index(drop=True)
        
        # Standardize columns
        merged = merged.rename(columns={
            "temperature_2m": "temperature_c",
            "relative_humidity_2m": "humidity_pct",
            "precipitation": "precipitation_mm",
            "precipitation_probability": "precipitation_probability_pct",
            "wind_speed_10m": "wind_speed_kmh",
            "wind_direction_10m": "wind_direction_deg"
        })
        
        # Reconcile uv_index if present in both
        if "uv_index_aq" in merged.columns:
            merged["uv_index"] = merged["uv_index"].combine_first(merged["uv_index_aq"])
            merged.drop(columns=["uv_index_aq"], inplace=True)
            
        if "us_aqi" in merged.columns and "pm2_5" in merged.columns:
            calc_aqi = merged["pm2_5"].apply(calculate_us_aqi_from_pm25)
            merged["aqi"] = merged["us_aqi"].combine_first(calc_aqi)
        elif "pm2_5" in merged.columns:
            merged["aqi"] = merged["pm2_5"].apply(calculate_us_aqi_from_pm25)
        else:
            merged["aqi"] = np.nan
            
        merged["source"] = "open_meteo_live"
        merged["city"] = self.city
        
        # If OpenWeather or AQICN API keys are set, fuse with them!
        fused = self.fuse_optional_secondary_sources(merged)
        return self.impute_missing_values(fused)

    def fuse_optional_secondary_sources(self, df_primary: pd.DataFrame) -> pd.DataFrame:
        """
        Fuses secondary data sources (OpenWeather, AQICN) into primary observations
        by time-aligning and taking weighted average / priority fallback.
        """
        now = datetime.now()
        
        # Secondary source 1: OpenWeatherMap (Current observation)
        if OPENWEATHER_API_KEY:
            try:
                ow_url = "https://api.openweathermap.org/data/2.5/weather"
                params = {"lat": self.lat, "lon": self.lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
                r = requests.get(ow_url, params=params, timeout=5)
                if r.status_code == 200:
                    ow_data = r.json()
                    temp_ow = ow_data["main"]["temp"]
                    hum_ow = ow_data["main"]["humidity"]
                    wind_speed_ow = ow_data["wind"]["speed"] * 3.6 # m/s to km/h
                    logger.info(f"OpenWeather observation received: Temp={temp_ow}°C, Humidity={hum_ow}%")
                    
                    # Fuse latest current timestamp with 50/50 weighted average
                    latest_idx = df_primary[df_primary["time"] <= pd.Timestamp(now)].index
                    if len(latest_idx) > 0:
                        idx = latest_idx[-1]
                        df_primary.loc[idx, "temperature_c"] = 0.5 * df_primary.loc[idx, "temperature_c"] + 0.5 * temp_ow
                        df_primary.loc[idx, "humidity_pct"] = 0.5 * df_primary.loc[idx, "humidity_pct"] + 0.5 * hum_ow
                        df_primary.loc[idx, "source"] = "fused_openmeteo_openweather"
            except Exception as e:
                logger.warning(f"Optional OpenWeather fetch failed: {e}")
                
        # Secondary source 2: AQICN (Current AQI observation)
        if AQICN_API_KEY:
            try:
                aqicn_url = f"https://api.waqi.info/feed/geo:{self.lat};{self.lon}/?token={AQICN_API_KEY}"
                r = requests.get(aqicn_url, timeout=5)
                if r.status_code == 200:
                    res_json = r.json()
                    if res_json.get("status") == "ok":
                        aqicn_val = float(res_json["data"]["aqi"])
                        logger.info(f"AQICN observation received: AQI={aqicn_val}")
                        latest_idx = df_primary[df_primary["time"] <= pd.Timestamp(now)].index
                        if len(latest_idx) > 0:
                            idx = latest_idx[-1]
                            df_primary.loc[idx, "aqi"] = 0.5 * df_primary.loc[idx, "aqi"] + 0.5 * aqicn_val
                            df_primary.loc[idx, "source"] = "fused_aqicn"
            except Exception as e:
                logger.warning(f"Optional AQICN fetch failed: {e}")
                
        return df_primary

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Documented Missing Value Imputation Strategy:
        1. Sort chronologically by time.
        2. Short Gaps (<= 3 hours): Linear time-interpolation to preserve continuity.
        3. Medium Gaps (4 to 24 hours): Rolling diurnal median for the specific hour of the day.
        4. Long Gaps (> 24 hours) / Boundary NaNs: Global median for the metric.
        5. Clamp values to physical meteorological bounds (e.g., Humidity: [0, 100], UV: [0, 16], AQI: [0, 500]).
        """
        df = df.copy()
        df = df.sort_values("time").reset_index(drop=True)
        
        target_metrics = [
            "aqi", "temperature_c", "precipitation_mm",
            "precipitation_probability_pct", "humidity_pct",
            "uv_index", "wind_speed_kmh", "wind_direction_deg"
        ]
        
        df["hour_of_day"] = df["time"].dt.hour
        
        for col in target_metrics:
            if col not in df.columns:
                df[col] = np.nan
                
            # Step 1: Linear interpolation for short sensor gaps (limit=3)
            df[col] = df[col].interpolate(method="linear", limit=3, limit_direction="both")
            
            # Step 2: Hourly diurnal group median
            hourly_medians = df.groupby("hour_of_day")[col].transform("median")
            df[col] = df[col].fillna(hourly_medians)
            
            # Step 3: Fallback global median
            global_median = df[col].median()
            if pd.isna(global_median):
                defaults = {
                    "aqi": 75.0, "temperature_c": 24.0, "precipitation_mm": 0.0,
                    "precipitation_probability_pct": 0.0, "humidity_pct": 55.0,
                    "uv_index": 3.0, "wind_speed_kmh": 12.0, "wind_direction_deg": 180.0
                }
                global_median = defaults.get(col, 0.0)
            df[col] = df[col].fillna(global_median)
            
        # Physical meteorological bounds enforcement
        df["humidity_pct"] = df["humidity_pct"].clip(0.0, 100.0)
        df["precipitation_mm"] = df["precipitation_mm"].clip(lower=0.0)
        df["precipitation_probability_pct"] = df["precipitation_probability_pct"].clip(0.0, 100.0)
        df["uv_index"] = df["uv_index"].clip(0.0, 16.0)
        df["wind_speed_kmh"] = df["wind_speed_kmh"].clip(lower=0.0)
        df["wind_direction_deg"] = df["wind_direction_deg"].clip(0.0, 360.0)
        df["aqi"] = df["aqi"].clip(0.0, 500.0)
        
        df.drop(columns=["hour_of_day"], inplace=True)
        return df

