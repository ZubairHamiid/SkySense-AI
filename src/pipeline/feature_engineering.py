import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

from config.settings import METRICS, FORECAST_HORIZON_HOURS

logger = logging.getLogger("FeatureEngineering")


def compute_features(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Computes time-based, lag, and derived (rate-of-change, rolling statistics) features
    from raw meteorological and pollutant observations.
    
    If is_training is True, generates target columns by shifting observations 24h ahead:
    target_metric = observation_metric(t + 24)
    """
    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)
    
    # 1. Time-Based & Cyclical Features
    dt = df["time"].dt
    df["hour"] = dt.hour
    df["day"] = dt.day
    df["month"] = dt.month
    df["day_of_week"] = dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_year"] = dt.dayofyear
    
    # Cyclical encodings (smooth circular continuity for time)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    
    # Wind vector decomposition (U and V components for linear ML models)
    if "wind_speed_kmh" in df.columns and "wind_direction_deg" in df.columns:
        rad = np.radians(df["wind_direction_deg"])
        df["wind_u"] = -df["wind_speed_kmh"] * np.sin(rad)
        df["wind_v"] = -df["wind_speed_kmh"] * np.cos(rad)
    
    # 2. Lag and Derived Features for each of the core metrics
    features_to_lag = [
        "aqi", "temperature_c", "precipitation_mm",
        "precipitation_probability_pct", "humidity_pct",
        "uv_index", "wind_speed_kmh"
    ]
    
    for metric in features_to_lag:
        if metric not in df.columns:
            continue
            
        # Lags
        df[f"{metric}_lag_1h"] = df[metric].shift(1)
        df[f"{metric}_lag_2h"] = df[metric].shift(2)
        df[f"{metric}_lag_3h"] = df[metric].shift(3)
        df[f"{metric}_lag_6h"] = df[metric].shift(6)
        df[f"{metric}_lag_12h"] = df[metric].shift(12)
        df[f"{metric}_lag_24h"] = df[metric].shift(24) # Naive 24h baseline anchor
        
        # Rate of change (First differences)
        df[f"{metric}_diff_1h"] = df[metric] - df[f"{metric}_lag_1h"]
        df[f"{metric}_diff_3h"] = df[metric] - df[f"{metric}_lag_3h"]
        df[f"{metric}_diff_24h"] = df[metric] - df[f"{metric}_lag_24h"]
        
        # Rolling statistics (3h, 6h, 12h, 24h)
        df[f"{metric}_rolling_mean_3h"] = df[metric].rolling(window=3, min_periods=1).mean()
        df[f"{metric}_rolling_mean_6h"] = df[metric].rolling(window=6, min_periods=1).mean()
        df[f"{metric}_rolling_mean_12h"] = df[metric].rolling(window=12, min_periods=1).mean()
        df[f"{metric}_rolling_mean_24h"] = df[metric].rolling(window=24, min_periods=1).mean()
        
        df[f"{metric}_rolling_std_6h"] = df[metric].rolling(window=6, min_periods=1).std().fillna(0.0)
        df[f"{metric}_rolling_std_24h"] = df[metric].rolling(window=24, min_periods=1).std().fillna(0.0)
        df[f"{metric}_rolling_min_24h"] = df[metric].rolling(window=24, min_periods=1).min()
        df[f"{metric}_rolling_max_24h"] = df[metric].rolling(window=24, min_periods=1).max()
        
    # Heat index proxy (perceived temperature when hot and humid)
    if "temperature_c" in df.columns and "humidity_pct" in df.columns:
        df["apparent_temp_proxy"] = (
            df["temperature_c"] + 0.33 * (df["humidity_pct"] / 100.0 * 6.105 * np.exp(17.27 * df["temperature_c"] / (237.7 + df["temperature_c"]))) - 4.0
        )
        
    # AQI Severity Category indicator (ordinal)
    if "aqi" in df.columns:
        df["aqi_category_num"] = pd.cut(
            df["aqi"],
            bins=[-np.inf, 50, 100, 150, 200, 300, np.inf],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(float).fillna(0.0)
        
    # 3. Create 24h Ahead Target Variables
    # As specified in Section 4 of PDF:
    # "create targets by shifting observations 24 hours ahead.
    # The newest 24 hours have no target yet, so they are used for forecasting only."
    for metric in METRICS:
        target_col = f"target_{metric}"
        # Shift -24 means observation at t+24 becomes the target for time t
        df[target_col] = df[metric].shift(-FORECAST_HORIZON_HOURS)
        
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Returns the list of input feature column names for model training and inference.
    Excludes timestamp, raw target columns, metadata, and ground truth forward labels.
    """
    excluded = [
        "time", "city", "source", "id", "created_at"
    ] + [f"target_{m}" for m in METRICS]
    
    feature_cols = [
        c for c in df.columns 
        if c not in excluded and not c.startswith("target_") and pd.api.types.is_numeric_dtype(df[c])
    ]
    return sorted(feature_cols)

