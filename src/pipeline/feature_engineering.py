import pandas as pd
import numpy as np
from typing import List
import logging

from config.settings import METRICS, FORECAST_HORIZON_HOURS, PAKISTAN_CITIES, DEFAULT_CITY

logger = logging.getLogger("FeatureEngineering")


def _compute_single_city_features(df_city: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Computes strictly causal time-based, lag, rolling, geolocation, and derived features
    for a SINGLE city sorted chronologically by time.
    Guarantees zero cross-city leakage and zero future peeking.
    """
    df = df_city.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    new_cols = {}

    # 0. Static Geolocation Features (allows multi-city model to distinguish altitude & climate)
    city_name = str(df["city"].iloc[0]) if "city" in df.columns and len(df) > 0 else DEFAULT_CITY
    city_meta = PAKISTAN_CITIES.get(city_name, PAKISTAN_CITIES[DEFAULT_CITY])
    new_cols["city_lat"] = np.full(len(df), float(city_meta["lat"]))
    new_cols["city_lon"] = np.full(len(df), float(city_meta["lon"]))
    new_cols["city_elevation_m"] = np.full(len(df), float(city_meta["elevation_m"]))

    # 1. Time-Based & Cyclical Features
    dt = df["time"].dt
    hour = dt.hour
    day = dt.day
    month = dt.month
    day_of_week = dt.dayofweek
    day_of_year = dt.dayofyear

    new_cols["hour"] = hour
    new_cols["day"] = day
    new_cols["month"] = month
    new_cols["day_of_week"] = day_of_week
    new_cols["is_weekend"] = (day_of_week >= 5).astype(int)
    new_cols["day_of_year"] = day_of_year

    # Cyclical encodings (smooth circular continuity for time)
    new_cols["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    new_cols["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    new_cols["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    new_cols["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    new_cols["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    new_cols["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Wind vector decomposition (U and V components for linear ML models)
    if "wind_speed_kmh" in df.columns and "wind_direction_deg" in df.columns:
        rad = np.radians(df["wind_direction_deg"])
        new_cols["wind_u"] = -df["wind_speed_kmh"] * np.sin(rad)
        new_cols["wind_v"] = -df["wind_speed_kmh"] * np.cos(rad)

    # 2. Strictly Backward-Looking Lag and Derived Features for each core metric
    features_to_lag = [
        "aqi", "temperature_c", "precipitation_mm",
        "precipitation_probability_pct", "humidity_pct",
        "uv_index", "wind_speed_kmh"
    ]

    for metric in features_to_lag:
        if metric not in df.columns:
            continue

        s = df[metric]
        lag_1 = s.shift(1)
        lag_2 = s.shift(2)
        lag_3 = s.shift(3)
        lag_6 = s.shift(6)
        lag_12 = s.shift(12)
        lag_24 = s.shift(24)

        new_cols[f"{metric}_lag_1h"] = lag_1
        new_cols[f"{metric}_lag_2h"] = lag_2
        new_cols[f"{metric}_lag_3h"] = lag_3
        new_cols[f"{metric}_lag_6h"] = lag_6
        new_cols[f"{metric}_lag_12h"] = lag_12
        new_cols[f"{metric}_lag_24h"] = lag_24

        # Rate of change (First differences, strictly backward)
        new_cols[f"{metric}_diff_1h"] = s - lag_1
        new_cols[f"{metric}_diff_3h"] = s - lag_3
        new_cols[f"{metric}_diff_24h"] = s - lag_24

        # Rolling statistics (strictly past/present: center=False by default)
        new_cols[f"{metric}_rolling_mean_3h"] = s.rolling(window=3, min_periods=1).mean()
        new_cols[f"{metric}_rolling_mean_6h"] = s.rolling(window=6, min_periods=1).mean()
        new_cols[f"{metric}_rolling_mean_12h"] = s.rolling(window=12, min_periods=1).mean()
        new_cols[f"{metric}_rolling_mean_24h"] = s.rolling(window=24, min_periods=1).mean()

        new_cols[f"{metric}_rolling_std_6h"] = s.rolling(window=6, min_periods=1).std().fillna(0.0)
        new_cols[f"{metric}_rolling_std_24h"] = s.rolling(window=24, min_periods=1).std().fillna(0.0)
        new_cols[f"{metric}_rolling_min_24h"] = s.rolling(window=24, min_periods=1).min()
        new_cols[f"{metric}_rolling_max_24h"] = s.rolling(window=24, min_periods=1).max()

    # Heat index proxy (perceived temperature when hot and humid)
    if "temperature_c" in df.columns and "humidity_pct" in df.columns:
        new_cols["apparent_temp_proxy"] = (
            df["temperature_c"]
            + 0.33 * (df["humidity_pct"] / 100.0 * 6.105 * np.exp(17.27 * df["temperature_c"] / (237.7 + df["temperature_c"])))
            - 4.0
        )

    # AQI Severity Category indicator (ordinal)
    if "aqi" in df.columns:
        new_cols["aqi_category_num"] = pd.cut(
            df["aqi"],
            bins=[-np.inf, 50, 100, 150, 200, 300, np.inf],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(float).fillna(0.0)

    # 3. Create 24h Ahead Target Variables (strictly within this single city)
    for metric in METRICS:
        if metric in df.columns:
            new_cols[f"target_{metric}"] = df[metric].shift(-FORECAST_HORIZON_HOURS)

    # Drop any overlapping pre-existing derived columns before clean single-shot concat
    drop_existing = [c for c in new_cols.keys() if c in df.columns]
    if drop_existing:
        df = df.drop(columns=drop_existing)

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def compute_features(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Computes strictly causal features per city to prevent cross-city lag/target leakage.
    If multiple cities are present in df, each city is processed independently and recombined.
    """
    if df.empty:
        return df.copy()

    if "city" in df.columns and df["city"].nunique() > 1:
        parts = []
        for _, grp in df.groupby("city", sort=False):
            parts.append(_compute_single_city_features(grp, is_training=is_training))
        return pd.concat(parts, ignore_index=True)

    return _compute_single_city_features(df, is_training=is_training)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Returns the list of input feature column names for model training and inference.
    Excludes timestamp, metadata, redundant duplicate us_aqi, and all forward target_* labels.
    """
    excluded = {
        "time", "city", "source", "id", "created_at", "us_aqi"
    } | {f"target_{m}" for m in METRICS}

    feature_cols = [
        c for c in df.columns
        if c not in excluded and not c.startswith("target_") and pd.api.types.is_numeric_dtype(df[c])
    ]
    return sorted(feature_cols)
