import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from config.settings import METRICS, FORECAST_HORIZON_HOURS, DEFAULT_CITY
from src.agents.state import AgentWeatherState
from src.pipeline.model_registry import ModelRegistry
from src.pipeline.feature_store import FeatureStoreManager
from src.pipeline.feature_engineering import get_feature_columns

logger = logging.getLogger("ForecastAgent")


def forecast_node(state: AgentWeatherState) -> AgentWeatherState:
    """
    LangGraph Node: Forecast Agent.
    Fetches the latest engineered features, queries the Model Registry for the champion
    model per metric, produces 24h-ahead predictions, logs everything into the
    Forecast Ledger, and updates the shared agent state.
    """
    logger.info("--- [Node: Forecast Agent] Starting 24h-ahead forecasting ---")
    
    fs = FeatureStoreManager()
    registry = ModelRegistry()
    city = state.get("city", DEFAULT_CITY)
    
    # 1. Fetch latest feature record for target city
    feat_df = fs.get_features(city=city)
    if feat_df.empty or ("city" in feat_df.columns and city.lower() not in feat_df["city"].str.lower().values):
        logger.info(f"Target city [{city}] features not pre-cached. Ingesting live atmospheric observations for {city}...")
        try:
            from src.pipeline.fetcher import DataFetcher
            from src.pipeline.feature_engineering import compute_features
            fetcher = DataFetcher(city=city)
            df_live = fetcher.fetch_live_recent(past_hours=48, forecast_hours=0)
            feat_df = compute_features(df_live, is_training=False)
            fs.save_raw_observations(df_live, city=city)
            fs.save_features(feat_df, city=city)
            logger.info(f"Live features successfully computed for [{city}].")
        except Exception as e:
            logger.warning(f"Live fetch for [{city}] failed: {e}. Falling back to default features.")
            feat_df = fs.get_features()

    if feat_df.empty:
        err = f"Feature store empty for {city}. Cannot run forecast agent."
        logger.error(err)
        state["errors"].append(err)
        return state
        
    feat_df = feat_df.sort_values("time").reset_index(drop=True)
    latest_row = feat_df.iloc[-1]
    latest_time = pd.to_datetime(latest_row["time"])
    target_time = latest_time + timedelta(hours=FORECAST_HORIZON_HOURS)
    
    feature_cols = get_feature_columns(feat_df)
    X_latest = pd.DataFrame([latest_row[feature_cols]]).fillna(0.0)
    
    forecasts_24h = {}
    baselines = {}
    models_used = {}
    
    # 2. Generate predictions for each of the 6 metrics
    for metric in METRICS:
        champ_tuple = registry.get_champion(metric)
        current_val = float(latest_row.get(metric, 0.0))
        # Naive baseline: persistence of current value 24 hours ahead
        baselines[metric] = round(current_val, 2)
        
        if champ_tuple:
            model, meta = champ_tuple
            try:
                expected_feats = meta.get("feature_names", feature_cols)
                X_in = pd.DataFrame([{f: float(latest_row.get(f, 0.0)) for f in expected_feats}]).fillna(0.0)
                pred = float(model.predict(X_in)[0])
                model_name = meta["model_name"]
                model_fam = meta["family"]
            except Exception as e:
                logger.warning(f"Model prediction failed for {metric} ({e}). Falling back to persistence.")
                pred = current_val
                model_name = "persistence_fallback"
                model_fam = "Baseline"
        else:
            logger.warning(f"No champion model found for {metric}. Using persistence baseline.")
            pred = current_val
            model_name = "persistence_baseline"
            model_fam = "Baseline"
            
        # Physical meteorological clamping
        if metric == "humidity_pct":
            pred = float(np.clip(pred, 5.0, 100.0))
        elif metric == "precipitation_mm":
            pred = float(max(0.0, pred))
        elif metric == "uv_index":
            pred = float(np.clip(pred, 0.0, 16.0))
        elif metric == "wind_speed_kmh":
            pred = float(max(0.0, pred))
        elif metric == "aqi":
            pred = float(np.clip(pred, 0.0, 500.0))
            
        forecasts_24h[metric] = round(pred, 2)
        models_used[metric] = f"{model_name} ({model_fam})"
        
        # 3. Log to Forecast Ledger
        fs.log_forecast(
            target_time=target_time.isoformat(),
            city=city,
            metric=metric,
            forecast_val=round(pred, 2),
            baseline_val=baselines[metric],
            model_family=model_fam,
            model_name=model_name
        )
        
    # 4. Synthesize 24-hour step trajectory (from hour 1 to hour 24)
    hourly_trajectory = []
    # Build diurnal curve interpolation for visualization
    for h in range(1, FORECAST_HORIZON_HOURS + 1):
        step_time = latest_time + timedelta(hours=h)
        alpha = h / FORECAST_HORIZON_HOURS
        step_record = {"hour_step": h, "time": step_time.strftime("%Y-%m-%d %H:%M")}
        
        for metric in METRICS:
            c_val = baselines[metric]
            t_val = forecasts_24h[metric]
            
            # Add diurnal cycle variation
            hour_of_day = step_time.hour
            diurnal_factor = 0.0
            if metric == "temperature_c":
                # Peak at 15:00, trough at 05:00
                diurnal_factor = 3.5 * np.sin(2 * np.pi * (hour_of_day - 9) / 24.0)
            elif metric == "aqi":
                # Peak at 08:00 and 21:00 (inversion / traffic), dip in afternoon
                diurnal_factor = 18.0 * np.cos(4 * np.pi * (hour_of_day - 8) / 24.0)
            elif metric == "uv_index":
                # UV only exists during daytime (06:00 - 18:00)
                diurnal_factor = max(0.0, np.sin(np.pi * (hour_of_day - 6) / 12.0)) if 6 <= hour_of_day <= 18 else 0.0
                
            interpolated = c_val + alpha * (t_val - c_val) + diurnal_factor
            if metric in ["humidity_pct", "precipitation_mm", "uv_index", "wind_speed_kmh", "aqi"]:
                interpolated = max(0.0, interpolated)
            if metric == "humidity_pct":
                interpolated = min(100.0, interpolated)
                
            step_record[metric] = round(float(interpolated), 2)
        hourly_trajectory.append(step_record)
        
    state["forecast_24h"] = forecasts_24h
    state["baseline_forecast"] = baselines
    state["champion_models_used"] = models_used
    state["forecast_hourly_trajectory"] = hourly_trajectory
    state["current_features"] = latest_row.to_dict()
    
    logger.info(f"Forecasts produced: {forecasts_24h}")
    return state

