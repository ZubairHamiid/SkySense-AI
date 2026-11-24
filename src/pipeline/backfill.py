import argparse
from datetime import datetime, timedelta
import pandas as pd
import logging

from config.settings import DEFAULT_CITY, METRICS, FORECAST_HORIZON_HOURS
from src.pipeline.fetcher import DataFetcher
from src.pipeline.feature_engineering import compute_features
from src.pipeline.feature_store import FeatureStoreManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("BackfillPipeline")


def run_backfill(days: int = 180, city: str = DEFAULT_CITY) -> pd.DataFrame:
    """
    Backfills historical hourly weather and pollutant data for the specified city,
    computes engineered features, generates 24h-ahead targets, and stores everything
    in the Feature Store.
    
    The newest 24 hours will have NaN targets as specified in Section 4 of PDF.
    """
    end_date = datetime.now().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=days)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    logger.info(f"=== Starting Historical Backfill for {city} ===")
    logger.info(f"Date Range: {start_str} to {end_str} ({days} days)")
    
    fetcher = DataFetcher(city=city)
    fs = FeatureStoreManager()
    
    # 1. Fetch historical data
    df_raw = fetcher.fetch_open_meteo_historical(start_date=start_str, end_date=end_str)
    
    # 2. Fetch live recent (past 48h + next 24h) to attach up-to-date tail
    try:
        df_live = fetcher.fetch_live_recent(past_hours=48, forecast_hours=24)
        # Combine historical with live, drop duplicates by timestamp
        combined = pd.concat([df_raw, df_live], ignore_index=True)
        combined = combined.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    except Exception as e:
        logger.warning(f"Live fetch during backfill encountered warning: {e}. Using historical data.")
        combined = df_raw
        
    logger.info(f"Total raw observations collected for {city}: {len(combined)} rows.")
    fs.save_raw_observations(combined, city=city)
    
    # 3. Compute Features and 24h Ahead Targets
    logger.info("Computing time-based, lag, and derived features...")
    df_features = compute_features(combined, is_training=True)
    if "city" not in df_features.columns:
        df_features["city"] = city
    
    # Check newest 24h targets
    tail_targets = df_features[f"target_{METRICS[0]}"].tail(FORECAST_HORIZON_HOURS)
    logger.info(f"Target check: Newest {FORECAST_HORIZON_HOURS} hours target null count = {tail_targets.isna().sum()} (Expected {FORECAST_HORIZON_HOURS})")
    
    # 4. Save to Feature Store
    fs.save_features(df_features, city=city)
    logger.info(f"Successfully saved {len(df_features)} feature rows to Feature Store!")
    logger.info(f"Feature set contains {len(df_features.columns)} total columns.")
    return df_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical weather and AQI observations")
    parser.add_argument("--days", type=int, default=180, help="Number of past days to backfill (default: 180)")
    parser.add_argument("--city", type=str, default=DEFAULT_CITY, help="City name")
    args = parser.parse_args()
    
    run_backfill(days=args.days, city=args.city)

