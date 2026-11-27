import argparse
import pandas as pd
import logging
from datetime import datetime

from config.settings import DEFAULT_CITY
from src.pipeline.fetcher import DataFetcher
from src.pipeline.feature_engineering import compute_features
from src.pipeline.feature_store import FeatureStoreManager
from src.agents.graph import run_agent_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("HourlyIngestionPipeline")


def run_hourly_ingestion(city: str = DEFAULT_CITY, trigger_agents: bool = True):
    """
    Executes the Hourly Feature Ingestion pipeline:
    1. Fetches weather and pollutant data for the city.
    2. Fuses & time-aligns sources into one record per timestamp.
    3. Imputes missing values with documented strategy.
    4. Computes time-based, lag, and derived features.
    5. Stores features and raw observations in Feature Store.
    6. Triggers LangGraph Agentic Layer (Forecast, Alert, Report, Drift).
    """
    logger.info(f"=== Starting Hourly Data & Feature Ingestion for {city} ===")
    
    fetcher = DataFetcher(city=city)
    fs = FeatureStoreManager()
    
    # 1. Fetch live recent OBSERVED hours only (forecast_hours=0 prevents future leakage)
    df_live = fetcher.fetch_live_recent(past_hours=48, forecast_hours=0)
    df_live = df_live[df_live["time"] <= pd.Timestamp.now()].copy()
    logger.info(f"Fetched {len(df_live)} live observed rows.")
    
    # 2. Merge with existing historical raw observations for this city
    existing_raw = fs.get_raw_observations(city=city)
    if not existing_raw.empty:
        combined = pd.concat([existing_raw, df_live], ignore_index=True)
        combined = combined.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    else:
        combined = df_live
        
    fs.save_raw_observations(combined, city=city)
    
    # 3. Compute Features and Targets
    logger.info("Computing engineered features...")
    df_features = compute_features(combined, is_training=True)
    fs.save_features(df_features, city=city)
    logger.info(f"Feature Store updated with {len(df_features)} total feature rows.")
    
    # 4. Trigger the LangGraph Agentic Reasoning Layer
    if trigger_agents:
        logger.info("Triggering LangGraph agentic reasoning cycle...")
        state = run_agent_pipeline(city=city)
        logger.info("Agent cycle completed successfully.")
        return state
        
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hourly Data & Feature Ingestion Pipeline")
    parser.add_argument("--city", type=str, default=DEFAULT_CITY, help="City name")
    parser.add_argument("--no-agents", action="store_true", help="Skip running agent reasoning layer")
    args = parser.parse_args()
    
    run_hourly_ingestion(city=args.city, trigger_agents=not args.no_agents)

