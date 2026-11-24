import pandas as pd
from typing import Optional, Dict, Any, List
import logging

from src.pipeline.local_store import LocalFeatureStore
from src.pipeline.hopsworks_store import HopsworksFeatureStore

logger = logging.getLogger("FeatureStoreManager")


class FeatureStoreManager:
    """
    Unified Feature Store Manager orchestrating both Local SQLite/Parquet Feature Store
    and optional Hopsworks Cloud Feature Store.
    """
    
    def __init__(self):
        self.local = LocalFeatureStore()
        self.hopsworks = HopsworksFeatureStore()
        
    def save_raw_observations(self, df: pd.DataFrame, city: Optional[str] = None):
        self.local.save_raw_observations(df, city=city)
        if self.hopsworks.is_configured():
            try:
                self.hopsworks.sync_features(df, feature_group_name="raw_weather_observations")
            except Exception as e:
                logger.warning(f"Hopsworks raw sync skipped: {e}")

    def save_features(self, df: pd.DataFrame, city: Optional[str] = None):
        self.local.save_features(df, city=city)
        if self.hopsworks.is_configured():
            try:
                self.hopsworks.sync_features(df, feature_group_name="weather_features")
            except Exception as e:
                logger.warning(f"Hopsworks feature sync skipped: {e}")

    def get_features(self, city: Optional[str] = None) -> pd.DataFrame:
        return self.local.get_features(city=city)

    def get_raw_observations(self, city: Optional[str] = None) -> pd.DataFrame:
        return self.local.get_raw_observations(city=city)

    # Ledger and logging proxies
    def log_forecast(self, *args, **kwargs):
        self.local.log_forecast(*args, **kwargs)

    def get_forecast_ledger(self, limit: int = 100) -> pd.DataFrame:
        return self.local.get_forecast_ledger(limit=limit)

    def log_agent_call(self, *args, **kwargs):
        self.local.log_agent_call(*args, **kwargs)

    def get_agent_call_log(self, limit: int = 100) -> pd.DataFrame:
        return self.local.get_agent_call_log(limit=limit)

    def get_failed_agent_calls(self) -> pd.DataFrame:
        return self.local.get_failed_agent_calls()

    def log_drift(self, *args, **kwargs):
        self.local.log_drift(*args, **kwargs)

    def get_drift_log(self, limit: int = 100) -> pd.DataFrame:
        return self.local.get_drift_log(limit=limit)

