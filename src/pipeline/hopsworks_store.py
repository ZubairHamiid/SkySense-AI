import os
import pandas as pd
import logging
from typing import Optional

from config.settings import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT

logger = logging.getLogger("HopsworksStore")


class HopsworksFeatureStore:
    """
    Hopsworks Free Tier Feature Store integration.
    Connects to Hopsworks cloud, manages feature groups, and synchronizes observations.
    """
    
    def __init__(self, api_key: Optional[str] = HOPSWORKS_API_KEY, project_name: str = HOPSWORKS_PROJECT):
        self.api_key = api_key
        self.project_name = project_name
        self.project = None
        self.fs = None
        
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())
        
    def connect(self) -> bool:
        if not self.is_configured():
            logger.info("Hopsworks API key not provided in .env. Operating in Local Feature Store mode.")
            return False
            
        try:
            import hopsworks
            logger.info(f"Connecting to Hopsworks project: {self.project_name}...")
            self.project = hopsworks.login(api_key_value=self.api_key, project=self.project_name)
            self.fs = self.project.get_feature_store()
            logger.info("Successfully connected to Hopsworks Feature Store!")
            return True
        except Exception as e:
            logger.warning(f"Could not connect to Hopsworks ({e}). Using Local Feature Store.")
            return False
            
    def sync_features(self, df: pd.DataFrame, feature_group_name: str = "weather_features", version: int = 1) -> bool:
        """Uploads feature dataframe to Hopsworks Feature Group."""
        if not self.fs and not self.connect():
            return False
            
        try:
            df_sync = df.copy()
            # Hopsworks expects unix timestamp in milliseconds or integer/string for time
            if "time" in df_sync.columns:
                df_sync["timestamp_ms"] = (df_sync["time"].astype("int64") // 10**6).astype("int64")
                
            fg = self.fs.get_or_create_feature_group(
                name=feature_group_name,
                version=version,
                primary_key=["timestamp_ms"],
                event_time="timestamp_ms",
                description="AgentWeather hourly engineered features and targets"
            )
            fg.insert(df_sync)
            logger.info(f"Successfully synced {len(df_sync)} records to Hopsworks Feature Group: {feature_group_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to sync to Hopsworks: {e}")
            return False

