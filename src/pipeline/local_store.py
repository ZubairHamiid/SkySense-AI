import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

from config.settings import DB_PATH, PROCESSED_DATA_DIR, RAW_DATA_DIR

logger = logging.getLogger("LocalFeatureStore")


class LocalFeatureStore:
    """
    Local SQLite and Parquet storage engine serving as a fast, reliable, zero-config
    Feature Store, Forecast Ledger, Agent Call Log, and Drift Log repository.
    """
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = str(db_path)
        self._init_tables()
        
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
        
    def _init_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Raw Observations
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_observations (
                time TEXT PRIMARY KEY,
                city TEXT,
                aqi REAL,
                temperature_c REAL,
                precipitation_mm REAL,
                precipitation_probability_pct REAL,
                humidity_pct REAL,
                uv_index REAL,
                wind_speed_kmh REAL,
                wind_direction_deg REAL,
                pm2_5 REAL,
                pm10 REAL,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # 2. Forecast Ledger: every forecast made by the model (time, metric, forecast)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS forecast_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                forecast_generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                target_time TEXT,
                city TEXT,
                metric TEXT,
                forecast_value REAL,
                naive_baseline_value REAL,
                model_family TEXT,
                model_name TEXT,
                actual_value REAL,
                forecast_error REAL
            )
            """)
            
            # 3. Agent Call Log: prompt, response, latency, and status of every LLM call
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_call_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                agent_name TEXT,
                prompt TEXT,
                response TEXT,
                latency_ms REAL,
                status TEXT,
                model_used TEXT,
                error_message TEXT
            )
            """)
            
            # 4. Drift Log: each drift event (time, metric, drift type, score, threshold, action taken)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metric TEXT,
                drift_type TEXT,
                score REAL,
                threshold REAL,
                drift_detected INTEGER,
                action_taken TEXT,
                details TEXT
            )
            """)
            conn.commit()
            
    def save_raw_observations(self, df: pd.DataFrame, city: Optional[str] = None):
        """Saves raw hourly observations into SQLite and Parquet with multi-city partitioning."""
        if df.empty:
            return
            
        df_save = df.copy()
        df_save["time"] = df_save["time"].astype(str)
        if city and "city" not in df_save.columns:
            df_save["city"] = city
            
        # Determine city target
        target_city = city or (df_save["city"].iloc[0] if "city" in df_save.columns else "Lahore")
        
        # Save city-specific Parquet and combined Parquet
        city_parquet = RAW_DATA_DIR / f"raw_observations_{target_city.lower()}.parquet"
        if city_parquet.exists():
            try:
                exist_city = pd.read_parquet(city_parquet)
                comb_city = pd.concat([exist_city, df], ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
                comb_city.to_parquet(city_parquet, index=False)
            except Exception:
                df.to_parquet(city_parquet, index=False)
        else:
            df.to_parquet(city_parquet, index=False)
        
        # Merge into global Parquet
        global_parquet = RAW_DATA_DIR / "raw_observations.parquet"
        if global_parquet.exists():
            try:
                existing_df = pd.read_parquet(global_parquet)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                if "city" in combined_df.columns:
                    combined_df = combined_df.drop_duplicates(subset=["time", "city"]).sort_values("time").reset_index(drop=True)
                else:
                    combined_df = combined_df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
                combined_df.to_parquet(global_parquet, index=False)
            except Exception:
                df.to_parquet(global_parquet, index=False)
        else:
            df.to_parquet(global_parquet, index=False)
        
        with self._get_connection() as conn:
            cols = [
                "time", "city", "aqi", "temperature_c", "precipitation_mm",
                "precipitation_probability_pct", "humidity_pct", "uv_index",
                "wind_speed_kmh", "wind_direction_deg", "pm2_5", "pm10", "source"
            ]
            avail_cols = [c for c in cols if c in df_save.columns]
            
            records = df_save[avail_cols].to_dict(orient="records")
            placeholders = ", ".join(["?" for _ in avail_cols])
            columns_str = ", ".join(avail_cols)
            
            query = f"""
            INSERT OR REPLACE INTO raw_observations ({columns_str})
            VALUES ({placeholders})
            """
            data_tuples = [tuple(r.get(c, None) for c in avail_cols) for r in records]
            conn.executemany(query, data_tuples)
            conn.commit()
        logger.info(f"Saved {len(df)} raw observations for [{target_city}] to Local Feature Store.")

    def save_features(self, df: pd.DataFrame, city: Optional[str] = None):
        """Saves processed feature table into Parquet."""
        if df.empty:
            return
        target_city = city or (df["city"].iloc[0] if "city" in df.columns else "Lahore")
        city_parquet = PROCESSED_DATA_DIR / f"engineered_features_{target_city.lower()}.parquet"
        if city_parquet.exists():
            try:
                exist_f = pd.read_parquet(city_parquet)
                comb_f = pd.concat([exist_f, df], ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
                comb_f.to_parquet(city_parquet, index=False)
            except Exception:
                df.to_parquet(city_parquet, index=False)
        else:
            df.to_parquet(city_parquet, index=False)
        
        global_parquet = PROCESSED_DATA_DIR / "engineered_features.parquet"
        df.to_parquet(global_parquet, index=False)
        logger.info(f"Saved {len(df)} engineered feature rows for [{target_city}] to {city_parquet}")
        
    def get_features(self, city: Optional[str] = None) -> pd.DataFrame:
        """Retrieves engineered features from storage for a specified city or all."""
        if city:
            city_parquet = PROCESSED_DATA_DIR / f"engineered_features_{city.lower()}.parquet"
            if city_parquet.exists():
                return pd.read_parquet(city_parquet)
                
        parquet_path = PROCESSED_DATA_DIR / "engineered_features.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            if city and "city" in df.columns:
                df_city = df[df["city"].str.lower() == city.lower()]
                if not df_city.empty:
                    return df_city
            return df
            
        raw_df = self.get_raw_observations(city=city)
        if not raw_df.empty:
            from src.pipeline.feature_engineering import compute_features
            feat_df = compute_features(raw_df, is_training=True)
            self.save_features(feat_df, city=city)
            return feat_df
        return pd.DataFrame()

    def get_raw_observations(self, city: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieves raw observations, optionally filtered by city and limit."""
        with self._get_connection() as conn:
            limit_clause = f" LIMIT {int(limit)}" if limit else ""
            order_clause = "DESC" if limit else "ASC"
            if city:
                df = pd.read_sql(f"SELECT * FROM raw_observations WHERE LOWER(city) = ? ORDER BY time {order_clause}{limit_clause}", conn, params=(city.lower(),))
                if df.empty:
                    # Fallback to general observations if specific city records are not yet filtered
                    df = pd.read_sql(f"SELECT * FROM raw_observations ORDER BY time {order_clause}{limit_clause}", conn)
            else:
                df = pd.read_sql(f"SELECT * FROM raw_observations ORDER BY time {order_clause}{limit_clause}", conn)
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"])
            if limit:
                df = df.sort_values("time").reset_index(drop=True)
        return df

    # --- Forecast Ledger Methods ---
    def log_forecast(self, target_time: str, city: str, metric: str,
                     forecast_val: float, baseline_val: float,
                     model_family: str, model_name: str,
                     actual_val: Optional[float] = None, error: Optional[float] = None):
        """Logs a generated forecast record into the Forecast Ledger."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO forecast_ledger 
            (target_time, city, metric, forecast_value, naive_baseline_value, model_family, model_name, actual_value, forecast_error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (str(target_time), city, metric, forecast_val, baseline_val, model_family, model_name, actual_val, error))
            conn.commit()
            
    def get_forecast_ledger(self, limit: int = 100) -> pd.DataFrame:
        """Fetches the latest forecast ledger records."""
        with self._get_connection() as conn:
            df = pd.read_sql(f"SELECT * FROM forecast_ledger ORDER BY id DESC LIMIT {limit}", conn)
        return df

    # --- Agent Call Log Methods ---
    def log_agent_call(self, agent_name: str, prompt: str, response: str,
                       latency_ms: float, status: str, model_used: str,
                       error_message: Optional[str] = None):
        """Logs an agent LLM execution or failure event."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO agent_call_log 
            (agent_name, prompt, response, latency_ms, status, model_used, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (agent_name, str(prompt), str(response), latency_ms, status, model_used, error_message))
            conn.commit()
            
    def get_agent_call_log(self, limit: int = 100) -> pd.DataFrame:
        """Fetches agent call logs."""
        with self._get_connection() as conn:
            df = pd.read_sql(f"SELECT * FROM agent_call_log ORDER BY id DESC LIMIT {limit}", conn)
        return df

    def get_failed_agent_calls(self) -> pd.DataFrame:
        """Fetches failed agent calls for failure audits."""
        with self._get_connection() as conn:
            df = pd.read_sql("SELECT * FROM agent_call_log WHERE status != 'SUCCESS' ORDER BY id DESC", conn)
        return df

    # --- Drift Log Methods ---
    def log_drift(self, metric: str, drift_type: str, score: float,
                  threshold: float, drift_detected: bool,
                  action_taken: str, details: str = ""):
        """Logs a data drift or model drift event."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO drift_log 
            (metric, drift_type, score, threshold, drift_detected, action_taken, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (metric, drift_type, float(score), float(threshold), 1 if drift_detected else 0, action_taken, details))
            conn.commit()
            
    def get_drift_log(self, limit: int = 100) -> pd.DataFrame:
        """Fetches drift logs."""
        with self._get_connection() as conn:
            df = pd.read_sql(f"SELECT * FROM drift_log ORDER BY id DESC LIMIT {limit}", conn)
        return df

