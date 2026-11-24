import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

from src.pipeline.feature_store import FeatureStoreManager
from config.settings import METRICS

logger = logging.getLogger("EDA")


class EDAManager:
    """
    Performs comprehensive Exploratory Data Analysis (EDA) on all 6 weather and air-quality metrics:
    - Summary descriptive statistics and skewness
    - Multi-metric correlation matrix (pollutants vs meteorological factors)
    - Diurnal (24-hour hourly cycle) patterns
    - Outlier detection (IQR method)
    """
    
    def __init__(self):
        self.fs = FeatureStoreManager()
        
    def get_eda_dataframe(self) -> pd.DataFrame:
        # Use Lahore parquet (fastest, representative dataset) instead of full 140K SQLite scan
        df = self.fs.get_features(city="Lahore")
        if df.empty:
            df = self.fs.get_features()
        return df

    def get_summary_statistics(self) -> pd.DataFrame:
        """Returns descriptive statistics, skewness, and missing rate for each metric."""
        df = self.get_eda_dataframe()
        if df.empty:
            return pd.DataFrame()
            
        metrics_present = [m for m in METRICS if m in df.columns]
        desc = df[metrics_present].describe().T
        desc["skewness"] = df[metrics_present].skew()
        desc["missing_pct"] = (df[metrics_present].isna().mean() * 100.0).round(2)
        desc["iqr"] = desc["75%"] - desc["25%"]
        
        cols = ["count", "mean", "std", "min", "25%", "50%", "75%", "max", "skewness", "missing_pct"]
        return desc[cols].round(2)

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Computes Pearson correlation matrix across weather metrics and pollutants."""
        df = self.get_eda_dataframe()
        if df.empty:
            return pd.DataFrame()
            
        cols = [m for m in METRICS if m in df.columns]
        for extra in ["pm2_5", "pm10", "precipitation_probability_pct"]:
            if extra in df.columns and extra not in cols:
                cols.append(extra)
                
        corr = df[cols].corr().round(3)
        return corr

    def get_diurnal_patterns(self) -> pd.DataFrame:
        """Computes mean value of each metric for each hour of the day (0 to 23)."""
        df = self.get_eda_dataframe()
        if df.empty or "time" not in df.columns:
            return pd.DataFrame()
            
        df_copy = df.copy()
        df_copy["hour"] = pd.to_datetime(df_copy["time"]).dt.hour
        
        metrics_present = [m for m in METRICS if m in df_copy.columns]
        hourly = df_copy.groupby("hour")[metrics_present].mean().round(2).reset_index()
        return hourly

    def detect_outliers_iqr(self) -> pd.DataFrame:
        """Identifies outliers using the standard Interquartile Range (IQR) rule: Q1 - 1.5*IQR or Q3 + 1.5*IQR."""
        df = self.get_eda_dataframe()
        if df.empty:
            return pd.DataFrame()
            
        rows = []
        metrics_present = [m for m in METRICS if m in df.columns]
        
        for m in metrics_present:
            s = df[m].dropna()
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = s[(s < lower_bound) | (s > upper_bound)]
            rows.append({
                "Metric": m,
                "Total Observations": len(s),
                "Q1 (25th)": round(q1, 2),
                "Q3 (75th)": round(q3, 2),
                "IQR": round(iqr, 2),
                "Lower Fence": round(lower_bound, 2),
                "Upper Fence": round(upper_bound, 2),
                "Outlier Count": len(outliers),
                "Outlier Pct (%)": round(len(outliers) / len(s) * 100.0, 2) if len(s) > 0 else 0.0
            })
            
        return pd.DataFrame(rows)

