import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import logging

from config.settings import METRICS, FORECAST_HORIZON_HOURS, PROCESSED_DATA_DIR
from src.pipeline.feature_store import FeatureStoreManager
from src.pipeline.feature_engineering import get_feature_columns
from src.pipeline.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TrainingPipeline")


class StatisticalModelWrapper:
    """
    Statistical time-series forecasting wrapper using AutoRegressive (AR) / lag modeling.
    Predicts metric 24-hours ahead based on statistical auto-correlation.
    """
    def __init__(self, metric: str, lags: int = 24):
        self.metric = metric
        self.lags = lags
        self.weights = None
        self.intercept = 0.0
        self.features_used: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        lag_cols = [c for c in X.columns if c == self.metric or c.startswith(f"{self.metric}_lag_")]
        if not lag_cols:
            lag_cols = [c for c in X.columns if "lag" in c][:10]

        self.features_used = lag_cols
        X_sub = X[self.features_used].fillna(0.0).values
        y_val = y.values

        X_ones = np.c_[np.ones(X_sub.shape[0]), X_sub]
        reg = 1e-3 * np.eye(X_ones.shape[1])
        beta = np.linalg.pinv(X_ones.T @ X_ones + reg) @ X_ones.T @ y_val
        self.intercept = float(beta[0])
        self.weights = beta[1:]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_sub = X[self.features_used].fillna(0.0).values
        return self.intercept + X_sub @ self.weights


def load_all_city_features(fs: FeatureStoreManager) -> pd.DataFrame:
    """
    Loads all per-city engineered parquet files from PROCESSED_DATA_DIR so models train
    across all monitored Pakistani cities, strictly excluding any future forecast rows.
    """
    city_files = sorted(PROCESSED_DATA_DIR.glob("engineered_features_*.parquet"))
    dfs = []
    now_ts = pd.Timestamp.now()

    for fpath in city_files:
        try:
            cdf = pd.read_parquet(fpath)
            if cdf.empty:
                continue
            cdf["time"] = pd.to_datetime(cdf["time"])
            # Strictly exclude API forecast rows or future timestamps from ground-truth training
            if "source" in cdf.columns:
                cdf = cdf[cdf["source"] != "open_meteo_forecast"]
            cdf = cdf[cdf["time"] <= now_ts].copy()
            if not cdf.empty:
                dfs.append(cdf)
        except Exception as e:
            logger.warning(f"Skipping {fpath.name}: {e}")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined)} observed rows across {combined['city'].nunique() if 'city' in combined.columns else 1} cities.")
        return combined

    df = fs.get_features()
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
        if "source" in df.columns:
            df = df[df["source"] != "open_meteo_forecast"]
        df = df[df["time"] <= now_ts].copy()
    return df


def split_chronological_with_embargo(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    embargo_hours: int = FORECAST_HORIZON_HOURS
):
    """
    Performs a strict GLOBAL chronological split with a 24-hour embargo/purge gap:
    - Drops first 24h warmup rows (where lag_24h is NaN) and last 24h rows (no target_*) per city.
    - Computes a single GLOBAL calendar timestamp cutoff across all cities so NO city's
      training rows overlap in calendar time with ANY other city's validation rows
      (prevents cross-city synoptic weather leakage).
    - Enforces: global max(train_time) + 24h <= global min(val_time).
    """
    target_cols = [f"target_{m}" for m in METRICS]
    lag24_cols = [f"{m}_lag_24h" for m in METRICS if f"{m}_lag_24h" in df.columns]

    cleaned_parts = []
    groups = df.groupby("city", sort=True) if "city" in df.columns else [("default", df)]
    for _, gdf in groups:
        gdf = gdf.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
        gdf = gdf.dropna(subset=target_cols + lag24_cols).reset_index(drop=True)
        if len(gdf) > embargo_hours + 48:
            cleaned_parts.append(gdf)

    full_df = pd.concat(cleaned_parts, ignore_index=True)
    unique_times = np.sort(full_df["time"].unique())
    cutoff_idx = int(len(unique_times) * train_ratio)
    train_end_time = pd.Timestamp(unique_times[cutoff_idx - 1])
    val_start_time = train_end_time + pd.Timedelta(hours=embargo_hours)

    train_df = full_df[full_df["time"] <= train_end_time].sort_values(["city", "time"] if "city" in full_df.columns else ["time"]).reset_index(drop=True)
    val_df = full_df[full_df["time"] >= val_start_time].sort_values(["city", "time"] if "city" in full_df.columns else ["time"]).reset_index(drop=True)

    # Hard global leakage assertion across ALL cities combined
    max_train_target_time = train_df["time"].max() + pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
    min_val_time = val_df["time"].min()
    assert max_train_target_time <= min_val_time, (
        f"Global temporal leakage detected: max train target {max_train_target_time} > min val time {min_val_time}"
    )

    return train_df, val_df


def train_and_evaluate_pipeline(reset_registry: bool = True) -> Dict[str, Any]:
    """
    Zero-leakage model training and evaluation pipeline:
    1. Loads observed multi-city historical data (strictly excluding future forecast rows).
    2. Applies per-city chronological 80/20 split with a strict 24-hour embargo gap.
    3. Fits imputation medians and scalers STRICTLY on training fold only.
    4. Evaluates 3 distinct model families (Statistical, Classical ML, Deep Learning) vs 24h Naive Baseline.
    5. Registers clean champion models in Model Registry.
    """
    fs = FeatureStoreManager()
    registry = ModelRegistry()
    if reset_registry:
        registry.reset_registry()

    df = load_all_city_features(fs)
    if df.empty:
        raise ValueError("Feature Store has no features. Please run backfill first!")

    train_df, val_df = split_chronological_with_embargo(
        df, train_ratio=0.80, embargo_hours=FORECAST_HORIZON_HOURS
    )
    logger.info(
        f"Leakage-Free Embargoed Split (gap={FORECAST_HORIZON_HOURS}h): "
        f"Train={len(train_df)} ({train_df['time'].min()} to {train_df['time'].max()}), "
        f"Val={len(val_df)} ({val_df['time'].min()} to {val_df['time'].max()})"
    )

    feature_cols = get_feature_columns(train_df)
    logger.info(f"Input features count: {len(feature_cols)}")

    # Strictly train-only median imputation (zero validation stats leakage)
    train_medians = train_df[feature_cols].median().fillna(0.0)
    X_train = train_df[feature_cols].fillna(train_medians)
    X_val = val_df[feature_cols].fillna(train_medians)

    training_summary = {}

    for metric in METRICS:
        logger.info(f"\n=======================================================")
        logger.info(f"  TRAINING & BENCHMARKING FOR METRIC: [{metric.upper()}]")
        logger.info(f"=======================================================")

        y_train = train_df[f"target_{metric}"]
        y_val = val_df[f"target_{metric}"]

        # 1. Naive Baseline (24h Persistence: y_hat(t+24) = y(t))
        y_val_baseline = val_df[metric].values
        baseline_scores = registry.evaluate_model(y_val.values, y_val_baseline)
        logger.info(
            f"[{metric}] NAIVE BASELINE (24h Persistence) -> "
            f"RMSE: {baseline_scores['rmse']}, MAE: {baseline_scores['mae']}, R²: {baseline_scores['r2']}"
        )

        candidates = []

        # --- Family 1: Statistical ---
        try:
            stat_model = StatisticalModelWrapper(metric=metric)
            stat_model.fit(X_train, y_train)
            stat_pred = stat_model.predict(X_val)
            stat_scores = registry.evaluate_model(y_val.values, stat_pred)
            candidates.append(("Statistical", "AutoRegressive_OLS", stat_model, stat_scores))
            logger.info(f"[{metric}] STATISTICAL (AutoReg) -> RMSE: {stat_scores['rmse']}, MAE: {stat_scores['mae']}, R²: {stat_scores['r2']}")
        except Exception as e:
            logger.warning(f"[{metric}] Statistical model error: {e}")

        # --- Family 2: Classical ML (Ridge, Random Forest, XGBoost) ---
        try:
            ridge_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0))
            ])
            ridge_pipeline.fit(X_train, y_train)
            ridge_pred = ridge_pipeline.predict(X_val)
            ridge_scores = registry.evaluate_model(y_val.values, ridge_pred)
            candidates.append(("Classical_ML", "Ridge_Regression", ridge_pipeline, ridge_scores))
            logger.info(f"[{metric}] CLASSICAL ML (Ridge) -> RMSE: {ridge_scores['rmse']}, MAE: {ridge_scores['mae']}, R²: {ridge_scores['r2']}")

            rf = RandomForestRegressor(
                n_estimators=80, max_depth=12, min_samples_split=10,
                max_features="sqrt", max_samples=0.6, random_state=42, n_jobs=-1
            )
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_val)
            rf_scores = registry.evaluate_model(y_val.values, rf_pred)
            candidates.append(("Classical_ML", "Random_Forest", rf, rf_scores))
            logger.info(f"[{metric}] CLASSICAL ML (Random Forest) -> RMSE: {rf_scores['rmse']}, MAE: {rf_scores['mae']}, R²: {rf_scores['r2']}")

            xgb_model = xgb.XGBRegressor(
                n_estimators=150, max_depth=5, learning_rate=0.06,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                random_state=42, n_jobs=-1
            )
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_val)
            xgb_scores = registry.evaluate_model(y_val.values, xgb_pred)
            candidates.append(("Classical_ML", "XGBoost", xgb_model, xgb_scores))
            logger.info(f"[{metric}] CLASSICAL ML (XGBoost) -> RMSE: {xgb_scores['rmse']}, MAE: {xgb_scores['mae']}, R²: {xgb_scores['r2']}")
        except Exception as e:
            logger.warning(f"[{metric}] Classical ML error: {e}")

        # --- Family 3: Deep Learning (MLP Neural Network) ---
        try:
            dl_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.01,
                    batch_size=512,
                    max_iter=40,
                    n_iter_no_change=5,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42
                ))
            ])
            dl_pipeline.fit(X_train, y_train)
            dl_pred = dl_pipeline.predict(X_val)
            dl_scores = registry.evaluate_model(y_val.values, dl_pred)
            candidates.append(("Deep_Learning", "Neural_Network_MLP", dl_pipeline, dl_scores))
            logger.info(f"[{metric}] DEEP LEARNING (MLP Neural Net) -> RMSE: {dl_scores['rmse']}, MAE: {dl_scores['mae']}, R²: {dl_scores['r2']}")
        except Exception as e:
            logger.warning(f"[{metric}] Deep learning error: {e}")

        # Register each candidate in Model Registry and select Champion
        for family, name, artifact, scores in candidates:
            registry.register_candidate(
                metric=metric,
                family=family,
                model_name=name,
                model_artifact=artifact,
                val_metrics=scores,
                baseline_metrics=baseline_scores,
                feature_names=feature_cols
            )

        champ_artifact, champ_meta = registry.get_champion(metric)
        logger.info(
            f"[{metric}] 🏆 CURRENT CHAMPION: {champ_meta['model_name']} ({champ_meta['family']}) - "
            f"RMSE: {champ_meta['val_metrics']['rmse']} vs Baseline: {baseline_scores['rmse']}"
        )
        training_summary[metric] = champ_meta

    logger.info("\n=== ZERO-LEAKAGE TRAINING & EVALUATION PIPELINE COMPLETED ===")
    return training_summary


if __name__ == "__main__":
    train_and_evaluate_pipeline()
