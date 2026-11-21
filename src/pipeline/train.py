import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from statsmodels.tsa.ar_model import AutoReg
import xgboost as xgb
import logging

from config.settings import METRICS, FORECAST_HORIZON_HOURS
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
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Use direct metric lags as statistical regressors
        lag_cols = [c for c in X.columns if c.startswith(f"{self.metric}_lag_")]
        if not lag_cols:
            lag_cols = [c for c in X.columns if "lag" in c][:10]
            
        self.features_used = lag_cols
        X_sub = X[self.features_used].fillna(0.0).values
        y_val = y.values
        
        # OLS closed-form solution: beta = (X^T X)^-1 X^T y
        X_ones = np.c_[np.ones(X_sub.shape[0]), X_sub]
        # Regularized pseudo-inverse for numerical stability
        reg = 1e-3 * np.eye(X_ones.shape[1])
        beta = np.linalg.pinv(X_ones.T @ X_ones + reg) @ X_ones.T @ y_val
        self.intercept = beta[0]
        self.weights = beta[1:]
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_sub = X[self.features_used].fillna(0.0).values
        return self.intercept + X_sub @ self.weights


def train_and_evaluate_pipeline() -> Dict[str, Any]:
    """
    Executes the comprehensive model training and evaluation pipeline:
    1. Fetches features and targets from Feature Store.
    2. Performs chronological time-based split (80% train, 20% validation).
    3. Evaluates 3 distinct model families (Statistical, Classical ML, Deep Learning) plus Naive Baseline.
    4. Computes RMSE, MAE, R² metrics.
    5. Saves and registers champion models in Model Registry.
    """
    fs = FeatureStoreManager()
    registry = ModelRegistry()
    
    df = fs.get_features()
    if df.empty:
        raise ValueError("Feature Store has no features. Please run backfill first!")
        
    logger.info(f"Loaded {len(df)} rows from Feature Store.")
    
    # Drop rows where target is NaN (e.g. the newest 24h reserved for inference)
    # Filter valid rows where targets exist
    train_val_df = df.dropna(subset=[f"target_{m}" for m in METRICS]).copy()
    train_val_df = train_val_df.sort_values("time").reset_index(drop=True)
    
    total_samples = len(train_val_df)
    logger.info(f"Usable training/validation samples with complete targets: {total_samples}")
    
    # Chronological Time-Based Split (80% train, 20% validation)
    split_idx = int(total_samples * 0.80)
    train_df = train_val_df.iloc[:split_idx].copy()
    val_df = train_val_df.iloc[split_idx:].copy()
    
    logger.info(f"Time-based split: Train={len(train_df)} ({train_df['time'].min()} to {train_df['time'].max()}), Val={len(val_df)} ({val_df['time'].min()} to {val_df['time'].max()})")
    
    feature_cols = get_feature_columns(train_df)
    logger.info(f"Input features count: {len(feature_cols)}")
    
    X_train = train_df[feature_cols].fillna(0.0)
    X_val = val_df[feature_cols].fillna(0.0)
    
    training_summary = {}
    
    for metric in METRICS:
        logger.info(f"\n=======================================================")
        logger.info(f"  TRAINING & BENCHMARKING FOR METRIC: [{metric.upper()}]")
        logger.info(f"=======================================================")
        
        y_train = train_df[f"target_{metric}"]
        y_val = val_df[f"target_{metric}"]
        
        # 1. Naive Baseline (Persistence: observation from 24 hours ago / lag_24h)
        # In 24h forecasting, predicting observation(t) as observation(t+24) is the benchmark
        y_val_baseline = val_df[metric].values
        baseline_scores = registry.evaluate_model(y_val.values, y_val_baseline)
        logger.info(f"[{metric}] NAIVE BASELINE (24h Persistence) -> RMSE: {baseline_scores['rmse']}, MAE: {baseline_scores['mae']}, R²: {baseline_scores['r2']}")
        
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
            
        # --- Family 2: Classical ML (Ridge & Random Forest & XGBoost) ---
        try:
            # 2a. Ridge Regression (Scalable linear ML)
            ridge_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0))
            ])
            ridge_pipeline.fit(X_train, y_train)
            ridge_pred = ridge_pipeline.predict(X_val)
            ridge_scores = registry.evaluate_model(y_val.values, ridge_pred)
            candidates.append(("Classical_ML", "Ridge_Regression", ridge_pipeline, ridge_scores))
            logger.info(f"[{metric}] CLASSICAL ML (Ridge) -> RMSE: {ridge_scores['rmse']}, MAE: {ridge_scores['mae']}, R²: {ridge_scores['r2']}")
            
            # 2b. Random Forest Regressor
            rf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_val)
            rf_scores = registry.evaluate_model(y_val.values, rf_pred)
            candidates.append(("Classical_ML", "Random_Forest", rf, rf_scores))
            logger.info(f"[{metric}] CLASSICAL ML (Random Forest) -> RMSE: {rf_scores['rmse']}, MAE: {rf_scores['mae']}, R²: {rf_scores['r2']}")

            # 2c. XGBoost Regressor
            xgb_model = xgb.XGBRegressor(
                n_estimators=120, max_depth=5, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
            )
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_val)
            xgb_scores = registry.evaluate_model(y_val.values, xgb_pred)
            candidates.append(("Classical_ML", "XGBoost", xgb_model, xgb_scores))
            logger.info(f"[{metric}] CLASSICAL ML (XGBoost) -> RMSE: {xgb_scores['rmse']}, MAE: {xgb_scores['mae']}, R²: {xgb_scores['r2']}")
        except Exception as e:
            logger.warning(f"[{metric}] Classical ML error: {e}")
            
        # --- Family 3: Deep Learning (Multi-Layer Perceptron Neural Network) ---
        try:
            dl_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    max_iter=150,
                    early_stopping=True,
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
        logger.info(f"[{metric}] 🏆 CURRENT CHAMPION: {champ_meta['model_name']} ({champ_meta['family']}) - RMSE: {champ_meta['val_metrics']['rmse']}")
        training_summary[metric] = champ_meta
        
    logger.info("\n=== TRAINING AND EVALUATION PIPELINE COMPLETED SUCCESSFULLY ===")
    return training_summary


if __name__ == "__main__":
    train_and_evaluate_pipeline()

