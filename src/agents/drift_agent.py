import numpy as np
import pandas as pd
from scipy import stats
import logging
from typing import Dict, Any, List

from config.settings import (
    METRICS,
    DATA_DRIFT_PVALUE_THRESHOLD,
    MODEL_DRIFT_ERROR_INCREASE_PCT
)
from src.agents.state import AgentWeatherState
from src.pipeline.feature_store import FeatureStoreManager
from src.pipeline.model_registry import ModelRegistry
from src.pipeline.feature_engineering import get_feature_columns

logger = logging.getLogger("DriftAgent")


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """
    Computes Population Stability Index (PSI) between reference and live distributions.
    PSI < 0.1: No change
    0.1 <= PSI < 0.2: Slight drift
    PSI >= 0.2: Significant drift
    """
    try:
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]
        if len(expected) < 10 or len(actual) < 10:
            return 0.0
            
        quantiles = np.linspace(0, 100, num_buckets + 1)
        bins = np.percentile(expected, quantiles)
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        expected_counts = np.histogram(expected, bins=bins)[0]
        actual_counts = np.histogram(actual, bins=bins)[0]
        
        # Add epsilon to prevent division by zero
        expected_pct = np.maximum(expected_counts / len(expected), 1e-4)
        actual_pct = np.maximum(actual_counts / len(actual), 1e-4)
        
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(round(psi, 4))
    except Exception:
        return 0.0


def check_data_drift(df_all: pd.DataFrame, reference_window: int = 1440, live_window: int = 48) -> Dict[str, Any]:
    """
    Compares live features against reference training distribution using
    two-sample Kolmogorov-Smirnov (KS) tests and PSI.
    """
    if len(df_all) < (live_window + 48):
        return {"data_drift_detected": False, "metric_reports": {}}
        
    df_clean = df_all.sort_values("time").reset_index(drop=True)
    live_df = df_clean.iloc[-live_window:]
    ref_df = df_clean.iloc[-(live_window + reference_window):-live_window]
    
    if len(ref_df) < 50:
        ref_df = df_clean.iloc[:-live_window]
        
    drift_detected = False
    metric_reports = {}
    
    for metric in METRICS:
        if metric not in df_clean.columns:
            continue
            
        ref_vals = ref_df[metric].dropna().values
        live_vals = live_df[metric].dropna().values
        
        if len(ref_vals) < 10 or len(live_vals) < 10:
            continue
            
        # Two-sample Kolmogorov-Smirnov test
        ks_res = stats.ks_2samp(ref_vals, live_vals)
        ks_stat = round(float(ks_res.statistic), 4)
        p_val = round(float(ks_res.pvalue), 4)
        psi_score = calculate_psi(ref_vals, live_vals)
        
        # Drift condition: p-value < threshold (significant distribution divergence)
        has_metric_drift = (p_val < DATA_DRIFT_PVALUE_THRESHOLD)
        if has_metric_drift:
            drift_detected = True
            
        metric_reports[metric] = {
            "ks_statistic": ks_stat,
            "p_value": p_val,
            "psi": psi_score,
            "threshold": DATA_DRIFT_PVALUE_THRESHOLD,
            "drift_detected": has_metric_drift
        }
        
    return {"data_drift_detected": drift_detected, "metric_reports": metric_reports}


def check_model_drift(fs: FeatureStoreManager, registry: ModelRegistry) -> Dict[str, Any]:
    """
    Compares recent forecast errors (where actual observations have landed)
    against the baseline validation error recorded in Model Registry.
    """
    ledger_df = fs.get_forecast_ledger(limit=300)
    raw_df = fs.get_raw_observations(limit=500)
    
    if ledger_df.empty or raw_df.empty:
        return {"model_drift_detected": False, "metric_reports": {}}
        
    # Match forecast target_time with raw observation time
    raw_lookup = raw_df.set_index(raw_df["time"].astype(str))
    drift_detected = False
    metric_reports = {}
    
    for metric in METRICS:
        champ_tuple = registry.get_champion(metric)
        if not champ_tuple:
            continue
            
        _, champ_meta = champ_tuple
        val_mae = champ_meta["val_metrics"].get("mae", 5.0)
        
        m_ledger = ledger_df[ledger_df["metric"] == metric].copy()
        errors = []
        
        for _, row in m_ledger.iterrows():
            t_str = str(row["target_time"])[:19]
            # Check if actual is recorded in lookup
            matches = raw_lookup[raw_lookup["time"].astype(str).str.startswith(t_str[:13])]
            if not matches.empty and metric in matches.columns:
                actual = matches[metric].values[0]
                pred = row["forecast_value"]
                if not np.isnan(actual) and not np.isnan(pred):
                    errors.append(abs(actual - pred))
                    
        if len(errors) >= 5:
            recent_mae = round(float(np.mean(errors)), 3)
            # Drift if error grew by more than threshold percent
            error_increase = (recent_mae - val_mae) / max(val_mae, 1e-4)
            has_drift = error_increase > MODEL_DRIFT_ERROR_INCREASE_PCT
            if has_drift:
                drift_detected = True
                
            metric_reports[metric] = {
                "recent_mae": recent_mae,
                "validation_mae": val_mae,
                "error_increase_pct": round(error_increase * 100.0, 1),
                "threshold_pct": round(MODEL_DRIFT_ERROR_INCREASE_PCT * 100.0, 1),
                "drift_detected": has_drift
            }
        else:
            metric_reports[metric] = {
                "recent_mae": val_mae,
                "validation_mae": val_mae,
                "error_increase_pct": 0.0,
                "threshold_pct": round(MODEL_DRIFT_ERROR_INCREASE_PCT * 100.0, 1),
                "drift_detected": False
            }
            
    return {"model_drift_detected": drift_detected, "metric_reports": metric_reports}


def drift_node(state: AgentWeatherState) -> AgentWeatherState:
    """
    LangGraph Node: Drift Agent.
    Evaluates both Data Drift and Model Drift.
    Logs each drift check into the Drift Log.
    If drift crosses thresholds, dispatches automated model retraining pipeline.
    """
    logger.info("--- [Node: Drift Agent] Running drift detection pipeline ---")
    
    fs = FeatureStoreManager()
    registry = ModelRegistry()
    feat_df = fs.get_features()
    
    # 1. Run Data Drift analysis
    data_drift_res = check_data_drift(feat_df)
    # 2. Run Model Drift analysis
    model_drift_res = check_model_drift(fs, registry)
    
    has_data_drift = data_drift_res["data_drift_detected"]
    has_model_drift = model_drift_res["model_drift_detected"]
    overall_drift = has_data_drift or has_model_drift
    
    logger.info(f"Drift Audit Results: Data Drift={has_data_drift}, Model Drift={has_model_drift}")
    
    # 3. Log into Drift Log for each metric
    for metric in METRICS:
        dd_info = data_drift_res["metric_reports"].get(metric, {})
        md_info = model_drift_res["metric_reports"].get(metric, {})
        
        # Log data drift event
        if dd_info:
            fs.log_drift(
                metric=metric,
                drift_type="DATA_DRIFT",
                score=dd_info["p_value"],
                threshold=dd_info["threshold"],
                drift_detected=dd_info["drift_detected"],
                action_taken="ALERT_DISPATCHED" if dd_info["drift_detected"] else "MONITORING_NORMAL",
                details=f"KS-stat: {dd_info['ks_statistic']}, PSI: {dd_info['psi']}"
            )
            
        # Log model drift event
        if md_info:
            fs.log_drift(
                metric=metric,
                drift_type="MODEL_DRIFT",
                score=md_info["error_increase_pct"],
                threshold=md_info["threshold_pct"],
                drift_detected=md_info["drift_detected"],
                action_taken="ALERT_DISPATCHED" if md_info["drift_detected"] else "MONITORING_NORMAL",
                details=f"Recent MAE: {md_info['recent_mae']} vs Val MAE: {md_info['validation_mae']}"
            )
            
    # 4. Trigger Retraining or Schedule if threshold is crossed
    triggered_retrain = False
    if overall_drift:
        logger.warning("🚨 DRIFT THRESHOLD BREACHED! Flagging automated model retraining in audit ledger...")
        triggered_retrain = True
        # Log retrain action into drift ledger
        fs.log_drift(
            metric="ALL",
            drift_type="SYSTEM_ACTION",
            score=1.0,
            threshold=1.0,
            drift_detected=True,
            action_taken="AUTOMATED_RETRAINING_SCHEDULED",
            details="Triggered by Drift Agent upon detection of statistical distribution divergence"
        )
            
    state["drift_status"] = {
        "overall_drift_detected": overall_drift,
        "data_drift": data_drift_res,
        "model_drift": model_drift_res
    }
    state["triggered_retraining"] = triggered_retrain
    
    return state

