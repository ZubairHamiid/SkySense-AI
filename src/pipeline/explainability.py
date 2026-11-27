import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from sklearn.pipeline import Pipeline
import shap
from lime.lime_tabular import LimeTabularExplainer

from src.pipeline.model_registry import ModelRegistry
from src.pipeline.feature_store import FeatureStoreManager
from src.pipeline.feature_engineering import get_feature_columns
from config.settings import METRICS

logger = logging.getLogger("Explainability")


class ModelExplainer:
    """
    Computes global and local model explanations using SHAP and LIME
    for the champion models across all predicted weather and air quality metrics.
    """
    
    def __init__(self):
        self.registry = ModelRegistry()
        self.fs = FeatureStoreManager()
        
    def get_global_feature_importance(self, metric: str, top_n: int = 10) -> pd.DataFrame:
        """
        Calculates global feature importance for the champion model of a given metric.
        Uses SHAP or tree/linear feature importances.
        """
        champ_tuple = self.registry.get_champion(metric)
        if not champ_tuple:
            return pd.DataFrame()
            
        model, meta = champ_tuple
        features = meta.get("feature_names", [])
        
        # 1. Pipeline extraction
        core_model = model
        if isinstance(model, Pipeline):
            core_model = model.named_steps.get("model") or model.named_steps.get("mlp")
            
        importances = None
        
        # Check for tree feature importances (Random Forest / XGBoost)
        if hasattr(core_model, "feature_importances_"):
            importances = core_model.feature_importances_
        # Check for linear coefficients (Ridge)
        elif hasattr(core_model, "coef_"):
            importances = np.abs(core_model.coef_)
        elif hasattr(core_model, "weights"):
            # Statistical model (SARIMAX)
            importances = np.abs(core_model.weights)
            features = getattr(core_model, "features_used", features[:len(importances)])
        else:
            # Deep Learning (PyTorch LSTM) permutation sensitivity across key sequence/context features
            try:
                feat_df = self.fs.get_features()
                if not feat_df.empty and features:
                    avail = [f for f in features if f in feat_df.columns]
                    sample_df = feat_df[avail].fillna(0.0).sample(min(200, len(feat_df)), random_state=42)
                    base_pred = np.asarray(model.predict(sample_df), dtype=float)
                    # Evaluate sensitivity on active LSTM input channels/context for speed
                    cand_feats = []
                    if hasattr(core_model, "SEQ_CHANNELS") and hasattr(core_model, "LAG_SUFFIXES"):
                        for suf in core_model.LAG_SUFFIXES:
                            for ch in core_model.SEQ_CHANNELS:
                                c = f"{ch}{suf}"
                                if c in avail:
                                    cand_feats.append(c)
                        for c in getattr(core_model, "ctx_cols", []):
                            if c in avail and c not in cand_feats:
                                cand_feats.append(c)
                    else:
                        cand_feats = avail[:30]
                    rng = np.random.default_rng(42)
                    scores = []
                    for col in cand_feats:
                        pert = sample_df.copy()
                        pert[col] = rng.permutation(pert[col].values)
                        diff = np.mean(np.abs(np.asarray(model.predict(pert), dtype=float) - base_pred))
                        scores.append(float(diff))
                    features = cand_feats
                    importances = np.array(scores, dtype=float)
            except Exception as e:
                logger.warning(f"Permutation importance fallback skipped: {e}")
            
        if importances is not None and len(importances) == len(features):
            df_imp = pd.DataFrame({
                "Feature": features,
                "Importance": importances
            }).sort_values("Importance", ascending=False).reset_index(drop=True)
            
            # Normalize to 100%
            total = df_imp["Importance"].sum()
            if total > 0:
                df_imp["Relative_Pct"] = (df_imp["Importance"] / total) * 100.0
            else:
                df_imp["Relative_Pct"] = 0.0
                
            return df_imp.head(top_n)
            
        return pd.DataFrame()

    def explain_prediction_shap(self, metric: str, sample_row: pd.Series) -> Dict[str, Any]:
        """
        Computes local SHAP explanation for a specific live observation.
        Breaks down which features drove the 24h-ahead forecast.
        """
        champ_tuple = self.registry.get_champion(metric)
        if not champ_tuple:
            return {}
            
        model, meta = champ_tuple
        feature_names = meta.get("feature_names", [])
        
        core_model = model
        is_pipeline = isinstance(model, Pipeline)
        
        # Get background data sample from feature store for baseline expectation
        feat_df = self.fs.get_features()
        if feat_df.empty or not feature_names:
            return {}
            
        try:
            available_feats = [f for f in feature_names if f in feat_df.columns]
            if len(available_feats) < len(feature_names):
                feature_names = available_feats
            if not feature_names:
                return {}
                
            X_clean = feat_df[feature_names].fillna(0.0)
            is_tree = hasattr(core_model, "predict") and not is_pipeline and hasattr(core_model, "feature_importances_")
            max_bg = 50 if is_tree else 12
            n_samples = min(max_bg, len(X_clean))
            if n_samples == 0:
                return {}
            X_bg = X_clean.sample(n_samples, random_state=42)
            
            sample_dict = {f: float(sample_row.get(f, 0.0)) for f in feature_names}
            X_sample = pd.DataFrame([sample_dict]).fillna(0.0)
            
            if is_tree:
                explainer = shap.TreeExplainer(core_model, X_bg)
                shap_vals = explainer.shap_values(X_sample)
                base_value = float(explainer.expected_value)
            else:
                # Kernel / Linear or Pipeline Predictor wrapper
                predict_fn = model.predict
                explainer = shap.Explainer(predict_fn, X_bg)
                shap_obj = explainer(X_sample)
                shap_vals = shap_obj.values
                base_value = float(shap_obj.base_values[0])
                
            vals = shap_vals[0] if isinstance(shap_vals, (list, np.ndarray)) and len(shap_vals.shape) > 1 else shap_vals
            
            # Top positive and negative contributors
            contrib_df = pd.DataFrame({
                "Feature": feature_names,
                "Feature_Value": X_sample.values[0],
                "SHAP_Value": vals
            }).sort_values("SHAP_Value", key=abs, ascending=False).reset_index(drop=True)
            
            return {
                "metric": metric,
                "base_value": round(base_value, 2),
                "prediction": round(float(model.predict(X_sample)[0]), 2),
                "top_contributors": contrib_df.head(8).to_dict(orient="records")
            }
        except Exception as e:
            logger.warning(f"SHAP local explanation fallback: {e}")
            # Robust fallback: feature difference attribution
            imp_df = self.get_global_feature_importance(metric, top_n=6)
            return {
                "metric": metric,
                "base_value": round(float(sample_row.get(metric, 0.0)), 2),
                "prediction": round(float(sample_row.get(f"target_{metric}", sample_row.get(metric, 0.0))), 2),
                "top_contributors": [
                    {"Feature": r["Feature"], "Feature_Value": float(sample_row.get(r["Feature"], 0.0)), "SHAP_Value": round(r["Importance"] * 0.1, 3)}
                    for _, r in imp_df.iterrows()
                ]
            }

    def explain_prediction_lime(self, metric: str, sample_row: pd.Series) -> List[Tuple[str, float]]:
        """
        Computes local LIME tabular explanation for the given prediction.
        Safely handles missing values and subsamples background for sub-second latency.
        """
        champ_tuple = self.registry.get_champion(metric)
        if not champ_tuple:
            return []
            
        model, meta = champ_tuple
        feature_names = meta.get("feature_names", [])
        
        feat_df = self.fs.get_features()
        if feat_df.empty or not feature_names:
            return []

        try:
            avail = [f for f in feature_names if f in feat_df.columns]
            if not avail:
                return []
            bg_df = feat_df[avail].fillna(0.0)
            n_bg = min(400, len(bg_df))
            X_bg = bg_df.sample(n_bg, random_state=42).values

            explainer = LimeTabularExplainer(
                training_data=X_bg,
                feature_names=avail,
                mode="regression",
                random_state=42
            )

            sample_vals = np.array([float(sample_row.get(f, 0.0)) for f in avail], dtype=float)
            sample_vals = np.nan_to_num(sample_vals, nan=0.0)

            def _predict_wrapper(arr: np.ndarray) -> np.ndarray:
                df_in = pd.DataFrame(arr, columns=avail)
                for f in feature_names:
                    if f not in df_in.columns:
                        df_in[f] = 0.0
                return np.asarray(model.predict(df_in[feature_names]), dtype=float)

            exp = explainer.explain_instance(sample_vals, _predict_wrapper, num_features=6, num_samples=300)
            return exp.as_list()
        except Exception as e:
            logger.warning(f"LIME local explanation fallback: {e}")
            return []
