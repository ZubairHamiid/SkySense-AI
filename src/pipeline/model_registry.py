import json
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

from config.settings import MODEL_REGISTRY_DIR, METRICS

logger = logging.getLogger("ModelRegistry")


class ModelRegistry:
    """
    Manages model versioning, champion model selection, performance tracking,
    and artifact persistence.
    """
    
    def __init__(self, registry_dir: Path = MODEL_REGISTRY_DIR):
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.registry_dir / "registry_metadata.json"
        self._load_metadata()
        
    def _load_metadata(self):
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading registry metadata: {e}. Reinitializing.")
                self.metadata = {"champions": {}, "history": []}
        else:
            self.metadata = {"champions": {}, "history": []}
            
    def _save_metadata(self):
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def reset_registry(self):
        """
        Resets champions and evaluation history before a fresh clean benchmark run,
        ensuring models evaluated on different/older splits never pollute champion selection.
        """
        self.metadata = {"champions": {}, "history": []}
        self._save_metadata()
        logger.info("Model Registry metadata reset for clean benchmark evaluation.")

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates RMSE, MAE, and R² for predictions."""
        # Filter out any NaN values
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if np.sum(mask) == 0:
            return {"rmse": 999.0, "mae": 999.0, "r2": -1.0}
            
        yt = y_true[mask]
        yp = y_pred[mask]
        
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae = float(mean_absolute_error(yt, yp))
        r2 = float(r2_score(yt, yp))
        return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}

    def register_candidate(
        self,
        metric: str,
        family: str,
        model_name: str,
        model_artifact: Any,
        val_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        feature_names: List[str]
    ) -> bool:
        """
        Evaluates whether candidate model outperforms existing champion on validation RMSE.
        If better, saves model artifact and updates champion in registry.
        """
        metric_key = metric.lower()
        is_champion = False
        
        # Determine artifact filename
        filename = f"{metric_key}_{family}_{model_name}.joblib"
        artifact_path = self.registry_dir / filename
        
        # Save candidate artifact
        joblib.dump(model_artifact, artifact_path)
        
        record = {
            "metric": metric_key,
            "family": family,
            "model_name": model_name,
            "artifact_file": filename,
            "val_metrics": val_metrics,
            "baseline_metrics": baseline_metrics,
            "feature_names": feature_names,
            "registered_at": pd.Timestamp.now().isoformat()
        }
        
        current_champ = self.metadata["champions"].get(metric_key)
        
        if current_champ is None:
            is_champion = True
            logger.info(f"[{metric_key}] Initial champion registered: {model_name} (RMSE: {val_metrics['rmse']})")
        else:
            prev_rmse = current_champ["val_metrics"]["rmse"]
            new_rmse = val_metrics["rmse"]
            # Outperforms previous champion if RMSE is lower
            if new_rmse < prev_rmse:
                is_champion = True
                logger.info(f"[{metric_key}] New champion dethroned old champion! Old RMSE: {prev_rmse} -> New RMSE: {new_rmse} ({model_name})")
            else:
                logger.info(f"[{metric_key}] Candidate {model_name} (RMSE: {new_rmse}) did not beat champion {current_champ['model_name']} (RMSE: {prev_rmse}).")
                
        if is_champion:
            record["is_champion"] = True
            self.metadata["champions"][metric_key] = record
            # Optional cloud sync to Hopsworks Model Registry when API key is configured
            try:
                from src.pipeline.hopsworks_store import HopsworksFeatureStore
                hw = HopsworksFeatureStore()
                if hw.is_configured():
                    hw.register_model(metric_key, model_name, str(artifact_path), val_metrics)
            except Exception as e:
                logger.warning(f"Hopsworks Model Registry sync skipped: {e}")
            
        self.metadata["history"].append(record)
        self._save_metadata()
        return is_champion

    def get_champion(self, metric: str) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """Loads and returns the champion model artifact and its metadata for a given metric."""
        metric_key = metric.lower()
        champ_meta = self.metadata["champions"].get(metric_key)
        if not champ_meta:
            return None
            
        artifact_path = self.registry_dir / champ_meta["artifact_file"]
        if not artifact_path.exists():
            logger.warning(f"Champion artifact file not found: {artifact_path}")
            return None
            
        model = joblib.load(artifact_path)
        return model, champ_meta

    def get_leaderboard(self) -> pd.DataFrame:
        """Constructs a leaderboard table of all evaluated models across metrics."""
        rows = []
        for item in self.metadata.get("history", []):
            metric = item["metric"]
            champ_info = self.metadata["champions"].get(metric, {})
            is_champ = (champ_info.get("model_name") == item["model_name"] and champ_info.get("family") == item["family"])
            
            rows.append({
                "Metric": metric,
                "Family": item["family"],
                "Model": item["model_name"],
                "Validation RMSE": item["val_metrics"]["rmse"],
                "Validation MAE": item["val_metrics"]["mae"],
                "Validation R²": item["val_metrics"]["r2"],
                "Baseline RMSE": item["baseline_metrics"]["rmse"],
                "Baseline MAE": item["baseline_metrics"]["mae"],
                "Beat Baseline?": "YES" if item["val_metrics"]["rmse"] < item["baseline_metrics"]["rmse"] else "NO",
                "Champion": "YES" if is_champ else "NO",
                "Evaluated At": item["registered_at"][:19].replace("T", " ")
            })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(["Metric", "Validation RMSE"], ascending=[True, True])

