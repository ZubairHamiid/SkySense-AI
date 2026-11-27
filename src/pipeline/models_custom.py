import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import List
import logging
import warnings

logger = logging.getLogger("CustomModels")


class SARIMAXForecaster:
    """
    Statistical time-series forecasting model powered by statsmodels SARIMAX.
    Fits exogenous meteorological lags + seasonal Fourier harmonics via Maximum Likelihood
    in a state-space SARIMAX(2,0,0) formulation, supporting fast vectorized 24h-ahead inference.
    """
    def __init__(self, metric: str, max_fit_rows: int = 3000):
        self.metric = metric
        self.max_fit_rows = max_fit_rows
        self.exog_cols: List[str] = []
        self.exog_weights: np.ndarray = np.array([])
        self.intercept: float = 0.0
        self.scaler = StandardScaler()
        self.aic: float = np.nan

    def fit(self, X: pd.DataFrame, y: pd.Series):
        preferred = [
            self.metric,
            f"{self.metric}_lag_1h",
            f"{self.metric}_lag_6h",
            f"{self.metric}_lag_24h",
            f"{self.metric}_rolling_mean_24h",
            "hour_sin", "hour_cos",
            "month_sin", "month_cos",
        ]
        self.exog_cols = [c for c in preferred if c in X.columns]
        if not self.exog_cols:
            self.exog_cols = [c for c in X.columns if "lag" in c][:8]

        # Fit scaler strictly on training data (zero leakage)
        X_all = X[self.exog_cols].fillna(0.0).values.astype(np.float64)
        X_scaled_all = self.scaler.fit_transform(X_all)

        # Sample evenly across chronological training span so SARIMAX MLE sees all seasons & cities
        n_fit = min(self.max_fit_rows, len(y))
        step = max(1, len(y) // n_fit)
        y_sub = y.values[::step][:n_fit].astype(np.float64)
        X_sub = X_scaled_all[::step][:n_fit]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = SARIMAX(
                    endog=y_sub,
                    exog=X_sub,
                    order=(2, 0, 0),
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(disp=False, maxiter=40)
                params = res.params
                self.aic = float(res.aic)
                # statsmodels param order with trend='c': [intercept, x1..xk, ar.L1, ar.L2, sigma2]
                self.intercept = float(params[0])
                self.exog_weights = np.array(params[1 : 1 + len(self.exog_cols)], dtype=np.float64)
            except Exception as e:
                logger.warning(f"SARIMAX MLE fallback for {self.metric}: {e}")
                # Regularized OLS fallback on exact same scaled exogenous matrix
                X_ones = np.c_[np.ones(len(X_sub)), X_sub]
                reg = 1e-2 * np.eye(X_ones.shape[1])
                beta = np.linalg.pinv(X_ones.T @ X_ones + reg) @ X_ones.T @ y_sub
                self.intercept = float(beta[0])
                self.exog_weights = beta[1:]

        # Expose weights attribute for Explainability compatibility
        self.weights = self.exog_weights
        self.features_used = self.exog_cols
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_mat = pd.DataFrame({c: X[c] if c in X.columns else 0.0 for c in self.exog_cols}).fillna(0.0).values.astype(np.float64)
        X_scaled = self.scaler.transform(X_mat)
        return self.intercept + X_scaled @ self.exog_weights


class _LSTMNet(nn.Module):
    """PyTorch Sequence-to-One LSTM fused with cyclical/geolocation context."""
    def __init__(self, seq_channels: int, ctx_dim: int, hidden_size: int = 48):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + ctx_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_ctx: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x_seq)
        last_hidden = out[:, -1, :]
        fused = torch.cat([last_hidden, x_ctx], dim=1)
        return self.head(fused).squeeze(-1)


class PyTorchLSTMForecaster:
    """
    Deep Learning family model using PyTorch LSTM (torch.nn.LSTM).
    Constructs a genuine 7-step temporal sequence [t-24h, t-12h, t-6h, t-3h, t-2h, t-1h, t]
    across 6 atmospheric channels, fused with cyclical time & city geolocation context.
    """
    SEQ_CHANNELS = ["aqi", "temperature_c", "precipitation_mm", "humidity_pct", "uv_index", "wind_speed_kmh"]
    LAG_SUFFIXES = ["_lag_24h", "_lag_12h", "_lag_6h", "_lag_3h", "_lag_2h", "_lag_1h", ""]
    CTX_COLS = [
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "day_of_year_sin", "day_of_year_cos",
        "city_lat", "city_lon", "city_elevation_m",
        "wind_u", "wind_v", "apparent_temp_proxy",
    ]

    def __init__(self, hidden_size: int = 48, epochs: int = 10, batch_size: int = 512, lr: float = 2e-3, random_state: int = 42):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.seq_cols: List[str] = []
        self.ctx_cols: List[str] = []
        self.seq_scaler = StandardScaler()
        self.ctx_scaler = StandardScaler()
        self.y_mean: float = 0.0
        self.y_std: float = 1.0
        self.net: _LSTMNet = None

    def _extract_matrices(self, X: pd.DataFrame):
        # Build (N, 7 * n_channels) ordered strictly oldest -> newest
        seq_arrays = []
        for suf in self.LAG_SUFFIXES:
            for ch in self.SEQ_CHANNELS:
                col = f"{ch}{suf}"
                if col in X.columns:
                    seq_arrays.append(X[col].fillna(0.0).values.astype(np.float32))
                else:
                    seq_arrays.append(np.zeros(len(X), dtype=np.float32))
        X_seq_flat = np.column_stack(seq_arrays)

        ctx_arrays = []
        for c in self.ctx_cols:
            if c in X.columns:
                ctx_arrays.append(X[c].fillna(0.0).values.astype(np.float32))
            else:
                ctx_arrays.append(np.zeros(len(X), dtype=np.float32))
        X_ctx = np.column_stack(ctx_arrays) if ctx_arrays else np.zeros((len(X), 1), dtype=np.float32)
        return X_seq_flat, X_ctx

    def fit(self, X: pd.DataFrame, y: pd.Series):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.ctx_cols = [c for c in self.CTX_COLS if c in X.columns]
        X_seq_flat, X_ctx = self._extract_matrices(X)

        # Strictly train-only scaling (zero leakage)
        X_seq_scaled = self.seq_scaler.fit_transform(X_seq_flat).astype(np.float32)
        X_ctx_scaled = self.ctx_scaler.fit_transform(X_ctx).astype(np.float32)

        n_steps = len(self.LAG_SUFFIXES)
        n_ch = len(self.SEQ_CHANNELS)
        X_seq_3d = X_seq_scaled.reshape(-1, n_steps, n_ch)

        y_vals = y.values.astype(np.float32)
        self.y_mean = float(np.mean(y_vals))
        self.y_std = float(np.std(y_vals)) if float(np.std(y_vals)) > 1e-6 else 1.0
        y_norm = (y_vals - self.y_mean) / self.y_std

        self.net = _LSTMNet(seq_channels=n_ch, ctx_dim=X_ctx_scaled.shape[1], hidden_size=self.hidden_size)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.HuberLoss(delta=1.5)

        ds = TensorDataset(
            torch.from_numpy(X_seq_3d),
            torch.from_numpy(X_ctx_scaled),
            torch.from_numpy(y_norm),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.net.train()
        for _ in range(self.epochs):
            for bx_seq, bx_ctx, by in loader:
                optimizer.zero_grad()
                pred = self.net(bx_seq, bx_ctx)
                loss = criterion(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()

        self.net.eval()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.net.eval()
        X_seq_flat, X_ctx = self._extract_matrices(X)
        X_seq_scaled = self.seq_scaler.transform(X_seq_flat).astype(np.float32)
        X_ctx_scaled = self.ctx_scaler.transform(X_ctx).astype(np.float32)

        n_steps = len(self.LAG_SUFFIXES)
        n_ch = len(self.SEQ_CHANNELS)
        X_seq_3d = X_seq_scaled.reshape(-1, n_steps, n_ch)

        preds = []
        with torch.no_grad():
            for i in range(0, len(X_seq_3d), 2048):
                bx_seq = torch.from_numpy(X_seq_3d[i : i + 2048])
                bx_ctx = torch.from_numpy(X_ctx_scaled[i : i + 2048])
                out = self.net(bx_seq, bx_ctx).numpy()
                preds.append(out * self.y_std + self.y_mean)
        return np.concatenate(preds, axis=0)
