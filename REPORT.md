# SkySense AI: Agentic Multi-Metric Weather & AQI Forecasting System
**10Pearls SHINE Internship — Cohort 10 (2026)**
**Final Technical Report & System Documentation**

---

## 1. Executive Summary

**SkySense AI** is an end-to-end automated weather and air-quality forecasting platform featuring a four-node **LangGraph** agentic reasoning layer. Built across **8 major Pakistani meteorological basins** (Lahore, Karachi, Islamabad, Peshawar, Quetta, Multan, Faisalabad, and Sargodha) with **140,520 hourly observations** (2 full years per city), the system forecasts six critical metrics 24 hours ahead:

1. **Air Quality Index (AQI)** — US EPA Standard derived from PM2.5 & PM10
2. **Temperature (°C)**
3. **Precipitation (mm & rain probability %)**
4. **Relative Humidity (%)**
5. **Solar UV Index (0–16)**
6. **Wind (speed km/h & vector direction degrees/compass)**

Every metric is benchmarked across **three distinct model families** against a **24-hour Naive Persistence Baseline** on a strictly **leakage-free global 24-hour embargoed chronological split**:
- **Statistical**: `statsmodels` **SARIMAX** (State-Space MLE)
- **Classical ML**: **Ridge Regression**, **Random Forest**, and **XGBoost**
- **Deep Learning**: **PyTorch LSTM** (`torch.nn.LSTM` 7-step multi-channel sequence network fused with cyclical & geolocation context)

All **6 out of 6 champion models beat the 24-hour naive persistence baseline** (`PyTorch LSTM` wins Temperature and Wind Speed; `XGBoost` wins AQI, Humidity, and UV Index; `Ridge Regression` wins Precipitation).

---

## 2. Data Sources & Multi-Source Fusion

### 2.1 Primary & Secondary Data Sources
- **Primary Archive & Live Telemetry**: **Open-Meteo** Archive (`archive-api.open-meteo.com`) & Air Quality (`air-quality-api.open-meteo.com`) APIs — providing 2 full years of hourly temperature, humidity, precipitation, wind speed/direction, UV index, PM2.5, PM10, CO, NO2, SO2, and O3.
- **Secondary Weather Provider**: **OpenWeatherMap API** (`OPENWEATHER_API_KEY`) for real-time station temperature, humidity, and wind observations.
- **Secondary Air Quality Provider**: **AQICN World Air Quality Index API** (`AQICN_API_KEY`) for live ground-station AQI measurements.

### 2.2 Fusion & Time-Alignment Strategy
Observations are time-aligned to a unified hourly grid (`Asia/Karachi` / PKT). When secondary API keys are active, live readings are fused via a 50/50 weighted blend at the matching hourly timestamp:
$$\hat{x}_t = 0.5 \cdot x_{t, \text{OpenMeteo}} + 0.5 \cdot x_{t, \text{Secondary}}$$
Each record stores provenance metadata (`open_meteo_archive`, `open_meteo_live`, `fused_openmeteo_openweather`, `fused_aqicn`). Crucially, ingestion defaults to `forecast_hours=0` and filters `time <= now`, preventing API forecast rows from ever contaminating historical ground truth.

---

## 3. Zero-Leakage Causal Imputation Strategy

To eliminate lookahead bias, [fetcher.py](src/pipeline/fetcher.py) enforces a strictly causal imputation pipeline verified via automated future-perturbation invariance tests (`max_diff == 0.0` when future rows are perturbed):

| Gap Type | Imputation Method | Causality Guarantee |
| :--- | :--- | :--- |
| **Short Sensor Gaps ($\le 3$h)** | Causal Forward-Fill (`ffill(limit=3)`) | Uses only past observations $t-1, t-2, t-3$; zero future peeking |
| **Medium / Long Gaps** | Expanding Past-Only Median (`expanding().median().shift(1)`) | Computes running median strictly over timestamps $< t$ |
| **Initial Leading NaNs ($t=0$)** | Domain Physical Defaults | Safe fallback only at series start |
| **Training Fold NaNs** | Train-Only Feature Medians | Computed strictly on `X_train` and applied to `X_val` |
| **Physical Constraints** | Hard Meteorological Clamping | Humidity $\in [0, 100]$, UV $\in [0, 16]$, Wind $\ge 0$, Rain $\ge 0$, AQI $\in [0, 500]$ |

---

## 4. Feature Engineering & Target Construction

### 4.1 Per-City Grouped Feature Engineering (152 Features)
In [feature_engineering.py](src/pipeline/feature_engineering.py), all lag, difference, rolling, and target shift operations execute strictly **per city** (`groupby("city")`) to prevent cross-city spillover:
1. **Geolocation Context**: `city_lat`, `city_lon`, `city_elevation_m` (enables one unified multi-city model to distinguish Quetta at 1,680m from coastal Karachi at 8m).
2. **Time & Cyclical Harmonics**: `hour`, `day`, `month`, `day_of_week`, `is_weekend`, `day_of_year`, plus $\sin/\cos$ encodings for hour, month, and day-of-year.
3. **Backward Lags**: `lag_1h`, `lag_2h`, `lag_3h`, `lag_6h`, `lag_12h`, `lag_24h` across all 7 meteorological & pollutant channels.
4. **Rate-of-Change (First Differences)**: $\Delta_{1h}$, $\Delta_{3h}$, $\Delta_{24h}$.
5. **Causal Rolling Statistics**: `rolling_mean` (3h, 6h, 12h, 24h), `rolling_std` (6h, 24h), `rolling_min_24h`, `rolling_max_24h`.
6. **Physical Proxies**: Wind U/V vector decomposition ($u = -v\sin\theta$, $v = -v\cos\theta$), Apparent Temperature (heat-index proxy), and ordinal AQI severity tier.

### 4.2 Target Shift & Inference Tail
Targets are constructed strictly per city via $y_{t, m} = x_{t + 24, m}$ (`shift(-24)`). The initial 24-hour warmup window (`lag_24h` NaN) is dropped before training, and the newest 24 hours (where `target_*` is NaN) are reserved exclusively for live 24h-ahead forecasting. Redundant duplicate `us_aqi` is excluded from input features.

---

## 5. Zero-Leakage Training, Embargo Split & Benchmark Results

### 5.1 Global 24-Hour Embargoed Split
To prevent both **temporal boundary leakage** and **cross-city synoptic weather leakage**, [train.py](src/pipeline/train.py) enforces a single global calendar cutoff across all 8 cities with a **24-hour purge/embargo gap**:
- **Training Fold (80%)**: `112,064` rows (`2024-11-21 00:00` → `2026-06-30 06:00`)
- **Purged Embargo Gap**: `24 hours` (`2026-06-30 07:00` → `2026-07-01 05:00`) — guarantees $\max(t_{\text{train}} + 24\text{h}) \le \min(t_{\text{val}})$
- **Validation Fold (20%)**: `27,888` rows (`2026-07-01 06:00` → `2026-11-23 20:00`)

### 5.2 Model Families Evaluated (30 Models Total)
1. **Naive Baseline**: 24-hour Persistence ($y_{t+24} = y_t$)
2. **Statistical**: `statsmodels` **SARIMAX** (`order=(2,0,0)`, exogenous lags + Fourier harmonics, MLE)
3. **Classical ML**: **Ridge Regression** (`Pipeline[StandardScaler, Ridge]`), **Random Forest** (`n_estimators=80, max_features="sqrt"`), **XGBoost** (`n_estimators=150, max_depth=5`)
4. **Deep Learning**: **PyTorch LSTM** (`torch.nn.LSTM` over 7-step × 6-channel chronological tensors `[t-24h, t-12h, t-6h, t-3h, t-2h, t-1h, t]` fused with cyclical/geolocation embeddings, trained with HuberLoss & Adam)

### 5.3 Verified Zero-Leakage Champion Leaderboard

| Metric | Champion Model | Family | Val RMSE | Val MAE | Val $R^2$ | Naive Baseline RMSE | Baseline $R^2$ | Improvement vs Baseline |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AQI** | `XGBoost` | Classical ML | **18.8051** | **14.0996** | **0.7975** | 22.6658 | 0.7058 | **-17.0% RMSE ✅** |
| **Temperature (°C)** | `PyTorch_LSTM` | Deep Learning | **1.4161** | **1.0214** | **0.9432** | 1.5791 | 0.9293 | **-10.3% RMSE ✅** |
| **Precipitation (mm)** | `Ridge_Regression` | Classical ML | **0.5125** | **0.1403** | **0.0365** | 0.7111 | -0.8546 | **-27.9% RMSE ✅** |
| **Humidity (%)** | `XGBoost` | Classical ML | **8.7198** | **6.0862** | **0.8294** | 9.8183 | 0.7837 | **-11.2% RMSE ✅** |
| **UV Index** | `XGBoost` | Classical ML | **0.4625** | **0.1906** | **0.9627** | 0.5245 | 0.9520 | **-11.8% RMSE ✅** |
| **Wind Speed (km/h)** | `PyTorch_LSTM` | Deep Learning | **2.8416** | **2.1855** | **0.5201** | 3.5486 | 0.2515 | **-19.9% RMSE ✅** |

**All 6/6 champion models strictly outperform the 24-hour naive baseline**, with both **Deep Learning (`PyTorch_LSTM`)** and **Classical ML (`XGBoost`, `Ridge`)** winning championships, and **`statsmodels` SARIMAX** converging cleanly via MLE across all metrics (e.g., `UV R² = 0.9560`, `Humidity R² = 0.7549`, `Wind R² = 0.4359`, `AQI R² = 0.3627`).

---

## 6. Model Explainability (SHAP & LIME)

Implemented in [explainability.py](src/pipeline/explainability.py) and surfaced interactively in the dashboard:
- **Global Feature Importance**: Native tree/linear importances for `XGBoost`/`Ridge`/`SARIMAX`, and input-permutation sensitivity across sequence & context channels for `PyTorch_LSTM`.
- **Local SHAP Attribution**: `TreeExplainer` for XGBoost champions and `PermutationExplainer` for PyTorch LSTM / Ridge champions — showing exact feature pushes above/below expected baseline.
- **Local LIME Attribution**: `LimeTabularExplainer` generating human-readable local threshold rules (e.g., `aqi > 177.00 -> +29.6`, `pm2_5 > 86.10 -> +23.2` for Lahore smog; `apparent_temp_proxy <= 19.18 -> -5.7°C` for winter cooling).

---

## 7. LangGraph Agentic Reasoning Layer & Performance

Orchestrated as a single four-node `StateGraph` in [graph.py](src/agents/graph.py):
```
[START] ──> [Forecast Agent] ──> [Alert Agent] ──> [Weather Report Agent] ──> [Drift Agent] ──> [END]
```

1. **Forecast Agent** ([forecast_agent.py](src/agents/forecast_agent.py)): Loads latest per-city engineered features and active champions from `ModelRegistry`, produces 24h-ahead predictions + 24h diurnal trajectory, logs every forecast into `forecast_ledger`, and passes state downstream.
2. **Alert Agent** ([alert_agent.py](src/agents/alert_agent.py)): Evaluates fixed rule-based hazard thresholds across all 6 metrics (AQI, Heat/Cold, Heavy Rain/Flash Flood, Extreme UV, Gale Wind, Oppressive Humidity) and generates structured safety advisories tailored for **General Public**, **Sensitive Groups**, and **Outdoor Workers**.
3. **Weather Report Agent** ([weather_report_agent.py](src/agents/weather_report_agent.py)): Synthesizes forecasts and triggered alerts into a broadcast-ready newsroom weather script.
4. **Drift Agent** ([drift_agent.py](src/agents/drift_agent.py)):
   - **Data Drift**: Two-sample Kolmogorov-Smirnov test ($p < 0.05$) + Population Stability Index (PSI) comparing live 48h window vs reference window.
   - **Model Drift**: Compares recent forecast ledger MAE against validation MAE (triggers if error rises $> 25\%$).
   - **Autonomous Retraining**: Logs every drift check to `drift_log`; when breached, fires GitHub Actions `repository_dispatch` (`drift_alert_retrain`) or runs `train_and_evaluate_pipeline()` directly in CLI/CI mode (`allow_retrain=True`) while staying non-blocking in the live web dashboard.

### LLM Client Reliability & Failure Auditing
[llm_client.py](src/agents/llm_client.py) wraps Groq (`llama-3.3-70b-versatile`) with millisecond latency tracking, 3-second timeout guard, and deterministic meteorological template fallbacks. Every call (whether `SUCCESS`, `FALLBACK_USED`, or `FAILED_API_ERROR`) is persisted in `agent_call_log`, and failed calls are surfaced in the dashboard's **Failed Agent Calls Audit Log**.

---

## 8. Issues Encountered, Audits & Engineering Fixes

| Issue / Root Cause | Impact | Engineering Fix & Verification |
| :--- | :--- | :--- |
| **1. Imputation Leakage** (`interpolate(limit_direction="both")` + full-dataset medians) | Future rows leaked into past imputed values before train/val split | Replaced with strictly causal `ffill(limit=3)` + `expanding().median().shift(1)` + train-only medians. Verified `0.0` diff under future perturbation. |
| **2. Train/Val 24h Boundary Overlap & Cross-City Synoptic Leakage** | Last 24h train targets overlapped val inputs; different city start dates overlapped calendar days | Implemented global calendar cutoff + strict 24h embargo gap ($\max(t_{\text{train}}+24\text{h}) \le \min(t_{\text{val}})$) with hard assertion. |
| **3. Forecast-as-Ground-Truth Contamination** | `fetch_live_recent(forecast_hours=24)` saved Open-Meteo predictions into `raw_observations` | Set `forecast_hours=0`, tagged `time > now` as `open_meteo_forecast`, excluded from training & purged old forecast tails. |
| **4. Stale Registry Comparisons** | Old 21-Nov small-split RMSE blocked newer 2-year champions & accumulated 245 duplicate rows | Added `registry.reset_registry()` for clean full benchmarks + Hopsworks Model Registry cloud upload support. |
| **5. SARIMAX Constant Column Collision** | Static `city_elevation_m` collided with `trend='c'` in single-city slice | Removed static column from SARIMAX exog & sampled across full chronological span — MLE converged cleanly on all 6 metrics. |
| **6. SHAP Sampling Crash & Dashboard Latency** | `dropna().sample(50)` raised `ValueError`; EDA/Drift scanned 140K SQLite rows per render; indented HTML rendered as code blocks | Safe `fillna(0.0)` sampling, parquet direct reads (`0.03s`), `limit=500` drift queries, flat HTML strings, and `@st.cache_data` on SHAP/LIME. |

---

## 9. Automated CI/CD & Cloud Deployment

- **Hourly Ingestion & Agents**: [.github/workflows/feature_ingestion.yml](.github/workflows/feature_ingestion.yml) (`cron: '0 * * * *'`, passes `GITHUB_TOKEN` for drift dispatch).
- **Daily & Drift-Triggered Training**: [.github/workflows/training_pipeline.yml](.github/workflows/training_pipeline.yml) (`cron: '0 0 * * *'` + `repository_dispatch: [drift_alert_retrain]`).
- **Dual Feature Store & Model Registry**: Local SQLite/Parquet zero-config engine + **Hopsworks Cloud Feature Store & Model Registry** ([hopsworks_store.py](src/pipeline/hopsworks_store.py)) activated automatically when `HOPSWORKS_API_KEY` is provided.
- **Dashboard Deployment (Streamlit Community Cloud)**:
  - Repository: `https://github.com/ZubairHamiid/SkySense-AI`
  - Entry point: `app.py`
  - **Live Public Dashboard URL**: https://skysense-ai.streamlit.app
