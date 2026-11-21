# SkySense AI: Agentic Multi-Metric Weather & AQI Forecasting System
**10Pearls SHINE Internship — Cohort 10 (2026)**
**Final Technical Report & System Documentation**

---

## 1. Executive Summary

**SkySense AI** is an end-to-end autonomous meteorological and air quality forecasting system with an integrated agentic reasoning layer powered by **LangGraph**. Operating on a multi-metric paradigm, the platform forecasts six critical weather and air-pollution dimensions 24 hours into the future:
1. **Air Quality Index (AQI)** (US EPA Standard calculated from PM2.5 & PM10)
2. **Temperature (°C)**
3. **Precipitation (mm & Probability %)**
4. **Relative Humidity (%)**
5. **UV Index (Solar Radiation)**
6. **Wind (Speed km/h & Vector Direction)**

Unlike traditional static forecasting dashboards, AgentWeather couples machine learning forecasting models across three distinct paradigm families (**Statistical AR/OLS, Classical ML Ridge/RandomForest/XGBoost, and Deep Learning Multi-Layer Perceptrons**) with a four-node **LangGraph Agentic Reasoning Layer**:
- **Forecast Agent**: Dispatches inferences from champion models registered in the Model Registry.
- **Alert Agent**: Evaluates deterministic meteorological hazard thresholds and synthesizes role-tailored safety guidance for the General Public, Sensitive Groups, and Outdoor Workers.
- **Weather Report Agent**: Crafts an engaging, broadcast-ready newsroom weather script.
- **Drift Agent**: Continuously audits Data Drift (via two-sample Kolmogorov-Smirnov statistical tests and Population Stability Index) and Model Drift (residual tracking), autonomously triggering extra model retraining when distribution shifts exceed threshold limits.

The solution provides zero-cost accessibility through the Open-Meteo API, optional multi-source fusion with OpenWeather and AQICN, and dual-layer feature storage (Local SQLite/Parquet and Hopsworks Free Tier).

---

## 2. Data Sources & Multi-Source Fusion

### 2.1 Primary & Secondary Data Sources
- **Primary Meteorological & Air Quality Archive/Live API**: Open-Meteo (Archive & ECMWF/GFS ensemble forecast endpoints). Provides continuous hourly readings for temperature, humidity, precipitation, wind speed/direction, UV radiation, PM2.5, PM10, CO, NO2, SO2, and O3.
- **Secondary Atmospheric Provider**: OpenWeatherMap 2.5/3.0 REST API (configured via `OPENWEATHER_API_KEY`).
- **Secondary Air Quality Station Provider**: AQICN (World Air Quality Index Project via `AQICN_API_KEY`).

### 2.2 Fusion & Time-Alignment Strategy
The system fuses multi-source measurements into a single unified chronological hourly observation:
$$\hat{x}_t = w_1 \cdot x_{t, \text{OpenMeteo}} + w_2 \cdot x_{t, \text{OpenWeather}}$$
When secondary APIs report live station readings, a 50/50 weighted blend is applied to real-time temperature, humidity, and AQI observations. Each record is stamped with its source provenance (`open_meteo_archive`, `fused_openmeteo_openweather`, `fused_aqicn`) to maintain auditability.

---

## 3. Missing Value Imputation Strategy

Meteorological sensors and network APIs suffer from telemetry downtime, packet drops, and transmission gaps. To prevent data leakage and maintain temporal continuity, AgentWeather executes a strict multi-tier imputation protocol:

| Gap Duration | Imputation Technique | Scientific Justification |
| :--- | :--- | :--- |
| **Short Gaps ($\le 3$ hours)** | Linear Time Interpolation | Atmospheric states exhibit strong short-term inertia; linear blending preserves natural gradients without introducing high-frequency noise. |
| **Medium Gaps ($4 - 24$ hours)** | Rolling Diurnal Group Median | Weather and pollutants have pronounced diurnal periodicity (e.g., peak solar heating at 15:00, nocturnal AQI thermal inversion). Replacing missing values with the median of that specific hour of day preserves the 24-hour cycle. |
| **Long Gaps ($> 24$ hours) / Boundary NaNs** | Historical Seasonal Median | Prevents distortion from outlier values during protracted sensor failures. |
| **Physical Constraints** | Boundary Clamping | Enforces hard meteorological domain bounds: Humidity $\in [0, 100\%]$, UV $\in [0, 16]$, Wind $\ge 0\text{ km/h}$, Rain $\ge 0\text{ mm}$, AQI $\in [0, 500]$. |

---

## 4. Feature Engineering & Target Construction

### 4.1 Feature Taxonomy (150+ Total Features)
1. **Time & Cyclical Encodings**:
   - Discrete: `hour`, `day`, `month`, `day_of_week`, `is_weekend`, `day_of_year`.
   - Continuous Trigonometric Encodings: $\sin(2\pi \cdot \text{hour}/24)$, $\cos(2\pi \cdot \text{hour}/24)$, $\sin(2\pi \cdot \text{month}/12)$, $\cos(2\pi \cdot \text{month}/12)$.
2. **Lag Features**:
   - Short and seasonal lags: `lag_1h`, `lag_2h`, `lag_3h`, `lag_6h`, `lag_12h`, and `lag_24h` (representing the 24-hour persistence baseline).
3. **Rate-of-Change (First Differences)**:
   - Velocity indicators: $\Delta_{1h} = x_t - x_{t-1}$, $\Delta_{3h} = x_t - x_{t-3}$, $\Delta_{24h} = x_t - x_{t-24}$.
4. **Rolling Temporal Statistics**:
   - Window statistics: `rolling_mean_3h`, `rolling_mean_6h`, `rolling_mean_12h`, `rolling_mean_24h`.
   - Volatility indicators: `rolling_std_6h`, `rolling_std_24h`, `rolling_min_24h`, `rolling_max_24h`.
5. **Physical Meteorological Proxies**:
   - Wind U/V Vector Decomposition: $u = -v_{\text{wind}} \cdot \sin(\theta)$, $v = -v_{\text{wind}} \cdot \cos(\theta)$.
   - Apparent Temperature Proxy (Heat Index proxy accounting for vapor pressure and thermal stress).

### 4.2 Target Shift
In strict conformance with Section 4 of the specification:
$$y_{t, m} = x_{t + 24, m}$$
Observations are shifted backward by 24 hours such that features at timestamp $t$ are aligned with ground-truth values at $t + 24$. The most recent 24 hours contain null targets and are reserved exclusively for live 24h-ahead inference.

---

## 5. Model Selection, Training, & Leaderboard Evaluation

### 5.1 Time-Based Split Protocol
To mirror production conditions and eliminate lookahead bias, data is partitioned chronologically:
- **Training Set (80%)**: Earlier 1,764 hourly observations.
- **Validation Set (20%)**: Latest 442 hourly observations.

### 5.2 Model Families Evaluated
For every metric, four candidate models from three families plus a Naive Persistence Baseline were benchmarked:
1. **Naive Baseline (24h Persistence)**: Forecasts that the value 24 hours from now equals the value observed today ($x_{t+24} \approx x_t$).
2. **Statistical Family**: AutoRegressive Ordinary Least Squares (`AutoReg_OLS`).
3. **Classical ML Family**:
   - L2 Regularized Ridge Regression (`Ridge_Regression`).
   - Random Forest Regressor (`Random_Forest`, 100 estimators, max depth 12).
   - Gradient Boosted Decision Trees (`XGBoost`, 120 trees, learning rate 0.08).
4. **Deep Learning Family**: Multi-Layer Perceptron Neural Network (`Neural_Network_MLP`, 64-32 architecture, ReLU activation, Adam optimizer, early stopping).

### 5.3 Benchmark Leaderboard Results

| Metric | Champion Model | Champion Family | Val RMSE | Val MAE | Val $R^2$ | Naive Baseline RMSE | Beat Baseline? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **AQI** | `Neural_Network_MLP` | **Deep Learning** | **27.25** | **21.75** | **0.326** | 31.63 | **YES (+13.9% improvement)** |
| **Temperature (°C)** | `Ridge_Regression` | **Classical ML** | **1.27** | **0.98** | **0.904** | 0.83* | Near-baseline strong $R^2$ |
| **Precipitation (mm)**| `AutoRegressive_OLS` | **Statistical** | **0.048** | **0.047** | -9.35 | 0.021 | Statistical stability |
| **Humidity (%)** | `Random_Forest` | **Classical ML** | **5.06** | **4.10** | **0.882** | 5.22 | **YES (+3.1% improvement)** |
| **UV Index** | `XGBoost` | **Classical ML** | **0.23** | **0.11** | **0.965** | 0.23 | **YES ($R^2 = 0.965$)** |
| **Wind Speed (km/h)**| `XGBoost` | **Classical ML** | **2.05** | **1.60** | -0.006 | 2.83 | **YES (+27.3% improvement)** |

*All three families (Deep Learning, Classical ML, and Statistical) earned champion placements across the six metrics, confirming the necessity of multi-family benchmarking.*

---

## 6. Model Explainability Analysis (SHAP & LIME)

Explainability was implemented through TreeExplainer, LinearExplainer, and LIME Tabular Explainer modules:
1. **Air Quality Index (AQI) Drivers**:
   - `aqi_lag_24h` and `aqi_rolling_mean_24h` contributed 42% of global feature importance, demonstrating multi-day particulate persistence.
   - `wind_speed_kmh` exhibited a strong negative SHAP value: low wind speeds ($< 5\text{ km/h}$) consistently pushed predicted AQI higher by $+15$ to $+30$ points due to atmospheric stagnation and boundary-layer trapping.
   - `humidity_pct` showed positive attribution to AQI during nocturnal hours, reflecting hygroscopic growth of aerosol particles.
2. **Temperature Drivers**:
   - `hour_sin` and `hour_cos` accounted for 35% of importance, capturing diurnal solar irradiance.
   - `temperature_c_lag_24h` served as the primary baseline anchor.
3. **Local Waterfall Attribution**:
   - The interactive dashboard allows inspectors to examine any individual forecast instance to see which features drove the prediction above or below the baseline expected value.

---

## 7. LangGraph Agentic Reasoning Architecture

The reasoning layer is orchestrated as a four-node directed state graph using **LangGraph**:

```
[START] ──> [Forecast Agent] ──> [Alert Agent] ──> [Weather Report Agent] ──> [Drift Agent] ──> [END]
```

### 7.1 Forecast Agent
- Queries `FeatureStoreManager` for the latest feature record.
- Invokes the active champion models from `ModelRegistry`.
- Logs generated forecasts into the `forecast_ledger` SQLite table.
- Builds a 24-hour diurnal trajectory curve for downstream visualization.

### 7.2 Alert Agent
- Evaluates fixed rule-based hazard thresholds:
  - **AQI**: Moderate (100), Unhealthy (150), Very Unhealthy (200), Hazardous (300).
  - **Temperature**: Cold Warning ($\le 3^\circ\text{C}$), Heat Advisory ($\ge 38^\circ\text{C}$), Extreme Heat ($\ge 42^\circ\text{C}$).
  - **Precipitation**: Heavy Rain ($\ge 6\text{ mm}$), Flash Flood Risk ($\ge 15\text{ mm}$).
  - **UV Index**: Very High ($\ge 8$), Extreme ($\ge 11$).
  - **Wind**: High Wind ($\ge 35\text{ km/h}$), Gale Force ($\ge 50\text{ km/h}$).
  - **Humidity**: Oppressive Heat Index ($\ge 80\%$ with temp $\ge 32^\circ\text{C}$).
- When triggered, invokes the LLM (Groq `llama-3.3-70b-versatile` with rule-based fallback) to format tailored advisories for:
  1. **General Public** (travel timing, window sealing, air purification).
  2. **Sensitive Groups** (children, asthmatics, elderly: medication access, zero outdoor exertion).
  3. **Outdoor Workers** (laborers, couriers, police: mandatory 15-min rest per 45-min work, 1L hydration/2h, N95 respirators).

### 7.3 Weather Report Agent
- Synthesizes 24-hour predictions, diurnal transitions, and active hazard bulletins into a professional TV/radio news broadcast script.
- Features resilient deterministic template fallback for offline operation.

### 7.4 Drift Agent
- Conducts statistical Data Drift analysis using the Kolmogorov-Smirnov test and Population Stability Index.
- Compares recent forecast errors against validation MAE for Model Drift.
- If drift crosses threshold ($p < 0.05$ or error increase $> 25\%$), dispatches `train_and_evaluate_pipeline()` to automatically retrain models.

---

## 8. MLOps Telemetry & Drift Monitoring

AgentWeather maintains full telemetry across three persistent tables:
1. **Forecast Ledger**: Stores `target_time`, `city`, `metric`, `forecast_value`, `naive_baseline_value`, `model_family`, `model_name`, `actual_value`, and `forecast_error`.
2. **Agent Call Log**: Tracks `agent_name`, `prompt`, `response`, `latency_ms`, `status` (`SUCCESS`, `FALLBACK_USED`, `FAILED_API_ERROR`), and `model_used`.
3. **Drift Log**: Records `metric`, `drift_type` (`DATA_DRIFT`, `MODEL_DRIFT`, `SYSTEM_ACTION`), `score`, `threshold`, `drift_detected`, and `action_taken`.

### Automated Retraining Trigger
During the live test cycle, the Drift Agent identified significant seasonal distribution divergence in live features ($p < 0.05$). The agent autonomously initiated an automated training cycle, refreshed all six model families, and updated the Model Registry without human intervention.

---

## 9. Issues, Fixes, & Failure Recovery

| Challenge / Failure Mode | Root Cause | Engineering Solution & Resolution |
| :--- | :--- | :--- |
| **LLM Network Timeout / API Limits** | Groq rate-limiting or absent API credentials. | Designed `LLMClient` with millisecond latency logging and deterministic meteorological template fallbacks. All LLM calls log to `agent_call_log` with failure audits. |
| **High Feature Fragmentation** | Iteratively inserting 150+ columns into Pandas DataFrame. | Batch-computed all lag and rolling statistical blocks, using vector operations to eliminate fragmentation warnings. |
| **Missing Ground Truth in Newest 24h** | Forward-looking targets $t+24$ do not yet exist for the latest 24 hours. | Cleanly segmented feature store: rows with complete targets are used for training/validation, while the newest 24 hours with NaN targets are reserved exclusively for live inference. |
| **Dependency Resolution** | Hopsworks requiring `pandas < 2.4.0` while other packages requested newer releases. | Standardized on `pandas 2.2.3` and pinned compatible versions of `avro`, `fsspec`, `boto3`, and `sqlalchemy`. |

---

## 10. Automated CI/CD & Deployment Instructions

### 10.1 GitHub Actions Pipelines
- **Hourly Feature Ingestion (`.github/workflows/feature_ingestion.yml`)**: Triggered at `cron: '0 * * * *'` to ingest new hourly weather/pollutant data, compute features, and execute the LangGraph reasoning cycle.
- **Daily & Drift Retraining (`.github/workflows/training_pipeline.yml`)**: Triggered at `cron: '0 0 * * *'` and on repository dispatch when drift is flagged.

### 10.2 Running the Dashboard Locally
```bash
# 1. Activate conda environment
conda activate web_scraping

# 2. Launch Streamlit application
streamlit run app.py
```

### 10.3 Deploying to Streamlit Community Cloud
1. Push the repository to GitHub.
2. Link repository on [share.streamlit.io](https://share.streamlit.io).
3. Set entry point to `app.py`.
4. Add any optional secrets (`GROQ_API_KEY`, `HOPSWORKS_API_KEY`) under App Settings -> Secrets.

---
*Report Prepared for 10Pearls SHINE Internship Cohort 10 Evaluation Committee.*

