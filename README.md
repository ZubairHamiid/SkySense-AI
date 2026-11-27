# ⛅ SkySense AI: Agentic Multi-Metric Weather & AQI Forecasting System
### 10Pearls SHINE Internship — Cohort 10 (2026)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch LSTM](https://img.shields.io/badge/Deep%20Learning-PyTorch%20LSTM-ee4c2c.svg)](https://pytorch.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Zero Leakage](https://img.shields.io/badge/Validation-Zero--Leakage%2024h%20Embargo-10b981.svg)](REPORT.md)

---

## 🌟 Project Overview

**SkySense AI** is an end-to-end automated weather and air-quality forecasting platform powered by a 4-node **LangGraph** agentic reasoning layer. Trained on **140,520 hourly observations** (2 full years across 8 major Pakistani cities: **Lahore, Karachi, Islamabad, Peshawar, Quetta, Multan, Faisalabad, Sargodha**), it forecasts six critical metrics 24 hours ahead:

1. **Air Quality Index (AQI)** — US EPA Standard derived from PM2.5 and PM10
2. **Temperature (°C)**
3. **Precipitation (mm & rain probability %)**
4. **Relative Humidity (%)**
5. **Solar UV Index**
6. **Wind (speed km/h & compass direction degrees)**

> 📄 **Full Technical Report**: See [REPORT.md](REPORT.md) for data fusion, zero-leakage causal imputation, embargoed validation, explainability, drift detection, and failure audit details.

---

## 🏛️ System Architecture

```mermaid
graph TD
    A[Open-Meteo + OpenWeather + AQICN APIs] -->|Causal Imputation| B[Per-City Feature Engineering<br>152 Features + 24h Target Shift]
    B -->|Features & Targets| C[(Feature Store<br>Local SQLite/Parquet + Hopsworks)]
    C -->|Global 24h Embargo Split| D[Multi-Family Training<br>SARIMAX · Ridge · RF · XGBoost · PyTorch LSTM]
    D -->|Champions| E[(Model Registry<br>Local + Hopsworks)]
    
    subgraph LangGraph Agentic Reasoning Layer
        F[Forecast Agent] -->|24h Predictions| G[Alert Agent<br>Public / Sensitive / Workers]
        G -->|Hazards| H[Weather Report Agent<br>News Broadcast Script]
        H -->|Report| I[Drift Agent<br>KS + PSI + MAE]
        I -->|Retrain / GitHub Dispatch| D
    end
    
    E --> F
    C --> F
    F -->|Forecast Ledger| C
    G -->|Agent Call & Failure Log| C
    I -->|Drift Event Log| C
    
    LangGraph Agentic Reasoning Layer --> J[Responsive Streamlit Dashboard<br>Alerts · News · Leaderboard · SHAP & LIME · EDA · Drift]
```

---

## 🏆 Zero-Leakage 8-City Benchmark Leaderboard (140,520 Rows)

Evaluated using a **strict global chronological split (80/20)** with a **24-hour embargo/purge gap** (`Train: 112,064 rows`, `Val: 27,888 rows`) and strictly causal imputation. **All 6/6 champions beat the 24-hour naive persistence baseline:**

| Metric | Champion Model | Family | Val RMSE | Val MAE | Val $R^2$ | Naive Baseline RMSE | Beat Baseline? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **AQI** | `XGBoost` | Classical ML | **18.81** | **14.10** | **0.7975** | 22.67 | ✅ **-17.0%** |
| **Temperature (°C)** | `PyTorch_LSTM` | Deep Learning | **1.42** | **1.02** | **0.9432** | 1.58 | ✅ **-10.3%** |
| **Precipitation (mm)** | `Ridge_Regression` | Classical ML | **0.51** | **0.14** | **0.0365** | 0.71 | ✅ **-27.9%** |
| **Humidity (%)** | `XGBoost` | Classical ML | **8.72** | **6.09** | **0.8294** | 9.82 | ✅ **-11.2%** |
| **UV Index** | `XGBoost` | Classical ML | **0.46** | **0.19** | **0.9627** | 0.52 | ✅ **-11.8%** |
| **Wind Speed (km/h)** | `PyTorch_LSTM` | Deep Learning | **2.84** | **2.19** | **0.5201** | 3.55 | ✅ **-19.9%** |

*(Statistical `statsmodels` SARIMAX MLE also benchmarked cleanly across all 6 metrics alongside Ridge, Random Forest, XGBoost, and PyTorch LSTM).*

---

## 🤖 The Four LangGraph Agents

| Agent | Responsibility | Fallback & Logging |
| :--- | :--- | :--- |
| **Forecast Agent** | Loads latest city features & champion models; generates 24h forecasts & hourly trajectory; logs every prediction to `forecast_ledger`. | 24h persistence fallback |
| **Alert Agent** | Evaluates fixed hazard rules across all 6 metrics; writes advisories for **General Public**, **Sensitive Groups**, and **Outdoor Workers**. | Deterministic 3-audience safety template; logged in `agent_call_log` |
| **Weather Report Agent** | Synthesizes forecasts and alerts into a TV/radio news-anchor weather broadcast script. | Broadcast script fallback; logged in `agent_call_log` |
| **Drift Agent** | Checks **Data Drift** (KS-test + PSI) & **Model Drift** (recent MAE vs val MAE); dispatches GitHub Actions `drift_alert_retrain` or retrains in CI/CLI mode. | Logged in `drift_log` |

---

## 🖥️ Dashboard Panels ([app.py](app.py))

1. **Cinematic Multi-Screen Weather Hero**: Live 24h forecast for all 6 metrics (AQI, Temp, Rain mm + Probability %, Humidity, UV, Wind speed + compass direction), glowing interactive spline wave, 12-hour forecast pills, and live major-city comparison cards.
2. **🤖 Agent Activity & Live Alerts**: Active hazard expanders with full 3-audience advisories, live **Forecast Ledger**, and dispatched champion model matrix.
3. **🎙️ Studio News Anchor Script**: Full broadcast script from `WeatherReportAgent`.
4. **🧠 Model Registry & Leaderboard**: Interactive Plotly RMSE comparison + full 30-model evaluation matrix.
5. **🔍 Explainability (SHAP & LIME)**: Global Feature Importance + Local SHAP Waterfall + Local LIME rule attributions.
6. **📊 Exploratory Data Analysis (EDA)**: Pearson correlation heatmap, 24h diurnal cycles, summary statistics & skewness, and IQR outlier audit.
7. **🛡️ Drift Audits & Failure Logs**: Per-metric KS-stat/PSI & MAE drift tables, `drift_log`, `agent_call_log`, and dedicated **Failed Agent Calls Audit Log**.

---

## 🚀 Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Configure API keys in .env (works zero-config via Open-Meteo & fallbacks)
cp .env.example .env

# 3. Run historical backfill, zero-leakage training, or hourly agent cycle
python run_pipeline.py --mode backfill --days 730 --city Lahore
python -m src.pipeline.train
python run_pipeline.py --mode agents --city Lahore

# 4. Launch Dashboard
streamlit run app.py
```

---

## ⚙️ Automated CI/CD (GitHub Actions)

- **[.github/workflows/feature_ingestion.yml](.github/workflows/feature_ingestion.yml)**: Runs hourly (`cron: '0 * * * *'`) to ingest live observations, engineer features, execute all 4 LangGraph agents, and fire `drift_alert_retrain` if drift breaches threshold.
- **[.github/workflows/training_pipeline.yml](.github/workflows/training_pipeline.yml)**: Runs daily (`cron: '0 0 * * *'`) and on `repository_dispatch: [drift_alert_retrain]` to retrain SARIMAX, Classical ML, and PyTorch LSTM models and update the Model Registry.

---

## 🌐 Live Deployed Dashboard

- **Public App URL:** https://skysense-ai.streamlit.app
- **GitHub Repository:** https://github.com/ZubairHamiid/SkySense-AI

### Redeploy / Self-Host on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
2. Click **New app** → Select `ZubairHamiid/SkySense-AI` → Branch `main` → Main file path `app.py`.
3. (Optional) Under **Advanced settings → Secrets**, add `GROQ_API_KEY`, `HOPSWORKS_API_KEY`, `OPENWEATHER_API_KEY`, `AQICN_API_KEY`.
4. Click **Deploy**.
