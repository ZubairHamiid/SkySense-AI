# SkySense AI — Agentic Multi-Metric Weather & AQI Forecasting System
### 10Pearls SHINE Internship — Cohort 10 (2026)

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-skysense--ai.streamlit.app-0ea5e9.svg)](https://skysense-ai.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch LSTM](https://img.shields.io/badge/Deep%20Learning-PyTorch%20LSTM-ee4c2c.svg)](https://pytorch.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-f97316.svg)](https://github.com/langchain-ai/langgraph)
[![Zero Leakage](https://img.shields.io/badge/Validation-Zero--Leakage%2024h%20Embargo-10b981.svg)](REPORT.md)

- **Live Public Dashboard:** https://skysense-ai.streamlit.app/
- **Full Technical Report:** [REPORT.md](REPORT.md)
- **Cloud Feature Store & Model Registry:** Hopsworks Project `skysense` (all 6 champion models registered)

---

## Project Overview

**SkySense AI** is an end-to-end automated weather and air-quality forecasting platform powered by a four-node **LangGraph** agentic reasoning layer. Trained on **140,520 hourly observations** (2 full years across 8 major Pakistani cities: **Lahore, Karachi, Islamabad, Peshawar, Quetta, Multan, Faisalabad, and Sargodha**), it automatically fetches fresh live telemetry every hour and forecasts six critical metrics 24 hours ahead:

1. **Air Quality Index (AQI)** — US EPA Standard derived from PM2.5 and PM10
2. **Temperature (°C)**
3. **Precipitation (mm and rain probability %)**
4. **Relative Humidity (%)**
5. **Solar UV Index**
6. **Wind (speed km/h and compass direction degrees)**

---

## System Architecture

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
    
    LangGraph Agentic Reasoning Layer --> J[Responsive Streamlit Dashboard<br>Alerts · Broadcast · Leaderboard · SHAP & LIME · EDA · Drift]
```

---

## Zero-Leakage 8-City Benchmark Leaderboard (140,520 Rows)

Evaluated using a **strict global chronological split (80/20)** with a **24-hour embargo/purge gap** (`Train: 112,064 rows`, `Val: 27,888 rows`), strictly causal imputation (`ffill` + past-only expanding median), and train-only feature scalers/medians. **All 6/6 champions beat the 24-hour naive persistence baseline:**

| Metric | Champion Model | Family | Val RMSE | Val MAE | Val R² | Naive Baseline RMSE | Improvement |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **AQI** | `XGBoost` | Classical ML | **18.81** | **14.10** | **0.7975** | 22.67 | **-17.0% RMSE** |
| **Temperature (°C)** | `PyTorch_LSTM` | Deep Learning | **1.42** | **1.02** | **0.9432** | 1.58 | **-10.3% RMSE** |
| **Precipitation (mm)** | `Ridge_Regression` | Classical ML | **0.51** | **0.14** | **0.0365** | 0.71 | **-27.9% RMSE** |
| **Humidity (%)** | `XGBoost` | Classical ML | **8.72** | **6.09** | **0.8294** | 9.82 | **-11.2% RMSE** |
| **UV Index** | `XGBoost` | Classical ML | **0.46** | **0.19** | **0.9627** | 0.52 | **-11.8% RMSE** |
| **Wind Speed (km/h)** | `PyTorch_LSTM` | Deep Learning | **2.84** | **2.19** | **0.5201** | 3.55 | **-19.9% RMSE** |

*(Statistical `statsmodels` SARIMAX MLE is also benchmarked cleanly across all 6 metrics alongside Ridge, Random Forest, XGBoost, and PyTorch LSTM.)*

---

## The Four LangGraph Agents

| Agent | Responsibility | Fallback & Logging |
| :--- | :--- | :--- |
| **Forecast Agent** | Auto-refreshes live hourly telemetry (Open-Meteo + OpenWeather + AQICN) whenever data is >2h old; runs active champion models for 24h forecasts & hourly trajectory; logs every prediction to `forecast_ledger`. | 24h persistence fallback |
| **Alert Agent** | Evaluates fixed hazard thresholds across all 6 metrics; generates tailored advisories via Groq (`openai/gpt-oss-20b`) for **General Public**, **Sensitive Groups**, and **Outdoor Workers**. | Hazard-specific deterministic template; logged in `agent_call_log` |
| **Weather Report Agent** | Synthesizes 24h forecasts and active alerts into a newsroom broadcast weather script via Groq (`openai/gpt-oss-20b`). | Broadcast script fallback; logged in `agent_call_log` |
| **Drift Agent** | Checks **Data Drift** (KS-test + PSI) and **Model Drift** (recent MAE vs validation MAE); dispatches GitHub Actions `drift_alert_retrain` or retrains directly in CI/CLI mode. | Logged in `drift_log` |

---

## Dashboard Panels ([app.py](app.py))

1. **Multi-Screen Atmospheric Hero**: Live 24h forecasts for all 6 metrics (AQI, Temp, Precip mm + probability %, Humidity, UV, Wind speed + compass direction), responsive spline trajectory wave, 12-hour forecast pills, dynamic condition-aware Targeted Advisories, and live multi-city comparison cards.
2. **Agent Activity & Alerts**: Active hazard advisories for all 3 audiences, live **Forecast Ledger**, and dispatched champion models.
3. **Weather Broadcast**: Full news-style weather report from `WeatherReportAgent`.
4. **Model Leaderboard**: Best Validation RMSE grouped bar chart across model families + complete 30-model evaluation table.
5. **Explainability (SHAP & LIME)**: Global Feature Importance (including PyTorch LSTM permutation sensitivity), Local SHAP Attribution, and Local LIME rules.
6. **Exploratory Analysis**: Pearson correlation heatmap, 24h diurnal cycles, summary statistics & skewness, and IQR outlier audit.
7. **Drift & Audit Logs**: Per-metric Data Drift (KS / PSI) and Model Drift (MAE) tables, `drift_log`, `agent_call_log`, and dedicated **Failed Agent Calls Audit Log**.

---

## Quickstart

```bash
# 1. Install dependencies (Python 3.11 recommended)
pip install -r requirements.txt

# 2. Configure API keys in .env (optional — works zero-config via Open-Meteo & fallbacks)
cp .env.example .env

# 3. Run historical backfill, zero-leakage training, or live agent cycle
python run_pipeline.py --mode backfill --days 730 --city Lahore
python -m src.pipeline.train
python run_pipeline.py --mode agents --city Lahore

# 4. Launch Dashboard
streamlit run app.py
```

---

## Automated CI/CD (GitHub Actions)

- **[.github/workflows/feature_ingestion.yml](.github/workflows/feature_ingestion.yml)**: Runs hourly (`cron: '0 * * * *'`) to ingest live observations, engineer features, execute all 4 LangGraph agents, and dispatch `drift_alert_retrain` if drift crosses threshold.
- **[.github/workflows/training_pipeline.yml](.github/workflows/training_pipeline.yml)**: Runs daily (`cron: '0 0 * * *'`) and on `repository_dispatch: [drift_alert_retrain]` to retrain SARIMAX, Classical ML, and PyTorch LSTM models and update the Model Registry.

---

## Deployment (Streamlit Community Cloud)

- **Live App URL:** https://skysense-ai.streamlit.app/
- **GitHub Repository:** https://github.com/ZubairHamiid/SkySense-AI
- **Secrets (App Settings -> Secrets):** `GROQ_API_KEY`, `GROQ_MODEL="openai/gpt-oss-20b"`, `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT="skysense"`, `OPENWEATHER_API_KEY`, `AQICN_API_KEY`.
