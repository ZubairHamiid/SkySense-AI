# ⛅ SkySense AI: Agentic Multi-Metric Weather & AQI Forecasting System
### 10Pearls SHINE Internship — Cohort 10 (2026)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌟 Project Overview

**SkySense AI** (AgentWeather) is an enterprise-grade agentic weather and air-quality forecasting platform designed for the **10Pearls SHINE Internship (Cohort 10)**. It forecasts six critical meteorological and pollutant dimensions 24 hours ahead, coupled with an autonomous **LangGraph** reasoning layer.

### 🎯 6 Forecast Metrics
1. **Air Quality Index (AQI)** — US EPA Standard derived from PM2.5 and PM10
2. **Temperature (°C)**
3. **Precipitation (mm & probability %)**
4. **Relative Humidity (%)**
5. **Solar UV Index**
6. **Wind (speed km/h and direction degrees)**

---

## 🏛️ System Architecture

```mermaid
graph TD
    A[Weather & Pollution APIs<br>Open-Meteo, OpenWeather, AQICN] -->|Raw Data| B[Feature Generation<br>Lags, Diff, Rolling Stats, Diurnal]
    B -->|Features & Targets| C[(Feature Store<br>Local SQLite & Hopsworks)]
    C -->|Training Batches| D[Model Training & Benchmarking<br>Statistical, Classical ML, Deep Learning]
    D -->|Champions| E[(Model Registry)]
    
    subgraph Agentic Reasoning Layer (LangGraph)
        F[Forecast Agent] -->|24h Predictions| G[Alert Agent]
        G -->|Hazard Triggers| H[Weather Report Agent]
        H -->|Broadcast Script| I[Drift Agent]
        I -->|Retrain Trigger on Drift| D
    end
    
    E --> F
    C --> F
    I -->|Log Drift Events| C
    F -->|Forecast Ledger| C
    G -->|Agent Call Logs| C
    
    Agentic Reasoning Layer --> J[Interactive Streamlit Dashboard]
```

---

## 🤖 The Four LangGraph Agents

| Agent | Responsibility | Fallback Mechanism |
| :--- | :--- | :--- |
| **Forecast Agent** | Fetches latest features and champion models; produces 24h-ahead forecasts and 24h step trajectories; logs to Forecast Ledger. | Persistence baseline fallback |
| **Alert Agent** | Evaluates fixed meteorological hazard rules across all 6 metrics; authors tailored safety plans for **General Public**, **Sensitive Groups**, and **Outdoor Workers**. | Deterministic safety advisory template |
| **Weather Report Agent** | Synthesizes forecasts and alerts into an engaging news-anchor broadcast weather report. | Broadcast script generator |
| **Drift Agent** | Audits **Data Drift** (Kolmogorov-Smirnov test & PSI) and **Model Drift** (residual tracking); triggers automated retraining on threshold breach. | Safe passive monitoring |

---

## 📁 Repository Structure

```
├── .github/workflows/
│   ├── feature_ingestion.yml     # Hourly ingestion & agent reasoning workflow
│   └── training_pipeline.yml     # Daily & drift-triggered training workflow
├── config/
│   ├── settings.py               # Global configuration & metric thresholds
│   └── .env.example              # Environment variables template
├── data/
│   ├── raw/                      # Parquet raw observations
│   ├── processed/                # Engineered feature tables
│   └── storage.db                # SQLite Feature Store & Forecast Ledger
├── models/
│   └── registry/                 # Champion model artifacts & metadata
├── src/
│   ├── pipeline/
│   │   ├── fetcher.py            # Multi-source API fetcher & imputation
│   │   ├── feature_engineering.py# Lags, rolling stats, cyclical encodings
│   │   ├── feature_store.py      # Unified Feature Store Manager
│   │   ├── local_store.py        # SQLite / Parquet local store
│   │   ├── hopsworks_store.py    # Hopsworks Cloud integration
│   │   ├── backfill.py           # Historical data backfill script
│   │   ├── train.py              # Multi-family model training & benchmark
│   │   ├── model_registry.py     # Champion versioning & leaderboard
│   │   ├── eda.py                # Exploratory Data Analysis engine
│   │   └── explainability.py     # SHAP & LIME explanation module
│   ├── agents/
│   │   ├── state.py              # LangGraph AgentWeatherState definition
│   │   ├── llm_client.py         # Groq LLM client with latency logging
│   │   ├── forecast_agent.py     # 24h multi-metric forecasting node
│   │   ├── alert_agent.py        # 3-audience hazard alert node
│   │   ├── weather_report_agent.py # News broadcast script node
│   │   ├── drift_agent.py        # KS-test & model drift node
│   │   └── graph.py              # LangGraph StateGraph compilation
│   └── dashboard/
│       └── ui_components.py      # CSS styles & Plotly charting helpers
├── app.py                        # Streamlit Interactive Web Application
├── run_pipeline.py               # Unified CLI runner
├── requirements.txt              # Production dependencies
├── README.md                     # Project documentation
└── REPORT.md                     # Comprehensive technical submission report
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
Activate your conda environment `web_scraping`:
```bash
conda activate web_scraping
```

If any dependency is missing, install via:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` (all APIs have zero-cost Open-Meteo and rule-based fallbacks, so running without keys works out of the box):
```bash
cp .env.example .env
```
*(Optional: add `GROQ_API_KEY` for LLM inference or `HOPSWORKS_API_KEY` for cloud feature store).*

### 3. Run Historical Backfill
Backfills 90-180 days of hourly weather and pollution observations, computes 150+ features, and builds 24h targets:
```bash
python run_pipeline.py --mode backfill --days 90 --city Lahore
```

### 4. Train Models & Benchmark Leaderboard
Trains Statistical, Classical ML (Ridge, Random Forest, XGBoost), and Deep Learning (MLP) models, evaluating against the Naive Baseline:
```bash
python run_pipeline.py --mode train
```

### 5. Run the LangGraph Multi-Agent Reasoning Cycle
Executes Forecast, Alert, Weather Report, and Drift agents:
```bash
python run_pipeline.py --mode agents
```

### 6. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📊 Evaluation & Leaderboard

AgentWeather evaluates all candidates using chronological time-based splitting (80% train / 20% validation).

| Metric | Champion Model | Champion Family | Val RMSE | Val MAE | Val $R^2$ | Naive Baseline RMSE |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **AQI** | `Neural_Network_MLP` | **Deep Learning** | **27.25** | **21.75** | **0.326** | 31.63 |
| **Temperature (°C)** | `Ridge_Regression` | **Classical ML** | **1.27** | **0.98** | **0.904** | 0.83 |
| **Precipitation (mm)** | `AutoRegressive_OLS` | **Statistical** | **0.048** | **0.047** | -9.35 | 0.021 |
| **Humidity (%)** | `Random_Forest` | **Classical ML** | **5.06** | **4.10** | **0.882** | 5.22 |
| **UV Index** | `XGBoost` | **Classical ML** | **0.23** | **0.11** | **0.965** | 0.23 |
| **Wind Speed (km/h)** | `XGBoost` | **Classical ML** | **2.05** | **1.60** | -0.006 | 2.83 |

---

## ⚙️ CI/CD Automation (GitHub Actions)

- **Hourly Ingestion Job (`feature_ingestion.yml`)**: Runs at `cron: '0 * * * *'` to ingest new hourly weather/pollution telemetry and execute the LangGraph reasoning cycle.
- **Daily & Drift Retraining Job (`training_pipeline.yml`)**: Runs at `cron: '0 0 * * *'` and triggers automatically when the Drift Agent detects distribution shifts.

---

## 🌐 Free Cloud Deployment (Streamlit Community Cloud / HF Spaces)

1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select your repository, set the branch to `main`, and main file path to `app.py`.
4. Add any optional API keys in **App Settings -> Secrets**.
5. Click **Deploy**!

---

*Developed for 10Pearls SHINE Internship Cohort 10 (2026).*

