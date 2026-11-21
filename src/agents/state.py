from typing import TypedDict, List, Dict, Any, Optional


class AgentWeatherState(TypedDict):
    """
    Shared state for the LangGraph agentic reasoning layer.
    Persisted in the Forecast Ledger and state stores between execution runs.
    """
    city: str
    run_timestamp: str
    
    # Ingestion & Features
    current_features: Dict[str, Any]
    
    # Forecast Agent outputs
    forecast_24h: Dict[str, float]               # 24h ahead predictions for all 6 metrics
    forecast_hourly_trajectory: List[Dict[str, Any]] # Hourly path (1h to 24h ahead)
    baseline_forecast: Dict[str, float]          # 24h naive persistence baseline values
    champion_models_used: Dict[str, str]         # Model names utilized for each metric
    
    # Alert Agent outputs
    alerts: List[Dict[str, Any]]                 # Triggered alerts (public, sensitive, workers)
    has_active_alerts: bool
    
    # Weather Report Agent outputs
    weather_report: str                          # News-anchor style weather broadcast script
    
    # Drift Agent outputs
    drift_status: Dict[str, Any]                 # Data drift and model drift evaluation metrics
    triggered_retraining: bool                   # Whether automated retraining was dispatched
    
    # Auditing and telemetry
    llm_logs: List[Dict[str, Any]]               # Prompt, response, latency, status
    errors: List[str]

