from datetime import datetime
import logging
from langgraph.graph import StateGraph, START, END

from src.agents.state import AgentWeatherState
from src.agents.forecast_agent import forecast_node
from src.agents.alert_agent import alert_node
from src.agents.weather_report_agent import weather_report_node
from src.agents.drift_agent import drift_node
from config.settings import DEFAULT_CITY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AgentWeatherGraph")


def build_weather_agent_graph():
    """
    Constructs the 4-agent LangGraph workflow:
    [START]
       │
       ▼
    [forecast_agent]
       │
       ▼
    [alert_agent]
       │
       ▼
    [weather_report_agent]
       │
       ▼
    [drift_agent]
       │
       ▼
     [END]
    """
    workflow = StateGraph(AgentWeatherState)
    
    # Register the 4 agent nodes
    workflow.add_node("forecast_agent", forecast_node)
    workflow.add_node("alert_agent", alert_node)
    workflow.add_node("weather_report_agent", weather_report_node)
    workflow.add_node("drift_agent", drift_node)
    
    # Define execution edges
    workflow.add_edge(START, "forecast_agent")
    workflow.add_edge("forecast_agent", "alert_agent")
    workflow.add_edge("alert_agent", "weather_report_agent")
    workflow.add_edge("weather_report_agent", "drift_agent")
    workflow.add_edge("drift_agent", END)
    
    return workflow.compile()


def run_agent_pipeline(city: str = DEFAULT_CITY) -> AgentWeatherState:
    """
    Executes the compiled LangGraph agent graph with an initialized state.
    """
    logger.info(f"=== Initiating LangGraph Multi-Agent Execution for {city} ===")
    app = build_weather_agent_graph()
    
    initial_state: AgentWeatherState = {
        "city": city,
        "run_timestamp": datetime.now().isoformat(),
        "current_features": {},
        "forecast_24h": {},
        "forecast_hourly_trajectory": [],
        "baseline_forecast": {},
        "champion_models_used": {},
        "alerts": [],
        "has_active_alerts": False,
        "weather_report": "",
        "drift_status": {},
        "triggered_retraining": False,
        "llm_logs": [],
        "errors": []
    }
    
    final_state = app.invoke(initial_state)
    logger.info(f"=== LangGraph Execution Completed Successfully! ===")
    return final_state


if __name__ == "__main__":
    state = run_agent_pipeline()
    print("\n--- FORECAST SUMMARY ---")
    print(state["forecast_24h"])
    print("\n--- ALERTS TRIGGERED ---")
    print(len(state["alerts"]))
    print("\n--- WEATHER REPORT ---")
    print(state["weather_report"])

