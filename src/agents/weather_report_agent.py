import logging
from datetime import datetime

from src.agents.state import AgentWeatherState
from src.agents.llm_client import LLMClient
from config.settings import DEFAULT_CITY

logger = logging.getLogger("WeatherReportAgent")


def generate_deterministic_report_fallback(forecast_24h: dict, alerts: list, city: str) -> str:
    """Professional news broadcast script fallback when LLM is unavailable."""
    now_str = datetime.now().strftime("%A, %B %d, %Y")
    
    aqi = forecast_24h.get("aqi", 85.0)
    temp = forecast_24h.get("temperature_c", 26.0)
    rain = forecast_24h.get("precipitation_mm", 0.0)
    hum = forecast_24h.get("humidity_pct", 55.0)
    uv = forecast_24h.get("uv_index", 4.0)
    wind = forecast_24h.get("wind_speed_kmh", 12.0)
    
    aqi_status = "Good" if aqi <= 50 else ("Moderate" if aqi <= 100 else ("Unhealthy for Sensitive Groups" if aqi <= 150 else ("Unhealthy" if aqi <= 200 else "Hazardous")))
    rain_desc = f"expect steady precipitation accumulating around {rain:.1f} mm" if rain > 1.0 else "dry skies are anticipated with negligible rainfall"
    
    alert_section = ""
    if alerts:
        alert_titles = ", ".join([a["title"] for a in alerts])
        alert_section = f"\n\n🚨 CRITICAL ADVISORY: Authorities and residents should note active alerts: {alert_titles}. Please take necessary safety precautions."
    else:
        alert_section = "\n\n🟢 MUNICIPAL NOTICE: No severe weather or air-hazard warnings are currently in effect."
        
    return f"""🎙️ AGENTWEATHER 24-HOUR BROADCAST BULLETIN
City: {city.upper()} | Broadcast Date: {now_str}
========================================================================

"Good day, {city}! This is your 24-hour AgentWeather newsroom briefing powered by our agentic forecasting engine.

Looking at the meteorological radar over the next 24 hours, temperatures will hold around {temp:.1f}°C, accompanied by a relative humidity of {hum:.0f}%. Winds are forecasted to breeze across the metropolitan basin at {wind:.1f} km/h, while daytime peak solar UV exposure will register an index of {uv:.1f}. On the precipitation front, {rain_desc}.

Turning to our environmental index, the Air Quality Index (AQI) is projected at {aqi:.0f}, placing regional atmospheric conditions into the '{aqi_status}' category. Citizens sensitive to airborne particulates should plan accordingly.{alert_section}

Stay safe, stay informed, and keep your AgentWeather dashboard close as our agents continue hourly surveillance of the skies!"
"""


def weather_report_node(state: AgentWeatherState) -> AgentWeatherState:
    """
    LangGraph Node: Weather Report Agent.
    Writes one engaging, television/radio news broadcast style weather report
    synthesizing the 24-hour multi-metric forecasts and any active alerts.
    """
    logger.info("--- [Node: Weather Report Agent] Composing broadcast weather script ---")
    
    city = state.get("city", DEFAULT_CITY)
    forecast_24h = state.get("forecast_24h", {})
    alerts = state.get("alerts", [])
    
    llm = LLMClient()
    
    prompt = f"""
Write an engaging, television/radio newsroom broadcast script for {city}.
Date of Broadcast: {datetime.now().strftime('%A, %B %d, %Y')}

Forecasted Conditions over the Next 24 Hours:
- Air Quality Index (AQI): {forecast_24h.get('aqi')}
- Temperature: {forecast_24h.get('temperature_c')}°C
- Precipitation: {forecast_24h.get('precipitation_mm')} mm
- Relative Humidity: {forecast_24h.get('humidity_pct')}%
- Peak Solar UV Index: {forecast_24h.get('uv_index')}
- Wind Speed: {forecast_24h.get('wind_speed_kmh')} km/h

Active Hazards / Safety Alerts:
{chr(10).join(['- ' + a['title'] for a in alerts]) if alerts else 'None. All conditions nominal.'}

Broadcast Guidelines:
- Open with a vibrant news anchor greeting and signature headline for {city}.
- Clearly communicate the expected changes in temperature, rain, wind, and especially air quality (AQI).
- Emphasize safety takeaways for morning commuters and outdoor workers if any alerts are active.
- Maintain an authoritative, engaging, TV-meteorologist tone.
"""
    
    system_instruction = "You are the Chief Meteorological Broadcaster for AgentWeather News. Deliver lively, articulate, informative weather reports."
    
    report_text, log_meta = llm.generate(
        prompt=prompt,
        system_instruction=system_instruction,
        agent_name="WeatherReportAgent",
        fallback_fn=lambda: generate_deterministic_report_fallback(forecast_24h, alerts, city)
    )
    
    state["llm_logs"].append(log_meta)
    state["weather_report"] = report_text
    logger.info("Weather report composed successfully.")
    return state

