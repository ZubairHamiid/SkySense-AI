import logging
from typing import Dict, Any, List

from src.agents.state import AgentWeatherState
from src.agents.llm_client import LLMClient
from config.settings import DEFAULT_CITY

logger = logging.getLogger("AlertAgent")


def evaluate_fixed_hazard_rules(forecast_24h: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Fixed meteorological and environmental safety rules checking all 6 forecast metrics.
    """
    hazards = []
    
    # 1. Air Quality Index (AQI) Rules
    aqi = forecast_24h.get("aqi", 0.0)
    if aqi > 300:
        hazards.append({
            "metric": "aqi",
            "level": "HAZARDOUS",
            "value": aqi,
            "unit": "AQI",
            "threshold": 300,
            "title": "Emergency Air Quality Alert: Hazardous Smog/Dust"
        })
    elif aqi > 200:
        hazards.append({
            "metric": "aqi",
            "level": "VERY_UNHEALTHY",
            "value": aqi,
            "unit": "AQI",
            "threshold": 200,
            "title": "Severe Air Pollution Warning: Very Unhealthy"
        })
    elif aqi > 150:
        hazards.append({
            "metric": "aqi",
            "level": "UNHEALTHY",
            "value": aqi,
            "unit": "AQI",
            "threshold": 150,
            "title": "Elevated Air Pollution: Unhealthy for General Public"
        })
    elif aqi > 100:
        hazards.append({
            "metric": "aqi",
            "level": "MODERATE_WARNING",
            "value": aqi,
            "unit": "AQI",
            "threshold": 100,
            "title": "Air Quality Advisory: Unhealthy for Sensitive Groups"
        })
        
    # 2. Temperature Rules
    temp = forecast_24h.get("temperature_c", 25.0)
    if temp >= 42.0:
        hazards.append({
            "metric": "temperature_c",
            "level": "CRITICAL_HEAT",
            "value": temp,
            "unit": "°C",
            "threshold": 42.0,
            "title": "Severe Heatwave Emergency Warning"
        })
    elif temp >= 38.0:
        hazards.append({
            "metric": "temperature_c",
            "level": "HEAT_ADVISORY",
            "value": temp,
            "unit": "°C",
            "threshold": 38.0,
            "title": "High Heatwave Advisory"
        })
    elif temp <= 3.0:
        hazards.append({
            "metric": "temperature_c",
            "level": "COLD_WARNING",
            "value": temp,
            "unit": "°C",
            "threshold": 3.0,
            "title": "Freezing Frost & Cold Wave Warning"
        })
        
    # 3. Precipitation Rules
    rain = forecast_24h.get("precipitation_mm", 0.0)
    if rain >= 15.0:
        hazards.append({
            "metric": "precipitation_mm",
            "level": "TORRENTIAL_RAIN",
            "value": rain,
            "unit": "mm",
            "threshold": 15.0,
            "title": "Torrential Downpour & Flash Flood Risk"
        })
    elif rain >= 6.0:
        hazards.append({
            "metric": "precipitation_mm",
            "level": "HEAVY_RAIN",
            "value": rain,
            "unit": "mm",
            "threshold": 6.0,
            "title": "Heavy Precipitation Advisory"
        })
        
    # 4. UV Index Rules
    uv = forecast_24h.get("uv_index", 0.0)
    if uv >= 11.0:
        hazards.append({
            "metric": "uv_index",
            "level": "EXTREME_UV",
            "value": uv,
            "unit": "UV",
            "threshold": 11.0,
            "title": "Extreme Solar UV Radiation Warning"
        })
    elif uv >= 8.0:
        hazards.append({
            "metric": "uv_index",
            "level": "VERY_HIGH_UV",
            "value": uv,
            "unit": "UV",
            "threshold": 8.0,
            "title": "Very High UV Radiation Advisory"
        })
        
    # 5. Wind Speed Rules
    wind = forecast_24h.get("wind_speed_kmh", 10.0)
    if wind >= 50.0:
        hazards.append({
            "metric": "wind_speed_kmh",
            "level": "GALE_FORCE",
            "value": wind,
            "unit": "km/h",
            "threshold": 50.0,
            "title": "Severe Gale Wind Warning: Structural & Flying Debris Hazard"
        })
    elif wind >= 35.0:
        hazards.append({
            "metric": "wind_speed_kmh",
            "level": "HIGH_WIND",
            "value": wind,
            "unit": "km/h",
            "threshold": 35.0,
            "title": "High Wind Speed Advisory"
        })
        
    # 6. Humidity Rules
    humidity = forecast_24h.get("humidity_pct", 50.0)
    if humidity >= 80.0 and temp >= 32.0:
        hazards.append({
            "metric": "humidity_pct",
            "level": "OPPRESSIVE_HUMIDITY",
            "value": humidity,
            "unit": "%",
            "threshold": 80.0,
            "title": "Oppressive Sultry Humidity / High Heat Index Advisory"
        })
        
    return hazards


def generate_deterministic_alert_fallback(hazards: List[Dict[str, Any]], city: str) -> str:
    """Deterministic, high-quality rule-based safety advisory when LLM is unavailable."""
    if not hazards:
        return (
            f"🟢 NORMAL CONDITIONS FOR {city.upper()}:\n"
            "All predicted weather and air quality parameters are within standard, safe thresholds. "
            "No active meteorological hazards are projected for the next 24 hours."
        )
        
    hazard_titles = [f"- {h['title']} ({h['value']} {h['unit']})" for h in hazards]
    hazard_summary = "\n".join(hazard_titles)
    
    return f"""🚨 METEOROLOGICAL HAZARD ALERT FOR {city.upper()}
==================================================
Active Hazard Triggers:
{hazard_summary}

1. 👥 GUIDANCE FOR THE GENERAL PUBLIC:
- Minimize unnecessary travel during peak hazardous windows.
- Seal windows and use indoor air purifiers/AC recirculation if AQI is elevated.
- Maintain adequate hydration and carry protective umbrellas or windwear.

2. 🫁 GUIDANCE FOR SENSITIVE GROUPS (Asthmatics, Elderly, Children):
- Avoid all strenuous outdoor exertion; remain in climate-controlled indoor environments.
- Keep rescue inhalers and essential pulmonary/cardiac medications within reach.
- Wear N95-grade particulate respirators if brief outdoor transit is mandatory.

3. 👷 GUIDANCE FOR OUTDOOR WORKERS (Laborers, Delivery Riders, Police):
- Implement mandatory 15-minute shaded rest breaks for every 45 minutes of heavy activity.
- Hydrate frequently with electrolyte-infused water (minimum 1 liter every 2 hours).
- Wear certified N95 dust/smog respirators, UV-rated eye protection, and high-visibility raincoats as applicable.
"""


def alert_node(state: AgentWeatherState) -> AgentWeatherState:
    """
    LangGraph Node: Alert Agent.
    Evaluates fixed rule-based hazard thresholds across all 6 metrics.
    If hazard triggers, invokes LLM with fallback to author structured advisories
    for General Public, Sensitive Groups, and Outdoor Workers.
    """
    logger.info("--- [Node: Alert Agent] Evaluating hazard thresholds ---")
    
    city = state.get("city", DEFAULT_CITY)
    forecast_24h = state.get("forecast_24h", {})
    
    hazards = evaluate_fixed_hazard_rules(forecast_24h)
    state["has_active_alerts"] = len(hazards) > 0
    state["alerts"] = hazards
    
    if not hazards:
        logger.info("No meteorological hazards triggered. Conditions nominal.")
        return state
        
    logger.info(f"Triggered {len(hazards)} hazard alerts: {[h['title'] for h in hazards]}")
    
    # Prompt LLM to author the tailored advice
    llm = LLMClient()
    hazard_bullets = "\n".join([f"- {h['title']} (Predicted: {h['value']} {h['unit']}, Safe Limit: {h['threshold']})" for h in hazards])
    
    prompt = f"""
Meteorological Alert Generation Request for City: {city}
The automated forecasting engine has detected the following hazard triggers for the next 24 hours:
{hazard_bullets}

Overall 24-hour Forecast Snapshot:
- AQI: {forecast_24h.get('aqi')}
- Temperature: {forecast_24h.get('temperature_c')}°C
- Precipitation: {forecast_24h.get('precipitation_mm')} mm
- Humidity: {forecast_24h.get('humidity_pct')}%
- UV Index: {forecast_24h.get('uv_index')}
- Wind Speed: {forecast_24h.get('wind_speed_kmh')} km/h

Write a comprehensive, professional emergency advisory structured strictly into three actionable sections:
1. 👥 GENERAL PUBLIC: Daily routine precautions, travel impact, indoor ventilation.
2. 🫁 SENSITIVE GROUPS: Specific health safeguards for elderly, children, asthmatics, and heart patients.
3. 👷 OUTDOOR WORKERS: Occupational safety instructions for construction workers, couriers, and traffic personnel (hydration, PPE, mandatory rest intervals).
"""
    
    system_instruction = "You are the Senior Safety Officer of AgentWeather. Issue precise, authoritative, life-saving weather and air-quality alerts."
    
    response_text, log_meta = llm.generate(
        prompt=prompt,
        system_instruction=system_instruction,
        agent_name="AlertAgent",
        fallback_fn=lambda: generate_deterministic_alert_fallback(hazards, city)
    )
    
    state["llm_logs"].append(log_meta)
    # Store synthesized advice inside hazards
    for h in state["alerts"]:
        h["detailed_advisory"] = response_text
        
    return state

