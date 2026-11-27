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
            "title": "Air Quality Emergency"
        })
    elif aqi > 200:
        hazards.append({
            "metric": "aqi",
            "level": "VERY_UNHEALTHY",
            "value": aqi,
            "unit": "AQI",
            "threshold": 200,
            "title": "Air Quality Warning"
        })
    elif aqi > 150:
        hazards.append({
            "metric": "aqi",
            "level": "UNHEALTHY",
            "value": aqi,
            "unit": "AQI",
            "threshold": 150,
            "title": "Air Quality Alert"
        })
    elif aqi > 100:
        hazards.append({
            "metric": "aqi",
            "level": "SENSITIVE_GROUPS",
            "value": aqi,
            "unit": "AQI",
            "threshold": 100,
            "title": "Air Quality Advisory"
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
    """Condition-aware rule-based safety advisory tailored specifically to the active hazards."""
    if not hazards:
        return (
            f"NORMAL CONDITIONS — {city.upper()}:\n"
            "All predicted weather and air quality parameters are within standard, safe thresholds. "
            "No active meteorological hazards are projected for the next 24 hours."
        )

    hazard_titles = [f"- {h['title']} ({h['value']} {h['unit']})" for h in hazards]
    hazard_summary = "\n".join(hazard_titles)
    metrics_triggered = {h["metric"] for h in hazards}
    levels_triggered = {h["level"] for h in hazards}

    pub_lines, sens_lines, work_lines = [], [], []

    if "aqi" in metrics_triggered:
        if "HAZARDOUS" in levels_triggered or "VERY_UNHEALTHY" in levels_triggered:
            pub_lines.append("- Avoid outdoor exertion and seal doors/windows during peak smog hours.")
            sens_lines.append("- Stay strictly indoors with air purification; keep rescue inhalers within reach.")
            work_lines.append("- Mandatory N95/P100 particulate respirators and frequent indoor recovery breaks.")
        elif "UNHEALTHY" in levels_triggered:
            pub_lines.append("- Limit prolonged outdoor commutes and wear a mask along heavy traffic corridors.")
            sens_lines.append("- Avoid strenuous outdoor activity; monitor breathing and cardiac symptoms.")
            work_lines.append("- Wear certified N95 masks and rotate high-exertion outdoor shifts every 45 min.")
        else:
            pub_lines.append("- General travel is safe; reduce prolonged vigorous exercise near major roads.")
            sens_lines.append("- Children, elderly, and asthmatics should shorten heavy outdoor activity.")
            work_lines.append("- Use particulate masks in dusty zones or roadside operations.")

    if "temperature_c" in metrics_triggered:
        if "COLD_WARNING" in levels_triggered:
            pub_lines.append("- Dress in insulated warm layers and protect exposed pipes/crops from frost.")
            sens_lines.append("- Keep elderly and young children in heated indoor spaces overnight.")
            work_lines.append("- Provide thermal outerwear, gloves, and warm break areas for early-morning crews.")
        else:
            pub_lines.append("- Avoid direct sun between 11 AM and 4 PM; wear light cotton clothing and hydrate.")
            sens_lines.append("- High heat-illness risk: stay in cooled environments and drink ORS/electrolytes.")
            work_lines.append("- Enforce 15-min shaded rest breaks every 30–45 min and 1L electrolyte water every 2h.")

    if "precipitation_mm" in metrics_triggered:
        pub_lines.append("- Expect slick roads and localized urban flooding; avoid low-lying underpasses.")
        sens_lines.append("- Stay dry indoors and plan medical/essential transit around peak rain bands.")
        work_lines.append("- Issue waterproof PPE, anti-slip boots, and pause elevated/electrical tasks.")

    if "uv_index" in metrics_triggered:
        pub_lines.append("- Apply broad-spectrum SPF 30+ sunscreen and wear UV-blocking sunglasses.")
        sens_lines.append("- Keep infants and sun-sensitive individuals out of direct midday sunlight.")
        work_lines.append("- Mandate UV-rated safety eyewear, long sleeves, and shaded midday scheduling.")

    if "wind_speed_kmh" in metrics_triggered:
        pub_lines.append("- Secure loose rooftop/balcony objects and expect dust gusts on open highways.")
        sens_lines.append("- Keep windows shut to block wind-blown dust and pollen.")
        work_lines.append("- Halt crane/scaffolding operations and wear wrap-around eye protection.")

    if "humidity_pct" in metrics_triggered:
        pub_lines.append("- High heat-index humidity — slow outdoor pace and stay in ventilated spaces.")
        sens_lines.append("- Monitor for heat exhaustion and stay in dehumidified/air-conditioned rooms.")
        work_lines.append("- Increase cooling/electrolyte intervals to offset reduced sweat evaporation.")

    return (
        f"**METEOROLOGICAL HAZARD ALERT — {city.upper()}**\n\n"
        f"**Active Hazard Triggers:**\n{hazard_summary}\n\n"
        "**1. GENERAL PUBLIC:**\n" + "\n".join(pub_lines) + "\n\n"
        "**2. SENSITIVE GROUPS (Asthmatics, Elderly, Children):**\n" + "\n".join(sens_lines) + "\n\n"
        "**3. OUTDOOR WORKERS (Laborers, Delivery Riders, Field Staff):**\n" + "\n".join(work_lines) + "\n"
    )


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

