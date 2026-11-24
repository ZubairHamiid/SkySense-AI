import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


def get_aqi_badge_color(aqi: float) -> Tuple[str, str, str]:
    """Returns (category_name, hex_background, text_color) based on US EPA AQI standard."""
    if aqi <= 50:
        return "Good", "#10b981", "#ffffff"
    elif aqi <= 100:
        return "Moderate", "#eab308", "#000000"
    elif aqi <= 150:
        return "Unhealthy for Sensitive", "#f97316", "#ffffff"
    elif aqi <= 200:
        return "Unhealthy", "#ef4444", "#ffffff"
    elif aqi <= 300:
        return "Very Unhealthy", "#a855f7", "#ffffff"
    else:
        return "Hazardous", "#881337", "#ffffff"


def inject_custom_css():
    """Injects high-end, bespoke Glassmorphism & Cyber-Meteorological styling."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Hero Atmospheric Card */
    .hero-weather-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.7) 50%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(16px);
        box-shadow: 0 12px 35px -10px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }
    .hero-weather-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.15) 0%, rgba(56, 189, 248, 0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    
    /* Metric Glass Cards */
    .metric-glass-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 12px;
        backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .metric-glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 12px 28px -5px rgba(56, 189, 248, 0.25);
    }
    
    /* Live Status Pulse */
    .pulse-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 2s infinite;
        margin-right: 8px;
        vertical-align: middle;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    /* Teleprompter Broadcast Box */
    .broadcast-teleprompter {
        background: linear-gradient(145deg, #090d16 0%, #0f172a 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-left: 6px solid #38bdf8;
        border-radius: 16px;
        padding: 24px 28px;
        color: #f1f5f9;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.05rem;
        line-height: 1.8;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        margin-bottom: 24px;
    }
    
    /* Actionable Advisory Cards */
    .advisory-badge-public {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(37, 99, 235, 0.05));
        border: 1px solid rgba(59, 130, 246, 0.35);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 14px;
    }
    .advisory-badge-sensitive {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.05));
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 14px;
    }
    .advisory-badge-workers {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.05));
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 14px;
    }
    
    /* Code/Timestamp tags */
    .chip-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.025em;
        text-transform: uppercase;
    }
    </style>
    """, unsafe_allow_html=True)


def plot_aqi_gauge(aqi_val: float) -> go.Figure:
    """Creates a high-precision circular speedometer gauge for AQI."""
    cat_name, bg_col, _ = get_aqi_badge_color(aqi_val)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=aqi_val,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>AIR QUALITY INDEX (AQI)</b><br><span style='font-size:0.85em;color:{bg_col}'>{cat_name}</span>", 'font': {'size': 14, 'color': '#f8fafc'}},
        number={'font': {'size': 42, 'color': bg_col, 'family': 'JetBrains Mono'}},
        gauge={
            'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "#475569", 'nticks': 6},
            'bar': {'color': bg_col, 'thickness': 0.28},
            'bgcolor': "rgba(15, 23, 42, 0.6)",
            'borderwidth': 1,
            'bordercolor': "rgba(255, 255, 255, 0.1)",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [50, 100], 'color': 'rgba(234, 179, 8, 0.2)'},
                {'range': [100, 150], 'color': 'rgba(249, 115, 22, 0.2)'},
                {'range': [150, 200], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [200, 300], 'color': 'rgba(168, 85, 247, 0.2)'},
                {'range': [300, 500], 'color': 'rgba(136, 19, 55, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.75,
                'value': aqi_val
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        height=220
    )
    return fig


def plot_circular_gauge(value: float, min_val: float, max_val: float, title: str, unit: str, color: str) -> go.Figure:
    """Creates a sleek circular metric gauge for Temperature, Humidity, or Wind."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{title}</b>", 'font': {'size': 13, 'color': '#cbd5e1'}},
        number={'suffix': f" {unit}", 'font': {'size': 32, 'color': '#ffffff', 'family': 'JetBrains Mono'}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickcolor': "#475569"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "rgba(15, 23, 42, 0.6)",
            'borderwidth': 1,
            'bordercolor': "rgba(255, 255, 255, 0.1)"
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=40, b=15),
        height=190
    )
    return fig


def plot_pakistan_weather_map(cities_info: Dict[str, Dict[str, Any]], active_city: str) -> go.Figure:
    """
    Renders an interactive geographical map of Pakistan showing monitored cities,
    their current AQI (color-coded), and temperature.
    """
    data = []
    for city_name, c in cities_info.items():
        aqi_val = c.get("aqi", 100)
        temp_val = c.get("temp", 22.0)
        cat_name, bg_col, _ = get_aqi_badge_color(aqi_val)
        is_active = (city_name.lower() == active_city.lower())
        
        data.append({
            "City": city_name,
            "lat": c["lat"],
            "lon": c["lon"],
            "AQI": aqi_val,
            "Temp": f"{temp_val:.1f}°C",
            "Status": cat_name,
            "Province": c.get("province", "Pakistan"),
            "Climate": c.get("climate", ""),
            "Color": bg_col,
            "Size": 28 if is_active else 18,
            "Symbol": "star" if is_active else "circle"
        })
        
    df_map = pd.DataFrame(data)
    
    fig = go.Figure()
    
    # Add city scatter markers
    fig.add_trace(go.Scattergeo(
        lon=df_map["lon"],
        lat=df_map["lat"],
        text=df_map["City"],
        customdata=np.stack((df_map["AQI"], df_map["Temp"], df_map["Status"], df_map["Climate"]), axis=-1),
        hovertemplate=(
            "<b>📍 %{text}</b><br>" +
            "AQI: <b>%{customdata[0]}</b> (%{customdata[2]})<br>" +
            "Temperature: <b>%{customdata[1]}</b><br>" +
            "Zone: <i>%{customdata[3]}</i><extra></extra>"
        ),
        mode="markers+text",
        textposition="top center",
        textfont=dict(color="#f8fafc", size=11, family="Plus Jakarta Sans"),
        marker=dict(
            size=df_map["Size"],
            color=df_map["Color"],
            line=dict(width=2, color="#ffffff"),
            opacity=0.9
        )
    ))
    
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        showcountries=True,
        countrycolor="rgba(255, 255, 255, 0.2)",
        showsubunits=True,
        subunitcolor="rgba(255, 255, 255, 0.1)",
        showland=True,
        landcolor="rgba(15, 23, 42, 0.95)",
        showocean=True,
        oceancolor="rgba(8, 12, 22, 0.95)",
        lataxis_range=[23, 38],
        lonaxis_range=[60, 78]
    )
    
    fig.update_layout(
        title="<b>🗺️ Pakistan Real-Time Environmental Monitoring Basins</b>",
        title_font=dict(size=14, color="#e2e8f0"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        height=340
    )
    return fig


def plot_metric_forecast_curve(df_trajectory: pd.DataFrame, metric: str, baseline_val: float) -> go.Figure:
    """Plots an ultra-sleek 24-hour forecast curve with baseline and shaded EPA levels."""
    titles = {
        "aqi": ("Air Quality Index (AQI)", "Index"),
        "temperature_c": ("Temperature (°C)", "°C"),
        "precipitation_mm": ("Precipitation Volume (mm)", "mm"),
        "humidity_pct": ("Relative Humidity (%)", "%"),
        "uv_index": ("Solar UV Radiation", "UV"),
        "wind_speed_kmh": ("Wind Speed (km/h)", "km/h")
    }
    colors = {
        "aqi": "#f97316",
        "temperature_c": "#ef4444",
        "precipitation_mm": "#38bdf8",
        "humidity_pct": "#06b6d4",
        "uv_index": "#eab308",
        "wind_speed_kmh": "#10b981"
    }
    
    title_str, unit = titles.get(metric, (metric.capitalize(), ""))
    line_color = colors.get(metric, "#6366f1")
    
    fig = go.Figure()
    
    # 24h Model Forecast Trajectory
    fig.add_trace(go.Scatter(
        x=df_trajectory["time"],
        y=df_trajectory[metric],
        mode="lines+markers",
        name="Champion Model 24h Trajectory",
        line=dict(color=line_color, width=3.5, shape="spline"),
        marker=dict(size=7, color=line_color, line=dict(width=1.5, color="#ffffff")),
        hovertemplate=f"%{{x}}<br><b>Forecast:</b> %{{y:.1f}} {unit}<extra></extra>"
    ))
    
    # Naive Persistence Baseline
    fig.add_trace(go.Scatter(
        x=df_trajectory["time"],
        y=[baseline_val] * len(df_trajectory),
        mode="lines",
        name=f"24h Persistence Baseline ({baseline_val:.1f} {unit})",
        line=dict(color="#64748b", width=2, dash="dash"),
        hovertemplate=f"<b>Baseline:</b> {baseline_val:.1f} {unit}<extra></extra>"
    ))
    
    # Add EPA bands for AQI
    if metric == "aqi":
        fig.add_hrect(y0=0, y1=50, fillcolor="#10b981", opacity=0.12, line_width=0, annotation_text="Good (0-50)")
        fig.add_hrect(y0=50, y1=100, fillcolor="#eab308", opacity=0.12, line_width=0, annotation_text="Moderate (51-100)")
        fig.add_hrect(y0=100, y1=150, fillcolor="#f97316", opacity=0.12, line_width=0, annotation_text="Sensitive (101-150)")
        fig.add_hrect(y0=150, y1=200, fillcolor="#ef4444", opacity=0.12, line_width=0, annotation_text="Unhealthy (151-200)")
        fig.add_hrect(y0=200, y1=300, fillcolor="#a855f7", opacity=0.12, line_width=0, annotation_text="Very Unhealthy (201-300)")
        fig.add_hrect(y0=300, y1=500, fillcolor="#881337", opacity=0.15, line_width=0, annotation_text="Hazardous (>300)")
        
    fig.update_layout(
        title=f"<b>{title_str}</b> — 24-Hour Horizon Trajectory",
        title_font=dict(size=15, color="#f1f5f9"),
        xaxis_title="Timeline (Hourly Steps)",
        yaxis_title=unit,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_model_comparison_bar(leaderboard_df: pd.DataFrame) -> go.Figure:
    """Renders a grouped comparison bar chart of models across metrics."""
    if leaderboard_df.empty:
        return go.Figure()
        
    fig = px.bar(
        leaderboard_df,
        x="Target Metric",
        y="Validation RMSE",
        color="Model Family",
        barmode="group",
        hover_data=["Model Name", "Validation R²", "Validation MAE"],
        title="<b>Champion & Candidate Validation RMSE Across Model Families</b>",
        template="plotly_dark",
        color_discrete_map={
            "Classical_ML": "#38bdf8",
            "Statistical": "#f59e0b",
            "Deep_Learning": "#a855f7",
            "Baseline": "#64748b"
        }
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=360
    )
    return fig
