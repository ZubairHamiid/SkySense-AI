import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List


def get_aqi_badge_color(aqi: float) -> tuple[str, str, str]:
    """Returns (category_name, hex_background, text_color) based on US EPA AQI standard."""
    if aqi <= 50:
        return "Good (0-50)", "#00E400", "#000000"
    elif aqi <= 100:
        return "Moderate (51-100)", "#FFFF00", "#000000"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups (101-150)", "#FF7E00", "#FFFFFF"
    elif aqi <= 200:
        return "Unhealthy (151-200)", "#FF0000", "#FFFFFF"
    elif aqi <= 300:
        return "Very Unhealthy (201-300)", "#8F3F97", "#FFFFFF"
    else:
        return "Hazardous (301-500)", "#7E0023", "#FFFFFF"


def inject_custom_css():
    """Injects modern, polished CSS styling into the Streamlit app."""
    st.markdown("""
    <style>
    /* Metric Card Glassmorphism */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .broadcast-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-left: 5px solid #38bdf8;
        border-radius: 8px;
        padding: 20px 24px;
        color: #f8fafc;
        font-family: 'Georgia', serif;
        font-size: 1.05rem;
        line-height: 1.7;
        margin-bottom: 25px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    .alert-card-public {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 12px;
    }
    
    .alert-card-sensitive {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 12px;
    }
    
    .alert-card-workers {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 12px;
    }
    
    .stat-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    </style>
    """, unsafe_allow_html=True)


def plot_metric_forecast_curve(df_trajectory: pd.DataFrame, metric: str, baseline_val: float) -> go.Figure:
    """Plots an interactive 24-hour forecast curve compared with the Naive Persistence Baseline."""
    titles = {
        "aqi": ("Air Quality Index (AQI)", "Index"),
        "temperature_c": ("Temperature Forecast (°C)", "°C"),
        "precipitation_mm": ("Precipitation Volume (mm)", "mm"),
        "humidity_pct": ("Relative Humidity (%)", "%"),
        "uv_index": ("Solar UV Index", "UV"),
        "wind_speed_kmh": ("Wind Speed (km/h)", "km/h")
    }
    
    colors = {
        "aqi": "#f97316",
        "temperature_c": "#ef4444",
        "precipitation_mm": "#3b82f6",
        "humidity_pct": "#06b6d4",
        "uv_index": "#eab308",
        "wind_speed_kmh": "#10b981"
    }
    
    title_str, unit = titles.get(metric, (metric.capitalize(), ""))
    line_color = colors.get(metric, "#6366f1")
    
    fig = go.Figure()
    
    # 1. 24h Model Forecast Trajectory
    fig.add_trace(go.Scatter(
        x=df_trajectory["time"],
        y=df_trajectory[metric],
        mode="lines+markers",
        name="Agent Forecast (24h Ahead)",
        line=dict(color=line_color, width=3.5, shape="spline"),
        marker=dict(size=6, color=line_color),
        hovertemplate=f"%{{x}}<br><b>Forecast:</b> %{{y:.1f}} {unit}<extra></extra>"
    ))
    
    # 2. Naive Persistence Baseline
    fig.add_trace(go.Scatter(
        x=df_trajectory["time"],
        y=[baseline_val] * len(df_trajectory),
        mode="lines",
        name="Naive Baseline (24h Persistence)",
        line=dict(color="#94a3b8", width=2, dash="dash"),
        hovertemplate=f"<b>Baseline:</b> {baseline_val:.1f} {unit}<extra></extra>"
    ))
    
    # Add EPA bands for AQI
    if metric == "aqi":
        fig.add_hrect(y0=0, y1=50, fillcolor="#00E400", opacity=0.1, line_width=0, annotation_text="Good (0-50)")
        fig.add_hrect(y0=50, y1=100, fillcolor="#FFFF00", opacity=0.1, line_width=0, annotation_text="Moderate (51-100)")
        fig.add_hrect(y0=100, y1=150, fillcolor="#FF7E00", opacity=0.1, line_width=0, annotation_text="Sensitive (101-150)")
        fig.add_hrect(y0=150, y1=200, fillcolor="#FF0000", opacity=0.1, line_width=0, annotation_text="Unhealthy (151-200)")
        fig.add_hrect(y0=200, y1=300, fillcolor="#8F3F97", opacity=0.1, line_width=0, annotation_text="Very Unhealthy (201-300)")
        fig.add_hrect(y0=300, y1=500, fillcolor="#7E0023", opacity=0.1, line_width=0, annotation_text="Hazardous (>300)")
        
    fig.update_layout(
        title=f"<b>{title_str}</b> — Next 24 Hours",
        xaxis_title="Timeline (Hourly Steps)",
        yaxis_title=unit,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

