import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple


def get_aqi_badge_color(aqi: float) -> Tuple[str, str, str]:
    """Returns (category_name, hex_background, text_color) based on US EPA AQI standard."""
    if aqi <= 50:
        return "Good", "#10b981", "#ffffff"
    elif aqi <= 100:
        return "Moderate", "#eab308", "#000000"
    elif aqi <= 150:
        return "Sensitive Groups", "#f97316", "#ffffff"
    elif aqi <= 200:
        return "Unhealthy", "#ef4444", "#ffffff"
    elif aqi <= 300:
        return "Very Unhealthy", "#a855f7", "#ffffff"
    else:
        return "Hazardous", "#e11d48", "#ffffff"


def get_weather_icon_and_desc(temp: float, aqi: float, rain: float, wind: float) -> Tuple[str, str]:
    """Generates weather icon emoji and descriptive text matching reference designs."""
    if rain > 1.0:
        return "⛈️", "Stormy with Rain Showers"
    elif rain > 0.1:
        return "🌧️", "Overcast with Light Rain"
    elif aqi > 200:
        return "🌫️", "Dense Smog & Hazardous Air"
    elif aqi > 120:
        return "🌤️", "Hazy Sunshine & Particulate Smog"
    elif temp > 32:
        return "☀️", "Clear Sky & High Solar Radiation"
    elif wind > 20:
        return "💨", "Breezy & Active Wind Currents"
    else:
        return "⛅", "Partly Cloudy with Mild Breeze"


def inject_cinematic_theme_css():
    """
    Injects responsive, glassmorphic UI matching the user's reference designs.
    Fully responsive across Mobile (<600px), Tablet (600-1024px), Laptop (1024-1600px),
    and Wide LCD/TV (>1600px).
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');
    
    /* Global Container & Background Reset */
    .stApp {
        background-color: #080b11;
        background-image: 
            radial-gradient(circle at 85% 15%, rgba(245, 158, 11, 0.12) 0%, transparent 45%),
            radial-gradient(circle at 15% 85%, rgba(56, 189, 248, 0.1) 0%, transparent 50%),
            linear-gradient(180deg, rgba(8, 11, 17, 0.72) 0%, rgba(8, 11, 17, 0.92) 100%),
            url('https://images.unsplash.com/photo-1534088568595-a066f410bcda?auto=format&fit=crop&w=2600&q=80');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #f8fafc;
    }
    
    /* Top Header Bar */
    .cinematic-topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        padding: 12px 24px;
        background: rgba(18, 24, 38, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        backdrop-filter: blur(25px);
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    
    /* Main Hero Atmospheric Glass Card */
    .hero-glass-frame {
        background: rgba(18, 24, 38, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 28px;
        padding: 32px 36px;
        backdrop-filter: blur(30px);
        box-shadow: 0 25px 60px -15px rgba(0, 0, 0, 0.7);
        position: relative;
        overflow: hidden;
        margin-bottom: 24px;
    }
    
    /* Side Phone-Mockup / Widget Card (From Image 1) */
    .widget-glass-card {
        background: rgba(18, 24, 38, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 24px;
        backdrop-filter: blur(25px);
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.5);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    /* Huge Temperature Display (From Images 1, 2, 3) */
    .hero-temp-title {
        font-size: 5.5rem;
        font-weight: 800;
        line-height: 1;
        color: #ffffff;
        letter-spacing: -0.04em;
        font-family: 'Plus Jakarta Sans', sans-serif;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
        margin: 0;
    }
    
    .hero-condition-text {
        font-size: 2.2rem;
        font-weight: 400;
        color: #e2e8f0;
        letter-spacing: -0.02em;
        margin: 12px 0 20px 0;
        line-height: 1.2;
    }
    
    /* Horizontal Hourly Pills (From Image 2 & 1) */
    .hourly-scroll-container {
        display: flex;
        gap: 14px;
        overflow-x: auto;
        padding: 8px 4px 16px 4px;
        scrollbar-width: thin;
        scrollbar-color: rgba(255, 255, 255, 0.2) transparent;
    }
    .hourly-scroll-container::-webkit-scrollbar {
        height: 6px;
    }
    .hourly-scroll-container::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
    }
    
    .hourly-pill {
        min-width: 82px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 14px 10px;
        text-align: center;
        backdrop-filter: blur(15px);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    .hourly-pill:hover, .hourly-pill.active {
        background: rgba(255, 255, 255, 0.14);
        border-color: rgba(255, 255, 255, 0.25);
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
    }
    
    /* Recently Searched City Cards (From Image 1) */
    .city-chip-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 14px 18px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.2s ease;
    }
    .city-chip-card:hover {
        background: rgba(255, 255, 255, 0.09);
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateX(4px);
    }
    
    /* Metric Mini-Pills */
    .stat-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 9999px;
        padding: 6px 14px;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
    
    /* Status Badge */
    .hazard-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
    }
    
    /* Responsive Media Queries */
    @media (max-width: 600px) {
        .hero-temp-title { font-size: 3.8rem; }
        .hero-condition-text { font-size: 1.6rem; }
        .hero-glass-frame { padding: 20px 16px; border-radius: 20px; }
        .widget-glass-card { padding: 16px; border-radius: 18px; }
        .cinematic-topbar { padding: 12px 16px; }
    }
    
    @media (min-width: 1920px) {
        .hero-temp-title { font-size: 6.5rem; }
        .hero-condition-text { font-size: 2.6rem; }
        .hero-glass-frame { padding: 42px 48px; border-radius: 36px; }
    }
    
    /* Streamlit Components Dark Override */
    .stSelectbox label, .stSlider label {
        color: #94a3b8 !important;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        border-radius: 16px;
        padding: 6px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        color: #94a3b8;
        padding: 10px 18px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.12) !important;
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)


def plot_cinematic_spline_curve(df_trajectory: pd.DataFrame, metric: str = "temperature_c", title_prefix: str = "24-Hour Horizon") -> go.Figure:
    """
    Renders the exact glowing, smooth spline wave temperature/AQI curve shown in Image 1 & Image 3.
    Features: smooth golden/cyan glowing spline line, luminous point highlight, and transparent background.
    """
    if df_trajectory.empty or metric not in df_trajectory.columns:
        return go.Figure()
        
    df_plot = df_trajectory.copy()
    df_plot["time_dt"] = pd.to_datetime(df_plot["time"])
    df_plot["time_label"] = df_plot["time_dt"].dt.strftime("%H:00")
    
    metric_labels = {
        "temperature_c": ("Temperature", "°C", "#f59e0b", "rgba(245, 158, 11, 0.25)"),
        "aqi": ("Air Quality Index", "AQI", "#f97316", "rgba(249, 115, 22, 0.25)"),
        "precipitation_mm": ("Precipitation", "mm", "#38bdf8", "rgba(56, 189, 248, 0.25)"),
        "humidity_pct": ("Humidity", "%", "#06b6d4", "rgba(6, 182, 212, 0.25)"),
        "uv_index": ("UV Radiation", "UV", "#eab308", "rgba(234, 179, 8, 0.25)"),
        "wind_speed_kmh": ("Wind Velocity", "km/h", "#10b981", "rgba(16, 185, 129, 0.25)")
    }
    
    name_str, unit, line_color, fill_color = metric_labels.get(
        metric, (metric.capitalize(), "", "#38bdf8", "rgba(56, 189, 248, 0.2)")
    )
    
    fig = go.Figure()
    
    # 1. Glowing Smooth Spline Wave Area Fill (Matching Image 1 & Image 3)
    fig.add_trace(go.Scatter(
        x=df_plot["time_label"],
        y=df_plot[metric],
        mode="lines",
        name=f"24h {name_str}",
        line=dict(color=line_color, width=3.5, shape="spline", smoothing=1.3),
        fill="tozeroy",
        fillcolor=fill_color,
        hovertemplate=f"<b>Time:</b> %{{x}}<br><b>{name_str}:</b> %{{y:.1f}} {unit}<extra></extra>"
    ))
    
    # 2. Glowing Milestone Markers
    step_indices = [0, 4, 8, 12, 16, 20, len(df_plot)-1]
    step_indices = [i for i in step_indices if i < len(df_plot)]
    
    fig.add_trace(go.Scatter(
        x=df_plot["time_label"].iloc[step_indices],
        y=df_plot[metric].iloc[step_indices],
        mode="markers+text",
        text=[f"{val:.0f}{unit}" for val in df_plot[metric].iloc[step_indices]],
        textposition="top center",
        textfont=dict(color="#ffffff", size=11, family="Plus Jakarta Sans"),
        marker=dict(
            size=9,
            color="#ffffff",
            line=dict(color=line_color, width=3)
        ),
        showlegend=False,
        hoverinfo="skip"
    ))
    
    fig.update_layout(
        title=f"<b>{title_prefix} — {name_str} Dynamics Wave</b>",
        title_font=dict(size=14, color="#cbd5e1", family="Plus Jakarta Sans"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            color="#64748b",
            tickfont=dict(family="JetBrains Mono", size=10, color="#94a3b8")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.05)",
            zeroline=False,
            color="#64748b",
            tickfont=dict(family="JetBrains Mono", size=10, color="#94a3b8"),
            ticksuffix=f" {unit}"
        ),
        margin=dict(l=10, r=10, t=40, b=20),
        height=260,
        hovermode="x unified"
    )
    return fig


def render_hourly_cards_html(df_trajectory: pd.DataFrame) -> str:
    """Renders the horizontal row of hourly cards matching Image 2."""
    if df_trajectory.empty:
        return ""
        
    cards_html = []
    # Take next 12-16 hours
    display_rows = df_trajectory.head(12)
    
    for idx, row in display_rows.iterrows():
        t = pd.to_datetime(row["time"]).strftime("%H:00")
        temp = float(row.get("temperature_c", 20.0))
        aqi = float(row.get("aqi", 100.0))
        rain = float(row.get("precipitation_mm", 0.0))
        icon, _ = get_weather_icon_and_desc(temp, aqi, rain, 5.0)
        
        is_active = (idx == 0)
        active_class = "active" if is_active else ""
        
        cards_html.append(f"""
        <div class="hourly-pill {active_class}">
            <div style="font-size: 0.75rem; color: #94a3b8; font-family: 'JetBrains Mono'; font-weight: 600;">{t}</div>
            <div style="font-size: 1.5rem; margin: 6px 0;">{icon}</div>
            <div style="font-size: 1rem; font-weight: 700; color: #ffffff;">{temp:.0f}°</div>
            <div style="font-size: 0.68rem; color: #38bdf8; margin-top: 2px;">{rain:.1f}mm</div>
        </div>
        """)
        
    return f"""
    <div class="hourly-scroll-container">
        {''.join(cards_html)}
    </div>
    """


def render_city_chips_html(cities_meta: Dict[str, Dict[str, Any]], active_city: str) -> str:
    """Renders the 'Recently Searched / Major Cities' cards on the side matching Image 1."""
    chips_html = []
    
    for cname, cdata in list(cities_meta.items())[:4]:
        temp = cdata.get("temp", 22.0)
        aqi = cdata.get("aqi", 110)
        icon, desc = get_weather_icon_and_desc(temp, aqi, 0.0, 5.0)
        is_active = (cname.lower() == active_city.lower())
        border_style = "border-color: rgba(56, 189, 248, 0.6); background: rgba(56, 189, 248, 0.1);" if is_active else ""
        
        chips_html.append(f"""
        <div class="city-chip-card" style="{border_style}">
            <div style="display:flex; align-items:center; gap: 12px;">
                <span style="font-size: 1.4rem;">{icon}</span>
                <div>
                    <div style="font-weight: 700; color: #ffffff; font-size: 0.95rem;">{cname}</div>
                    <div style="font-size: 0.72rem; color: #94a3b8;">{cdata.get('province', 'Pakistan')}</div>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.15rem; font-weight: 800; color: #ffffff; font-family: 'JetBrains Mono';">{temp:.0f}°</div>
                <div style="font-size: 0.7rem; color: #38bdf8;">AQI {aqi}</div>
            </div>
        </div>
        """)
        
    return "".join(chips_html)


def plot_model_comparison_bar(leaderboard_df: pd.DataFrame) -> go.Figure:
    """
    Renders a grouped comparison bar chart of models across metrics.
    Robust to column naming variations (e.g., 'Metric' vs 'Target Metric').
    """
    if leaderboard_df.empty:
        return go.Figure()
        
    df = leaderboard_df.copy()
    x_col = "Metric" if "Metric" in df.columns else ("Target Metric" if "Target Metric" in df.columns else df.columns[0])
    color_col = "Family" if "Family" in df.columns else ("Model Family" if "Model Family" in df.columns else "Model")
    y_col = "Validation RMSE" if "Validation RMSE" in df.columns else df.columns[3]
    
    hover_cols = [c for c in ["Model", "Validation R²", "Validation MAE"] if c in df.columns]
    
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        barmode="group",
        hover_data=hover_cols,
        title="<b>Validation RMSE Comparison Across Model Families</b>",
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
        height=320,
        xaxis_title="Atmospheric Metric",
        yaxis_title="Validation RMSE"
    )
    return fig

