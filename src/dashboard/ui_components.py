import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import textwrap
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


def _svg_icon(kind: str, size: int = 22, color: str = "#e2e8f0") -> str:
    """Returns clean, minimalist inline SVG icons (zero Unicode emojis)."""
    icons = {
        "storm": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M19 16.9A5 5 0 0 0 18 7h-1.26a8 8 0 1 0-11.62 9"/><polyline points="13 11 9 17 15 17 11 23"/></svg>',
        "rain": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 16.2A4.5 4.5 0 0 0 17.5 8h-1.8A7 7 0 1 0 4 14.9"/><line x1="8" y1="19" x2="8" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/><line x1="16" y1="19" x2="16" y2="21"/></svg>',
        "smog": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 14h16"/><path d="M6 18h12"/><path d="M8 10h8"/><path d="M17.5 10a4.5 4.5 0 0 0-8.6-1.8A3.5 3.5 0 0 0 5 11.5"/></svg>',
        "haze": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="10" r="4"/><path d="M3 16h18"/><path d="M5 20h14"/></svg>',
        "sun": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>',
        "wind": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2"/></svg>',
        "cloud": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"/></svg>',
        "pin": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>',
    }
    return icons.get(kind, icons["cloud"])


def get_weather_icon_and_desc(temp: float, aqi: float, rain: float, wind: float, icon_size: int = 26) -> Tuple[str, str]:
    """Generates clean minimalist SVG weather icon and descriptive text (zero emojis)."""
    if rain > 1.0:
        return _svg_icon("storm", icon_size, "#38bdf8"), "Stormy with Rain Showers"
    elif rain > 0.1:
        return _svg_icon("rain", icon_size, "#38bdf8"), "Overcast with Light Rain"
    elif aqi > 200:
        return _svg_icon("smog", icon_size, "#a855f7"), "Dense Smog & Hazardous Air"
    elif aqi > 120:
        return _svg_icon("haze", icon_size, "#f59e0b"), "Hazy Sky & Particulate Smog"
    elif temp > 32:
        return _svg_icon("sun", icon_size, "#f59e0b"), "Clear Sky & High Solar Radiation"
    elif wind > 20:
        return _svg_icon("wind", icon_size, "#38bdf8"), "Breezy & Active Wind Currents"
    else:
        return _svg_icon("cloud", icon_size, "#cbd5e1"), "Partly Cloudy with Mild Breeze"


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
    """Renders the horizontal row of hourly cards matching Image 2. No leading whitespace to prevent markdown code blocks."""
    if df_trajectory.empty:
        return ""
        
    cards_html = []
    display_rows = df_trajectory.head(12)
    
    for idx, row in display_rows.iterrows():
        t = pd.to_datetime(row["time"]).strftime("%H:00")
        temp = float(row.get("temperature_c", 20.0))
        aqi = float(row.get("aqi", 100.0))
        rain = float(row.get("precipitation_mm", 0.0))
        icon, _ = get_weather_icon_and_desc(temp, aqi, rain, 5.0, icon_size=22)
        active_class = "active" if idx == 0 else ""
        cards_html.append(
            f'<div class="hourly-pill {active_class}">'
            f'<div style="font-size:0.75rem;color:#94a3b8;font-family:\'JetBrains Mono\';font-weight:600;">{t}</div>'
            f'<div style="display:flex;justify-content:center;align-items:center;margin:8px 0;">{icon}</div>'
            f'<div style="font-size:1rem;font-weight:700;color:#ffffff;">{temp:.0f}°</div>'
            f'<div style="font-size:0.68rem;color:#38bdf8;margin-top:2px;">{rain:.1f} mm</div>'
            f'</div>'
        )
    return '<div class="hourly-scroll-container">' + "".join(cards_html) + '</div>'


def render_city_chips_html(cities_meta: Dict[str, Dict[str, Any]], active_city: str) -> str:
    """Renders the 'Recently Searched / Major Cities' cards on the side matching Image 1. No leading whitespace."""
    chips_html = []
    
    for cname, cdata in list(cities_meta.items())[:4]:
        temp = cdata.get("temp", 22.0)
        aqi = cdata.get("aqi", 110)
        icon, desc = get_weather_icon_and_desc(temp, aqi, 0.0, 5.0, icon_size=22)
        is_active = (cname.lower() == active_city.lower())
        border_style = "border-color:rgba(56,189,248,0.6);background:rgba(56,189,248,0.1);" if is_active else ""
        province = cdata.get("province", "Pakistan")
        chips_html.append(
            f'<div class="city-chip-card" style="{border_style}">'
            f'<div style="display:flex;align-items:center;gap:12px;">'
            f'<span style="display:inline-flex;align-items:center;">{icon}</span>'
            f'<div>'
            f'<div style="font-weight:700;color:#ffffff;font-size:0.95rem;">{cname}</div>'
            f'<div style="font-size:0.72rem;color:#94a3b8;">{province}</div>'
            f'</div></div>'
            f'<div style="text-align:right;">'
            f'<div style="font-size:1.15rem;font-weight:800;color:#ffffff;font-family:\'JetBrains Mono\';">{temp:.0f}°</div>'
            f'<div style="font-size:0.7rem;color:#38bdf8;">AQI {aqi:.0f}</div>'
            f'</div></div>'
        )
    return "".join(chips_html)


def build_dynamic_advisories(forecast_24h: Dict[str, Any], alerts: List[Dict[str, Any]] = None) -> Tuple[str, str, str]:
    """
    Dynamically synthesizes tailored 1-line advisories for General Public, Sensitive Groups,
    and Outdoor Workers based on the selected city's actual 24h forecast across all 6 metrics.
    """
    aqi = float(forecast_24h.get("aqi", 75.0))
    temp = float(forecast_24h.get("temperature_c", 22.0))
    rain = float(forecast_24h.get("precipitation_mm", 0.0))
    hum = float(forecast_24h.get("humidity_pct", 55.0))
    uv = float(forecast_24h.get("uv_index", 1.0))
    wind = float(forecast_24h.get("wind_speed_kmh", 8.0))

    pub_parts, sens_parts, work_parts = [], [], []

    # 1. Air Quality tier
    if aqi > 300:
        pub_parts.append("Hazardous smog — stay indoors & keep windows tightly sealed.")
        sens_parts.append("Emergency: avoid all outdoor air; run HEPA purifier & keep inhalers ready.")
        work_parts.append("Suspend non-essential outdoor work; N95/P100 respirator mandatory.")
    elif aqi > 200:
        pub_parts.append("Very unhealthy air — avoid outdoor exertion & limit peak-hour travel.")
        sens_parts.append("Stay indoors with air filtration; keep rescue medications accessible.")
        work_parts.append("Mandatory N95 particulate PPE & frequent indoor/shaded recovery breaks.")
    elif aqi > 150:
        pub_parts.append("Unhealthy AQI — reduce prolonged outdoor activity & wear a mask in traffic.")
        sens_parts.append("Avoid strenuous outdoor activity; monitor respiratory symptoms closely.")
        work_parts.append("Wear N95 dust/smog masks & rotate heavy tasks every 45 minutes.")
    elif aqi > 100:
        pub_parts.append("Moderate haze — normal travel fine; cut back intense roadside exercise.")
        sens_parts.append("Asthmatics, kids & elderly should shorten prolonged outdoor exertion.")
        work_parts.append("Use particulate masks near heavy traffic or dusty construction zones.")
    elif aqi > 50:
        pub_parts.append("Acceptable air quality — comfortable for daily commutes and outdoor errands.")
        sens_parts.append("Low risk; unusually sensitive individuals should monitor symptoms.")
        work_parts.append("Standard outdoor operations safe; routine hydration recommended.")
    else:
        pub_parts.append("Clean air — ideal conditions for outdoor exercise & natural ventilation.")
        sens_parts.append("Safe atmospheric quality for children, elderly, and respiratory groups.")
        work_parts.append("Clear air — normal outdoor work shifts with standard safety gear.")

    # 2. Temperature / Thermal modifier
    if temp >= 40.0:
        pub_parts.append("Extreme heat — avoid midday sun (11 AM–4 PM) & hydrate.")
        sens_parts.append("High heatstroke risk — stay in cooled rooms & drink ORS.")
        work_parts.append("Mandatory 15-min shaded hydration breaks every 30 min.")
    elif temp >= 35.0:
        pub_parts.append("Hot weather — wear light cotton & stay hydrated.")
        sens_parts.append("Limit direct afternoon sun exposure.")
        work_parts.append("Schedule heavy labor early morning & keep cool water on site.")
    elif temp <= 12.0:
        pub_parts.append(f"Chilly ({temp:.0f}°C) — wear warm layers, especially after sunset.")
        sens_parts.append("Protect elderly & children from cold air to prevent chills.")
        work_parts.append("Thermal jackets & gloves recommended for morning/night shifts.")

    # 3. Rain / Wind / UV modifiers
    if rain >= 6.0:
        pub_parts.append("Heavy rain expected — watch for waterlogging & slick roads.")
        sens_parts.append("Avoid wet/cold exposure during downpours.")
        work_parts.append("Waterproof PPE, slip-resistant boots & pause elevated electrical work.")
    elif rain >= 0.5:
        pub_parts.append("Light rain possible — carry an umbrella.")
        work_parts.append("Keep raincoats accessible & check footing on wet surfaces.")

    if uv >= 8.0:
        pub_parts.append("Very high UV — apply SPF 30+ sunscreen & sunglasses.")
        work_parts.append("UV-rated eyewear & wide-brim helmets required outdoors.")

    if wind >= 30.0:
        pub_parts.append(f"Gusty winds ({wind:.0f} km/h) — secure loose outdoor items.")
        work_parts.append("Halt crane/scaffold work & wear dust goggles.")
    elif hum >= 80.0 and temp >= 30.0:
        pub_parts.append("High muggy humidity — pace outdoor walking.")
        work_parts.append("Increase electrolyte intake for high heat-index stress.")

    return " ".join(pub_parts[:2]), " ".join(sens_parts[:2]), " ".join(work_parts[:2])


def render_left_panel_html(
    selected_city: str,
    aqi_val: float,
    cat_name: str,
    bg_col: str,
    text_col: str,
    cities_summary: Dict[str, Any],
    forecast_24h: Dict[str, Any] = None,
    alerts: List[Dict[str, Any]] = None,
) -> str:
    """Renders the left widget card with dynamic condition-aware advisories and zero leading whitespace."""
    chips_html = render_city_chips_html(cities_summary, selected_city)
    f_dict = forecast_24h if forecast_24h else {"aqi": aqi_val}
    pub_adv, sens_adv, work_adv = build_dynamic_advisories(f_dict, alerts)

    # Dynamic border colors based on severity
    pub_col = "#ef4444" if aqi_val > 200 else ("#f59e0b" if aqi_val > 100 else "#10b981")
    sens_col = "#ef4444" if aqi_val > 150 else ("#f59e0b" if aqi_val > 100 else "#38bdf8")
    work_col = "#ef4444" if aqi_val > 200 else ("#f59e0b" if aqi_val > 100 else "#3b82f6")

    html = (
        '<div class="widget-glass-card"><div>'
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">'
        '<span style="font-size:0.8rem;color:#94a3b8;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">Air & Hazard Status</span>'
        f'<span style="background:{bg_col}25;border:1px solid {bg_col};color:{bg_col};padding:3px 10px;border-radius:9999px;font-size:0.75rem;font-weight:700;">{cat_name.upper()}</span>'
        '</div>'
        '<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:18px;padding:18px;margin-bottom:20px;">'
        '<div style="display:flex;justify-content:space-between;align-items:center;">'
        '<div>'
        '<div style="font-size:0.75rem;color:#94a3b8;">AQI Metric (24h Ahead)</div>'
        f'<div style="font-size:2.4rem;font-weight:800;color:{bg_col};font-family:\'JetBrains Mono\';line-height:1.1;">{aqi_val:.0f}</div>'
        '</div>'
        '<div style="text-align:right;">'
        f'<span style="background:{bg_col};color:{text_col};padding:4px 12px;border-radius:9999px;font-size:0.75rem;font-weight:800;">{cat_name.split(" ")[0]}</span>'
        '<div style="font-size:0.7rem;color:#94a3b8;margin-top:6px;">PM2.5 / PM10 EPA</div>'
        '</div></div></div>'
        '<div style="margin-bottom:20px;">'
        f'<div style="font-size:0.76rem;color:#94a3b8;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;">Targeted Advisories — {selected_city}</div>'
        f'<div style="background:rgba(59,130,246,0.08);border-left:3px solid {pub_col};border-radius:8px;padding:10px 12px;margin-bottom:8px;font-size:0.8rem;color:#e2e8f0;"><span style="font-size:0.68rem;font-weight:700;letter-spacing:0.06em;color:#94a3b8;text-transform:uppercase;display:block;margin-bottom:2px;">General Public</span>{pub_adv}</div>'
        f'<div style="background:rgba(245,158,11,0.08);border-left:3px solid {sens_col};border-radius:8px;padding:10px 12px;margin-bottom:8px;font-size:0.8rem;color:#e2e8f0;"><span style="font-size:0.68rem;font-weight:700;letter-spacing:0.06em;color:#94a3b8;text-transform:uppercase;display:block;margin-bottom:2px;">Sensitive Groups</span>{sens_adv}</div>'
        f'<div style="background:rgba(239,68,68,0.08);border-left:3px solid {work_col};border-radius:8px;padding:10px 12px;font-size:0.8rem;color:#e2e8f0;"><span style="font-size:0.68rem;font-weight:700;letter-spacing:0.06em;color:#94a3b8;text-transform:uppercase;display:block;margin-bottom:2px;">Outdoor Workers</span>{work_adv}</div>'
        '</div>'
        '<div>'
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
        '<span style="font-size:0.8rem;color:#94a3b8;font-weight:600;text-transform:uppercase;">Major Basins</span>'
        '<span style="font-size:0.75rem;color:#38bdf8;">Real-Time</span>'
        '</div>'
        f'{chips_html}'
        '</div></div></div>'
    )
    return html


def degrees_to_compass(deg: float) -> str:
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((float(deg) + 22.5) // 45) % 8
    return dirs[idx]


def render_hero_header_html(
    selected_city: str,
    city_info: Dict[str, Any],
    temp_val: float,
    high_temp: float,
    low_temp: float,
    weather_icon: str,
    weather_desc: str,
    wind_val: float,
    humidity_val: float,
    rain_val: float,
    uv_val: float,
    aqi_val: float,
    wind_dir_deg: float = 180.0,
    rain_prob_pct: float = 0.0,
) -> str:
    """Renders the top portion of the right hero frame with zero indentation to prevent code blocks."""
    now_str = datetime.now().strftime('%A, %B %d')
    province = city_info.get('province', 'Pakistan')
    elevation = city_info.get('elevation_m', 200)
    climate = city_info.get('climate', 'Continental')
    compass = degrees_to_compass(wind_dir_deg)

    pin_svg = _svg_icon("pin", 16, "#38bdf8")
    html = (
        '<div class="hero-glass-frame">'
        '<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;margin-bottom:12px;">'
        '<div style="display:flex;align-items:center;gap:8px;">'
        f'<span style="display:inline-flex;align-items:center;">{pin_svg}</span>'
        f'<span style="font-size:1.15rem;font-weight:700;color:#f8fafc;">{selected_city}, {province}, Pakistan</span>'
        f'<span style="color:#64748b;font-size:0.9rem;">({now_str})</span>'
        '</div>'
        f'<div style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);border-radius:9999px;padding:4px 14px;font-size:0.78rem;color:#94a3b8;">Elevation: {elevation}m &middot; {climate}</div>'
        '</div>'
        '<div style="display:flex;align-items:baseline;gap:24px;flex-wrap:wrap;">'
        f'<div class="hero-temp-title">{temp_val:.0f}°</div>'
        f'<div style="color:#94a3b8;font-size:1.1rem;font-weight:600;font-family:\'JetBrains Mono\';">H <span style="color:#ffffff;">{high_temp:.0f}°</span> &nbsp;&middot;&nbsp; L <span style="color:#ffffff;">{low_temp:.0f}°</span></div>'
        '</div>'
        '<div class="hero-condition-text" style="display:flex;align-items:center;gap:12px;">'
        f'<span style="display:inline-flex;align-items:center;">{weather_icon}</span><span><b>{weather_desc}</b></span>'
        '</div>'
        '<div style="display:flex;gap:10px;flex-wrap:wrap;">'
        f'<div class="stat-pill"><span style="color:#94a3b8;font-size:0.72rem;font-weight:700;letter-spacing:0.05em;">WIND</span> <b>{wind_val:.1f} km/h ({compass} {wind_dir_deg:.0f}°)</b></div>'
        f'<div class="stat-pill"><span style="color:#94a3b8;font-size:0.72rem;font-weight:700;letter-spacing:0.05em;">HUMIDITY</span> <b>{humidity_val:.0f}%</b></div>'
        f'<div class="stat-pill"><span style="color:#94a3b8;font-size:0.72rem;font-weight:700;letter-spacing:0.05em;">PRECIP</span> <b>{rain_val:.1f} mm ({rain_prob_pct:.0f}%)</b></div>'
        f'<div class="stat-pill"><span style="color:#94a3b8;font-size:0.72rem;font-weight:700;letter-spacing:0.05em;">UV INDEX</span> <b>{uv_val:.1f}</b></div>'
        f'<div class="stat-pill"><span style="color:#94a3b8;font-size:0.72rem;font-weight:700;letter-spacing:0.05em;">AQI</span> <b>{aqi_val:.0f}</b></div>'
        '</div>'
        '</div>'
    )
    return html


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

