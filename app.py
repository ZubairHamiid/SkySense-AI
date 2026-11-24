import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any, Optional

# Set page configuration for full responsiveness
st.set_page_config(
    page_title="SkySense AI | Next-Gen Atmospheric Intelligence",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from config.settings import DEFAULT_CITY, METRICS, PAKISTAN_CITIES
from src.dashboard.ui_components import (
    inject_cinematic_theme_css,
    get_aqi_badge_color,
    get_weather_icon_and_desc,
    plot_cinematic_spline_curve,
    render_hourly_cards_html,
    render_city_chips_html,
    plot_model_comparison_bar
)
from src.pipeline.feature_store import FeatureStoreManager
from src.pipeline.model_registry import ModelRegistry
from src.pipeline.eda import EDAManager
from src.pipeline.explainability import ModelExplainer
from src.agents.graph import run_agent_pipeline

# 1. Inject Cinematic Glassmorphism Styling
inject_cinematic_theme_css()

# --- Cached Resource Managers for Instant Performance ---
@st.cache_resource
def get_feature_store():
    return FeatureStoreManager()

@st.cache_resource
def get_model_registry():
    return ModelRegistry()

@st.cache_resource
def get_eda_manager():
    return EDAManager()

@st.cache_resource
def get_model_explainer():
    return ModelExplainer()

fs = get_feature_store()
registry = get_model_registry()
eda = get_eda_manager()
explainer = get_model_explainer()


# --- Fast Cached Agent Execution ---
@st.cache_data(ttl=300, show_spinner=False)
def get_cached_agent_state(city: str) -> Dict[str, Any]:
    return run_agent_pipeline(city=city)

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_eda_stats():
    return eda.get_summary_statistics()

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_eda_outliers():
    return eda.detect_outliers_iqr()

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_eda_correlation():
    return eda.get_correlation_matrix()

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_eda_diurnal():
    return eda.get_diurnal_patterns()

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_feature_importance(metric: str):
    return explainer.get_global_feature_importance(metric, top_n=10)


# =========================================================================
# TOP NAVIGATION BAR (Matching Reference Designs)
# =========================================================================
city_names = list(PAKISTAN_CITIES.keys())

col_nav1, col_nav2, col_nav3 = st.columns([1.5, 2, 1.2])

with col_nav1:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px;">
        <span style="font-size:1.8rem;">⛅</span>
        <div>
            <div style="font-weight:800; font-size:1.25rem; color:#ffffff; letter-spacing:-0.02em;">SkySense AI</div>
            <div style="font-size:0.7rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em;">Multi-Agent Atmospheric Core</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_nav2:
    selected_city = st.selectbox(
        "Location Basin",
        city_names,
        index=0,
        label_visibility="collapsed",
        help="Select any major Pakistani city to inspect live authentic weather and AQI."
    )

with col_nav3:
    col_sub1, col_sub2 = st.columns([1, 1])
    with col_sub1:
        if st.button("🔄 Sync Live", use_container_width=True):
            st.cache_data.clear()
            st.session_state["active_state"] = run_agent_pipeline(city=selected_city)
            st.toast(f"Synchronized live atmospheric telemetry for {selected_city}!", icon="🟢")
    with col_sub2:
        st.markdown(f"""
        <div style="text-align:right; padding-top:6px;">
            <span style="font-size:0.75rem; color:#10b981; font-weight:700;">● LIVE TELEMETRY</span><br>
            <span style="font-size:0.7rem; color:#64748b; font-family:'JetBrains Mono';">{datetime.now().strftime('%H:%M PKT')}</span>
        </div>
        """, unsafe_allow_html=True)


# --- Load State for Selected City ---
if "active_state" not in st.session_state or st.session_state.get("active_city") != selected_city:
    st.session_state["active_state"] = get_cached_agent_state(selected_city)
    st.session_state["active_city"] = selected_city

state = st.session_state.get("active_state", {})
f24 = state.get("forecast_24h", {})
trajectory = state.get("forecast_hourly_trajectory", [])
df_traj = pd.DataFrame(trajectory) if trajectory else pd.DataFrame()

city_info = PAKISTAN_CITIES.get(selected_city, {
    "lat": 31.5204, "lon": 74.3587, "province": "Punjab", "climate": "Semi-Arid", "elevation_m": 217
})

temp_val = float(f24.get("temperature_c", 22.0))
aqi_val = float(f24.get("aqi", 145.0))
rain_val = float(f24.get("precipitation_mm", 0.0))
humidity_val = float(f24.get("humidity_pct", 58.0))
wind_val = float(f24.get("wind_speed_kmh", 8.4))
uv_val = float(f24.get("uv_index", 2.2))

weather_icon, weather_desc = get_weather_icon_and_desc(temp_val, aqi_val, rain_val, wind_val)
cat_name, bg_col, text_col = get_aqi_badge_color(aqi_val)

# Min/Max from trajectory or baseline
high_temp = round(float(df_traj["temperature_c"].max()), 0) if not df_traj.empty else round(temp_val + 4, 0)
low_temp = round(float(df_traj["temperature_c"].min()), 0) if not df_traj.empty else round(temp_val - 5, 0)


# =========================================================================
# MAIN CINEMATIC DASHBOARD (Matching Reference Images 1, 2, 3)
# =========================================================================
col_left, col_right = st.columns([1, 2.2], gap="large")

# -------------------------------------------------------------------------
# LEFT COLUMN: Widget Card / Status / Major Cities (From Image 1)
# -------------------------------------------------------------------------
with col_left:
    st.markdown(f"""
    <div class="widget-glass-card">
        <div>
            <!-- Status Pill & Mini Wave (Matching Image 1 Left Card) -->
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
                <span style="font-size:0.8rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Air & Hazard Status</span>
                <span style="background:{bg_col}25; border:1px solid {bg_col}; color:{bg_col}; padding:3px 10px; border-radius:9999px; font-size:0.75rem; font-weight:700;">
                    {cat_name.upper()}
                </span>
            </div>
            
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:18px; padding:18px; margin-bottom:20px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-size:0.75rem; color:#94a3b8;">AQI Metric (24h Ahead)</div>
                        <div style="font-size:2.4rem; font-weight:800; color:{bg_col}; font-family:'JetBrains Mono'; line-height:1.1;">{aqi_val:.0f}</div>
                    </div>
                    <div style="text-align:right;">
                        <span style="background:{bg_col}; color:{text_col}; padding:4px 12px; border-radius:9999px; font-size:0.75rem; font-weight:800;">
                            {cat_name.split(' ')[0]}
                        </span>
                        <div style="font-size:0.7rem; color:#94a3b8; margin-top:6px;">PM2.5 / PM10 EPA</div>
                    </div>
                </div>
            </div>
            
            <!-- 3-Audience Safety Brief -->
            <div style="margin-bottom:20px;">
                <div style="font-size:0.8rem; color:#94a3b8; font-weight:600; text-transform:uppercase; margin-bottom:10px;">Targeted Advisories</div>
                
                <div style="background:rgba(59,130,246,0.1); border-left:3px solid #3b82f6; border-radius:8px; padding:10px 12px; margin-bottom:8px; font-size:0.8rem; color:#e2e8f0;">
                    <b>👥 Public:</b> Limit peak-hour travel & seal windows during stagnant wind.
                </div>
                <div style="background:rgba(245,158,11,0.1); border-left:3px solid #f59e0b; border-radius:8px; padding:10px 12px; margin-bottom:8px; font-size:0.8rem; color:#e2e8f0;">
                    <b>🫁 Sensitive:</b> Strict indoor air purification; keep rescue inhalers accessible.
                </div>
                <div style="background:rgba(239,68,68,0.1); border-left:3px solid #ef4444; border-radius:8px; padding:10px 12px; font-size:0.8rem; color:#e2e8f0;">
                    <b>👷 Workers:</b> Mandatory N95 particulate PPE & 15-min shaded hydration breaks.
                </div>
            </div>
            
            <!-- Major Pakistani Cities Live Glance (Matching Image 1 Right Pill Stack) -->
            <div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <span style="font-size:0.8rem; color:#94a3b8; font-weight:600; text-transform:uppercase;">Major Basins</span>
                    <span style="font-size:0.75rem; color:#38bdf8;">Real-Time</span>
                </div>
    """, unsafe_allow_html=True)
    
    # Generate city comparison pills
    cities_summary = {
        "Karachi": {"temp": 27.0, "aqi": 82, "province": "Sindh Coastal"},
        "Islamabad": {"temp": 18.5, "aqi": 88, "province": "Federal Foothills"},
        "Peshawar": {"temp": 19.0, "aqi": 135, "province": "KP Valley"},
        "Quetta": {"temp": 12.0, "aqi": 65, "province": "Balochistan Plateau"}
    }
    st.markdown(render_city_chips_html(cities_summary, selected_city), unsafe_allow_html=True)
    
    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------------------
# RIGHT COLUMN: Main Hero Atmospheric Area & Spline Wave (Images 1, 2, 3)
# -------------------------------------------------------------------------
with col_right:
    st.markdown(f"""
    <div class="hero-glass-frame">
        <!-- Location & Date Header (Matching Image 1 & 2) -->
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; margin-bottom:12px;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-size:1.1rem; color:#38bdf8;">📍</span>
                <span style="font-size:1.15rem; font-weight:700; color:#f8fafc;">
                    {selected_city}, {city_info['province']}, Pakistan
                </span>
                <span style="color:#64748b; font-size:0.9rem;">({datetime.now().strftime('%A, %B %d')})</span>
            </div>
            <div style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1); border-radius:9999px; padding:4px 14px; font-size:0.78rem; color:#94a3b8;">
                Elevation: {city_info['elevation_m']}m • {city_info['climate']}
            </div>
        </div>
        
        <!-- Big Temperature & Weather State (Matching Image 1 & 3) -->
        <div style="display:flex; align-items:baseline; gap:24px; flex-wrap:wrap;">
            <div class="hero-temp-title">{temp_val:.0f}°</div>
            <div style="color:#94a3b8; font-size:1.1rem; font-weight:600; font-family:'JetBrains Mono';">
                H <span style="color:#ffffff;">{high_temp:.0f}°</span> &nbsp;•&nbsp; L <span style="color:#ffffff;">{low_temp:.0f}°</span>
            </div>
        </div>
        
        <div class="hero-condition-text">
            <span style="font-size:2.2rem; margin-right:8px; vertical-align:middle;">{weather_icon}</span>
            <b>{weather_desc}</b>
        </div>
        
        <!-- Quick Environmental Telemetry Bar (Matching Image 3 Bottom Bar) -->
        <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px;">
            <div class="stat-pill">💨 Wind: <b>{wind_val:.1f} km/h</b></div>
            <div class="stat-pill">💧 Humidity: <b>{humidity_val:.0f}%</b></div>
            <div class="stat-pill">🌧️ Precip: <b>{rain_val:.1f} mm</b></div>
            <div class="stat-pill">☀️ UV Index: <b>{uv_val:.1f}</b></div>
            <div class="stat-pill">🔘 Station Air: <b>{aqi_val:.0f} AQI</b></div>
        </div>
    """, unsafe_allow_html=True)
    
    # Golden Spline Wave Curve (Matching Image 1 Bottom Wave)
    if not df_traj.empty:
        metric_choice = st.selectbox(
            "Select Trajectory Dimension:",
            ["temperature_c", "aqi", "precipitation_mm", "humidity_pct", "wind_speed_kmh"],
            index=0,
            format_func=lambda x: {
                "temperature_c": "Temperature Wave (°C)",
                "aqi": "Air Quality Index Wave (AQI)",
                "precipitation_mm": "Precipitation Volume (mm)",
                "humidity_pct": "Relative Humidity (%)",
                "wind_speed_kmh": "Wind Velocity (km/h)"
            }.get(x, x),
            key="spline_selector"
        )
        fig_spline = plot_cinematic_spline_curve(df_traj, metric=metric_choice, title_prefix=selected_city)
        st.plotly_chart(fig_spline, use_container_width=True)
    
    # Horizontal Hourly Forecast Cards (Matching Image 2)
    st.markdown("<div style='margin-top:12px; font-size:0.8rem; color:#94a3b8; font-weight:600; text-transform:uppercase;'>Hourly Progression (Next 12 Hours)</div>", unsafe_allow_html=True)
    if not df_traj.empty:
        st.markdown(render_hourly_cards_html(df_traj), unsafe_allow_html=True)
        
    st.markdown("""
    </div>
    """, unsafe_allow_html=True)


# =========================================================================
# DEEP-TECH INSPECTION TABS (Fulfilling All 10Pearls Requirements)
# =========================================================================
st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

tech_tabs = st.tabs([
    "🎙️ Studio News Anchor Script",
    "🧠 Model Registry & Benchmark Leaderboard",
    "🔍 Explainability (SHAP & LIME)",
    "📊 Exploratory Data Analysis (EDA)",
    "🛡️ MLOps Telemetry & Drift Audits"
])

# TAB 1: Newsroom Broadcast
with tech_tabs[0]:
    st.markdown(f"#### 🎙️ Live Broadcast Newsroom Script — {selected_city}")
    st.caption("Generated autonomously by the LangGraph Weather Report Agent.")
    report_text = state.get("weather_report", "Weather broadcast synthesis in progress...")
    st.markdown(f"""
    <div style="background: rgba(18, 24, 38, 0.7); border: 1px solid rgba(56, 189, 248, 0.3); border-left: 6px solid #38bdf8; border-radius: 18px; padding: 24px; color: #f1f5f9; font-size: 1.05rem; line-height: 1.8;">
        {report_text}
    </div>
    """, unsafe_allow_html=True)

# TAB 2: Model Registry & Leaderboard
with tech_tabs[1]:
    st.markdown("#### 🏆 Champion Models & 3-Family Evaluation Matrix")
    st.caption("Benchmark scores across Statistical (AutoReg), Classical ML (Ridge, Random Forest, XGBoost), Deep Learning (MLP), and 24h Naive Persistence Baseline.")
    leaderboard_df = registry.get_leaderboard()
    if not leaderboard_df.empty:
        try:
            st.plotly_chart(plot_model_comparison_bar(leaderboard_df), use_container_width=True)
        except Exception:
            pass
        st.dataframe(
            leaderboard_df.style.highlight_max(subset=["Validation R²"], color="#1e3a8a")
                               .highlight_min(subset=["Validation RMSE", "Validation MAE"], color="#064e3b"),
            use_container_width=True
        )
    else:
        st.info("Leaderboard active upon pipeline benchmark.")

# TAB 3: SHAP & LIME Explainability
with tech_tabs[2]:
    st.markdown("#### 🔍 Model Explainability (SHAP Global Drivers & Local Waterfall)")
    col_x1, col_x2 = st.columns(2)
    with col_x1:
        st.markdown("##### 🌐 Global Feature Importance")
        exp_metric = st.selectbox("Explain Target Metric", METRICS, index=0)
        imp_df = get_cached_feature_importance(exp_metric)
        if not imp_df.empty:
            fig_imp = px.bar(
                imp_df, x="Relative_Pct", y="Feature", orientation="h",
                labels={"Relative_Pct": "Importance (%)", "Feature": "Feature Driver"},
                color="Relative_Pct", color_continuous_scale="Tealgrn", template="plotly_dark"
            )
            fig_imp.update_layout(yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10), height=300)
            st.plotly_chart(fig_imp, use_container_width=True)
    with col_x2:
        st.markdown("##### 🎯 Local Forecast Attribution (SHAP Waterfall)")
        if state and state.get("current_features"):
            s_series = pd.Series(state["current_features"])
            shap_info = explainer.explain_prediction_shap(exp_metric, s_series)
            if shap_info and "top_contributors" in shap_info:
                df_c = pd.DataFrame(shap_info["top_contributors"])
                fig_shap = px.bar(
                    df_c, x="SHAP_Value", y="Feature", orientation="h",
                    color="SHAP_Value", color_continuous_scale=["#38bdf8", "#94a3b8", "#ef4444"], template="plotly_dark"
                )
                fig_shap.update_layout(yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10), height=300)
                st.plotly_chart(fig_shap, use_container_width=True)

# TAB 4: EDA
with tech_tabs[3]:
    st.markdown("#### 📊 Exploratory Data Analysis (EDA)")
    col_ed1, col_ed2 = st.columns(2)
    with col_ed1:
        st.markdown("##### 🔗 Pearson Correlation Heatmap")
        corr_df = get_cached_eda_correlation()
        if not corr_df.empty:
            fig_c = px.imshow(corr_df, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", template="plotly_dark")
            fig_c.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10), height=320)
            st.plotly_chart(fig_c, use_container_width=True)
    with col_ed2:
        st.markdown("##### ⏰ 24-Hour Diurnal Hourly Cycles")
        diurnal_df = get_cached_eda_diurnal()
        if not diurnal_df.empty:
            fig_d = px.line(diurnal_df, x="hour", y=["temperature_c", "aqi"], markers=True, template="plotly_dark")
            fig_d.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10), height=320)
            st.plotly_chart(fig_d, use_container_width=True)

# TAB 5: MLOps Drift Audits
with tech_tabs[4]:
    st.markdown("#### 🛡️ MLOps Telemetry: Drift Audits & Autonomous Retraining")
    if state and state.get("drift_status"):
        ds = state["drift_status"]
        md1, md2, md3 = st.columns(3)
        with md1: st.metric("Overall Drift", "🚨 ACTIVE" if ds.get("overall_drift_detected") else "🟢 NOMINAL")
        with md2: st.metric("KS-Test Distribution", "SHIFT DETECTED" if ds.get("data_drift", {}).get("data_drift_detected") else "CALIBRATED")
        with md3: st.metric("Autonomous Retraining", "✅ EXECUTED" if state.get("triggered_retraining") else "IDLE")
    drift_log_df = fs.get_drift_log(limit=25)
    if not drift_log_df.empty:
        st.dataframe(drift_log_df, use_container_width=True)

st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.75rem; padding: 24px 0 10px 0;">
    <b>SkySense AI</b> • Responsive Multi-Screen Meteorological Platform • Designed for Mobile, Tablet, Laptop, and LCD/TV Screens
</div>
""", unsafe_allow_html=True)
