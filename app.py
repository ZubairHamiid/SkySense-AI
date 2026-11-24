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
    plot_model_comparison_bar,
    render_left_panel_html,
    render_hero_header_html
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

@st.cache_data(ttl=1800, show_spinner=False)
def get_cached_eda_stats():
    return eda.get_summary_statistics()

@st.cache_data(ttl=1800, show_spinner=False)
def get_cached_eda_outliers():
    return eda.detect_outliers_iqr()

@st.cache_data(ttl=1800, show_spinner=False)
def get_cached_eda_correlation():
    return eda.get_correlation_matrix()

@st.cache_data(ttl=1800, show_spinner=False)
def get_cached_eda_diurnal():
    return eda.get_diurnal_patterns()

@st.cache_data(ttl=1800, show_spinner=False)
def get_cached_feature_importance(metric: str):
    return explainer.get_global_feature_importance(metric, top_n=10)

@st.cache_data(ttl=300, show_spinner=False)
def get_city_quick_stats(city: str) -> Dict[str, float]:
    """Returns latest temp and AQI for a city directly from parquet - no agent pipeline needed."""
    try:
        df = fs.get_features(city=city)
        if df.empty:
            return {"temp": 22.0, "aqi": 100.0}
        last = df.sort_values("time").iloc[-1]
        return {
            "temp": float(last.get("temperature_c", 22.0)),
            "aqi": float(last.get("aqi", 100.0))
        }
    except Exception:
        return {"temp": 22.0, "aqi": 100.0}


# =========================================================================
# TOP NAVIGATION BAR (Matching Reference Designs)
# =========================================================================
city_names = list(PAKISTAN_CITIES.keys())

col_nav1, col_nav2, col_nav3 = st.columns([1.5, 2, 1.2])

with col_nav1:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;">'
        '<span style="font-size:1.8rem;">⛅</span>'
        '<div>'
        '<div style="font-weight:800;font-size:1.25rem;color:#ffffff;letter-spacing:-0.02em;">SkySense AI</div>'
        '<div style="font-size:0.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.05em;">Multi-Agent Atmospheric Core</div>'
        '</div></div>',
        unsafe_allow_html=True
    )

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
        now_pkt = datetime.now().strftime('%H:%M PKT')
        st.markdown(
            f'<div style="text-align:right;padding-top:6px;">'
            f'<span style="font-size:0.75rem;color:#10b981;font-weight:700;">● LIVE TELEMETRY</span><br>'
            f'<span style="font-size:0.7rem;color:#64748b;font-family:\'JetBrains Mono\';">{now_pkt}</span>'
            f'</div>',
            unsafe_allow_html=True
        )


# --- Load State for Selected City ---
if "active_state" not in st.session_state or st.session_state.get("active_city") != selected_city:
    with st.spinner(f"Loading atmospheric data for {selected_city}..."):
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

# --- Live city summary: real data from city parquets (fast, no pipeline) ---
_city_meta = {
    "Karachi": "Sindh Coastal", "Islamabad": "Federal Foothills",
    "Peshawar": "KP Valley", "Quetta": "Balochistan Plateau",
    "Multan": "Punjab South", "Faisalabad": "Central Punjab",
    "Sargodha": "Punjab Agri Belt", "Lahore": "Punjab Capital",
}
_sidebar_cities = [c for c in ["Karachi", "Islamabad", "Peshawar", "Quetta"] if c != selected_city][:4]
cities_summary = {}
for c in _sidebar_cities:
    _stats = get_city_quick_stats(c)
    cities_summary[c] = {
        "temp": _stats["temp"],
        "aqi": _stats["aqi"],
        "province": _city_meta.get(c, "Pakistan")
    }


# =========================================================================
# MAIN CINEMATIC DASHBOARD (Matching Reference Images 1, 2, 3)
# =========================================================================
col_left, col_right = st.columns([1, 2.2], gap="large")

# -------------------------------------------------------------------------
# LEFT COLUMN: Widget Card / Status / Major Cities (From Image 1)
# -------------------------------------------------------------------------
with col_left:
    st.markdown(render_left_panel_html(selected_city, aqi_val, cat_name, bg_col, text_col, cities_summary), unsafe_allow_html=True)


# -------------------------------------------------------------------------
# RIGHT COLUMN: Main Hero Atmospheric Area & Spline Wave (Images 1, 2, 3)
# -------------------------------------------------------------------------
with col_right:
    st.markdown(render_hero_header_html(selected_city, city_info, temp_val, high_temp, low_temp, weather_icon, weather_desc, wind_val, humidity_val, rain_val, uv_val, aqi_val), unsafe_allow_html=True)
    
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
    st.markdown('<div style="margin-top:12px;font-size:0.8rem;color:#94a3b8;font-weight:600;text-transform:uppercase;">Hourly Progression (Next 12 Hours)</div>', unsafe_allow_html=True)
    if not df_traj.empty:
        st.markdown(render_hourly_cards_html(df_traj), unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)


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
    st.markdown(
        f'<div style="background:rgba(18,24,38,0.7);border:1px solid rgba(56,189,248,0.3);border-left:6px solid #38bdf8;border-radius:18px;padding:24px;color:#f1f5f9;font-size:1.05rem;line-height:1.8;">'
        f'{report_text}'
        f'</div>',
        unsafe_allow_html=True
    )

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
            try:
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
            except Exception as e:
                st.info(f"SHAP explanation unavailable: {e}")

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

st.markdown(
    '<div style="text-align:center;color:#64748b;font-size:0.75rem;padding:24px 0 10px 0;">'
    '<b>SkySense AI</b> • Responsive Multi-Screen Meteorological Platform • Designed for Mobile, Tablet, Laptop, and LCD/TV Screens'
    '</div>',
    unsafe_allow_html=True
)
