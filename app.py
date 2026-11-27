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


# --- Fast Cached Execution ---
@st.cache_data(ttl=300, show_spinner=False)
def get_cached_agent_state(city: str) -> Dict[str, Any]:
    return run_agent_pipeline(city=city, allow_retrain=False)

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

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_shap_local(metric: str, city: str, features_items: tuple):
    s_series = pd.Series(dict(features_items))
    return explainer.explain_prediction_shap(metric, s_series)

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_lime_local(metric: str, city: str, features_items: tuple):
    s_series = pd.Series(dict(features_items))
    return explainer.explain_prediction_lime(metric, s_series)

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
# TOP NAVIGATION BAR
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
            st.session_state["active_state"] = run_agent_pipeline(city=selected_city, allow_retrain=False)
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
cur_feats = state.get("current_features", {})
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
wind_dir_deg = float(cur_feats.get("wind_direction_deg", 180.0) or 180.0)
rain_prob_pct = float(cur_feats.get("precipitation_probability_pct", min(100.0, rain_val * 25.0)) or 0.0)

weather_icon, weather_desc = get_weather_icon_and_desc(temp_val, aqi_val, rain_val, wind_val)
cat_name, bg_col, text_col = get_aqi_badge_color(aqi_val)

# Min/Max from trajectory or baseline
high_temp = round(float(df_traj["temperature_c"].max()), 0) if not df_traj.empty else round(temp_val + 4, 0)
low_temp = round(float(df_traj["temperature_c"].min()), 0) if not df_traj.empty else round(temp_val - 5, 0)

# --- Live city summary from real city parquets ---
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
# MAIN CINEMATIC DASHBOARD
# =========================================================================
col_left, col_right = st.columns([1, 2.2], gap="large")

with col_left:
    st.markdown(
        render_left_panel_html(
            selected_city, aqi_val, cat_name, bg_col, text_col,
            cities_summary, forecast_24h=f24, alerts=state.get("alerts", [])
        ),
        unsafe_allow_html=True
    )

with col_right:
    st.markdown(
        render_hero_header_html(
            selected_city, city_info, temp_val, high_temp, low_temp,
            weather_icon, weather_desc, wind_val, humidity_val, rain_val,
            uv_val, aqi_val, wind_dir_deg=wind_dir_deg, rain_prob_pct=rain_prob_pct
        ),
        unsafe_allow_html=True
    )

    if not df_traj.empty:
        metric_choice = st.selectbox(
            "Select Trajectory Dimension:",
            ["temperature_c", "aqi", "precipitation_mm", "humidity_pct", "uv_index", "wind_speed_kmh"],
            index=0,
            format_func=lambda x: {
                "temperature_c": "Temperature Wave (°C)",
                "aqi": "Air Quality Index Wave (AQI)",
                "precipitation_mm": "Precipitation Volume (mm)",
                "humidity_pct": "Relative Humidity (%)",
                "uv_index": "Solar UV Index",
                "wind_speed_kmh": "Wind Velocity (km/h)"
            }.get(x, x),
            key="spline_selector"
        )
        fig_spline = plot_cinematic_spline_curve(df_traj, metric=metric_choice, title_prefix=selected_city)
        st.plotly_chart(fig_spline, use_container_width=True)

    st.markdown('<div style="margin-top:12px;font-size:0.8rem;color:#94a3b8;font-weight:600;text-transform:uppercase;">Hourly Progression (Next 12 Hours)</div>', unsafe_allow_html=True)
    if not df_traj.empty:
        st.markdown(render_hourly_cards_html(df_traj), unsafe_allow_html=True)


# =========================================================================
# DEEP-TECH INSPECTION TABS (Fulfilling All 10Pearls Specification Panels)
# =========================================================================
st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

tech_tabs = st.tabs([
    "🤖 Agent Activity & Alerts",
    "🎙️ Studio News Anchor Script",
    "🧠 Model Registry & Leaderboard",
    "🔍 Explainability (SHAP & LIME)",
    "📊 Exploratory Data Analysis (EDA)",
    "🛡️ Drift Audits & Failure Logs"
])

# TAB 1: Agent Activity Panel & Active Multi-Audience Alerts
with tech_tabs[0]:
    st.markdown(f"#### 🤖 LangGraph Agent Activity & Active Hazards — {selected_city}")
    alerts = state.get("alerts", [])
    champs_used = state.get("champion_models_used", {})

    ag1, ag2, ag3, ag4 = st.columns(4)
    with ag1:
        st.metric("Forecast Agent", f"{len(f24)} / 6 Metrics")
    with ag2:
        st.metric("Alert Agent", f"🚨 {len(alerts)} Active" if alerts else "🟢 Nominal")
    with ag3:
        st.metric("Weather Report Agent", "✅ Broadcast Ready" if state.get("weather_report") else "Pending")
    with ag4:
        ds_top = state.get("drift_status", {})
        st.metric("Drift Agent", "🚨 Drift Flagged" if ds_top.get("overall_drift_detected") else "🟢 Calibrated")

    if alerts:
        st.markdown("##### 🚨 Triggered Meteorological Hazards & 3-Audience Advisories")
        for idx_a, alert in enumerate(alerts):
            with st.expander(f"[{alert.get('level', 'ALERT')}] {alert.get('title')} — {alert.get('value')} {alert.get('unit')} (Limit: {alert.get('threshold')})", expanded=(idx_a == 0)):
                st.markdown(alert.get("detailed_advisory", "Advisory generated by AlertAgent."))
    else:
        st.success(f"🟢 No hazardous conditions triggered for {selected_city} over the next 24 hours.")

    col_led1, col_led2 = st.columns([1.2, 1])
    with col_led1:
        st.markdown("##### 📒 Forecast Ledger (Recent Logged Predictions)")
        ledger_df = fs.get_forecast_ledger(limit=25)
        if not ledger_df.empty:
            st.dataframe(ledger_df, use_container_width=True, height=260)
        else:
            st.info("Forecast ledger will populate on agent execution.")
    with col_led2:
        st.markdown("##### 🧬 Active Champion Models Dispatched")
        if champs_used:
            df_models = pd.DataFrame([
                {"Metric": k, "Champion Dispatched": v, "24h Forecast": f24.get(k), "24h Baseline": state.get("baseline_forecast", {}).get(k)}
                for k, v in champs_used.items()
            ])
            st.dataframe(df_models, use_container_width=True, height=260)

# TAB 2: Newsroom Broadcast
with tech_tabs[1]:
    st.markdown(f"#### 🎙️ Live Broadcast Newsroom Script — {selected_city}")
    st.caption("Generated autonomously by the LangGraph Weather Report Agent (Groq LLM + deterministic broadcast fallback).")
    report_text = state.get("weather_report", "Weather broadcast synthesis in progress...")
    st.markdown(
        f'<div style="background:rgba(18,24,38,0.7);border:1px solid rgba(56,189,248,0.3);border-left:6px solid #38bdf8;border-radius:18px;padding:24px;color:#f1f5f9;font-size:1.05rem;line-height:1.8;white-space:pre-wrap;">'
        f'{report_text}'
        f'</div>',
        unsafe_allow_html=True
    )

# TAB 3: Model Registry & Leaderboard
with tech_tabs[2]:
    st.markdown("#### 🏆 Champion Models & 3-Family Evaluation Matrix")
    st.caption("Zero-leakage global 24h-embargoed benchmarks across Statistical (SARIMAX), Classical ML (Ridge, Random Forest, XGBoost), Deep Learning (PyTorch LSTM), and 24h Naive Persistence Baseline.")
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

# TAB 4: SHAP & LIME Explainability
with tech_tabs[3]:
    st.markdown("#### 🔍 Model Explainability (Global Importance, SHAP Waterfall & LIME)")
    exp_metric = st.selectbox("Explain Target Metric", METRICS, index=0, key="exp_metric_sel")
    col_x1, col_x2, col_x3 = st.columns(3)

    # Prepare hashable feature tuple for caching
    numeric_feat_items = tuple(sorted(
        (k, float(v)) for k, v in cur_feats.items()
        if isinstance(v, (int, float, np.number)) and not pd.isna(v)
    ))

    with col_x1:
        st.markdown("##### 🌐 Global Feature Importance")
        imp_df = get_cached_feature_importance(exp_metric)
        if not imp_df.empty:
            fig_imp = px.bar(
                imp_df, x="Relative_Pct", y="Feature", orientation="h",
                labels={"Relative_Pct": "Importance (%)", "Feature": "Driver"},
                color="Relative_Pct", color_continuous_scale="Tealgrn", template="plotly_dark"
            )
            fig_imp.update_layout(yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10), height=300)
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Global importance available for tree/linear champions.")

    with col_x2:
        st.markdown("##### 🎯 Local SHAP Attribution")
        if numeric_feat_items:
            try:
                shap_info = get_cached_shap_local(exp_metric, selected_city, numeric_feat_items)
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

    with col_x3:
        st.markdown("##### 🧪 Local LIME Attribution")
        if numeric_feat_items:
            try:
                lime_list = get_cached_lime_local(exp_metric, selected_city, numeric_feat_items)
                if lime_list:
                    df_lime = pd.DataFrame(lime_list, columns=["Rule", "Weight"])
                    fig_lime = px.bar(
                        df_lime, x="Weight", y="Rule", orientation="h",
                        color="Weight", color_continuous_scale=["#38bdf8", "#94a3b8", "#f59e0b"], template="plotly_dark"
                    )
                    fig_lime.update_layout(yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10), height=300)
                    st.plotly_chart(fig_lime, use_container_width=True)
                else:
                    st.info("LIME attribution unavailable for current metric.")
            except Exception as e:
                st.info(f"LIME explanation unavailable: {e}")

# TAB 5: Comprehensive EDA
with tech_tabs[4]:
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

    col_ed3, col_ed4 = st.columns(2)
    with col_ed3:
        st.markdown("##### 📋 Summary Statistics & Skewness")
        stats_df = get_cached_eda_stats()
        if not stats_df.empty:
            st.dataframe(stats_df, use_container_width=True)
    with col_ed4:
        st.markdown("##### 📐 IQR Outlier Audit")
        outliers_df = get_cached_eda_outliers()
        if not outliers_df.empty:
            st.dataframe(outliers_df, use_container_width=True)

# TAB 6: MLOps Drift Audits, Agent Call Log & Failure Log
with tech_tabs[5]:
    st.markdown("#### 🛡️ MLOps Telemetry: Drift Audits, Agent Call Logs & Failure Tracking")
    if state and state.get("drift_status"):
        ds = state["drift_status"]
        md1, md2, md3, md4 = st.columns(4)
        with md1: st.metric("Overall Drift", "🚨 ACTIVE" if ds.get("overall_drift_detected") else "🟢 NOMINAL")
        with md2: st.metric("Data Drift (KS/PSI)", "SHIFT DETECTED" if ds.get("data_drift", {}).get("data_drift_detected") else "CALIBRATED")
        with md3: st.metric("Model Drift (MAE)", "DEGRADED" if ds.get("model_drift", {}).get("model_drift_detected") else "STABLE")
        with md4: st.metric("Retraining Dispatch", "✅ TRIGGERED" if state.get("triggered_retraining") else "IDLE")

        # Per-metric KS-test & PSI breakdown
        dd_rep = ds.get("data_drift", {}).get("metric_reports", {})
        md_rep = ds.get("model_drift", {}).get("metric_reports", {})
        if dd_rep or md_rep:
            drift_rows = []
            for m in METRICS:
                d_m = dd_rep.get(m, {})
                m_m = md_rep.get(m, {})
                drift_rows.append({
                    "Metric": m,
                    "KS Stat": d_m.get("ks_statistic"),
                    "KS p-value": d_m.get("p_value"),
                    "PSI": d_m.get("psi"),
                    "Data Drift?": "🚨 YES" if d_m.get("drift_detected") else "🟢 NO",
                    "Recent MAE": m_m.get("recent_mae"),
                    "Val MAE": m_m.get("validation_mae"),
                    "Model Drift?": "🚨 YES" if m_m.get("drift_detected") else "🟢 NO",
                })
            st.markdown("##### 📉 Per-Metric Data Drift (KS / PSI) & Model Drift (MAE)")
            st.dataframe(pd.DataFrame(drift_rows), use_container_width=True)

    col_log1, col_log2 = st.columns(2)
    with col_log1:
        st.markdown("##### 📡 Drift Event Audit Log")
        drift_log_df = fs.get_drift_log(limit=25)
        if not drift_log_df.empty:
            st.dataframe(drift_log_df, use_container_width=True, height=250)
        else:
            st.info("No drift events recorded yet.")

    with col_log2:
        st.markdown("##### 📞 LLM Agent Call Log (Prompt, Latency & Status)")
        call_log_df = fs.get_agent_call_log(limit=25)
        if not call_log_df.empty:
            st.dataframe(call_log_df[["timestamp", "agent_name", "model_used", "latency_ms", "status", "error_message"]], use_container_width=True, height=250)
        else:
            st.info("No LLM calls logged yet.")

    st.markdown("##### ⚠️ Failed Agent Calls / Fallback Audit Log")
    failed_df = fs.get_failed_agent_calls()
    if not failed_df.empty:
        st.dataframe(failed_df[["timestamp", "agent_name", "model_used", "latency_ms", "status", "error_message"]].head(20), use_container_width=True, height=200)
    else:
        st.success("🟢 Zero failed LLM API calls recorded.")

st.markdown(
    '<div style="text-align:center;color:#64748b;font-size:0.75rem;padding:24px 0 10px 0;">'
    '<b>SkySense AI</b> • Responsive Multi-Screen Meteorological Platform • Built with LangGraph, PyTorch LSTM, SARIMAX, XGBoost, SHAP & LIME'
    '</div>',
    unsafe_allow_html=True
)
