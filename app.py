import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="SkySense AI | 10Pearls SHINE Cohort 10",
    page_icon="⛅",
    layout="wide",
    initial_sidebar_state="expanded"
)

from config.settings import DEFAULT_CITY, METRICS
from src.dashboard.ui_components import inject_custom_css, get_aqi_badge_color, plot_metric_forecast_curve
from src.pipeline.feature_store import FeatureStoreManager
from src.pipeline.model_registry import ModelRegistry
from src.pipeline.eda import EDAManager
from src.pipeline.explainability import ModelExplainer
from src.agents.graph import run_agent_pipeline

# Inject Custom CSS
inject_custom_css()

# Initialize managers
fs = FeatureStoreManager()
registry = ModelRegistry()
eda = EDAManager()
explainer = ModelExplainer()

# --- Sidebar Controls ---
st.sidebar.image("https://img.icons8.com/clouds/200/weather.png", width=110)
st.sidebar.title("SkySense AI 🛰️")
st.sidebar.caption("**10Pearls SHINE Internship Cohort 10**")
st.sidebar.markdown("---")

selected_city = st.sidebar.selectbox("Select Target City", ["Lahore", "Karachi", "Islamabad", "Peshawar", "Quetta"], index=0)
st.sidebar.info(f"📍 **Monitoring Basin**: {selected_city}\n\n⏱️ **Forecast Horizon**: 24 Hours Ahead\n\n🤖 **Reasoning Core**: LangGraph Multi-Agent")

if st.sidebar.button("⚡ Run Live Agent Cycle", use_container_width=True, type="primary"):
    with st.spinner(f"Executing LangGraph Agents for {selected_city}..."):
        st.session_state["agent_state"] = run_agent_pipeline(city=selected_city)
        st.success("Agentic reasoning cycle completed!")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Agent Architecture:**
- 📡 Ingestion Pipeline (Hourly)
- 🔮 Forecast Agent (24h Ahead)
- 🚨 Alert Agent (Safety Advisories)
- 🎙️ Weather Report Agent (News Anchor)
- 🛡️ Drift Agent (Data & Model Drift)
""")

# --- Main App Execution ---
# Check if state exists in session_state, or load from latest Feature Store / run agent cycle
if "agent_state" not in st.session_state:
    with st.spinner("Initializing AgentWeather system..."):
        try:
            st.session_state["agent_state"] = run_agent_pipeline(city=selected_city)
        except Exception as e:
            st.warning(f"Feature store initializing or awaiting backfill ({e}).")
            st.session_state["agent_state"] = None

state = st.session_state.get("agent_state")

# Header Section
st.markdown(f"# ⛅ AgentWeather: Agentic Multi-Metric Weather & AQI Forecasting")
st.markdown(f"**Target City:** `{selected_city}` | **System Status:** 🟢 Active | **Last Reasoning Cycle:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")

# --- Top Key Metrics Row ---
if state and state.get("forecast_24h"):
    f24 = state["forecast_24h"]
    aqi_val = f24.get("aqi", 0.0)
    cat_name, bg_col, text_col = get_aqi_badge_color(aqi_val)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#94a3b8; font-size:0.85rem;">AQI (24h Ahead)</span>
            <h2 style="margin:4px 0; color:{bg_col};">{aqi_val:.0f}</h2>
            <span class="stat-badge" style="background:{bg_col}; color:{text_col}; font-size:0.7rem;">{cat_name.split(' ')[0]}</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#94a3b8; font-size:0.85rem;">Temperature</span>
            <h2 style="margin:4px 0; color:#ef4444;">{f24.get('temperature_c', 0.0):.1f}°C</h2>
            <span style="color:#94a3b8; font-size:0.75rem;">Ambient Forecast</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#94a3b8; font-size:0.85rem;">Precipitation</span>
            <h2 style="margin:4px 0; color:#38bdf8;">{f24.get('precipitation_mm', 0.0):.1f} mm</h2>
            <span style="color:#94a3b8; font-size:0.75rem;">Rain Volume</span>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#94a3b8; font-size:0.85rem;">Humidity</span>
            <h2 style="margin:4px 0; color:#06b6d4;">{f24.get('humidity_pct', 0.0):.0f}%</h2>
            <span style="color:#94a3b8; font-size:0.75rem;">Moisture Saturation</span>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#94a3b8; font-size:0.85rem;">UV Index</span>
            <h2 style="margin:4px 0; color:#eab308;">{f24.get('uv_index', 0.0):.1f}</h2>
            <span style="color:#94a3b8; font-size:0.75rem;">Solar Radiation</span>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown(f"""
        <div class="metric-card">
            <span style="color:#94a3b8; font-size:0.85rem;">Wind Speed</span>
            <h2 style="margin:4px 0; color:#10b981;">{f24.get('wind_speed_kmh', 0.0):.1f} km/h</h2>
            <span style="color:#94a3b8; font-size:0.75rem;">Surface Wind</span>
        </div>
        """, unsafe_allow_html=True)

# --- Navigation Tabs ---
tabs = st.tabs([
    "🎙️ Weather Report Broadcast",
    "📈 24-Hour Forecasts",
    "🚨 Agent Activity & Alerts",
    "🧠 Model Leaderboard & Explainability",
    "🔍 Exploratory Data Analysis (EDA)",
    "🛡️ MLOps, Drift & Call Audits"
])

# =========================================================================
# TAB 1: News-Style Weather Report Broadcast
# =========================================================================
with tabs[0]:
    st.subheader("🎙️ Newsroom Weather Report Broadcast")
    if state and state.get("weather_report"):
        st.markdown(f"""
        <div class="broadcast-box">
            {state['weather_report'].replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Weather report broadcast will appear here once the agent reasoning cycle completes.")
        
    st.markdown("### ⚠️ Active Safety Bulletins & Alerts")
    if state and state.get("alerts"):
        for alert in state["alerts"]:
            st.error(f"**{alert['title']}** — Predicted Value: `{alert['value']} {alert['unit']}` (Safe Threshold: `{alert['threshold']}`)")
            if "detailed_advisory" in alert:
                with st.expander("View Tailored Action Plan (Public, Sensitive, Workers)", expanded=True):
                    st.markdown(alert["detailed_advisory"])
    else:
        st.success("🟢 **No Active Hazards Triggered**: All 6 predicted meteorological and pollutant metrics are within safe environmental limits.")

# =========================================================================
# TAB 2: 24-Hour Forecasts (All Six Metrics)
# =========================================================================
with tabs[1]:
    st.subheader("📈 24-Hour Multi-Metric Forecasting Trajectories")
    st.caption("Interactive hourly step predictions for the next 24 hours compared against the Naive 24-Hour Persistence Baseline.")
    
    if state and state.get("forecast_hourly_trajectory"):
        df_traj = pd.DataFrame(state["forecast_hourly_trajectory"])
        baselines = state.get("baseline_forecast", {})
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.plotly_chart(plot_metric_forecast_curve(df_traj, "aqi", baselines.get("aqi", 75.0)), use_container_width=True)
            st.plotly_chart(plot_metric_forecast_curve(df_traj, "temperature_c", baselines.get("temperature_c", 25.0)), use_container_width=True)
            st.plotly_chart(plot_metric_forecast_curve(df_traj, "precipitation_mm", baselines.get("precipitation_mm", 0.0)), use_container_width=True)
        with col_m2:
            st.plotly_chart(plot_metric_forecast_curve(df_traj, "humidity_pct", baselines.get("humidity_pct", 50.0)), use_container_width=True)
            st.plotly_chart(plot_metric_forecast_curve(df_traj, "uv_index", baselines.get("uv_index", 4.0)), use_container_width=True)
            st.plotly_chart(plot_metric_forecast_curve(df_traj, "wind_speed_kmh", baselines.get("wind_speed_kmh", 12.0)), use_container_width=True)
            
        with st.expander("📄 View Hourly Step Forecast Data Table"):
            st.dataframe(df_traj, use_container_width=True)
    else:
        st.warning("No trajectory data currently available. Please click 'Run Live Agent Cycle' in the sidebar.")

# =========================================================================
# TAB 3: Agent Activity & Alerts Panel
# =========================================================================
with tabs[2]:
    st.subheader("🚨 Agent Activity & Reasoning Panel")
    st.caption("Inspect live decision-making, hazard rule triggers, and role-specific safety advisories.")
    
    col_a1, col_a2 = st.columns([1, 1])
    
    with col_a1:
        st.markdown("#### 🔄 LangGraph Reasoning Flow")
        st.markdown("""
        ```mermaid
        graph TD
            A[Start Ingestion & Feature Sync] --> B[Forecast Agent]
            B -->|24h Multi-Metric Predictions| C[Alert Agent]
            C -->|Evaluates Fixed Hazard Rules| D{Hazard Triggered?}
            D -->|Yes| E[Author Structured 3-Audience Advisory]
            D -->|No| F[Flag Nominal Conditions]
            E --> G[Weather Report Agent]
            F --> G[Weather Report Agent]
            G -->|Synthesizes Broadcast Script| H[Drift Agent]
            H -->|Data Drift & Model Drift Audit| I{Drift > Threshold?}
            I -->|Yes| J[Trigger Automated Retraining Run]
            I -->|No| K[Log Nominal State]
            J --> L[End Cycle]
            K --> L[End Cycle]
        ```
        """)
        
        st.markdown("#### 🎯 Active Champion Models Employed")
        if state and state.get("champion_models_used"):
            st.json(state["champion_models_used"])

    with col_a2:
        st.markdown("#### 🛡️ Role-Specific Hazard Advisories")
        if state and state.get("alerts"):
            for idx, alert in enumerate(state["alerts"]):
                st.markdown(f"##### Hazard #{idx+1}: {alert['title']}")
                st.markdown(f"""
                <div class="alert-card-public">
                    <b>👥 General Public Guidance:</b><br>
                    Limit non-essential commute during peak hazard hours. Keep windows secured and run indoor filters.
                </div>
                <div class="alert-card-sensitive">
                    <b>🫁 Sensitive Groups Guidance (Children, Asthmatics, Elderly):</b><br>
                    Stay strictly in purified indoor air. Keep prescribed respiratory bronchodilators accessible. Wear N95 masks if transit is unavoidable.
                </div>
                <div class="alert-card-workers">
                    <b>👷 Outdoor Workers Guidance (Laborers, Delivery Riders, Police):</b><br>
                    Mandatory 15-minute shaded rest breaks every 45 minutes. Drink 500ml water every hour. Utilize N95 particulate PPE and cooling bandanas.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("All environmental indicators are within safe thresholds. No active advisories.")

# =========================================================================
# TAB 4: Model Leaderboard & Explainability (SHAP & LIME)
# =========================================================================
with tabs[3]:
    st.subheader("🧠 Model Leaderboard & Explainability (SHAP / LIME)")
    st.caption("Benchmark scores across Statistical, Classical ML, and Deep Learning model families compared to Naive 24h Baseline.")
    
    # 1. Leaderboard Table
    leaderboard_df = registry.get_leaderboard()
    if not leaderboard_df.empty:
        st.markdown("### 🏆 Comprehensive Model Evaluation Leaderboard")
        st.dataframe(
            leaderboard_df.style.highlight_max(subset=["Validation R²"], color="#1e3a8a")
                               .highlight_min(subset=["Validation RMSE", "Validation MAE"], color="#064e3b"),
            use_container_width=True
        )
    else:
        st.info("No trained models recorded in registry yet. Run `python -m src.pipeline.train` to generate benchmark leaderboard.")
        
    st.markdown("---")
    st.markdown("### 🔍 Model Explainability Analysis")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.markdown("#### 🌐 Global Feature Importance (Top Impact Drivers)")
        exp_metric = st.selectbox("Select Metric for Importance Analysis", METRICS, index=0)
        imp_df = explainer.get_global_feature_importance(exp_metric, top_n=10)
        
        if not imp_df.empty:
            fig_imp = px.bar(
                imp_df,
                x="Relative_Pct",
                y="Feature",
                orientation="h",
                title=f"Top 10 Global Feature Predictors for [{exp_metric.upper()}]",
                labels={"Relative_Pct": "Relative Importance (%)", "Feature": "Feature Name"},
                color="Relative_Pct",
                color_continuous_scale="Viridis",
                template="plotly_dark"
            )
            fig_imp.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance will be computed once models are registered.")

    with col_exp2:
        st.markdown("#### 🎯 Local Forecast Attribution (SHAP Waterfall)")
        if state and state.get("current_features"):
            sample_series = pd.Series(state["current_features"])
            shap_info = explainer.explain_prediction_shap(exp_metric, sample_series)
            
            if shap_info and "top_contributors" in shap_info:
                st.markdown(f"**Baseline Expected Value:** `{shap_info['base_value']}` | **Final 24h Forecast:** `{shap_info['prediction']}`")
                df_contrib = pd.DataFrame(shap_info["top_contributors"])
                
                fig_shap = px.bar(
                    df_contrib,
                    x="SHAP_Value",
                    y="Feature",
                    orientation="h",
                    title=f"SHAP Attribution Breakdown for 24h {exp_metric.upper()} Forecast",
                    labels={"SHAP_Value": "SHAP Impact (Pushes Prediction Up/Down)", "Feature": "Feature"},
                    color="SHAP_Value",
                    color_continuous_scale=["#3b82f6", "#94a3b8", "#ef4444"],
                    template="plotly_dark"
                )
                fig_shap.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_shap, use_container_width=True)
            else:
                st.info("Local SHAP attribution ready.")
        else:
            st.info("Run an agent cycle to compute local prediction attribution.")

# =========================================================================
# TAB 5: Exploratory Data Analysis (EDA)
# =========================================================================
with tabs[4]:
    st.subheader("🔍 Exploratory Data Analysis (EDA)")
    st.caption("Deep-dive historical statistical distributions, pollutant correlations, diurnal cycles, and outlier diagnostics.")
    
    col_eda1, col_eda2 = st.columns(2)
    
    with col_eda1:
        st.markdown("#### 📊 Descriptive Statistics & Skewness")
        stats_df = eda.get_summary_statistics()
        if not stats_df.empty:
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("No raw observations found for EDA. Run historical backfill first.")
            
    with col_eda2:
        st.markdown("#### 🚨 Outlier Detection (Interquartile Range Rule)")
        outlier_df = eda.detect_outliers_iqr()
        if not outlier_df.empty:
            st.dataframe(outlier_df, use_container_width=True)
        else:
            st.info("Outlier diagnostics ready upon backfill.")
            
    st.markdown("---")
    col_eda3, col_eda4 = st.columns(2)
    
    with col_eda3:
        st.markdown("#### 🔗 Meteorological & Pollutant Correlation Heatmap")
        corr_df = eda.get_correlation_matrix()
        if not corr_df.empty:
            fig_corr = px.imshow(
                corr_df,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Cross-Metric Pearson Correlation Heatmap",
                template="plotly_dark"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.caption("Notice negative correlation between wind speed and AQI (air stagnation amplifies smog accumulation).")
            
    with col_eda4:
        st.markdown("#### ⏰ Diurnal (24-Hour Cycle) Patterns")
        diurnal_df = eda.get_diurnal_patterns()
        if not diurnal_df.empty:
            sel_diurnal = st.multiselect("Select Metrics to View Diurnal Rhythm", METRICS, default=["temperature_c", "aqi"])
            if sel_diurnal:
                fig_diurnal = px.line(
                    diurnal_df,
                    x="hour",
                    y=sel_diurnal,
                    markers=True,
                    title="Average Diurnal Hourly Cycle (00:00 to 23:00)",
                    labels={"hour": "Hour of Day (0-23)", "value": "Mean Value"},
                    template="plotly_dark"
                )
                st.plotly_chart(fig_diurnal, use_container_width=True)

# =========================================================================
# TAB 6: MLOps, Drift Monitoring & Call Audits
# =========================================================================
with tabs[5]:
    st.subheader("🛡️ MLOps Telemetry: Drift Detection & Agent Failure Audits")
    st.caption("Continuous monitoring of Data Drift (Kolmogorov-Smirnov & PSI), Model Drift (residual degradation), and LLM call audits.")
    
    # 1. Drift Status
    if state and state.get("drift_status"):
        ds = state["drift_status"]
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.metric("Overall Drift Detected", "🚨 YES" if ds.get("overall_drift_detected") else "🟢 NO")
        with col_d2:
            st.metric("Data Drift Status", "DRIFT DETECTED" if ds.get("data_drift", {}).get("data_drift_detected") else "NOMINAL")
        with col_d3:
            st.metric("Automated Retraining Triggered", "✅ EXECUTED" if state.get("triggered_retraining") else "IDLE")
            
    # 2. Drift Events Log
    st.markdown("### 📋 Historical Drift Audit Log")
    drift_log_df = fs.get_drift_log(limit=50)
    if not drift_log_df.empty:
        st.dataframe(drift_log_df, use_container_width=True)
    else:
        st.info("No drift events logged yet.")
        
    st.markdown("---")
    st.markdown("### 📞 Agent Call Telemetry & Failure Audits")
    call_log_df = fs.get_agent_call_log(limit=50)
    if not call_log_df.empty:
        # Success vs Failure summary
        col_s1, col_s2, col_s3 = st.columns(3)
        total_calls = len(call_log_df)
        failed_calls = len(call_log_df[call_log_df["status"] != "SUCCESS"])
        avg_latency = call_log_df["latency_ms"].mean()
        
        with col_s1:
            st.metric("Total Agent LLM Calls", total_calls)
        with col_s2:
            st.metric("Failed / Fallback Calls", failed_calls, delta=f"{(failed_calls/max(1, total_calls))*100:.1f}% Fail Rate")
        with col_s3:
            st.metric("Average Latency", f"{avg_latency:.1f} ms")
            
        st.markdown("#### Full Agent Call Log")
        st.dataframe(call_log_df, use_container_width=True)
    else:
        st.info("Agent call logs will be registered upon agent execution.")

st.markdown("---")
st.caption("© 2026 AgentWeather | 10Pearls SHINE Internship Cohort 10 | Enterprise Agentic Weather & AQI Forecasting System")

