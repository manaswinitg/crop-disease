"""
Environmental Microclimate Analytics Component.
Interactively collects temperature, humidity, and soil moisture telemetry,
evaluates disease proliferation risk via Random Forest, and models Vapor Pressure Deficit (VPD).
"""

import streamlit as st
import pandas as pd
from src.environmental_analysis import analyze_environmental_conditions


def render_environmental_view():
    st.subheader("🌦️ Environmental Microclimate Risk Analytics")
    st.markdown("""
    Plant pathogens thrive under specific microclimates. This module analyzes ambient **Temperature**, 
    **Relative Humidity**, and **Soil Moisture** using an agronomic **Random Forest model** to predict environmental disease vulnerability.
    """)

    # Telemetry source selector
    col_src1, col_src2 = st.columns([2, 1])
    with col_src1:
        data_source = st.radio(
            "Telemetry Telemetry Mode:",
            options=["Simulated Microclimate Sensor", "Hardware IoT Telemetry (Actual Measured Data)"],
            index=0,
            horizontal=True
        )
    with col_src2:
        st.info(f"Active Mode: **{'SIMULATED DATA' if 'Simulated' in data_source else 'ACTUAL MEASURED DATA'}**")

    st.markdown("---")

    # Quick scenario presets
    st.markdown("#### Climate Scenario Presets")
    preset_cols = st.columns(4)

    preset_values = None
    with preset_cols[0]:
        if st.button("🌱 Optimal Mild Day", use_container_width=True):
            preset_values = (22.0, 58.0, 62.0)
    with preset_cols[1]:
        if st.button("🌧️ Cool & Humid (Scab Risk)", use_container_width=True):
            preset_values = (18.0, 88.0, 78.0)
    with preset_cols[2]:
        if st.button("🔥 Warm & Steamy (Blight Risk)", use_container_width=True):
            preset_values = (28.0, 89.0, 82.0)
    with preset_cols[3]:
        if st.button("☀️ Heat & Drought Stress", use_container_width=True):
            preset_values = (38.5, 32.0, 18.0)

    # Defaults or preset
    default_temp = preset_values[0] if preset_values else st.session_state.get("env_temp", 24.0)
    default_humid = preset_values[1] if preset_values else st.session_state.get("env_humid", 65.0)
    default_moist = preset_values[2] if preset_values else st.session_state.get("env_moist", 55.0)

    # Parameter input sliders
    st.markdown("#### Parameter Adjustments")
    c_temp, c_humid, c_moist = st.columns(3)

    with c_temp:
        temp = st.slider("Ambient Temperature (°C):", min_value=10.0, max_value=45.0, value=float(default_temp), step=0.5)
        st.session_state["env_temp"] = temp
        st.caption("Favorable fungal range: 18°C – 28°C")

    with c_humid:
        humidity = st.slider("Relative Humidity (%):", min_value=20.0, max_value=98.0, value=float(default_humid), step=1.0)
        st.session_state["env_humid"] = humidity
        st.caption("Spore germination threshold: > 75%")

    with c_moist:
        moisture = st.slider("Root Zone Soil Moisture (%):", min_value=10.0, max_value=95.0, value=float(default_moist), step=1.0)
        st.session_state["env_moist"] = moisture
        st.caption("Target field capacity: 45% – 70%")

    # Run Environmental Analytics
    source_tag = "actual_sensor" if "Actual" in data_source else "simulated"
    env_result = analyze_environmental_conditions(
        temperature=temp,
        humidity=humidity,
        soil_moisture=moisture,
        data_source=source_tag
    )
    st.session_state["current_environmental"] = env_result

    st.markdown("---")
    st.subheader("Microclimate Risk Diagnostic Summary")

    badge_colors = {
        "Low": "#2e7d32",
        "Moderate": "#f57f17",
        "High": "#e65100",
        "Extreme": "#b71c1c"
    }
    lvl_color = badge_colors.get(env_result["risk_level"], "#333333")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div style="background-color: {lvl_color}; padding: 12px; border-radius: 8px; color: white; text-align: center;">
            <div style="font-size: 0.8rem; text-transform: uppercase;">Pathogen Pressure</div>
            <div style="font-size: 1.4rem; font-weight: 700; margin-top: 2px;">{env_result['risk_level']} Risk</div>
            <div style="font-size: 0.85rem;">Badge: {env_result['risk_badge']}</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.metric(
            label="Environmental Risk Score",
            value=f"{env_result['environmental_risk_score']:.1f} / 100",
            delta="Pathogen Favorability",
            delta_color="inverse"
        )

    with m3:
        st.metric(
            label="Vapor Pressure Deficit (VPD)",
            value=f"{env_result['vpd_kpa']} kPa",
            delta="Optimal: 0.8 - 1.2 kPa"
        )

    with m4:
        st.metric(
            label="Random Forest Confidence",
            value=f"{max(env_result['class_probabilities'])*100:.1f}%",
            delta="Ensemble Consensus"
        )

    # Diagnostic Cards
    st.markdown("#### Microclimate Biophysical Indicators")
    d1, d2, d3 = st.columns(3)

    diag = env_result["diagnostics"]
    with d1:
        st.info(f"**Temperature Assessment:** {diag['temperature_status']} ({temp}°C)")
    with d2:
        if diag['humidity_status'] == "Pathogen Favorable":
            st.error(f"**Humidity Assessment:** {diag['humidity_status']} ({humidity}%)")
        else:
            st.success(f"**Humidity Assessment:** {diag['humidity_status']} ({humidity}%)")
    with d3:
        if diag['soil_moisture_status'] in ["Waterlogged", "Drought Stress"]:
            st.warning(f"**Soil Moisture Assessment:** {diag['soil_moisture_status']} ({moisture}%)")
        else:
            st.success(f"**Soil Moisture Assessment:** {diag['soil_moisture_status']} ({moisture}%)")
