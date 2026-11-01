"""
Prediction History and Farm Scout Logbook Component.
Maintains persistent field diagnostic records, plots temporal crop health index trends,
and supports CSV exports for agronomy record-keeping.
"""

from pathlib import Path
import json
import streamlit as st
import pandas as pd
from src.backend import load_history, save_history


def render_history_view():
    st.subheader("📜 Farm Scout Diagnostic Logbook & History")
    st.markdown("""
    Review field diagnosis events, inspect microclimate trends, and export diagnostic logs for farm compliance and record-keeping.
    """)

    records = load_history()

    if not records:
        st.info("ℹ️ No diagnostic history recorded yet. In the 'Crop Health Assessment' tab, click **Log Diagnosis to Farm Scout History** after analyzing a leaf.")
        return

    df = pd.DataFrame(records)

    # Top KPI Metrics
    total_assessments = len(df)
    healthy_count = int(df["is_healthy"].sum()) if "is_healthy" in df.columns else 0
    infected_count = total_assessments - healthy_count
    avg_chi = float(df["crop_health_index"].mean()) if "crop_health_index" in df.columns else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Assessments Logged", value=total_assessments)
    with k2:
        st.metric("Average Health Index (CHI)", value=f"{avg_chi:.1f}/100")
    with k3:
        st.metric("Healthy Foliage Scans", value=healthy_count, delta=f"{(healthy_count/total_assessments)*100:.1f}%")
    with k4:
        st.metric("Infected / Diseased Scans", value=infected_count, delta=f"{(infected_count/total_assessments)*100:.1f}%", delta_color="inverse")

    st.markdown("---")

    # Temporal Trend Chart
    st.markdown("### Crop Health Index (CHI) Progression Trend")
    if "crop_health_index" in df.columns and len(df) > 1:
        trend_df = df.iloc[::-1].reset_index(drop=True)  # Chronological order
        trend_df["Event Index"] = trend_df.index + 1
        st.line_chart(data=trend_df.set_index("Event Index")[["crop_health_index"]])

    st.markdown("---")

    # Table Filter & Actions
    col_filter, col_act1, col_act2 = st.columns([2, 1, 1])

    with col_filter:
        filter_status = st.selectbox("Filter Records by Status:", options=["All Records", "Healthy Foliage Only", "Infected Foliage Only"])

    filtered_df = df.copy()
    if filter_status == "Healthy Foliage Only":
        filtered_df = filtered_df[filtered_df["is_healthy"] == True]
    elif filter_status == "Infected Foliage Only":
        filtered_df = filtered_df[filtered_df["is_healthy"] == False]

    with col_act1:
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV Log",
            data=csv_data,
            file_name="crop_health_farm_scout_log.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_act2:
        if st.button("🗑️ Clear Logbook", use_container_width=True):
            save_history([])
            st.rerun()

    # Display Table
    st.markdown(f"#### Diagnostic Records ({len(filtered_df)})")

    display_cols = [c for c in ["timestamp", "predicted_class", "crop_health_index", "overall_status", "confidence", "temperature", "humidity", "soil_moisture", "environmental_risk_level", "data_source"] if c in filtered_df.columns]
    renamed_cols = {
        "timestamp": "Timestamp",
        "predicted_class": "Visual Diagnosis",
        "crop_health_index": "CHI (/100)",
        "overall_status": "Overall Health Status",
        "confidence": "Vision Confidence",
        "temperature": "Temp (°C)",
        "humidity": "RH (%)",
        "soil_moisture": "Soil Moist (%)",
        "environmental_risk_level": "Climate Risk",
        "data_source": "Telemetry Source"
    }

    styled_df = filtered_df[display_cols].rename(columns=renamed_cols)
    st.dataframe(styled_df, use_container_width=True)
