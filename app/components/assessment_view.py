"""
Unified Crop Health Assessment & Agronomic Treatment Component.
Fuses deep learning visual inferences with microclimate risk analytics into the Crop Health Index (CHI)
and generates targeted disease mitigation and prevention protocols.
"""

from datetime import datetime
import streamlit as st
from src.environmental_analysis import compute_unified_crop_health
from src.backend import load_history, save_history


def render_assessment_view():
    st.subheader("🌿 Unified Crop Health Assessment & Agronomic Roadmap")
    st.markdown("""
    This module synthesizes **leaf visual pathology** and **microclimate risk** into a single 
    actionable metric: the **Crop Health Index (CHI)**, generating tailored agronomic treatment directives.
    """)

    # Check prerequisites
    pred = st.session_state.get("current_prediction")
    env = st.session_state.get("current_environmental")

    if not pred:
        st.warning("⚠️ No leaf diagnosis active. Please run an image diagnosis in the 'Leaf Disease Diagnosis' tab first.")
        st.info("Tip: You can select a test preset sample from that tab with one click!")
        return

    if not env:
        st.warning("⚠️ Microclimate parameters uninitialized. Please adjust weather parameters in the 'Environmental Analytics' tab.")
        return

    # Compute Unified Health Assessment
    assessment = compute_unified_crop_health(pred, env)

    st.markdown("---")

    # Hero Assessment Card
    chi = assessment["crop_health_index"]
    status = assessment["overall_status"]
    status_color = assessment["status_color"]

    color_map = {
        "green": ("#2e7d32", "Optimal Vitality", "Foliage exhibits high vigor and low disease susceptibility."),
        "yellow": ("#f57f17", "Suboptimal / Caution", "Elevated environmental risk or emerging foliar stress detected."),
        "orange": ("#e65100", "Moderate Disease Infection", "Active pathogen presence identified; proactive treatment advised."),
        "red": ("#b71c1c", "Critical Outbreak Alert", "Severe pathogen colonization accelerated by favorable ambient microclimate.")
    }

    bg_color, subhead, desc = color_map.get(status_color, ("#333333", "Status", ""))

    st.markdown(f"""
    <div style="background-color: {bg_color}; padding: 22px 28px; border-radius: 12px; color: white; margin-bottom: 25px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span style="background: rgba(255,255,255,0.22); padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">
                    {subhead}
                </span>
                <h2 style="margin: 8px 0 4px 0; font-size: 2.1rem; font-weight: 800;">{status}</h2>
                <p style="margin: 0; font-size: 1.05rem; opacity: 0.95;">{desc}</p>
            </div>
            <div style="text-align: right; min-width: 150px; margin-top: 10px;">
                <div style="font-size: 0.9rem; opacity: 0.85; text-transform: uppercase;">Crop Health Index</div>
                <div style="font-size: 3.2rem; font-weight: 900; line-height: 1.1;">{chi}<span style="font-size: 1.6rem; font-weight: 500;">/100</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Diagnostic Inputs Overview
    st.markdown("#### Multimodal Input Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"**Visual Diagnosis:**")
        st.write(f"{pred['display_name']} ({pred['confidence']*100:.1f}%)")
    with c2:
        st.markdown(f"**Pathology State:**")
        st.write("Healthy Foliage" if pred["is_healthy"] else "Infected Foliage")
    with c3:
        st.markdown(f"**Microclimate Risk:**")
        st.write(f"{env['risk_level']} ({env['environmental_risk_score']:.1f}/100)")
    with c4:
        st.markdown(f"**Data Telemetry Source:**")
        st.write(f"{env.get('data_source', 'simulated').upper()}")

    st.markdown("---")

    # Agronomic Action Plan
    st.subheader("📋 Prescriptive Agronomic Treatment & Preventive Measures")
    st.markdown("""
    The following actionable steps are generated based on the specific pathogen identified and ambient environmental favorability:
    """)

    for idx, rec in enumerate(assessment["recommendations"], 1):
        st.markdown(f"**{idx}.** {rec}")

    st.markdown("---")

    # Log to History Button
    col_save, _ = st.columns([1, 2])
    with col_save:
        if st.button("💾 Log Diagnosis to Farm Scout History", use_container_width=True):
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "crop_health_index": chi,
                "overall_status": status,
                "predicted_class": pred["display_name"],
                "confidence": round(pred["confidence"], 4),
                "is_healthy": pred["is_healthy"],
                "temperature": env.get("temperature", st.session_state.get("env_temp", 24.0)),
                "humidity": env.get("humidity", st.session_state.get("env_humid", 65.0)),
                "soil_moisture": env.get("soil_moisture", st.session_state.get("env_moist", 55.0)),
                "environmental_risk_level": env["risk_level"],
                "data_source": env.get("data_source", "simulated"),
                "recommendations": assessment["recommendations"]
            }
            records = load_history()
            records.insert(0, record)
            save_history(records[:100])
            st.success("Successfully logged assessment into persistent Farm Scout history!")
