"""
Home / Executive Dashboard Component.
Displays project overview, system statistics, architecture pipeline, and target pathology guide.
"""

import streamlit as st


def render_dashboard_view():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1b4d3e, #2e7d32); padding: 26px; border-radius: 12px; color: white; margin-bottom: 25px;">
        <h1 style="margin:0; font-size: 2.2rem; font-weight: 700;">🌱 AI-Based Crop Health Monitoring System</h1>
        <p style="margin: 8px 0 0 0; font-size: 1.05rem; opacity: 0.92;">
            Next-Generation Deep Learning Plant Disease Diagnosis Fused with Microclimate Environmental Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="MobileNetV2 Test Accuracy", value="98.61%", delta="+8.33% vs CNN")
    with col2:
        st.metric(label="Custom CNN Test Accuracy", value="90.28%", delta="Trained from Scratch")
    with col3:
        st.metric(label="Diagnostic Categories", value="6 Classes", delta="PlantVillage Benchmark")
    with col4:
        st.metric(label="Inference Latency (CPU)", value="25.5 ms", delta="Real-time Edge Ready")

    st.markdown("---")

    # Architecture Overview
    st.subheader("System Architecture & Multimodal Fusion Pipeline")
    st.markdown("""
    The system bridges computer vision and agronomic sensor analytics to prevent catastrophic crop failure through early diagnosis:
    """)

    col_arch1, col_arch2 = st.columns([1, 1])
    with col_arch1:
        st.markdown("""
        #### 1. Deep Learning Vision Engine
        - **Convolutional Feature Extraction:** High-resolution lesion localization via multi-stage convolutions, batch normalization, and dropout.
        - **Transfer Learning (MobileNetV2):** Exploits inverted residual bottlenecks pre-trained on ImageNet for rapid feature convergence.
        - **Explainable AI (Grad-CAM):** Generates gradient-weighted class activation maps to visualize symptomatic necrotic spots and chlorosis directly on the leaf lamina.
        """)

    with col_arch2:
        st.markdown("""
        #### 2. Environmental Analytics Engine
        - **Microclimate Ingestion:** Captures Ambient Temperature, Relative Humidity, and Soil Moisture.
        - **Biophysical Modeling:** Computes Vapor Pressure Deficit (VPD in kPa), a primary driver of transpiration and spore germination.
        - **Random Forest Pathology Risk Classifier:** 120-estimator ensemble predicting disease proliferation favorability (97.8% validation accuracy).
        - **Unified Crop Health Index (CHI):** Algorithmic fusion of vision and climate risk (0-100 score).
        """)

    st.markdown("---")

    # Target Pathologies
    st.subheader("Target Crop Pathology Profiles")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        **🍎 Apple Scab (*Venturia inaequalis*)**
        - *Symptoms:* Olive-green velvety spots progressing to corky dark lesions on leaf surfaces.
        - *Favorable Weather:* Cool, wet weather (15°C–24°C) with prolonged canopy leaf wetness.
        - *Treatment:* Protectant fungicides (Captan/Mancozeb) and leaf litter clearing.
        """)
        st.markdown("""
        **🍎 Apple Black Rot (*Botryosphaeria obtusa*)**
        - *Symptoms:* Frog-eye circular brown spots with purple margins on foliage and fruit mummification.
        - *Favorable Weather:* Warm (24°C–30°C) and highly humid conditions.
        - *Treatment:* Canopy pruning of dead twigs and sterol-inhibiting fungicides.
        """)

    with c2:
        st.markdown("""
        **🍎 Cedar Apple Rust (*Gymnosporangium*)**
        - *Symptoms:* Bright yellow-orange circular lesions on upper leaf surfaces with gelatinous spore horns.
        - *Favorable Weather:* Mild spring temperatures (12°C–22°C) coupled with rainfall.
        - *Treatment:* Alternate host separation (Juniper) and early-season protective sprays.
        """)
        st.markdown("""
        **🥔 Potato Early Blight (*Alternaria solani*)**
        - *Symptoms:* Concentric dark 'target-spot' rings surrounded by chlorotic yellow halos.
        - *Favorable Weather:* Warm (24°C–30°C) with alternating wet and dry foliage cycles.
        - *Treatment:* Chlorothalonil foliar application and drip irrigation instead of overhead sprinklers.
        """)

    with c3:
        st.markdown("""
        **🌿 Apple Healthy Reference Tissue**
        - *Characteristics:* High photosynthetic chlorophyll vigor, uniform emerald-green lamina, intact venation.
        - *Agronomic Target:* Maintain optimal canopy transpiration and prophylactic bio-protection.
        """)
        st.markdown("""
        **🌿 Potato Healthy Reference Tissue**
        - *Characteristics:* Robust turgor pressure, absence of necrotic lesions, vibrant leaf margins.
        - *Agronomic Target:* Balanced nitrogen fertilization and proactive moisture monitoring.
        """)
