"""
Main Streamlit Application for AI-Based Crop Health Monitoring System.

Coordinates:
- Navigation across Executive Dashboard, Diagnosis & Grad-CAM,
  Environmental Analytics, Unified Assessment, Evaluation Hub, and History.
- Custom enterprise styling and theme configuration.
- Global session state management.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.dashboard_view import render_dashboard_view
from app.components.diagnosis_view import render_diagnosis_view
from app.components.environmental_view import render_environmental_view
from app.components.assessment_view import render_assessment_view
from app.components.evaluation_view import render_evaluation_view
from app.components.history_view import render_history_view

# Page Configuration
st.set_page_config(
    page_title="AI Crop Health Monitoring System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS Styling
st.markdown("""
<style>
    /* Main container and font styles */
    .main {
        background-color: #fcfdfc;
    }
    
    /* Metrics customization */
    div[data-testid="stMetricValue"] {
        font-weight: 700;
        color: #1b4d3e;
    }
    
    /* Sidebar branding */
    .sidebar-header {
        text-align: center;
        padding: 10px 0 15px 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 15px;
    }
    .sidebar-brand-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #1b4d3e;
        margin-top: 5px;
    }
    .sidebar-tagline {
        font-size: 0.8rem;
        color: #555555;
    }
    
    /* Custom buttons */
    div.stButton > button:first-child {
        border-radius: 6px;
        font-weight: 600;
    }
    
    /* Card style */
    .custom-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #e8ede8;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initializes session state defaults."""
    if "env_temp" not in st.session_state:
        st.session_state["env_temp"] = 24.0
    if "env_humid" not in st.session_state:
        st.session_state["env_humid"] = 65.0
    if "env_moist" not in st.session_state:
        st.session_state["env_moist"] = 55.0


def main():
    init_session_state()

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <span style="font-size: 2.5rem;">🌱</span>
            <div class="sidebar-brand-title">CropHealth AI</div>
            <div class="sidebar-tagline">Deep Learning & Microclimate Analytics</div>
        </div>
        """, unsafe_allow_html=True)

        selected_page = st.radio(
            "Navigation Modules:",
            options=[
                "🏠 Executive Overview",
                "🔬 Leaf Disease Diagnosis & Grad-CAM",
                "🌦️ Environmental Microclimate Analytics",
                "🌿 Unified Crop Health Assessment",
                "📊 Experimental Evaluation Hub",
                "📜 Farm Scout History & Logbook",
            ],
            index=0
        )

        st.markdown("---")
        st.markdown("### System Telemetry & Status")
        st.markdown("""
        - **Vision Engines:** MobileNetV2 & Custom CNN
        - **Microclimate Engine:** Random Forest (120 trees)
        - **Inference Runtime:** PyTorch CPU (Optimized)
        - **Explainability:** Grad-CAM Layer Visualizer
        - **Benchmark Split:** 480 Genuine PlantVillage Images
        """)

        st.caption("Version 1.0.0 | Production Release")

    # Routing
    if "Executive Overview" in selected_page:
        render_dashboard_view()
    elif "Leaf Disease Diagnosis" in selected_page:
        render_diagnosis_view()
    elif "Environmental Microclimate" in selected_page:
        render_environmental_view()
    elif "Unified Crop Health Assessment" in selected_page:
        render_assessment_view()
    elif "Experimental Evaluation" in selected_page:
        render_evaluation_view()
    elif "Farm Scout History" in selected_page:
        render_history_view()


if __name__ == "__main__":
    main()
