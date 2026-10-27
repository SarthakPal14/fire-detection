# app.py - Integrated Fire Detection System
import os
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO

# ---------------------------------------------------
# PAGE CONFIGURATION & STYLING
# ---------------------------------------------------
st.set_page_config(
    page_title="Integrated Fire Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 10px;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔥 Integrated AI Forest Fire Detection & Risk System")
st.markdown("A multi-tier early warning system combining **Environmental Fire Risk Analysis**, **Deep Learning Image Classification**, and **YOLOv8 Semantic Fire/Smoke Segmentation**.")

# ---------------------------------------------------
# LOCATE AND LOAD MODELS
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_models():
    cnn_path = os.path.join(MODELS_DIR, "fire_model.h5")
    rf_path = os.path.join(MODELS_DIR, "model_rf.pkl")
    xgb_path = os.path.join(MODELS_DIR, "model_xgb.pkl")
    seg_path = os.path.join(MODELS_DIR, "best.pt")

    for path in [cnn_path, rf_path, xgb_path, seg_path]:
        if not os.path.exists(path):
            st.error(f"Missing model file: `{path}`")
            return None, None, None, None

    cnn_model = tf.keras.models.load_model(cnn_path)
    rf_model = joblib.load(rf_path)
    xgb_model = joblib.load(xgb_path)
    seg_model = YOLO(seg_path)

    return cnn_model, rf_model, xgb_model, seg_model

with st.spinner("Loading AI models (MobileNetV2, Random Forest, XGBoost, YOLOv8)..."):
    cnn_model, rf_model, xgb_model, seg_model = load_models()

if None in (cnn_model, rf_model, xgb_model, seg_model):
    st.stop()

# ---------------------------------------------------
# SIDEBAR CONTROLS & NAVIGATION
# ---------------------------------------------------
st.sidebar.title("⚙️ System Control")
mode = st.sidebar.radio(
    "Execution Mode",
    ["Pipeline Flow (Step-by-Step)", "Direct Vision Testing (Image / Segmentation)"],
    index=0
)

st.sidebar.info("""
**System Architecture:**
1. **Model 2 (Tabular ML)**: Predicts Fire Weather Index (FWI) from weather parameters.
2. **Model 1 (MobileNetV2 CNN)**: Classifies whether camera feed has fire.
3. **Model 3 (YOLOv8-Seg)**: Pinpoints and masks fire/smoke regions.
""")

# ---------------------------------------------------
# MODE 1: STEP-BY-STEP INTEGRATED PIPELINE
# ---------------------------------------------------
if mode == "Pipeline Flow (Step-by-Step)":
    st.header("Step 1: Environmental Fire Risk Prediction (Model 2)")
    st.caption("Input real-time weather and Canadian Forest Fire Weather Index (FWI) components to calculate wildfire risk.")

    # Preset selection for quick evaluation
    preset = st.selectbox(
        "Select Weather Condition Preset:",
        ["Custom Values", "🚨 Extreme Heatwave & Drought (High Risk)", "⚠️ Warm & Dry Summer (Moderate Risk)", "✅ Cool & Rainy (Low Risk)"]
    )

    defaults = {
        "temp": 40.0, "rh": 15.0, "ws": 25.0,
        "rain": 0.0, "ffmc": 90.0, "dmc": 60.0,
        "dc": 200.0, "isi": 15.0, "bui": 80.0
    }

    if preset == "🚨 Extreme Heatwave & Drought (High Risk)":
        defaults = {"temp": 42.0, "rh": 12.0, "ws": 30.0, "rain": 0.0, "ffmc": 92.5, "dmc": 65.0, "dc": 250.0, "isi": 18.0, "bui": 85.0}
    elif preset == "⚠️ Warm & Dry Summer (Moderate Risk)":
        defaults = {"temp": 30.0, "rh": 40.0, "ws": 18.0, "rain": 0.0, "ffmc": 80.0, "dmc": 25.0, "dc": 100.0, "isi": 7.0, "bui": 30.0}
    elif preset == "✅ Cool & Rainy (Low Risk)":
        defaults = {"temp": 18.0, "rh": 85.0, "ws": 10.0, "rain": 12.0, "ffmc": 50.0, "dmc": 5.0, "dc": 20.0, "isi": 1.0, "bui": 8.0}

    col1, col2, col3 = st.columns(3)
    with col1:
        temp = st.number_input("Temperature (°C)", value=float(defaults["temp"]), min_value=-10.0, max_value=60.0, step=0.5)
        rh = st.number_input("Relative Humidity (%)", value=float(defaults["rh"]), min_value=0.0, max_value=100.0, step=1.0)
        ws = st.number_input("Wind Speed (km/h)", value=float(defaults["ws"]), min_value=0.0, max_value=120.0, step=0.5)

    with col2:
        rain = st.number_input("Rainfall (mm)", value=float(defaults["rain"]), min_value=0.0, max_value=500.0, step=0.1)
        ffmc = st.number_input("FFMC (Fine Fuel Moisture Code)", value=float(defaults["ffmc"]), min_value=0.0, max_value=105.0, step=0.5)
        dmc = st.number_input("DMC (Duff Moisture Code)", value=float(defaults["dmc"]), min_value=0.0, max_value=300.0, step=1.0)

    with col3:
        dc = st.number_input("DC (Drought Code)", value=float(defaults["dc"]), min_value=0.0, max_value=1000.0, step=5.0)
        isi = st.number_input("ISI (Initial Spread Index)", value=float(defaults["isi"]), min_value=0.0, max_value=100.0, step=0.5)
        bui = st.number_input("BUI (Build Up Index)", value=float(defaults["bui"]), min_value=0.0, max_value=300.0, step=1.0)

    if "risk_calculated" not in st.session_state:
        st.session_state.risk_calculated = False

    if st.button("🔥 Predict Fire Risk", type="primary", use_container_width=True):
        X = pd.DataFrame([[temp, rh, ws, rain, ffmc, dmc, dc, isi, bui]],
                         columns=["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "DC", "ISI", "BUI"])

        rf_pred = float(rf_model.predict(X)[0])
        xgb_pred = float(xgb_model.predict(X)[0])
        combined = (rf_pred + xgb_pred) / 2.0

        st.session_state.risk_calculated = True
        st.session_state.rf_val = rf_pred
        st.session_state.xgb_val = xgb_pred
        st.session_state.risk = combined

    if st.session_state.risk_calculated:
        rf_pred = st.session_state.rf_val
        xgb_pred = st.session_state.xgb_val
        risk = st.session_state.risk

        st.markdown("---")
        st.subheader("📊 Fire Weather Index (FWI) Results")

        m1, m2, m3 = st.columns(3)
        m1.metric("🌲 Random Forest FWI", f"{rf_pred:.2f}")
        m2.metric("⚡ XGBoost FWI", f"{xgb_pred:.2f}")
        m3.metric("🤝 Ensemble Risk Score", f"{risk:.2f}")

        if risk < 5:
            st.success("✅ **Low Fire Risk**: Atmospheric & fuel conditions are currently unfavorable for fire outbreak.")
        elif risk < 15:
            st.warning("⚠️ **Moderate Fire Risk**: Caution advised. Weather permits fire propagation if ignition occurs.")
            st.info("💡 You may proceed to Step 2 below to verify live camera or drone feeds for early detection.")
        else:
            st.error("🚨 **HIGH FIRE RISK DETECTED**: Environmental factors indicate critical danger! Activating visual verification.")

        # Show visual detection for Moderate and High Risk (or user bypass)
        if risk >= 5:
            st.markdown("---")
            st.header("Step 2: Optical Fire Image Detection (Model 1)")
            st.caption("Verifying on-ground imagery using MobileNetV2 CNN Deep Learning Classifier.")

            file = st.file_uploader("Upload Surveillance / Drone Image", type=["jpg", "jpeg", "png"], key="pipeline_file")

            if file is not None:
                image = Image.open(file).convert("RGB")
                
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    st.image(image, caption="Uploaded Camera Feed", use_container_width=True)

                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                pred = float(cnn_model.predict(img_array)[0][0])

                with v_col2:
                    st.subheader("Model 1 Classification Output")
                    # In binary dataset: 0 = FIRE, 1 = NON_FIRE
                    is_fire = pred < 0.5
                    confidence = (1.0 - pred) if is_fire else pred

                    if is_fire:
                        st.error(f"🔥 **FIRE DETECTED!** Confidence: {confidence * 100:.2f}%")
                        st.progress(confidence)
                    else:
                        st.success(f"✅ **NO FIRE DETECTED** Confidence: {confidence * 100:.2f}%")
                        st.progress(confidence)

                if is_fire:
                    st.markdown("---")
                    st.header("Step 3: Fire & Smoke Localization (Model 3 - YOLOv8)")
                    st.caption("Segmenting flame boundaries and smoke plumes with YOLOv8 Instance Segmentation.")

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        image.save(tmp.name)
                        results = seg_model(tmp.name)

                    result_img = results[0].plot()
                    st.image(result_img, caption="Segmented Fire & Smoke Mask Output", use_container_width=True)

                    # Display detected boxes/classes if any
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        st.write(f"🔍 **Detected {len(boxes)} visual hazard region(s).**")
                    else:
                        st.info("Segmentation model processed the image with no explicit high-confidence bounding polygon.")

# ---------------------------------------------------
# MODE 2: DIRECT VISION TESTING
# ---------------------------------------------------
else:
    st.header("Direct Optical Fire Detection & Segmentation")
    st.caption("Test Model 1 (Classification) and Model 3 (YOLO Segmentation) directly on any image without running environmental calculations.")

    sample_col, upload_col = st.columns([1, 2])
    
    with sample_col:
        st.subheader("Quick Test Samples")
        sample_choice = st.radio(
            "Select a pre-loaded sample:",
            ["Custom Upload", "Fire Scene (House on Fire)", "Forest Fire Sample"]
        )

    with upload_col:
        image = None
        workspace_root = os.path.dirname(BASE_DIR)
        
        if sample_choice == "Fire Scene (House on Fire)":
            sample_path = os.path.join(workspace_root, "testing", "house-on-fire-engulfed-in-flames-in-open-field-at-sunset-photo.jpg")
            if os.path.exists(sample_path):
                image = Image.open(sample_path).convert("RGB")
        elif sample_choice == "Forest Fire Sample":
            sample_path = os.path.join(workspace_root, "fire project", "test99.jpg")
            if os.path.exists(sample_path):
                image = Image.open(sample_path).convert("RGB")
        
        file = st.file_uploader("Or Upload an Image File", type=["jpg", "jpeg", "png"], key="direct_file")
        if file is not None:
            image = Image.open(file).convert("RGB")

    if image is not None:
        st.markdown("---")
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("1. MobileNetV2 CNN Classification")
            st.image(image, caption="Input Image", use_container_width=True)
            
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = float(cnn_model.predict(img_array)[0][0])
            is_fire = pred < 0.5
            confidence = (1.0 - pred) if is_fire else pred

            if is_fire:
                st.error(f"🔥 **FIRE DETECTED** (Confidence: {confidence * 100:.2f}%)")
            else:
                st.success(f"✅ **NO FIRE DETECTED** (Confidence: {confidence * 100:.2f}%)")

        with c2:
            st.subheader("2. YOLOv8 Segmentation Output")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                results = seg_model(tmp.name)

            result_img = results[0].plot()
            st.image(result_img, caption="Segmented Detection Mask", use_container_width=True)