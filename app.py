import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# 1. SET PAGE CONFIG
st.set_page_config(
    page_title="Plant Disease Detector", 
    page_icon="🌿", 
    layout="wide"  
)

# 2. LOAD ASSETS
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model('PlantVillageEFV2B0_Checkpoint.keras')
    with open('class_names_checkpoint.json', 'r') as f:
        class_names = json.load(f)
    return model, class_names

try:
    model, class_names = load_model_and_classes()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

# 3. UI HEADER
st.markdown("<h1 style='text-align: center;'>🌿 Plant Disease Diagnostic System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Powered by EfficientNetV2B0 • 38 Disease Categories</p>", unsafe_allow_html=True)
st.divider()

# 4. MAIN INTERFACE
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Upload Leaf Sample")
    uploaded_file = st.file_uploader("Drag and drop or click to upload", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Input Image for Analysis', use_container_width=True)

with col2:
    st.subheader("🔍 Automated Diagnosis")
    
    if uploaded_file is not None:
        with st.spinner('Running Neural Inference...'):
            # Image Preprocessing
            img = image.resize((128, 128))
            img_array = np.array(img)
            if img_array.shape[-1] == 4: img_array = img_array[:,:,:3]
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Prediction
            predictions = model.predict(img_array)
            class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            result_class = class_names[class_idx]

        # High-Impact Results
        display_name = result_class.replace("___", ": ").replace("_", " ")
        
        if "healthy" in result_class.lower():
            st.success(f"### Result: {display_name}")
        else:
            st.warning(f"### Detected: {display_name}")

        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Confidence Score", f"{confidence:.1f}%")
        m_col2.metric("Model Status", "96.4% Acc")
        
        st.progress(int(confidence))
        

    else:
        # Placeholder when no image is uploaded
        st.info("Please upload a leaf image in the left panel to begin the diagnostic process.")

# 5. FOOTER 
st.markdown("<br><hr><center><small>EfficientNetV2B0 | PlantVillage Dataset | 2025 Project</small></center>", unsafe_allow_html=True)