import streamlit as st
import torch
from PIL import Image
import numpy as np
import os

# --- Dependency Check and Import ---
try:
    from prediction import pred_class
except ImportError:
    st.warning("`prediction.py` not found. Using a placeholder prediction function. Please ensure the file exists and is correct.")
    def pred_class(model, image, class_names):
        st.info("Displaying random prediction results as `prediction.py` is missing.")
        import time
        time.sleep(2)
        random_probs = np.random.rand(len(class_names))
        random_probs /= np.sum(random_probs)
        return random_probs

# --- Page Configuration ---
st.set_page_config(
    page_title="Eye Diseases Classification",
    layout="centered"
)

# --- Fixed Model Path (Only MobileNetV3) ---
model_path = "mobilenetv3_large_100_checkpoint_fold0 (2).pt"

# --- Model Loading ---
@st.cache_resource
def load_model(model_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(model_path):
        st.error(f"ไม่พบไฟล์โมเดล (Model file not found): '{model_path}'")
        st.stop()
        
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล (Error loading model): {e}")
        st.stop()

# --- Main Application UI ---
st.title('👁️ Eye Diseases Classification')
st.header('Please upload an image of an eye')

# Load model once
model = load_model(model_path)

# Image uploader
uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        if st.button('Predict'):
            with st.spinner('Analyzing the image...'):
                class_names = [
                    'Cataract', 
                    'Diabetic Retinopathy', 
                    'Glaucoma', 
                    'Hypertension',
                    'Macular Degeneration',
                    'Myopia',
                    'Normal'
                ]
                probabilities = pred_class(model, image, class_names)
            
            st.success("Prediction Complete!")
            st.write("## Prediction Result")
            
            max_prob_index = np.argmax(probabilities)
            for i, class_name in enumerate(class_names):
                prob = probabilities[i] * 100
                if i == max_prob_index:
                    st.markdown(
                        f"**<span style='color: #28a745; font-size: 1.1em;'>➡️ {class_name}: {prob:.2f}%</span>**",
                        unsafe_allow_html=True
                    )
                else:
                    st.write(f"{class_name}: {prob:.2f}%")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผลภาพ (An error occurred during image processing): {e}")



