import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
from prediction import pred_class

# --- Page Configuration ---
st.set_page_config(
    page_title="EuroSAT Land Classification",
    layout="centered"
)

# --- Model Selection ---
model_options = {
    "DenseNet121": r"C:\Users\USER\OneDrive\microplastic-website\web-Classification\densenet121_checkpoint_fold1.pt",
    "MobileNetV3 Large": r"C:\Users\USER\OneDrive\microplastic-website\web-Classification\mobilenetv3_large_100_checkpoint_fold0.pt"
}
selected_model_name = st.selectbox("เลือกโมเดล", list(model_options.keys()))
selected_model_path = model_options[selected_model_name]

# --- Model Loading ---
@st.cache_resource
def load_model(model_path):
    """
    Loads the pre-trained PyTorch model.
    Checks if the model file exists before loading.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(model_path):
        st.error(f"ไม่พบไฟล์โมเดล: '{model_path}'")
        st.info("กรุณาตรวจสอบให้แน่ใจว่าไฟล์โมเดลอยู่ในโฟลเดอร์เดียวกับ `streamlit_app.py`")
        st.stop()
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        model = model.float()
        return model
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        st.stop()

# --- Main Application ---

st.title('EuroSAT Land Cover Classification')
st.header('Please upload an image')

# Load Model (ใช้ฟังก์ชันที่ cache ไว้)
model = load_model(selected_model_path)

# Display image & Prediction
uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)
        if st.button('Predict'):
            with st.spinner('Analyzing the image...'):
                class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
                               'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
                               'River', 'SeaLake']
                probabilities = pred_class(model, image, class_names)
            st.success("Prediction Complete!")
            st.write("## Prediction Result")
            max_prob_index = np.argmax(probabilities)
            for i, class_name in enumerate(class_names):
                prob = probabilities[i] * 100
                if i == max_prob_index:
                    st.markdown(f"**<span style='color: #28a745;'>{class_name}: {prob:.2f}%</span>**", unsafe_allow_html=True)
                else:
                    st.write(f"{class_name}: {prob:.2f}%")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผลภาพ: {e}")

