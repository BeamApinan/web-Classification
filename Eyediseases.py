import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import pytorch_lightning as pl

# --- Page Configuration ---
st.set_page_config(
    page_title="Eye Diseases Classification",
    layout="centered"
)

# --- Model Path ---
mobilenetv3_ckpt_path = r"mobilenetv3_large_100_checkpoint_fold0 (2).pt"

# --- Device ---
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --- Load Lightning checkpoint ---
@st.cache_resource
def load_model(ckpt_path):
    if not os.path.exists(ckpt_path):
        st.error(f"ไม่พบไฟล์โมเดล: '{ckpt_path}'")
        st.stop()
    try:
        # โหลด checkpoint โดยตรง
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # ตรวจสอบว่าเป็น state_dict ของ Lightning หรือเต็ม model
        if 'state_dict' in checkpoint:
            # สร้างโครงสร้าง MobileNetV3
            model = models.mobilenet_v3_large(weights=None)
            # ปรับ classifier ให้ตรงกับ 7 class
            model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 7)
            # โหลด state_dict จาก Lightning checkpoint
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                # Lightning มักบวก prefix 'model.' แก้ไข prefix
                if k.startswith('model.'):
                    k = k.replace('model.', '')
                state_dict[k] = v
            model.load_state_dict(state_dict)
        else:
            # โหลด model เต็ม (trust source)
            model = checkpoint
        
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
        st.stop()

# --- Prediction Function ---
def pred_class(model, image, class_names):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    return probabilities

# --- Main Application UI ---
st.title('👁️ Eye Diseases Classification')
st.header('Please upload an image of an eye')

# Load model
model = load_model(mobilenetv3_ckpt_path)

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
                    st.markdown(f"**<span style='color: #28a745; font-size: 1.1em;'>➡️ {class_name}: {prob:.2f}%</span>**", unsafe_allow_html=True)
                else:
                    st.write(f"{class_name}: {prob:.2f}%")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผลภาพ: {e}")
