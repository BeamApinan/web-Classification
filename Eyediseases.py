import streamlit as st
import torch
from PIL import Image
import numpy as np
import os

# --- Dependency Check and Import ---
# This part is for the prediction function, assuming it's in a separate file.
# If 'prediction.py' is not available, a placeholder function is created.
try:
    from prediction import pred_class
except ImportError:
    st.warning("`prediction.py` not found. Using a placeholder prediction function. Please ensure the file exists and is correct.")
    # Placeholder function if prediction.py is missing
    def pred_class(model, image, class_names):
        # This is a dummy function. It will return random probabilities.
        # Replace this with your actual prediction logic if needed.
        st.info("Displaying random prediction results as `prediction.py` is missing.")
        # Simulate some processing
        import time
        time.sleep(2)
        random_probs = np.random.rand(len(class_names))
        random_probs /= np.sum(random_probs) # Normalize to sum to 1
        return random_probs

# --- Page Configuration ---
st.set_page_config(
    page_title="Eye Diseases Classification",
    layout="centered"
)

# --- Model Selection ---
# You can add more models here as needed.
model_options = {
    "✨mobilenetv3✨ (แนะนำ)": r"C:\Users\USER\OneDrive\microplastic-website\web-Classification\mobilenetv3_large_100_checkpoint_fold0 (2).pt",
    "ResNet50": r"C:\Users\USER\OneDrive\microplastic-website\web-Classification\resnet50_checkpoint_fold1.pt",
}

selected_model_name = st.selectbox("เลือกโมเดล (Select Model)", list(model_options.keys()))
selected_model_path = model_options[selected_model_name]

# --- Model Loading ---
# Cache the model loading to improve performance.
@st.cache_resource
def load_model(model_path):
    """Loads the PyTorch model from the specified path."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(model_path):
        st.error(f"ไม่พบไฟล์โมเดล (Model file not found): '{model_path}'")
        st.info("กรุณาตรวจสอบว่าเส้นทางไฟล์ถูกต้องและไฟล์โมเดลมีอยู่จริง (Please verify the file path is correct and the model file exists).")
        st.stop()
        
    try:
        # --- FIX ---
        # Set weights_only=False as required by newer PyTorch versions for loading full model files.
        # This is safe because you trust the source of your model file.
        model = torch.load(model_path, map_location=device, weights_only=False)
        # If the saved file is a state dictionary, you would first initialize your model structure:
        # model = YourModelClass(*args, **kwargs)
        # model.load_state_dict(torch.load(model_path, map_location=device))
        
        model.to(device)
        model.eval() # Set the model to evaluation mode
        return model
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล (Error loading model): {e}")
        st.stop()

# --- Main Application UI ---
st.title('👁️ Eye Diseases Classification')
st.header('Please upload an image of an eye')

# Load the selected model
model = load_model(selected_model_path)

# Image uploader
uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image).convert('RGB')
        
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Prediction button
        if st.button('Predict'):
            with st.spinner('Analyzing the image...'):
                
                # !!! IMPORTANT !!!
                # Please change these class names to match the actual eye diseases
                # that your model is trained to predict.
                class_names = [
                    'Cataract', 
                    'Diabetic Retinopathy', 
                    'Glaucoma', 
                    'Hypertension',
                    'Macular Degeneration',
                    'Myopia',
                    'Normal'
                ]
                
                # Get prediction probabilities
                probabilities = pred_class(model, image, class_names)
            
            # Display results
            st.success("Prediction Complete!")
            st.write("## Prediction Result")
            
            max_prob_index = np.argmax(probabilities)
            
            for i, class_name in enumerate(class_names):
                prob = probabilities[i] * 100
                # Highlight the class with the highest probability
                if i == max_prob_index:
                    st.markdown(f"**<span style='color: #28a745; font-size: 1.1em;'>➡️ {class_name}: {prob:.2f}%</span>**", unsafe_allow_html=True)
                else:
                    st.write(f"{class_name}: {prob:.2f}%")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผลภาพ (An error occurred during image processing): {e}")

