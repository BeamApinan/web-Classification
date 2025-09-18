from typing import List, Tuple
import torch
import torchvision.transforms as T
from PIL import Image

def pred_class(model: torch.nn.Module,
               image: Image.Image,
               class_names: List[str],
               image_size: Tuple[int, int] = (224, 224)):
    """
    Function to predict the class of an image using a trained model.
    
    Args:
        model: The trained PyTorch model.
        image (PIL.Image.Image): The input image.
        class_names (list): A list of class names.
        image_size (tuple): The size to resize the image to.

    Returns:
        numpy.ndarray: A 1D array of probabilities for each class.
    """
    
    # 1. Create transformation for the image
    image_transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    
    ### Predict on image ### 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Make sure the model is on the target device
    model.to(device)
    model = model.half()  # ถ้าอยากใช้ float16 จริงๆ

    # 3. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # 4. Transform and add an extra dimension to the image
        # (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(image).unsqueeze(dim=0)
        # บังคับให้ input เป็น float32
        transformed_image = transformed_image.to(device=device, dtype=torch.float32)
        # 5. Make a prediction on the image
        target_image_pred = model(transformed_image)

    # 6. Convert logits -> prediction probabilities (using torch.softmax)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    
    # 7. Squeeze to remove batch dimension and convert to a 1D numpy array
    # เพื่อให้เข้ากันได้กับโค้ดใน streamlit_app.py
    prob_array = target_image_pred_probs.squeeze().cpu().numpy()

    return prob_array

