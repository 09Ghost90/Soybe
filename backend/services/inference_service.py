import cv2
import numpy as np 
from backend.routes.inference_routes import classify_image

# Converter bytes -> array NumPy -> Imagem OpenCV
def run_inference(model_name: str, imagem_bytes: bytes) -> dict:

    # Enviando bytes para a rota/modelo (inference_routes.py)
    resultado = classify_image(imagem_bytes)
    
    resultado = {
        "model_name": model_name,
        "label": resultado["class_name"],
        "confidence": resultado["confidence"]
    }

    return resultado
