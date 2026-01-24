import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from torchvision import models
from fastapi import APIRouter, UploadFile, File, Form

"""router = APIRouter(prefix="/inferencia", tags=["inferencia"])

# Rota de inferência
@router.post("")
async def inferencia(
    model: str = Form(...),
    files: list[UploadFile] = File(...)
):    
    return {"model": model, "files": [f.filename for f in files]}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
"""
classes = {
    0: "Ardidos",
    1: "Chochos",
    2: "Esverdeados",
    3: "Mamonas",
    4: "Normais",
    5: "Picadas",
    # 6: "Purpuras",
    # 7: "Quebrados"
}

# Configurações
num_classes = 8 # Ardidos, Chochos, Esverdeados, Mamonas, Normais, Picadas, Purpuras, Quebrados
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsando dispositivo: {device}")

# Carregando EfficientNet-B0 pré-treinado
model = models.efficientnet_b0(pretrained=False)  # False se você tiver pesos salvos
# Substitui a última camada fully connected pelo número de classes
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
PATH = os.path.join(PROJECT_ROOT, "models/efficientnet.pth")

# Carrega pesos
if os.path.exists(PATH):
    print(f"Arquivo de pesos encontrado. Carregando {PATH}...")
    model.load_state_dict(torch.load(PATH, map_location=device))
else:
    raise FileNotFoundError(f"Arquivo de pesos não encontrado: {PATH}")

model = model.to(device)
model.eval()

# Transformação da imagem (pré-processamento)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # EfficientNet usa 224x224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],  # médias ImageNet
        std=[0.5, 0.5, 0.5]    # std ImageNet
    )
])

# Função de inferência
def predict_single_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        probabilities = F.softmax(output, dim=1)
        confidence = probabilities[0][predicted_class].item()

    return predicted_class, confidence

# Teste
if __name__ == "__main__":
    image_path = os.path.join(PROJECT_ROOT, "src/models/test/intact_test.jpg")

    if not os.path.exists(image_path):
        print(f"Arquivo {image_path} não encontrado!")
    else:
        predicted_class, confidence = predict_single_image(image_path)
        print(f"Classe predita: {classes[predicted_class]}, Confiança: {confidence:.4f}")
