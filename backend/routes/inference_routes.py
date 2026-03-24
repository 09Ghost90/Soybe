import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from torchvision import models
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "../"))

classes = {
    0: "Broken soybeans",
    1: "Immature soybeans",
    2: "Intact soybeans",
    3: "Skin-damaged soybeans",
    4: "Spotted soybeans"
}

# Configurações
num_classes = 5 # Broken, Immature, Intact, Skin-damaged, Spotted
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    print(f"Usando CUDA no backend: {torch.cuda.get_device_name(0)}")
else:
    cpu_threads = os.cpu_count() or 1
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(max(1, cpu_threads // 2))
    print(f"Usando CPU no backend: {cpu_threads} threads")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_CONFIGS = {
    "EfficientNetB0": {
        "builder": models.efficientnet_b0,
        "input_size": 224,
        "weight_candidates": [
            os.path.join(PROJECT_ROOT, "network/models/efficientnet.pth"),
            os.path.join(PROJECT_ROOT, "network/models/efficientnet_b0.pth"),
        ],
    },
    "EfficientNetB7": {
        "builder": models.efficientnet_b7,
        "input_size": 600,
        "weight_candidates": [
            os.path.join(PROJECT_ROOT, "network/models/efficientnet_b7.pth"),
            os.path.join(WORKSPACE_ROOT, "models/soybean_model_efficientnet_b7.pth"),
        ],
    },
}

_MODEL_CACHE: dict[str, tuple[torch.nn.Module, transforms.Compose]] = {}


def _build_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _resolve_weight_path(weight_candidates: list[str]) -> str:
    for path in weight_candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Nenhum peso encontrado. Caminhos testados: {weight_candidates}")


def _load_model(model_name: str) -> tuple[torch.nn.Module, transforms.Compose]:
    config = MODEL_CONFIGS[model_name]
    model = config["builder"](weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    path = _resolve_weight_path(config["weight_candidates"])
    print(f"Carregando pesos de {path} para {model_name}...")
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()

    transform = _build_transform(config["input_size"])
    return model, transform


def get_model_and_transform(model_name: str) -> tuple[torch.nn.Module, transforms.Compose]:
    if model_name not in MODEL_CONFIGS:
        modelos = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Modelo {model_name} não suportado. Opções: {modelos}")

    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = _load_model(model_name)

    return _MODEL_CACHE[model_name]

# Função chamada pelo Inference_Service (endpoint)
def classify_image(image_bytes: bytes, model_name: str) -> dict:
    """
    Recebe bytes da imagem e retorna a classificação

    Bytes -> Numpy array -> Imagem PIL
    """
    model, transform = get_model_and_transform(model_name)

    nparr = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Erro de decodificação
    if img_cv is None:
        raise ValueError("Não foi possível decodificar a imagem.")
    
    # Converter OpenCV (BGR) para PIL (RGB)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img_rgb)

    # Pré-processamento
    input_tensor = transform(image).unsqueeze(0).to(device, non_blocking=device.type == "cuda")

    # Inferência
    with torch.inference_mode():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        probabilities = F.softmax(output, dim=1)
        confidence = probabilities[0][predicted_class].item()

    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "class_name": classes[predicted_class]
    }