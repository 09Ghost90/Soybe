import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

"""
1 - Carrega o modelo e os pesos
2 - Pré-processa a imagem
3 - Passa o tensor pelo modelo em modo de avaliação
4 - Torch.argmax ou torch.max para obter a classe
"""

classes = {
    0: "Broken soybeans",
    1: "Immature soybeans",
    2: "Intact soybeans",
    3: "Skin-damaged soybeans",
    4: "Spotted soybeans"
}
 
# Hiperparâmetros do modelo
in_channels = 3
num_class = 5
PATH = "./models/soybean_model_full[2].pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n\nsUsando dispositivo: {device}")

class SoybeanCNN(torch.nn.Module):
    def __init__(self, in_channels, num_class):
        super(SoybeanCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, num_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.global_avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x
    
# Modelo e os pesos
model = SoybeanCNN(in_channels, num_class).to(device)
if os.path.exists(PATH):
    print(f"Arquivo de pesos encontrado. Carregando {PATH}...")
    model.load_state_dict(torch.load(PATH, map_location=device))
else:
    raise FileNotFoundError(f"Arquivo de pesos não encontrado: {PATH}")

model.eval()

# Pré-Processar a imagem teste
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict_single_image(image_path):
    """
    Realiza inferência em uma única imagem.
    """
    # Carregue e processe a imagem
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inferência
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        probabilities = F.softmax(output, dim=1)
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence

if __name__ == "__main__":
    image_path = "./SOYBE/test/spotted_test.jpg"

    if not os.path.exists(image_path):
        print(f"Arquivo {image_path} não encontrado!")
    else:
        predicted_class, confidence = predict_single_image(image_path)
        print(f"Classe predita: {classes[predicted_class]}, Confiança: {confidence:.4f}\n\n")