import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from torch.utils.tensorboard import SummaryWriter
import time

import mlflow
import mlflow.pytorch

# # Implementando metricas da matriz de confusão
# import sklearn
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Classificação Supervisionada de Imagens de Soja usando CNN

# TensorBoard
writer = SummaryWriter()

# Hiperparâmetros
in_channels = 3
batch_size = 32
num_epochs = 20
num_class = 5 # Broken, Immature, Intact, Skin-damaged, Spotted
patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0
learning_rate = 1e-4
PATH = "./models/soybean_model_full[4].pth"
data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/dataset"))

data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

image_dataset = datasets.ImageFolder(root=data_root, transform=data_transforms)

train_size = int(0.8 * len(image_dataset))
val_size = int(0.1 * len(image_dataset))
test_size = len(image_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(image_dataset, [train_size, val_size, test_size])

# Loader -> Remover gargalo da CPU em preparar os dados
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Embaralha os dados
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # Não embaralha os dados
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # Não embaralha os dados

# Modelo CNN
class SoybeanCNN(nn.Module):
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
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # MaxPool pega o valor Max em uma região 2x2
        
        # Global Avarage Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, num_class) # Fica menor pq perde a dependência que o MaxPool2d ainda mantem do tamanho da imagem

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x)) 
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x) 
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.global_avg_pool(x) # Saida (batch, 128, 1, 1) -> Ele pega a média e reduz para um [1x1]
        x = x.reshape(x.shape[0], -1) # Achataa para (batch, 128)
        x = self.dropout(x)
        # x = F.relu(self.fc1(x))
        x = self.fc1(x) # Não usar ReLu na última camada, pois o CrossEntropyLoss já aplica a Softmax
        return x

# Utilização de gradiente -> Cálculo 3

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model = SoybeanCNN(in_channels, num_class).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Treinamento do Modelo
def train_model(num_epochs):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    with mlflow.start_run():  # melhor envolver o treino inteiro
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("model_architecture", "SoybeanCNN")

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for data, targets in tqdm(train_loader):
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_train_loss = running_loss / len(train_loader)

            # validação
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(device), targets.to(device)
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            epoch_val_loss = val_loss / len(val_loader)

            # Logs para TensorBoard
            writer.add_scalar("Loss/Treino", epoch_train_loss, epoch)
            writer.add_scalar("Loss/Validação", epoch_val_loss, epoch)

            # Logs para MLflow
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)

            print(f"Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

            # early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), PATH)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping na epoch {epoch+1}")
                    break

        writer.close()

        # loga o modelo treinado no MLflow
        mlflow.pytorch.log_model(model, "soybean_model")
        mlflow.log_artifact(PATH)

if __name__ == "__main__":
    start = time.time()
    train_model(num_epochs)
    end = time.time()
    print(end - start)

    # Melhores pesos salvos
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()

    # Avaliação do Modelo -> Garante verificar as métricas após o treinamento.
    acc = Accuracy(task="multiclass", num_classes=num_class).to(device)
    precision = Precision(task="multiclass", num_classes=num_class).to(device)
    recall = Recall(task="multiclass", num_classes=num_class).to(device)
    f1 = F1Score(task="multiclass", num_classes=num_class).to(device)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            acc.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            f1.update(preds, labels)

    accuracy = acc.compute()
    prec = precision.compute()
    rec = recall.compute()
    f1_score = f1.compute()

    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {prec:.4f}")
    print(f"Revocação: {rec:.4f}")
    print(f"F1-Score: {f1_score:.4f}")

    with open(f"resultado_[4]{os.path.basename(data_root)}.txt", "w", encoding="utf-8") as f:
        print(f"Acurácia: {accuracy:.4f}", file=f)
        print(f"Precisão: {prec:.4f}", file=f)
        print(f"Revocação: {rec:.4f}", file=f)
        print(f"F1-Score: {f1_score:.4f}", file=f)

    print("Conteúdo salvo...")
    torch.save(model.state_dict(), PATH)
