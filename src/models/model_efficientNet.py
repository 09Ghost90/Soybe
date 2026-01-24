import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.transforms import RandAugment
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

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
PATH = "./models/soybean_model_efficientnet.pth"
data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/dataset"))

data_transforms = T.Compose([
    T.Resize((224,224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

image_dataset = ImageFolder(root=data_root, transform=data_transforms)

train_size = int(0.8 * len(image_dataset))
val_size = int(0.1 * len(image_dataset))
test_size = len(image_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(image_dataset, [train_size, val_size, test_size])

# Loader -> Remover gargalo da CPU em preparar os dados
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Embaralha os dados
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # Não embaralha os dados
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # Não embaralha os dados

# Modelo EfficientNet pré-treinado
model = torchvision.models.efficientnet_b0(pretrained=True)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, num_class)  # Ajusta a camada final
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Treinamento do Modelo
def train_model(num_epochs):
    best_val_loss = float('inf')
    epochs_no_improve = 0

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

if __name__ == "__main__":
    start = time.time()
    train_model(num_epochs)
    end = time.time()
    print(end - start)

    # Melhores pesos salvos
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()

    # Avaliação do Modelo -> Garante verificar as métricas após o treinamento.
    acc = Accuracy(task="multiclass", num_classes=num_class, average="macro").to(device)  # macro para acurácia agregada
    precision = Precision(task="multiclass", num_classes=num_class, average=None).to(device)
    recall = Recall(task="multiclass", num_classes=num_class, average=None).to(device)
    f1 = F1Score(task="multiclass", num_classes=num_class, average=None).to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Atualiza as métricas
            acc.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            f1.update(preds, labels)

            # Matriz de Confusão
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # computa métricas
    accuracy = acc.compute().item()
    prec_per_class = precision.compute().cpu().numpy()  # vetor de tamanho num_class
    rec_per_class = recall.compute().cpu().numpy()
    f1_per_class = f1.compute().cpu().numpy()

    print(f"Acurácia (macro): {accuracy:.4f}\n")

    class_names = image_dataset.classes if hasattr(image_dataset, "classes") else [f"class_{i}" for i in range(num_class)]

    for i, cname in enumerate(class_names):
        print(f"Classe: {cname:15s}  Precision: {prec_per_class[i]:.4f}  Recall: {rec_per_class[i]:.4f}  F1: {f1_per_class[i]:.4f}")

    print("\n\nClassification Report (sklearn):\n")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix (raw counts):\n", cm)

    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    print("\nConfusion Matrix (normalized by true class / recall):\n", np.round(cm_norm, 3))
    print("Conteúdo salvo...")
    torch.save(model.state_dict(), PATH)
