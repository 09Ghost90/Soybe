import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SoyNetGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SoyNet - Classificação de Soja")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Variáveis
        self.image_path = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Adicionar as classes -> Aqui!
        self.classes = {
            0: "Broken soybeans (Grãos Quebrados)",
            1: "Immature soybeans (Grãos Imaturos)",
            2: "Intact soybeans (Grãos Intactos)",
            3: "Skin-damaged soybeans (Grãos com Dano na Pele)",
            4: "Spotted soybeans (Grãos Manchados)"
        }
        
        self.available_models = self.scan_available_models()
        self.setup_ui()

    def scan_available_models(self):
        available_models = {}
        PATH = "../models"

        if not os.path.exists(PATH):
            print("Pasta models/ não encontrada!")
            return available_models
        
        files = os.listdir(PATH)
        pth_files = [f for f in files if f.endswith('.pth')]
            
        for file in pth_files:
            if 'efficientnet' in file.lower():
                model_type = "efficientnet"
            else:
                model_type = "cnn"
            
            display_name = file.replace('.pth', '')

            available_models[display_name] = {
                "path": os.path.join(PATH, file),
                "type": model_type
            }

            print(f"Modelo encontrado: {display_name} | Arquivo: {file} | Tipo: {model_type}")

        return available_models
    
    def setup_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Título
        title_label = ttk.Label(main_frame, text="SoyNet - Classificação de Soja", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Seção de seleção de modelo
        ttk.Label(main_frame, text="Selecione o Modelo:", font=("Arial", 10, "bold")).grid(
            row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(main_frame, textvariable=self.model_var, 
                                       values=list(self.available_models.keys()),
                                       state="readonly", width=50)
        self.model_combo.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        self.model_combo.set(list(self.available_models.keys())[0])  # Primeiro modelo como padrão
        
        # Botão para carregar modelo
        self.load_model_btn = ttk.Button(main_frame, text="Carregar Modelo", 
                                        command=self.load_model)
        self.load_model_btn.grid(row=3, column=0, columnspan=2, pady=(0, 20))
        
        # Seção de seleção de imagem
        ttk.Label(main_frame, text="Selecione a Imagem:", font=("Arial", 10, "bold")).grid(
            row=4, column=0, sticky=tk.W, pady=(0, 5))
        
        self.select_image_btn = ttk.Button(main_frame, text="Escolher Imagem", 
                                          command=self.select_image)
        self.select_image_btn.grid(row=5, column=0, sticky=tk.W, pady=(0, 10))
        
        # Label para mostrar o nome da imagem selecionada
        self.image_name_label = ttk.Label(main_frame, text="Nenhuma imagem selecionada", 
                                         foreground="gray")
        self.image_name_label.grid(row=5, column=1, sticky=tk.W, padx=(10, 0), pady=(0, 10))
        
        # Frame para preview da imagem
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.grid(row=6, column=0, columnspan=2, pady=(0, 20))
        
        self.image_label = ttk.Label(self.image_frame, text="Preview da imagem aparecerá aqui")
        self.image_label.pack()
        
        # Botão de inferência
        self.predict_btn = ttk.Button(main_frame, text="Realizar Inferência", 
                                     command=self.predict_image, state="disabled")
        self.predict_btn.grid(row=7, column=0, columnspan=2, pady=(0, 20))
        
        # Frame para resultados
        self.result_frame = ttk.LabelFrame(main_frame, text="Resultado da Classificação", padding="10")
        self.result_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.result_label = ttk.Label(self.result_frame, text="Execute a inferência para ver o resultado", 
                                     font=("Arial", 11))
        self.result_label.pack()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set(f"Dispositivo: {self.device}")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var, 
                                     foreground="blue", font=("Arial", 8))
        self.status_label.grid(row=9, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        
    def load_model(self):
        """Carrega o modelo selecionado"""
        try:
            selected_model = self.model_var.get()
            if not selected_model:
                messagebox.showerror("Erro", "Selecione um modelo primeiro!")
                return
            
            model_info = self.available_models[selected_model]
            model_path = model_info["path"]
            model_type = model_info["type"]
            
            if not os.path.exists(model_path):
                messagebox.showerror("Erro", f"Arquivo do modelo não encontrado: {model_path}")
                return
            
            self.status_var.set("Carregando modelo...")
            self.root.update()
            
            # Criar o modelo baseado no tipo
            if model_type == "cnn":
                self.model = self.create_cnn_model()
            else:  # efficientnet
                self.model = self.create_efficientnet_model()
            
            # Carregar os pesos
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.status_var.set(f"Modelo carregado: {selected_model} | Dispositivo: {self.device}")
            messagebox.showinfo("Sucesso", "Modelo carregado com sucesso!")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar modelo: {str(e)}")
            self.status_var.set(f"Erro ao carregar modelo | Dispositivo: {self.device}")
    
    def create_cnn_model(self):
        """Cria modelo CNN personalizado"""
        class SoybeanCNN(torch.nn.Module):
            def __init__(self, in_channels=3, num_class=5):
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
        
        return SoybeanCNN()
    
    def create_efficientnet_model(self):
        """Cria modelo EfficientNet"""
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 5)
        return model
    
    def select_image(self):
        """Seleciona uma imagem para inferência"""
        file_types = [
            ('Imagens', '*.jpg *.jpeg *.png *.bmp *.tiff *.tif'),
            ('JPEG', '*.jpg *.jpeg'),
            ('PNG', '*.png'),
            ('Todos os arquivos', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Selecionar Imagem",
            filetypes=file_types,
            initialdir="../data/processed"
        )
        
        if filename:
            self.image_path = filename
            # Atualizar label com nome do arquivo
            image_name = os.path.basename(filename)
            self.image_name_label.config(text=image_name, foreground="black")
            
            # Mostrar preview da imagem
            self.show_image_preview(filename)
            
            # Habilitar botão de inferência se modelo estiver carregado
            if self.model is not None:
                self.predict_btn.config(state="normal")
    
    def show_image_preview(self, image_path):
        """Mostra preview da imagem selecionada"""
        try:
            image = Image.open(image_path)
            # Redimensionar para preview
            image.thumbnail((300, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Manter referência
            
        except Exception as e:
            self.image_label.config(text=f"Erro ao carregar preview: {str(e)}")
    
    def predict_image(self):
        """Realiza inferência na imagem selecionada"""
        if not self.image_path:
            messagebox.showerror("Erro", "Selecione uma imagem primeiro!")
            return
        
        if self.model is None:
            messagebox.showerror("Erro", "Carregue um modelo primeiro!")
            return
        
        try:
            self.status_var.set("Realizando inferência...")
            self.root.update()
            
            # Pré-processamento
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            # Carregar e processar imagem
            image = Image.open(self.image_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Inferência
            with torch.no_grad():
                output = self.model(input_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
                probabilities = F.softmax(output, dim=1)
                confidence = probabilities[0][predicted_class].item()
            
            # Mostrar resultado
            class_name = self.classes[predicted_class]
            confidence_percent = confidence * 100
            
            result_text = f"Classe Predita: {class_name}\nConfiança: {confidence_percent:.2f}%"
            self.result_label.config(text=result_text, foreground="green", font=("Arial", 12, "bold"))
            
            self.status_var.set(f"Inferência concluída | Dispositivo: {self.device}")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro durante inferência: {str(e)}")
            self.status_var.set(f"Erro na inferência | Dispositivo: {self.device}")

def main():
    root = tk.Tk()
    app = SoyNetGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()