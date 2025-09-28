# -*- coding: utf-8 -*-
"""
Pré-processamento de imagens de grãos
-------------------------------------
Este script faz:
1. Tratamento das imagens (filtro + binarização)
2. Separação de cada grão individual em arquivos separados
3. Identificação e remoção de outliers (grãos fora do padrão)
4. Processamento em lote de várias imagens

Bibliotecas usadas:
- OpenCV (cv2): processamento de imagem
- NumPy: operações numéricas
- skimage.metrics (SSIM): comparação de similaridade entre imagens
- shutil: mover arquivos (usado nos outliers)
- os: manipulação de diretórios
"""

import os
import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim
from shutil import move

# -------------------------------
# 1. Função para tratamento da imagem
# -------------------------------
def Tratamento_imagem(imagem_path):
    """
    Aplica pré-processamento em uma imagem:
    - Converte para CMYK
    - Usa apenas o canal Y (amarelo)
    - Aplica filtro e binarização (threshold)
    - Salva a imagem tratada
    
    Inputs:
        imagem_path (str) → caminho da imagem original
    Output:
        Retorna 1 apenas como confirmação
        (a imagem tratada é salva em disco)
    """

    # Pasta de saída para as imagens tratadas
    output_atual = "./output"
    # Nome do arquivo de saída (pega os últimos dígitos do nome original)
    output_path = f"{output_atual}/{imagem_path[-13:-4]}.png"

    # Lê a imagem em RGB
    rgb = cv.imread(imagem_path)
    # Normaliza os valores para [0,1]
    rgbdash = rgb.astype(np.float32) / 255.

    # Converte para CMYK
    K = 1 - np.max(rgbdash, axis=2)
    C = (1 - rgbdash[..., 2] - K) / (1 - K)
    M = (1 - rgbdash[..., 1] - K) / (1 - K)
    Y = (1 - rgbdash[..., 0] - K) / (1 - K)

    # Junta canais em uma imagem CMYK
    CMYK = (np.dstack((C, M, Y, K)) * 255).astype(np.uint8)

    # Seleciona apenas o canal Y (amarelo)
    Y, _, _, _ = cv.split(CMYK)

    # Converte Y para BGR para aplicar o filtro
    Y_3channel = cv.cvtColor(Y, cv.COLOR_GRAY2BGR)

    # Filtro de suavização
    filtro = cv.pyrMeanShiftFiltering(Y_3channel, 30, 90)

    # Converte para escala de cinza
    gray = cv.cvtColor(filtro, cv.COLOR_BGR2GRAY)

    # Aplica threshold binário (Otsu) para destacar os grãos
    tratado = cv.threshold(
        gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU
    )[1]

    # Salva a imagem tratada
    cv.imwrite(output_path, tratado)

    # Exibindo a imagem
    cv.imshow("Imagem tratada", tratado)
    cv.waiKey(0)
    cv.destroyAllWindows()

    return 1

# -------------------------------
# 2. Função para separar os grãos
# -------------------------------
def Separar_graos(imagem_tratada, imagem_original, tipo):
    """
    Separa os grãos de uma imagem já tratada e recorta cada um deles.
    
    Inputs:
        imagem_tratada (str)  → caminho da imagem binária (tratada)
        imagem_original (str) → caminho da imagem original (colorida)
        tipo (str)            → tipo de grão (normal, quebrado, etc.)
    Output:
        Retorna 1 apenas como confirmação
        (os grãos são salvos em disco)
    """

    # Pasta de saída (separada por tipo de grão)
    output_dir = os.path.join("./dados_separados", tipo)

    # Nome base para salvar os grãos
    buffer_path = os.path.basename(imagem_original).replace(".JPG", "")

    # Lê as imagens
    img = cv.imread(imagem_tratada)   # tratada (binária)
    original = cv.imread(imagem_original) # original (colorida)

    # Detecta bordas usando Canny
    edge = cv.Canny(img, 100, 230, L2gradient=True)

    # Encontra contornos (cada grão é um contorno)
    cont, _ = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    for i, contour in enumerate(cont):
        # Encontra o círculo mínimo que envolve o contorno
        center, radius = cv.minEnclosingCircle(contour)

        # Define margens para o recorte
        x_margin = 10
        x1 = int(center[0] - radius - x_margin)
        y1 = int(center[1] - radius - x_margin)
        x2 = int(center[0] + radius + x_margin)
        y2 = int(center[1] + radius + x_margin)

        # Garante que os limites estão dentro da imagem
        h, w = img.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        # Recorta o grão da imagem original
        grain_cropped = original[y1:y2, x1:x2]

        # Salva o grão
        output_path = os.path.join(output_dir, f"grao_{buffer_path}_{i+1}.png")
        cv.imwrite(output_path, grain_cropped)

    return 1

# -------------------------------
# 3. Função para tratar várias imagens
# -------------------------------
def Tratamento_imagens(dir_path, tipo):
    """
    Processa todas as imagens de uma pasta:
    - Aplica o tratamento
    - Separa os grãos
    
    Inputs:
        dir_path (str) → diretório com imagens originais
        tipo (str)     → tipo de grão
    """
    output_path = "/content/drive/MyDrive/Dados_tratados"

    if not os.path.isdir(dir_path):
        raise ValueError("Diretório inválido.")

    for filename in os.listdir(dir_path):
        if filename.endswith(('.JPG', '.jpeg', '.png')):
            imagem_path = os.path.join(dir_path, filename)
            print(f"Processando: {filename}")

            # Passo 1: tratamento
            Tratamento_imagem(imagem_path)

            # Caminho da imagem tratada gerada
            imagem_tratada = os.path.join(output_path, f"{filename[:-4]}.png")

            # Passo 2: separação dos grãos
            Separar_graos(imagem_tratada, imagem_path, tipo)

    return 1

# -------------------------------
# 4. Função para encontrar outliers
# -------------------------------
def achar_outliers(dir_path):
    """
    Identifica grãos fora do padrão comparando com um grão "controle"
    usando SSIM (Structural Similarity Index).
    
    Inputs:
        dir_path (str) → pasta com grãos separados
    """
    # Grão de referência (controle)
    path_controle = "/content/drive/MyDrive/Dados_outliers/controleN.png"
    controle = cv.imread(path_controle)

    print("Entrando na função achar_outliers")

    # Pasta de destino para grãos fora do padrão
    dir_out = "/content/drive/MyDrive/Dados_outliers/normais"

    # Dimensões do controle
    dy, dx, _ = controle.shape

    for imagem in os.listdir(dir_path):
        grao = cv.imread(f"{dir_path}/{imagem}")

        # Remove grãos muito pequenos
        if grao.shape[0] < 100 or grao.shape[1] < 100:
            move(f"{dir_path}/{imagem}", dir_out)
            continue

        # Redimensiona o grão para o mesmo tamanho do controle
        grao = cv.resize(grao, (dx, dy))

        # Calcula a similaridade estrutural
        score, dif = ssim(controle, grao, full=True, channel_axis=2)

        # Se a similaridade for baixa, move para a pasta de outliers
        if score <= 0.60:
            move(f"{dir_path}/{imagem}", dir_out)

    return 1

# -------------------------------
# 5. Execução principal
# -------------------------------

# Tratamento de uma imagem
Tratamento_imagem("./dataset/Grão Bom - 9.3 Umidade - 18-03-2025/940nm_2500ex_0001.bmp")

# Separação dos grãos dessa imagem
Separar_graos(
    "dataset/Grão Bom - 9.3 Umidade - 18-03-2025/940nm_2500ex_0001.bmp",
    "dataset/Dados_grider/940nm_2500ex_0001.bmp",
    "normal"
)

# Identificação de outliers na pasta de grãos separados
achar_outliers("dataset/Grão Bom - 9.3 Umidade - 18-03-2025")

# Mostra os arquivos salvos
# print(os.listdir("/content/drive/MyDrive/Dados_separados/normal"))

# Processa todas as imagens do diretório
Tratamento_imagens("dataset/Grão Bom - 9.3 Umidade - 18-03-2025", "normal")
