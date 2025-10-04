# -------------------------------------
# Pré-processamento de imagens de grãos (Versão 4.0)
# Baseado na lógica original do usuário, mas totalmente automatizado.
# -------------------------------------
"""
Este script faz:
1. Tratamento das imagens para criar uma máscara (lógica CMYK original).
2. Separação e recorte preciso de cada grão individual.
3. Remoção do fundo de cada grão, deixando-o preto.
4. Filtro de outliers (sujeira) com base no tamanho do recorte.
5. Processamento em lote de todas as categorias e imagens, priorizando NEF sobre JPG.
"""

import os
import cv2
import numpy as np
import rawpy
from shutil import move

# --- 1. CONFIGURAÇÃO DOS DIRETÓRIOS ---
BASE_DIR = os.getcwd()
ORIGINAIS_DIR = os.path.join(BASE_DIR, "1_IMAGENS_ORIGINAIS")
TRATADAS_DIR = os.path.join(BASE_DIR, "2_IMAGENS_TRATADAS")
RECORTADOS_DIR = os.path.join(BASE_DIR, "3_GRAOS_RECORTADOS")
OUTLIERS_DIR = os.path.join(BASE_DIR, "4_OUTLIERS")

# --- 2. FUNÇÕES DE PROCESSAMENTO (ADAPTADAS DO SCRIPT ORIGINAL) ---

def ler_imagem(path):
    """Lê imagens nos formatos JPG, PNG ou NEF."""
    try:
        if path.lower().endswith(".nef"):
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess()
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(path)
            if img is None: raise FileNotFoundError
            return img
    except Exception:
        print(f"    [ERRO] Falha ao ler a imagem: {path}")
        return None

def tratar_imagem(imagem_original_path):
    """Cria a máscara binária usando a lógica original de conversão CMYK."""
    print(f"  [Passo 1/2] Tratando a imagem: {os.path.basename(imagem_original_path)}")
    os.makedirs(TRATADAS_DIR, exist_ok=True)

    nome_base = os.path.splitext(os.path.basename(imagem_original_path))[0]
    output_path = os.path.join(TRATADAS_DIR, f"{nome_base}_mascara.png")

    rgb = ler_imagem(imagem_original_path)
    if rgb is None: return None

    # Lógica de conversão CMYK do script original
    rgbdash = rgb.astype(np.float32) / 255.
    K = 1 - np.max(rgbdash, axis=2)
    # Evita divisão por zero
    K[K == 1] = 1 - 1e-6
    C = (1 - rgbdash[..., 2] - K) / (1 - K)
    M = (1 - rgbdash[..., 1] - K) / (1 - K)
    Y = (1 - rgbdash[..., 0] - K) / (1 - K)
    CMYK = (np.dstack((C, M, Y, K)) * 255).astype(np.uint8)
    _, _, Y_channel, _ = cv2.split(CMYK)

    # Threshold com Otsu para criar a máscara
    _, mascara = cv2.threshold(Y_channel, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imwrite(output_path, mascara)
    print(f"    Máscara salva em: {output_path}")
    return output_path

def separar_e_filtrar_graos(imagem_tratada_path, imagem_original_path, tipo_grao):
    """Recorta os grãos com precisão e filtra outliers por tamanho."""
    print(f"  [Passo 2/2] Separando grãos e filtrando...")

    dir_graos_bons = os.path.join(RECORTADOS_DIR, tipo_grao)
    dir_outliers = os.path.join(OUTLIERS_DIR, tipo_grao)
    os.makedirs(dir_graos_bons, exist_ok=True)
    os.makedirs(dir_outliers, exist_ok=True)

    mascara_geral = cv2.imread(imagem_tratada_path, cv2.IMREAD_GRAYSCALE)
    original = ler_imagem(imagem_original_path)
    if mascara_geral is None or original is None: return

    # Encontra os contornos na máscara geral
    contornos, _ = cv2.findContours(mascara_geral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    nome_base = os.path.splitext(os.path.basename(imagem_original_path))[0]
    cont_graos = 0
    cont_outliers = 0

    for i, contorno in enumerate(contornos):
        # Pega o retângulo que envolve o grão para definir a área de recorte
        x, y, w, h = cv2.boundingRect(contorno)

        # Recorta o pedaço da imagem original
        grao_recortado = original[y:y+h, x:x+w]

        # Se o recorte for vazio, pula para o próximo
        if grao_recortado.size == 0:
            continue

        # --- LÓGICA DE RECORTE PRECISO (DO SEU SCRIPT ORIGINAL) ---
        # 1. Cria uma nova máscara preta, do mesmo tamanho do recorte
        mascara_local = np.zeros(grao_recortado.shape[:2], dtype=np.uint8)
        # 2. Move o contorno para as coordenadas locais do recorte (0,0)
        contorno_local = contorno - (x, y)
        # 3. Desenha o contorno preenchido em branco na máscara local
        cv2.drawContours(mascara_local, [contorno_local], -1, (255), thickness=cv2.FILLED)
        # 4. Usa a máscara local para extrair o grão do recorte (bitwise_and)
        grao_sem_fundo = cv2.bitwise_and(grao_recortado, grao_recortado, mask=mascara_local)
        # --- FIM DA LÓGICA DE RECORTE ---

        # Filtro por tamanho (outlier se for menor que 160px em largura ou altura)
        if w < 135 or h < 135:
            output_path = os.path.join(dir_outliers, f"{nome_base}_outlier_{i+1}.png")
            cv2.imwrite(output_path, grao_sem_fundo)
            cont_outliers += 1
        else:
            output_path = os.path.join(dir_graos_bons, f"{nome_base}_grao_{i+1}.png")
            cv2.imwrite(output_path, grao_sem_fundo)
            cont_graos += 1

    print(f"    Análise concluída: {cont_graos} grãos salvos, {cont_outliers} outliers movidos.")

# --- 3. FUNÇÃO PRINCIPAL DE EXECUÇÃO ---

def main():
    """Orquestra todo o processo de forma automatizada."""
    print("--- INICIANDO PROCESSAMENTO DE IMAGENS DE SOJA (v4.0) ---")

    if not os.path.isdir(ORIGINAIS_DIR) or not os.listdir(ORIGINAIS_DIR):
        print(f"\n[ERRO] O diretório '1_imagens_originais' está vazio ou não existe.")
        return

    for tipo_grao in os.listdir(ORIGINAIS_DIR):
        dir_categoria = os.path.join(ORIGINAIS_DIR, tipo_grao)

        if os.path.isdir(dir_categoria):
            print(f"\nProcessando categoria: '{tipo_grao}'")
            
            arquivos_processados = set()
            todos_os_arquivos = os.listdir(dir_categoria)
            
            # Prioriza arquivos .NEF
            arquivos_nef = [f for f in todos_os_arquivos if f.lower().endswith('.nef')]
            for nome_imagem in arquivos_nef:
                nome_base = os.path.splitext(nome_imagem)[0]
                imagem_original_path = os.path.join(dir_categoria, nome_imagem)
                
                imagem_tratada_path = tratar_imagem(imagem_original_path)
                if imagem_tratada_path:
                    separar_e_filtrar_graos(imagem_tratada_path, imagem_original_path, tipo_grao)
                
                arquivos_processados.add(nome_base)

            # Processa outros formatos se não houver .NEF
            outros_arquivos = [f for f in todos_os_arquivos if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for nome_imagem in outros_arquivos:
                nome_base = os.path.splitext(nome_imagem)[0]
                
                if nome_base not in arquivos_processados:
                    imagem_original_path = os.path.join(dir_categoria, nome_imagem)
                    
                    imagem_tratada_path = tratar_imagem(imagem_original_path)
                    if imagem_tratada_path:
                        separar_e_filtrar_graos(imagem_tratada_path, imagem_original_path, tipo_grao)
                    
                    arquivos_processados.add(nome_base)
                else:
                    print(f"  [INFO] Pulando '{nome_imagem}', pois uma versão de maior qualidade (.NEF) já foi processada.")

    print("\n--- PROCESSAMENTO FINALIZADO ---")

if __name__ == "__main__":
    main()
