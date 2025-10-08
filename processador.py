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
import sys
import cv2
import numpy as np
import rawpy
from shutil import move
from skimage.metrics import structural_similarity as ssim

# --- 1. CONFIGURAÇÃO DOS DIRETÓRIOS ---
BASE_DIR = os.getcwd()
ORIGINAIS_DIR = os.path.join(BASE_DIR, "1_IMAGENS_ORIGINAIS")
TRATADAS_DIR = os.path.join(BASE_DIR, "2_IMAGENS_TRATADAS")
RECORTADOS_DIR = os.path.join(BASE_DIR, "3_GRAOS_RECORTADOS")
OUTLIERS_DIR = os.path.join(BASE_DIR, "4_OUTLIERS")
BASE_GRAO_DIR = os.path.join(BASE_DIR, "5_GRAO_REFERENCIA")

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

def separar_graos(imagem_tratada_path, imagem_original_path, tipo_grao):
    """Apenas recortas os grãos e os salva temporariamente."""
    print(f" [Passo 2/3] Separando todos os objetos detectados...")

    dir_graos_temp = os.path.join(RECORTADOS_DIR, tipo_grao, "_temp")
    os.makedirs(dir_graos_temp, exist_ok=True)

    mascara_geral = cv2.imread(imagem_tratada_path, cv2.IMREAD_GRAYSCALE)
    original = ler_imagem(imagem_original_path)
    if mascara_geral is None or original is None: 
        return 0
    
    contornos, _ = cv2.findContours(mascara_geral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    nome_base = os.path.splitext(os.path.basename(imagem_original_path))[0]
    contador_objetos = 0

    for i, contorno in enumerate(contornos):
        if cv2.contourArea(contorno) < 100:
            continue
        x, y, w, h = cv2.boundingRect(contorno)
        grao_recortado = original[y:y+h, x:x+w]
        if grao_recortado.size == 0:
            continue
        mascara_local = np.zeros(grao_recortado.shape[:2], dtype=np.uint8)
        contorno_local = contorno - [x, y]
        cv2.drawContours(mascara_local, [contorno_local], -1, (255), thickness=cv2.FILLED)

        b, g, r = cv2.split(grao_recortado)
        grao_transparente = cv2.merge((b, g, r, mascara_local))

        # Salvar o grão recortado e sem fundo
        output_path = os.path.join(dir_graos_temp, f"{nome_base}_obj_{i+1}.png")
        cv2.imwrite(output_path, grao_transparente)
        contador_objetos += 1

    print(f"  [INFO] {contador_objetos} objetos recortados e salvos temporariamente.")
    return contador_objetos

def separar_graos(imagem_tratada_path, imagem_original_path, tipo_grao):
    """Recorta cada grão individual com fundo transparente (alpha)."""
    print(f" [Passo 2/3] Separando todos os objetos detectados...")

    dir_graos_temp = os.path.join(RECORTADOS_DIR, tipo_grao, "_temp")
    os.makedirs(dir_graos_temp, exist_ok=True)

    mascara_geral = cv2.imread(imagem_tratada_path, cv2.IMREAD_GRAYSCALE)
    original = ler_imagem(imagem_original_path)
    if mascara_geral is None or original is None:
        return 0

    contornos, _ = cv2.findContours(mascara_geral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    nome_base = os.path.splitext(os.path.basename(imagem_original_path))[0]
    contador_objetos = 0

    for i, contorno in enumerate(contornos):
        if cv2.contourArea(contorno) < 100:
            continue
        x, y, w, h = cv2.boundingRect(contorno)
        grao_recortado = original[y:y+h, x:x+w]
        if grao_recortado.size == 0:
            continue
        mascara_local = np.zeros(grao_recortado.shape[:2], dtype=np.uint8)
        contorno_local = contorno - [x, y]
        cv2.drawContours(mascara_local, [contorno_local], -1, (255), thickness=cv2.FILLED)

        # Cria imagem com alpha
        b, g, r = cv2.split(grao_recortado)
        grao_transparente = cv2.merge((b, g, r, mascara_local))  # alpha = máscara

        output_path = os.path.join(dir_graos_temp, f"{nome_base}_obj_{i+1}.png")
        cv2.imwrite(output_path, grao_transparente)
        contador_objetos += 1

    print(f"  [INFO] {contador_objetos} objetos recortados e salvos temporariamente (fundo transparente).")
    return contador_objetos


# --- Função auxiliar para comparar apenas os pixels visíveis (alpha) ---

def ssim_masked(imgA, imgB):
    """
    Calcula SSIM apenas sobre os pixels visíveis (alpha > 0) nas duas imagens.
    Suporta PNG com fundo transparente (BGRA) ou BGR.
    """
    if imgA is None or imgB is None:
        return 0.0

    # garante BGRA
    if imgA.shape[2] == 3:
        alphaA = np.any(imgA > 10, axis=2).astype(np.uint8) * 255
        imgA = cv2.merge((*cv2.split(imgA), alphaA))
    if imgB.shape[2] == 3:
        alphaB = np.any(imgB > 10, axis=2).astype(np.uint8) * 255
        imgB = cv2.merge((*cv2.split(imgB), alphaB))

    alphaA = imgA[..., 3] > 0
    alphaB = imgB[..., 3] > 0
    mask = np.logical_and(alphaA, alphaB)

    grayA = cv2.cvtColor(imgA[..., :3], cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB[..., :3], cv2.COLOR_BGR2GRAY)

    if np.count_nonzero(mask) == 0:
        return 0.0  # nada a comparar

    # aplica máscara
    valsA = grayA[mask]
    valsB = grayB[mask]

    # normaliza tamanhos
    m = min(len(valsA), len(valsB))
    if m == 0:
        return 0.0

    return ssim(valsA[:m], valsB[:m], data_range=255)


def achar_outliers(tipo_grao, grao_base_controle, debug=False):
    """
    Versão com suporte a transparência.
    Compara apenas pixels visíveis (alpha > 0) e filtra por tamanho e similaridade.
    """
    print(f"  [Passo 3/3] Verificando outliers na categoria '{tipo_grao}'...")

    dir_origem = os.path.join(RECORTADOS_DIR, tipo_grao, "_temp")
    dir_destino_outliers = os.path.join(OUTLIERS_DIR, tipo_grao)
    os.makedirs(dir_destino_outliers, exist_ok=True)

    if not os.path.isdir(dir_origem):
        print(f"    [AVISO] Diretório de origem não encontrado: {dir_origem}")
        return

    base = grao_base_controle
    if base is None:
        print("    [ERRO] Grão de referência inválido.")
        return

    dy, dx = base.shape[0], base.shape[1]
    THRESH_SSIM = 0.60

    for nome_arquivo in list(os.listdir(dir_origem)):
        caminho_origem = os.path.join(dir_origem, nome_arquivo)
        if not nome_arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        grao_atual = cv2.imread(caminho_origem, cv2.IMREAD_UNCHANGED)
        if grao_atual is None:
            continue

        # filtro de tamanho
        if grao_atual.shape[0] < 80 or grao_atual.shape[1] < 80:
            move(caminho_origem, os.path.join(dir_destino_outliers, nome_arquivo))
            if debug:
                print(f"    [Tamanho] {nome_arquivo} movido (muito pequeno).")
            continue

        # redimensiona mantendo proporção
        grao_redimensionado = cv2.resize(grao_atual, (dx, dy), interpolation=cv2.INTER_AREA)

        score = ssim_masked(base, grao_redimensionado)

        if debug:
            print(f"    [DEBUG] {nome_arquivo} -> SSIM_masked={score:.3f}")

        if score <= THRESH_SSIM:
            move(caminho_origem, os.path.join(dir_destino_outliers, nome_arquivo))
            if debug:
                print(f"    [Outlier] {nome_arquivo} movido (score={score:.3f}).")


# --- 3. FUNÇÃO PRINCIPAL DE EXECUÇÃO ---

def main():
    modo_execucao = "completo"
    for arg in sys.argv[1:]:
        if arg.startswith("--modo="):
            modo_execucao = arg.split("=")[1]

    print(f"--- INICIANDO PROCESSAMENTO (Modo: {modo_execucao.upper()}) ---")

    caminho_base = os.path.join(BASE_GRAO_DIR, "grao_base.png")
    if not os.path.exists(caminho_base):
        print(f"\n[ERRO FATAL] Imagem de referência não encontrada em: '{caminho_base}'")
        return
    grao_base_controle = cv2.imread(caminho_base)
    print(f"Grão de referência '{caminho_base}' carregado com sucesso.")

    if modo_execucao == "outliers":
        if not os.path.isdir(RECORTADOS_DIR):
            print(f"[ERRO] A pasta '3_graos_recortados' não existe. Rode o modo completo primeiro.")
            return
        categorias = [d for d in os.listdir(RECORTADOS_DIR) if os.path.isdir(os.path.join(RECORTADOS_DIR, d))]
    else:
        if not os.path.isdir(ORIGINAIS_DIR):
            print(f"\n[ERRO] O diretório '1_imagens_originais' não existe ou está vazio.")
            return
        categorias = os.listdir(ORIGINAIS_DIR)

    for tipo_grao in categorias:
        print(f"\nProcessando categoria: '{tipo_grao}'")
        
        if modo_execucao == "completo":
            dir_categoria = os.path.join(ORIGINAIS_DIR, tipo_grao)
            if not os.path.isdir(dir_categoria): continue

            dir_recortados_cat = os.path.join(RECORTADOS_DIR, tipo_grao)
            if os.path.isdir(dir_recortados_cat):
                for f in os.listdir(dir_recortados_cat):
                    if os.path.isfile(os.path.join(dir_recortados_cat, f)):
                        os.remove(os.path.join(dir_recortados_cat, f))
            
            arquivos_processados = set()
            todos_os_arquivos = os.listdir(dir_categoria)
            arquivos_nef = [f for f in todos_os_arquivos if f.lower().endswith('.nef')]
            outros_arquivos = [f for f in todos_os_arquivos if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for nome_imagem in arquivos_nef + outros_arquivos:
                nome_base = os.path.splitext(nome_imagem)[0]
                if nome_base in arquivos_processados: continue
                
                imagem_original_path = os.path.join(dir_categoria, nome_imagem)
                imagem_tratada_path = tratar_imagem(imagem_original_path)
                if imagem_tratada_path:
                    separar_graos(imagem_tratada_path, imagem_original_path, tipo_grao)
                
                arquivos_processados.add(nome_base)
        
        achar_outliers(tipo_grao, grao_base_controle)

    print("\n--- PROCESSAMENTO FINALIZADO ---")


if __name__ == "__main__":
    main()
