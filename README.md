# Arquitetura da CNN
Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPool

# Fluxo da Inferência atualmente:
Imagem -> Preprocessamento -> Modelo -> Saída -> Softmax -> Classe + Confiança

- CNN -> Cria o modelo da rede neural  
- model.load_state_dict() -> Carregamos os pesos treinados  
- model.eval() -> Prepara o modelo para inferência  
- preprocess_imagem -> Lê, redimensiona, normaliza e transforma a imagem em tensor  
- predict_single_image -> Faz inferência em uma imagem e retorna classe + confiança  
- predict_batch -> Inferência em várias imagens e retorna lista de classes e confianças  

# Next Stage 
* Implementar CIFAR-10

    - Conteúdo: 60k de imagens coloridas  
    - Dimensões: Cada imagem tem 32x32 pixels  
    - Classes: 10 classes de objetos, com 6k imagens por classe  
    - Divisão: 50k imagens para treino e 10k para teste  

    Estrutura da rede CNN do MNIST:  
    convolução -> Pooling -> Convolução -> pooling -> fully connected  
    Dropout, ReLU, MaxPooling  

    Classes do CIFAR-10:  
    1. airplane  
    2. automobile  
    3. bird  
    4. cat  
    5. deer  
    6. dog  
    7. frog  
    8. horse  
    9. ship  
    10. truck  

# Resultados
* CNN -> Utilizando optimizer Adam:  
  - Acurácia: 0.9858  
  - Precisão: 0.9858  
  - Revocação: 0.9858  

* CNN -> Utilizando optimizer SGD:  
  - Acurácia: 0.9713  
  - Precisão: 0.9713  
  - Revocação: 0.9713  

# Empregar CNN + Reinforcement Learning
- CNN -> É utilizada para extrair características de imagens (feature extractor)  
- RL -> Usa essas características como entrada para decidir ações, receber recompensas e aprender uma política ótima.  

# Criar o dataset para os grãos de soja
1. Label Studio → rotular cada grão (via bounding box + classe).  
2. Detector de objetos (YOLOv8, Faster R-CNN) para automatizar localização dos grãos.  
3. Classificador CNN para prever a classe do grão.  
4. Avaliar métricas (acurácia, recall por classe).  
5. Depois que tiver bom desempenho, integrar RL se quiser otimizar decisões no processo industrial.

---

# Outras arquiteturas

- CNN  
- VGG-Net  
- Transfer Learning  
- Gradiente Descendente  
- Backpropagation (porta de entrada)  
- Consultar outras literaturas  
- MobileNet V2  
- MaxViT com Segmentação  

---

# Gráfico de Benchmark - Quantidade de dados

- Métricas: Acurácia, Precisão, Revocação  
- Relação: Quantidade de dados utilizada para treinamento  

# Dados coletados

**Total Inicial: 5513 imagens**  
- Broken soybeans: 1002  
- Immature soybeans: 1125  
- Intact soybeans: 1201  
- Skin-damaged soybeans: 1127  
- Spotted soybeans: 1058  

---

## Redução dos dados por porcentagem

| Categoria             | Inicial | -20% | -40% | -50% | -60% | -70% | -80% | -90% | -95%|
|-----------------------|---------|------|------|------|------|------|------|------|------
| Broken soybeans       | 1002    | 802  | 601  | 501  | 401  | 301  | 200  | 100  | 50  |
| Immature soybeans     | 1125    | 900  | 675  | 563  | 450  | 338  | 225  | 113  | 56  |
| Intact soybeans       | 1201    | 961  | 721  | 600  | 480  | 360  | 240  | 120  | 60  |
| Skin-damaged soybeans | 1127    | 902  | 676  | 564  | 451  | 338  | 225  | 113  | 56  |
| Spotted soybeans      | 1058    | 846  | 635  | 529  | 423  | 317  | 211  | 106  | 53  |


**Total após redução:**  
- 20% redução: 4410 imagens  
- 40% redução: 3308 imagens  
- 50% redução: 2757 imagens  
- 60% redução: 2205 imagens
- 70% redução: 1654 imagens
- 80% redução: 1103 imagens
- 90% redução: 551 imagens
- 95% redução: 276 imagens

**Tempo para treinamento**
- 100% - 5513 Imagens: 82.57 Minutos
- 80% - 4410 Imagens: 
- 60% - 3308 Imagens:
- 50% - 2757 Imagens:
- 40% - 2205 Imagens: 32.20 Minutos
- 30%: 22 minutos e 1 segundo
- 20%: 18 minutos e 8 segundos
- 10%:  27 min 09 s

## Reduzindo a quantidade de batchs de treinamentos

## Métricas Importantes
Accuracy
False Negatives
False Positives
True Negatives
True Positives
Precision
Recall
AUC
Confusion Matrix
Roc_curve

# Problemas Multiclasses:
  * Precisam ter multineuronios para cenários multiclasses.
  

# Val Loss estático:
  Taxa de aprendizado inadequada
  Problema de gradiente (vanishing e exploding)

# Melhorias
  Adicionar mais camadas convolucionais para obter mais informações
  Adicionar BatchNorm
  Remover ReLu na ultima camada, pois para classificação multiclasse o CrossEntropyLoss já aplica a Softmax