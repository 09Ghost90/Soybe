# SoyNet - Interface de Classificação de Soja

Esta é uma interface gráfica simples para realizar inferência em imagens de grãos de soja usando modelos treinados.

## Funcionalidades

- **Seleção de Modelo**: Escolha entre diferentes modelos treinados:
  - CNN Personalizada (várias versões)
  - EfficientNet-B0 (94% e 96% de acurácia)

- **Seleção de Imagem**: Navegue e selecione imagens para classificação

- **Preview da Imagem**: Visualize a imagem selecionada antes da inferência

- **Classificação**: Obtenha a classe predita e o nível de confiança

## Classes Disponíveis

1. **Broken soybeans** (Grãos Quebrados)
2. **Immature soybeans** (Grãos Imaturos) 
3. **Intact soybeans** (Grãos Intactos)
4. **Skin-damaged soybeans** (Grãos com Dano na Pele)
5. **Spotted soybeans** (Grãos Manchados)

## (Adicionar Mais Classes)

## Como Usar

### Pré-requisitos

1. **Ambiente Python**: Certifique-se de ter Python 3.x instalado
2. **Dependências**: Execute os comandos abaixo para instalar as dependências

```bash
# Instale o tkinter (interface gráfica)
sudo apt-get update
sudo apt-get install python3-tk

# Ative o ambiente virtual
source soynet-env/bin/activate

# Instale dependências Python (se necessário)
pip install torch torchvision pillow
```

### Executando a Interface

1. **Abra um terminal** no diretório do projeto

2. **Ative o ambiente virtual**:
```bash
source soynet-env/bin/activate
```

3. **Execute a interface**:
```bash
python soynet_gui.py
```

### Usando a Interface

1. **Carregar Modelo**:
   - Selecione um modelo na lista suspensa
   - Clique em "Carregar Modelo"
   - Aguarde a confirmação de carregamento

2. **Selecionar Imagem**:
   - Clique em "Escolher Imagem"
   - Navegue até a imagem desejada (recomendado: `data/processed/`)
   - Visualize o preview da imagem

3. **Realizar Inferência**:
   - Com modelo carregado e imagem selecionada
   - Clique em "Realizar Inferência"
   - Veja o resultado com a classe predita e confiança

## Estrutura dos Modelos

### CNN Personalizada
- Arquitetura: 3 camadas convolucionais + pooling + classificador
- Entrada: Imagens 224x224 RGB
- Saída: 5 classes de soja

### EfficientNet-B0
- Arquitetura: EfficientNet pré-treinado adaptado
- Entrada: Imagens 224x224 RGB  
- Saída: 5 classes de soja

## Dicas de Uso

- **Melhor Performance**: Use GPU se disponível (será detectada automaticamente)
- **Imagens Recomendadas**: JPG, PNG com boa resolução
- **Modelos Recomendados**: EfficientNet-B0 96% para melhor acurácia
- **Localização das Imagens**: Use imagens da pasta `data/processed/` para testes

## Troubleshooting

**Erro de tkinter**: Instale com `sudo apt-get install python3-tk`
**Erro de PIL/Pillow**: Use o ambiente virtual com `source soynet-env/bin/activate`
**Modelo não encontrado**: Verifique se os arquivos .pth estão na pasta `models/`
**Erro de CUDA**: O sistema irá usar CPU automaticamente se GPU não disponível

## Apresentação

Esta interface foi criada para facilitar a demonstração do projeto SoyNet, permitindo:
- Teste rápido de diferentes modelos
- Classificação interativa de imagens
- Visualização clara dos resultados
- Interface intuitiva para apresentações

---

**Desenvolvido para apresentação do projeto SoyNet**