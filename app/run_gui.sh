#!/bin/bash

# Script para executar a interface SoyNet
# Uso: ./run_gui.sh

echo "=== SoyNet - Interface de Classificação de Soja ==="
echo ""

# Verificar se o ambiente virtual existe
if [ ! -d "../soynet-env" ]; then
    echo "❌ Ambiente virtual 'soynet-env' não encontrado!"
    echo "   Certifique-se de estar no diretório correto do projeto."
    exit 1
fi

# Verificar se o arquivo da interface existe
if [ ! -f "soynet_gui.py" ]; then
    echo "❌ Arquivo 'soynet_gui.py' não encontrado!"
    echo "   Certifique-se de estar no diretório correto do projeto."
    exit 1
fi

# Verificar se há modelos disponíveis
if [ ! -d "../models" ] || [ -z "$(ls -A ../models/*.pth 2>/dev/null)" ]; then
    echo "⚠️  Atenção: Nenhum modelo (.pth) encontrado na pasta 'models/'!"
    echo "   A interface ainda funcionará, mas você precisará de modelos para fazer inferência."
    echo ""
fi

echo "✅ Verificações concluídas. Iniciando interface..."
echo ""

# Ativar ambiente virtual e executar interface
source ../soynet-env/bin/activate

echo "🚀 Executando SoyNet GUI..."
echo "   Dispositivo: $(python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')")"
echo ""

python soynet_gui.py

echo ""
echo "Interface finalizada. Obrigado por usar SoyNet! 🌱"