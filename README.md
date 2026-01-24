# SoyNet  Site com Backend (FastAPI) e Frontend (React)

Projeto para classificação de grãos de soja com API de inferência em Python (FastAPI) e interface web em React/TypeScript.

## Visão Geral
- Backend: ackend/ (FastAPI) expõe endpoints para inferência.
- Frontend: rontend/ (React + Vite) interface web para upload e visualização dos resultados.
- Modelos: models/ (arquivos .pth), usados pela camada de serviços.
- Dados: data/ (dataset processado) mantido no repositório original.

## Estrutura do Projeto

`
backend/               # API (FastAPI)
  main.py              # Inicialização do FastAPI, CORS, endpoints
  schemas.py           # Contratos de entrada/saída (Pydantic)
  routes/
    inference_routes.py# Exemplo/apoio para testes (opcional)
  services/
    inference_service.py # Lógica de negócio de inferência
  schemas/
    inference_schema.py  # Esquemas específicos de inferência

frontend/              # Interface Web (React/Vite)
  src/                 # Código da aplicação
  package.json         # Dependências e scripts

models/                # Pesos dos modelos (.pth)
data/                  # Dataset
`

## Pré-requisitos
- Python 3.10+ e pip
- Node.js 18+

## Instalação

1) Dependências do backend

`powershell
pip install -r requirements.txt
`

2) Dependências do frontend

`powershell
cd frontend
npm install
`

## Como Rodar

Backend (porta padrão 8000):

`powershell
uvicorn backend.main:app --reload
`

Frontend (Vite, porta padrão 5173):

`powershell
cd frontend
npm run dev
`

## Endpoints

- POST /inferencia
  - Form-data: model_name (string), iles (um ou mais arquivos de imagem)
  - Resposta: lista de objetos com ilename, classification, model_used

Exemplo de chamada (PowerShell):

`powershell
curl -Method POST -Uri http://127.0.0.1:8000/inferencia 
  -Form @{ model_name = 'EfficientNetB0'; files = Get-Item .\path\to\image.jpg }
`

## Desenvolvimento
- Convenção de imports internos: usar prefixo ackend. (ex.: ackend.services.inference_service).
- A camada services/ concentra a regra de negócio; rotas apenas delegam.
- Veja o guia detalhado em ackend/README.md.

## Licença
Este repositório contém dados e modelos proprietários; use com cuidado e evite commitar credenciais.
