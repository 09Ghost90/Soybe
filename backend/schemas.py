# O arquivo schemas.py defini os modelos de dados usados na API

from pydantic import BaseModel
from typing import List

# BaseModel: modelo base para requisições e respostas
# Definição do contrato da API

# Aqui definimos o formato da requisição que a API espera receber do frontend
class InferenceRequest(BaseModel):
    model_name: str
    images: List[str]
    
# Aqui definimos o formato da resposta que a API vai enviar para o frontend
class InferenceResponse(BaseModel):
    filename: str
    classification: str
    confidence: float
    model_used: str