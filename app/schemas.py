# O arquivo schemas.py defini os modelos de dados usados na API

from pydantic import BaseModel
from typing import List

# BaseModel: modelo base para requisições e respostas
# Definição do contrato da API

class InferenceRequest(BaseModel):
    model_name: str
    images: List[str]
    
class InferenceResponse(BaseModel):
    filename: str
    classification: str
    model_used: str