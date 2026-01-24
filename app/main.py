from fastapi import FastAPI, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from app.schemas import InferenceResponse
from app.services.inference import run_inference

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False, # True: permite cookies, False: não permite cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/inferencia")
async def options_inferencia():
    return Response(status_code=200)

@app.post("/inferencia", response_model=List[InferenceResponse])
async def inferencia(
    model_name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    results = []
    

    for f in files:
        contents = await f.read()  # bytes do arquivo

        # TODO: aqui você vai chamar o modelo futuramente
        
        # De forma dinamica futuramente carrega o modelo baseado no model_name e passa os bytes para inferência
        
        # Endpoint chama service (função de inferência). Service nunca conhece Endpoint        
        prediction = run_inference(model_name, contents)

        results.append({
            "filename": f.filename,
            "classification": prediction["label"],
            "model_used": prediction["model_name"],
        })

        await f.close()

    return results

@app.get("/home")
async def root():
    return {"message": "API de Inferência de Imagens está rodando!"}
