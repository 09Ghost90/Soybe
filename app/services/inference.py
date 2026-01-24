def run_inference(model_name: str, file_bytes: bytes) -> dict:
    # Função mock de inferência
    return {
        "label": "Normais",
        "confidence": 0.95,
        "model_name": model_name
    }