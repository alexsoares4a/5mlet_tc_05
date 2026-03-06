import pandas as pd
import joblib
import logging
from fastapi import FastAPI, HTTPException
from app.schemas import StudentData, PredictionResponse
from pathlib import Path

# Configurar logging para monitoramento contínuo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API de Predição de Defasagem Escolar - Passos Mágicos",
    description="API para identificação precoce de risco escolar."
)

# Caminho do modelo
MODEL_PATH = Path("models/model.joblib")

# Carregar o artefato globalmente na inicialização
if MODEL_PATH.exists():
    artifact = joblib.load(MODEL_PATH)
    model = artifact['model']
    preprocessor = artifact['preprocessor']
    logger.info("Modelo e Preprocessor carregados com sucesso.")
else:
    logger.error("Arquivo model.joblib não encontrado!")
    model = None

@app.get("/health")
def health_check():
    """Verifica se a API e o modelo estão operacionais."""
    if model:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}

@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentData):
    """Realiza a predição do risco de defasagem."""
    if not model:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    try:
        # 1. Converter entrada Pydantic para DataFrame
        input_df = pd.DataFrame([student.dict()])
        
        # 2. Log da requisição (Requisito de Monitoramento)
        logger.info(f"Predição solicitada para RA: {student.RA}")
        
        # 3. Pré-processamento automático via o objeto salvo
        X_processed = preprocessor.transform(input_df)
        
        # 4. Predição
        prediction = model.predict(X_processed)[0]
        
        # 5. Lógica de Negócio para o Risco
        # Se D < 0, há atraso. Definimos níveis baseados no valor
        if prediction < -2:
            risco = "ALTO"
            msg = "Intervenção pedagógica urgente necessária."
        elif prediction < 0:
            risco = "MODERADO"
            msg = "Acompanhamento preventivo recomendado."
        else:
            risco = "BAIXO"
            msg = "Estudante em nível de adequação ideal."

        return {
            "RA": student.RA,
            "defasagem_estimada": round(prediction, 2),
            "risco": risco,
            "mensagem": msg
        }

    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=str(e))