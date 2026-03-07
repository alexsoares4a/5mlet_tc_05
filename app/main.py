# =============================================================================
# app/main.py - API FastAPI para Predição de Defasagem Escolar
# =============================================================================

import pandas as pd
import joblib
import logging
from fastapi import FastAPI, HTTPException
from app.schemas import StudentData, PredictionResponse
from pathlib import Path
from datetime import datetime

# Configurar logging
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'api.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API de Predição de Defasagem Escolar - Passos Mágicos",
    description="API para identificação precoce de risco escolar através de indicadores acadêmicos.",
    version="1.0.0"
)

# Carregamento dos artefatos do modelo
MODEL_PATH = Path("models/model.joblib")
model = None
preprocessor = None
expected_features = None

if MODEL_PATH.exists():
    try:
        artifact = joblib.load(MODEL_PATH)
        model = artifact['model']
        preprocessor = artifact['preprocessor']
        
        # Captura as features exatas do treinamento para garantir a ordem no predict
        if hasattr(preprocessor, 'numerical_columns') and hasattr(preprocessor, 'categorical_columns'):
            expected_features = preprocessor.numerical_columns + preprocessor.categorical_columns
        else:
            expected_features = getattr(preprocessor, 'feature_columns', None)
            
        logger.info("Modelo e Preprocessor carregados com sucesso.")
    except Exception as e:
        logger.error(f"Erro crítico ao carregar artefatos: {e}")
else:
    logger.error(f"Arquivo {MODEL_PATH} não encontrado!")

@app.get("/health")
def health_check():
    """Verifica se a API e o modelo estão operacionais."""
    return {
        "status": "healthy" if model and preprocessor else "unhealthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentData):
    """
    Realiza a predição da defasagem escolar e classifica o nível de risco.
    """
    if not model or not preprocessor:
        logger.error("Erro: Modelo ou Preprocessor ausente.")
        raise HTTPException(status_code=503, detail="Modelo não disponível no servidor.")

    try:
        # 1. Converter entrada Pydantic para Dicionário e capturar RA
        input_dict = student.model_dump()
        ra = input_dict.pop('ra', 'RA-000')
        
        # 2. Criar DataFrame e garantir colunas minúsculas
        input_df = pd.DataFrame([input_dict])
        input_df.columns = input_df.columns.str.lower()
        
        # 3. Proteção de Ordem das Features (Crucial para Random Forest)
        if expected_features:
            # Seleciona apenas as colunas que existem tanto no modelo quanto na entrada
                cols_to_use = [c for c in expected_features if c in input_df.columns]
                input_df = input_df[cols_to_use]
        
        logger.info(f"Processando predição para RA: {ra}")

        # 4. Transformação (Preprocessamento)
        X_processed = preprocessor.transform(input_df)
        
        # 5. Executar Predição
        prediction = float(model.predict(X_processed)[0])
        
        # 6. Lógica de Negócio Calibrada (Ajustada para a realidade do seu modelo)
        # O limite de -0.80 foi definido com base nas métricas de treinamento
        if prediction <= -0.8:
            risco = "ALTO"
            msg = "Intervenção pedagógica urgente necessária."
        elif prediction < 0:
            risco = "MODERADO"
            msg = "Acompanhamento preventivo recomendado."
        else:
            risco = "BAIXO"
            msg = "Estudante em nível de adequação ideal."

        logger.info(f"Predição concluída | RA: {ra} | Valor: {prediction:.2f} | Risco: {risco}")

        # 7. Retorno formatado conforme PredictionResponse
        return {
            "ra": ra,
            "defasagem_estimada": round(prediction, 2),
            "risco": risco,
            "mensagem": msg
        }

    except KeyError as e:
        logger.error(f"Coluna ausente na entrada: {e}")
        raise HTTPException(status_code=400, detail=f"Erro de formato: campo {e} ausente ou incorreto.")
    except Exception as e:
        logger.error(f"Erro inesperado: {type(e).__name__} - {e}")
        raise HTTPException(status_code=500, detail="Erro interno no processamento da predição.")