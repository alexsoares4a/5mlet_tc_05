# =============================================================================
# src/train.py - Pipeline Integrado de Treinamento
# =============================================================================

import pandas as pd
import numpy as np
import joblib
import logging
import json
import os
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Adiciona a raiz do projeto ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importando o seu preprocessor customizado
from src.preprocessing import DataPreprocessor

# Configuração de Logging para Monitoramento [cite: 22, 28]
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(MODEL_DIR / "train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainModel:
    """Classe para gerenciar o treinamento integrado com o DataPreprocessor."""
    
    def __init__(self, config: dict = None):
        self.config = config or {
            'test_size': 0.2,
            'random_state': 42,
            'n_estimators': 150,
            'max_depth': 12
        }
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.metrics = {}

    def run_pipeline(self, data_path: str):
        logger.info("=" * 60)
        logger.info("INICIANDO PIPELINE DE TREINAMENTO")
        logger.info("=" * 60)

        try:
            # 1. Carregamento
            df_raw = pd.read_csv(data_path)
            logger.info(f"Dados brutos carregados: {df_raw.shape}")

            # 2. Pré-processamento
            X_processed, y = self.preprocessor.fit_transform(df_raw)

            # Remover RA se existir
            if 'RA' in X_processed.columns:
                X_processed = X_processed.drop(columns=['RA'])

            # 3. Split
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, 
                test_size=self.config['test_size'], 
                random_state=self.config['random_state']
            )

            # 4. Treinamento
            self.model = RandomForestRegressor(
                n_estimators=self.config['n_estimators'],
                max_depth=self.config['max_depth'],
                random_state=self.config['random_state'],
                n_jobs=-1
            )
            
            logger.info(f"Treinando RandomForest com {X_train.shape[0]} amostras...")
            self.model.fit(X_train, y_train)

            # 5. Avaliação
            y_pred = self.model.predict(X_test)
            
            self.metrics = {
                'MAE': float(mean_absolute_error(y_test, y_pred)),
                'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'R2': float(r2_score(y_test, y_test)),
                'CV_RMSE': None  # Será calculado abaixo
            }

            # Cross-Validation
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            self.metrics['CV_RMSE'] = float(np.sqrt(-cv_scores.mean()))
            
            logger.info(f"Métricas Finais - MAE: {self.metrics['MAE']:.4f}, R2: {self.metrics['R2']:.4f}")
            logger.info(f"Validação Cruzada (RMSE): {self.metrics['CV_RMSE']:.4f}")

            # 6. Salvamento
            self.save_artifacts()
            
        except Exception as e:
            logger.error(f"Erro no pipeline: {str(e)}")
            raise

    def save_artifacts(self):
        """Salva o modelo e o preprocessor para uso na API."""
        model_path = MODEL_DIR / "model.joblib"
        metrics_path = MODEL_DIR / "metrics.json"

        # Salva o dicionário de artefatos contendo o modelo e o preprocessor já 'fitado'
        artifact = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'features': self.preprocessor.feature_columns,
            'metrics': self.metrics  # Adicionado para referência
        }
        
        joblib.dump(artifact, model_path)
        
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
            
        logger.info(f"Artefatos e métricas salvos em {MODEL_DIR}")

def main():
    data_input = 'data/processed/dataset_consolidado_eda.csv'
    trainer = TrainModel()
    trainer.run_pipeline(data_input)

if __name__ == "__main__":
    main()