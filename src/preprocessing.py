# =============================================================================
# src/preprocessing.py - Versão Final para Produção
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict, List

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.numerical_columns: List[str] = []
        self.is_fitted = False
        

    def define_features(self) -> List[str]:
        # Usar nomes padronizados gerados pelo notebook de EDA
        self.numerical_columns = ['inde', 'iaa', 'ieg', 'ips', 'ida', 'ipp', 'ipv']
        self.categorical_columns = ['pedra', 'fase', 'turma', 'instituicao_ensino']
        
        # Ponto_Virada é booleana/categórica
        self.feature_columns = self.numerical_columns + self.categorical_columns
        return self.feature_columns


    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Iniciando limpeza dos dados...")
        df_clean = df.copy()

        # Forçar minúsculas em todos os nomes de colunas
        df_clean.columns = [c.lower().strip() for c in df_clean.columns]

        # Mapeamento de colunas com sufixo de ano para padrão
        year_mapping = {
            'inde_2022': 'inde', 'inde_2023': 'inde', 'inde_2024': 'inde',
            'pedra_2022': 'pedra', 'pedra_2023': 'pedra', 'pedra_2024': 'pedra',
            'fase_2022': 'fase', 'fase_2023': 'fase', 'fase_2024': 'fase',
            'turma_2022': 'turma', 'turma_2023': 'turma', 'turma_2024': 'turma',
            'instituicao_ensino_aluno_2022': 'instituicao_ensino',
            'instituicao_ensino_aluno_2023': 'instituicao_ensino',
            'instituicao_ensino_aluno_2024': 'instituicao_ensino',
            'defasagem_2022': 'defasagem', 'defasagem_2023': 'defasagem', 'defasagem_2024': 'defasagem',
            'ponto_virada_2022': 'ponto_virada', 'ponto_virada_2023': 'ponto_virada', 'ponto_virada_2024': 'ponto_virada',
            'idade_aluno_2022': 'idade', 'idade_aluno_2023': 'idade', 'idade_aluno_2024': 'idade',
            'iaa_2022': 'iaa', 'iaa_2023': 'iaa', 'iaa_2024': 'iaa',
            'ieg_2022': 'ieg', 'ieg_2023': 'ieg', 'ieg_2024': 'ieg',
            'ips_2022': 'ips', 'ips_2023': 'ips', 'ips_2024': 'ips',
            'ida_2022': 'ida', 'ida_2023': 'ida', 'ida_2024': 'ida',
            'ipp_2022': 'ipp', 'ipp_2023': 'ipp', 'ipp_2024': 'ipp',
            'ipv_2022': 'ipv', 'ipv_2023': 'ipv', 'ipv_2024': 'ipv',
        }
        df_clean = df_clean.rename(columns=year_mapping)

        # Mapeamento adicional de colunas
        mapping = {'instituição de ensino': 'instituicao_ensino', 'instituicao_ensino': 'instituicao_ensino'}
        df_clean = df_clean.rename(columns=mapping)
        
        # Substituir valores especiais por NaN
        error_patterns = ['ERROR:#DIV/0!', 'ERROR:#N/A', 'INCLUIR', 'ERROR', 'NAN']
        df_clean = df_clean.replace(error_patterns, np.nan)
        
        # O alvo do Datathon é a Defasagem
        target_col = 'defasagem'
        
        if target_col in df_clean.columns:
            # Converter para numérico caso existam strings
            df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
            df_clean = df_clean.dropna(subset=[target_col])
        
        # Manter apenas colunas úteis
        cols_to_keep = ['ra'] + self.feature_columns + [target_col]
        available_cols = [col for col in cols_to_keep if col in df_clean.columns]

        return df_clean[available_cols]


    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        logger.info("Codificando variáveis categóricas...")
        df_encoded = df.copy()
        
        # 1. Encoding Ordinal para 'Pedra' (Mapeamento oficial Passos Mágicos)
        pedra_mapping = {'quartzo': 1, 'ágata': 2, 'ametista': 3, 'topázio': 4,
                        'quartzo ': 1, 'ágata ': 2, 'ametista ': 3, 'topázio ': 4}
        if 'pedra' in df_encoded.columns:
            df_encoded['pedra'] = df_encoded['pedra'].astype(str).str.lower().str.strip().map(pedra_mapping).fillna(0)

        # 2. Label Encoding para as demais (Fase, Turma, etc.)
        for col in self.categorical_columns:
            if col == 'pedra' or col not in df_encoded.columns:
                continue
            
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                # Tratar categorias novas em produção com valor -1
                encoder = self.label_encoders[col]
                df_encoded[col] = df_encoded[col].astype(str).apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
        
        # 3. Tratar Ponto_Virada
        if 'ponto_virada' in df_encoded.columns:
            df_encoded['ponto_virada'] = df_encoded['ponto_virada'].map({
                'sim': 1, 'não': 0, 'true': 1, 'false': 0,
                'sim ': 1, 'não ': 0, 'true ': 1, 'false ': 0
            }).fillna(0)
        
        return df_encoded


    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        self.define_features()
        df_clean = self.clean_data(df)
        
        y = df_clean['defasagem'].copy()
        X = df_clean.drop(columns=['defasagem'])

        # Pipeline de transformações
        X = self.encode_categorical(X, fit=True)
        
        # Remover RA antes de escalar e imputar
        ra_col = X['ra'] if 'ra' in X.columns else None
        X_feats = X.drop(columns=['ra']) if 'ra' in X.columns else X
        
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X_feats), columns=X_feats.columns)
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X_imputed), columns=X_imputed.columns)
        
        self.is_fitted = True

        return X_scaled, y


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Preprocessor não ajustado. Chame fit_transform primeiro.")
        
        try:
            df_clean = self.clean_data(df)
            X = self.encode_categorical(df_clean.drop(columns=['defasagem'], errors='ignore'), fit=False)
            X_feats = X.drop(columns=['ra'], errors='ignore')
            X_imputed = self.imputer.transform(X_feats)
            return pd.DataFrame(self.scaler.transform(X_imputed), columns=X_feats.columns)
        except Exception as e:
            logger.error(f"Erro no transform: {str(e)}")
            raise


    def save(self, filepath: str) -> None:
        """Salva o preprocessor em disco."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        artifact = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'imputer': self.imputer,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(artifact, filepath)
        logger.info(f"Preprocessor salvo em {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DataPreprocessor':
        """Carrega o preprocessor do disco."""
        artifact = joblib.load(filepath)
        
        preprocessor = cls()
        preprocessor.scaler = artifact['scaler']
        preprocessor.label_encoders = artifact['label_encoders']
        preprocessor.imputer = artifact['imputer']
        preprocessor.feature_columns = artifact['feature_columns']
        preprocessor.categorical_columns = artifact['categorical_columns']
        preprocessor.numerical_columns = artifact['numerical_columns']
        preprocessor.is_fitted = artifact['is_fitted']
        
        logger.info(f"Preprocessor carregado de {filepath}")
        
        return preprocessor