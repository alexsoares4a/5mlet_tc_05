# =============================================================================
# src/preprocessing.py - Versão Sincronizada com EDA
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict, List

logging.basicConfig(level=logging.INFO)
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
        self.numerical_columns = ['inde', 'iaa', 'ieg', 'ips', 'ida', 'ipp', 'ipv', 'idade']
        self.categorical_columns = ['pedra', 'fase', 'turma', 'instituicao_ensino']
        
        # Ponto_Virada é booleana/categórica
        self.feature_columns = self.numerical_columns + self.categorical_columns + ['ponto_virada']
        return self.feature_columns

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Iniciando limpeza dos dados...")
        df_clean = df.copy()

        # Forçar minúsculas em todos os nomes de colunas
        df_clean.columns = [c.lower().strip() for c in df_clean.columns]

        # Se o DataFrame vier com "Instituição de ensino" (do CSV original), renomeamos
        mapping = {'Instituição de ensino': 'instituicao_ensino', 'INSTITUICAO_ENSINO': 'instituicao_ensino'}
        df_clean = df_clean.rename(columns=mapping)
        
        # O alvo do Datathon é a Defasagem (conforme definido no EDA)
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
        pedra_mapping = {'QUARTZO': 1, 'ÁGATA': 2, 'AMETISTA': 3, 'TOPÁZIO': 4}
        if 'pedra' in df_encoded.columns:
            df_encoded['pedra'] = df_encoded['pedra'].astype(str).str.upper().map(pedra_mapping).fillna(0)

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
            df_encoded['ponto_virada'] = df_encoded['ponto_virada'].map({'SIM': 1, 'NÃO': 0, 'TRUE': 1, 'FALSE': 0}).fillna(0)
        
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
        if not self.is_fitted: raise ValueError("Preprocessor não ajustado.")
        df_clean = self.clean_data(df)
        X = self.encode_categorical(df_clean.drop(columns=['defasagem'], errors='ignore'), fit=False)
        X_feats = X.drop(columns=['ra'], errors='ignore')
        X_imputed = self.imputer.transform(X_feats)
        return pd.DataFrame(self.scaler.transform(X_imputed), columns=X_feats.columns)