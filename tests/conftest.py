# tests/conftest.py
"""
Fixtures para os testes unitários do projeto Passos Mágicos.
"""

import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from src.preprocessing import DataPreprocessor
from src.train import TrainModel


@pytest.fixture
def sample_data():
    """
    Gera um DataFrame de exemplo para testes de pré-processamento.
    """
    data = {
        'RA': ['RA-001', 'RA-002', 'RA-003', 'RA-004', 'RA-005'],
        'INDE': [5.5, 6.5, 7.5, 8.5, 9.5],
        'IAA': [5.0, 6.0, 7.0, 8.0, 9.0],
        'IEG': [5.5, 6.5, 7.5, 8.5, 9.5],
        'IPS': [5.0, 6.0, 7.0, 8.0, 9.0],
        'IDA': [5.5, 6.5, 7.5, 8.5, 9.5],
        'IPP': [5.0, 6.0, 7.0, 8.0, 9.0],
        'IPV': [5.5, 6.5, 7.5, 8.5, 9.5],
        'Pedra': ['Quartzo', 'Ágata', 'Ametista', 'Topázio', 'Topázio'],
        'Fase': ['FASE 1', 'FASE 2', 'FASE 3', 'FASE 4', 'FASE 5'],
        'Turma': ['A', 'B', 'C', 'A', 'B'],
        'Instituição_Ensino': ['Escola A', 'Escola B', 'Escola A', 'Escola B', 'Escola A'],
        'Defasagem': [-1, -0.5, 0, 0.5, 1]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_nulls():
    """
    Gera um DataFrame com valores nulos para testes de imputação.
    """
    data = {
        'RA': ['RA-001', 'RA-002', 'RA-003', 'RA-004', 'RA-005'],
        'INDE': [5.5, np.nan, 7.5, np.nan, 9.5],
        'IAA': [5.0, 6.0, np.nan, 8.0, 9.0],
        'IEG': [5.5, 6.5, 7.5, np.nan, 9.5],
        'IPS': [5.0, 6.0, 7.0, 8.0, np.nan],
        'IDA': [5.5, 6.5, 7.5, 8.5, 9.5],
        'IPP': [5.0, 6.0, 7.0, 8.0, 9.0],
        'IPV': [5.5, 6.5, 7.5, 8.5, 9.5],
        'Pedra': ['Quartzo', 'Ágata', 'Ametista', 'Topázio', 'Topázio'],
        'Fase': ['FASE 1', 'FASE 2', 'FASE 3', 'FASE 4', 'FASE 5'],
        'Turma': ['A', 'B', 'C', 'A', 'B'],
        'Instituição_Ensino': ['Escola A', 'Escola B', 'Escola A', 'Escola B', 'Escola A'],
        'Defasagem': [-1, -0.5, 0, 0.5, 1]
    }
    return pd.DataFrame(data)


@pytest.fixture
def preprocessor():
    """
    Retorna uma instância do DataPreprocessor.
    """
    return DataPreprocessor()


@pytest.fixture
def trained_preprocessor(sample_data):
    """
    Retorna um preprocessor já ajustado com dados de exemplo.
    """
    preprocessor = DataPreprocessor()
    preprocessor.fit_transform(sample_data)
    return preprocessor


@pytest.fixture
def client():
    """
    Retorna um cliente de teste para a API FastAPI.
    """
    return TestClient(app)


@pytest.fixture
def valid_prediction_payload():
    """
    Retorna um payload válido para testes de predição.
    """
    return {
        "RA": "RA-001",
        "INDE": 7.5,
        "IAA": 7.0,
        "IEG": 7.5,
        "IPS": 7.0,
        "IDA": 7.0,
        "IPP": 7.0,
        "IPV": 7.0,
        "Pedra": "Ametista",
        "Fase": "FASE 5",
        "Turma": "A",
        "Instituição_Ensino": "Escola Pública"
    }


@pytest.fixture
def invalid_prediction_payload():
    """
    Retorna um payload inválido para testes de validação.
    """
    return {
        "RA": "RA-001",
        "INDE": 15.0,  # Fora do range (0-10)
        "IAA": 7.0,
        "IEG": 7.5,
        "IPS": 7.0,
        "IDA": 7.0,
        "IPP": 7.0,
        "IPV": 7.0,
        "Pedra": "Ametista",
        "Fase": "FASE 5",
        "Turma": "A",
        "Instituição_Ensino": "Escola Pública"
    }