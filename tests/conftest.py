# Dados de exemplo

import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    """Gera um DataFrame fictício para testes de pré-processamento."""
    return pd.DataFrame({
        'PEDRA': ['Quartzo', 'Ágata', 'Ametista', 'Topázio', 'ERROR'],
        'DEFA': [0, -1, -3, 0, np.nan],
        'INDE': [5.0, 6.0, 7.5, 9.0, 'ERROR'],
        'IDA': [7.0, 8.0, 9.0, 10.0, 5.0]
    })