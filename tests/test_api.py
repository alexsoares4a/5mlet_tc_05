# tests/test_api.py
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_health():
    """Verifica se o endpoint de saúde está ativo e retornando os metadados."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "timestamp" in data

def test_predict_risco_baixo():
    with patch("app.main.model.predict", return_value=[0.5]): # Força Risco BAIXO
        payload = {
            "ra": "RA-BAIXO", "inde": 9.0, "iaa": 9.0, "ieg": 9.0, "ips": 9.0,
            "ida": 9.0, "ipp": 9.0, "ipv": 9.0,
            "pedra": "TOPÁZIO", "fase": "FASE 8", "turma": "A",
            "instituicao_ensino": "Escola Pública"
        }
        response = client.post("/predict", json=payload)
        assert response.json()["risco"] == "BAIXO"

def test_predict_risco_moderado():
    with patch("app.main.model.predict", return_value=[-0.5]): # Força Risco MODERADO
        payload = {
            "ra": "RA-MODERADO", "inde": 5.0, "iaa": 5.0, "ieg": 5.0, "ips": 5.0,
            "ida": 5.0, "ipp": 5.0, "ipv": 5.0,
            "pedra": "ÁGATA", "fase": "FASE 4", "turma": "B",
            "instituicao_ensino": "Escola Pública"
        }
        response = client.post("/predict", json=payload)
        assert response.json()["risco"] == "MODERADO"

def test_predict_risco_alto():
    with patch("app.main.model.predict", return_value=[-1.5]): # Força Risco ALTO
        payload = {
            "ra": "RA-ALTO", "inde": 1.0, "iaa": 1.0, "ieg": 1.0, "ips": 1.0,
            "ida": 1.0, "ipp": 1.0, "ipv": 1.0,
            "pedra": "QUARTZO", "fase": "FASE 1", "turma": "C",
            "instituicao_ensino": "Escola Estadual"
        }
        response = client.post("/predict", json=payload)
        assert response.json()["risco"] == "ALTO"