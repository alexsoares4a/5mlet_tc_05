from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_health():
    """Verifica se o endpoint de saúde está ativo."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint():
    """Testa a predição enviando dados simulados de um estudante."""
    payload = {
        "INDE": 7.5,
        "IDA": 8.0,
        "IEG": 9.0,
        "IAA": 8.5,
        "IPS": 7.0,
        "IPP": 8.0,
        "IPV": 9.0,
        "Pedra_Nivel": 3
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()