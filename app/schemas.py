from pydantic import BaseModel, Field
from typing import Optional

class StudentData(BaseModel):
    """
    Schema para input de predição.
    """
    # Identificação
    ra: Optional[str] = "RA-000"
    
    # Features do Modelo (lowercase com underscore - como foi treinado)
    inde: float = Field(..., ge=0, le=10, description="Índice de Desenvolvimento Educacional")
    iaa: float = Field(..., ge=0, le=10, description="Indicador de Auto Avaliação")
    ieg: float = Field(..., ge=0, le=10, description="Indicador de Engajamento")
    ips: float = Field(..., ge=0, le=10, description="Indicador Psicossocial")
    ida: float = Field(..., ge=0, le=10, description="Indicador de Aprendizagem")
    ipp: float = Field(..., ge=0, le=10, description="Indicador Psicopedagógico")
    ipv: float = Field(..., ge=0, le=10, description="Indicador de Ponto de Virada")
    # idade: int = Field(..., ge=5, le=30, description="Idade do aluno")
    pedra: str = Field(..., description="Quartzo, Ágata, Ametista, Topázio")
    fase: str = Field(..., description="Ex: FASE 1, FASE 2")
    turma: str = Field(..., description="Ex: A, B, C")
    instituicao_ensino: str = Field(..., description="Nome da escola")
    # ponto_virada: str = Field(..., description="Sim ou Não")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ra": "RA-001",
                "inde": 7.5,
                "iaa": 7.0,
                "ieg": 7.5,
                "ips": 7.0,
                "ida": 7.0,
                "ipp": 7.0,
                "ipv": 7.0,
                # "idade": 15,
                "pedra": "Ametista",
                "fase": "FASE 5",
                "turma": "A",
                "instituicao_ensino": "Escola Pública",
                # "ponto_virada": "Sim"
            }
        }

class PredictionResponse(BaseModel):
    """Schema para resposta da predição."""
    ra: str
    defasagem_estimada: float
    risco: str
    mensagem: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "ra": "RA-001",
                "defasagem_estimada": -1.5,
                "risco": "MODERADO",
                "mensagem": "Acompanhamento preventivo recomendado."
            }
        }