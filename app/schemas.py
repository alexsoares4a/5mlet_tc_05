from pydantic import BaseModel, Field
from typing import Optional

class StudentData(BaseModel):
    # Identificação
    RA: Optional[str] = "RA-000"
    
    # Indicadores Acadêmicos e Psicossociais (0 a 10)
    ra: Optional[str] = "RA-000"
    inde: float = Field(..., ge=0, le=10)
    ida: float = Field(..., ge=0, le=10)
    ieg: float = Field(..., ge=0, le=10)
    iaa: float = Field(..., ge=0, le=10)
    ips: float = Field(..., ge=0, le=10)
    ipp: float = Field(..., ge=0, le=10)
    ipv: float = Field(..., ge=0, le=10)
    
    # Perfil e Categorias
    pedra: str = Field(..., description="Ex: Quartzo, Ágata, Ametista, Topázio")
    fase: str = Field(..., description="Ex: FASE 1, FASE 2")
    turma: str = Field(..., description="Ex: A, B, C")


class PredictionResponse(BaseModel):
    RA: str
    defasagem_estimada: float
    risco: str
    mensagem: str