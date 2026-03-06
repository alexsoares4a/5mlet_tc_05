# Aqui validamos se o cálculo da Defasagem e do IAN segue as regras de negócio da Passos Mágicos.
# Regra do IAN: D=Fase Efetiva−Fase Ideal.

from src.features import calculate_ian # Supondo que você moveu a lógica para cá

def test_calculate_ian_logic():
    """Valida se a pontuação do IAN segue a tabela oficial."""
    # D >= 0 -> IAN 10 (Em fase) 
    assert calculate_ian(0) == 10 
    # 0 > D >= -2 -> IAN 5 (Moderada) 
    assert calculate_ian(-2) == 5 
    # D < -2 -> IAN 2.5 (Severa) 
    assert calculate_ian(-3) == 2.5