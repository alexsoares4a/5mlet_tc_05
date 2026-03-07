# Passos Mágicos - Predição de Defasagem Escolar

---

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests Coverage](https://img.shields.io/badge/coverage-84.15%25-green.svg)](#)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](#)

> **Datathon FIAP - Machine Learning Engineering**  
> Solução de engenharia de ML para estimar o risco de defasagem escolar de estudantes da Associação Passos Mágicos.

---

## 1️⃣ Visão Geral do Projeto

### 🎯 Objetivo
Desenvolver um modelo preditivo capaz de estimar o **risco de defasagem escolar** de cada estudante, permitindo intervenções pedagógicas preventivas e personalizadas. O modelo auxilia a equipe da Passos Mágicos a identificar precocemente alunos em situação de vulnerabilidade educacional.

### 💡 Solução Proposta
Pipeline completa de Machine Learning com as seguintes etapas:
1. **Pré-processamento robusto**: Padronização de colunas, tratamento de nulos, encoding de categóricas
2. **Feature engineering**: Criação de `Pedra_Nivel` (classificação ordinal baseada no INDE)
3. **Treinamento validado**: RandomForest Regressor com cross-validation (5 folds)
4. **API em produção**: Endpoint `/predict` para inferência em tempo real
5. **Monitoramento contínuo**: Logs estruturados para rastreabilidade

### 🛠️ Stack Tecnológica

| Categoria | Tecnologias |
|-----------|------------|
| **Linguagem** | Python 3.12 |
| **ML/Data** | scikit-learn==1.8.0, pandas==2.3.3, numpy>=1.24.0,<2.0.0 |
| **API** | FastAPI==0.128.0, Pydantic==2.12.5, Uvicorn==0.40.0 |
| **Serialização** | joblib==1.5.3 |
| **Testes** | pytest==9.0.2, pytest-cov==7.0.0 (≥80% coverage) |
| **Empacotamento** | Docker (python:3.12-slim) |
| **Deploy** | Local |
| **Monitoramento** | Logging estruturado + dashboard de drift (Evidently) |
| **Notebooks** | Jupyter==1.1.1, matplotlib==3.8.0, seaborn==0.13.0 |

### 📊 Métrica de Avaliação do Modelo
- **R² (Coeficiente de Determinação)**: Mede a proporção da variância da target explicada pelo modelo
- **RMSE (Raiz do Erro Quadrático Médio)**: Penaliza erros grandes, ideal para regressão
- **MAE (Erro Absoluto Médio)**: Interpretação direta em unidades da target
- **CV_RMSE**: Validação cruzada para garantir robustez em dados não vistos

> ✅ O modelo é confiável para produção pois: (1) utiliza validação cruzada, (2) métricas são salvas e versionadas, (3) pipeline é reproduzível via Docker, (4) testes unitários garantem qualidade do código.

---

## 2️⃣ Estrutura do Projeto
```
5mlet_tc_05/
├── data/
│   ├── processed/
│   │   └── dataset_consolidado_eda.csv    # Dataset processado pelo EDA
│   └── raw/
│       └── BASE DE DADOS PEDE 2024 - DATATHON.xlsx  # Fonte original
│
├── models/
│   ├── model.joblib          # Artefato: modelo + preprocessor + features
│   ├── metrics.json          # Métricas de avaliação (MAE, RMSE, R², CV_RMSE)
│   └── train.log             # Logs do treinamento
│
├── reports/                  # Relatórios de monitoramento
│   └── drift_report.html     # Painel de drift (Evidently AI)
│
├── logs/                     # Logs de execução
│   ├── api.log               # Logs da API FastAPI
│   ├── train.log             # Logs do treinamento
│   └── monitoring.log        # Logs do monitoramento de drift
│
├── notebook/
│   └── eda_g.ipynb           # EDA exploratório e feature engineering
│
├── src/                      # Pipeline de Machine Learning
│   ├── init.py
│   ├── preprocessing.py      # Limpeza, encoding, scaling (DataPreprocessor)
│   ├── train.py              # Treinamento e salvamento (TrainModel)
│   ├── eval.py               # Métricas de avaliação
│   ├── monitoring.py         # Automação de drift com Evidently
│   ├── utils.py              # Funções auxiliares
│   └── config.py             # Hiperparâmetros e caminhos
│
├── app/                      # Camada de serviço (API)
│   ├── init.py
│   ├── main.py               # FastAPI: endpoints /health e /predict
│   └── schemas.py            # Pydantic: StudentData, PredictionResponse
│
├── tests/                    # Testes unitários (≥80% coverage)
│   ├── conftest.py           # Fixtures reutilizáveis
│   ├── test_api.py           # Testes dos endpoints
│   ├── test_preprocessing.py # Testes do preprocessor
│   └── test_train.py         # Testes do treinamento
├── video/
│   └── dt_passos_magicos.mp4 # Vídeo Explicativo
│
├── Dockerfile                # Empacotamento da solução
├── .dockerignore             # Arquivos ignorados no build Docker
├── .gitignore                # Arquivos ignorados pelo Git
├── pytest.ini                # Configuração do pytest + coverage
├── requirements.txt          # Dependências do projeto
└── README.md                 # Este arquivo

```


---

## 3️⃣ Instruções de Deploy

### 📋 Pré-requisitos
- Python 3.12+
- Docker e Docker Compose (opcional, para containerização)
- Git

### 🚀 Opção A: Execução Local (Sem Docker)

```bash
# 1. Clonar repositório e entrar na pasta
git clone https://github.com/alexsoares4a/5mlet_tc_05.git
cd 5mlet_tc_05

# 2. Criar e ativar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\Activate  # Windows

# 3. Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt

# 4. Treinar o modelo (gera models/model.joblib)
python src/train.py

# 5. Iniciar a API
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 6. Acessar documentação Swagger
http://localhost:8000/docs

```
### 🐳 Opção B: Execução com Docker (Recomendado)
```bash
# 1. Build da imagem
docker build -t passos-magicos-api .

# 2. Rodar o container
docker run -d -p 8000:8000 --name container-pm api-passos-magicos

# 3. Verificar logs em tempo real
docker logs -f passos-magicos

# 4. Testar health check
curl http://localhost:8000/health

# 5. Acessar a API
# http://localhost:8000
# http://localhost:8000/docs

```
### 🧪 Rodar Testes com Coverage
```bash
# Ativar ambiente virtual
source venv/bin/activate  # ou .\venv\Scripts\Activate

# Executar testes com relatório de coverage
pytest --cov=src --cov=app --cov-report=html --cov-fail-under=80 -v

# Abrir relatório HTML no navegador
start htmlcov\index.html  # Windows
# ou
open htmlcov/index.html  # Mac/Linux
```

### 📊 Executar Monitoramento de Drift
```bash
# Ativar ambiente virtual
source venv/bin/activate

# Executar monitoramento
python src/monitoring.py

# Abrir relatório de drift
# Windows:
start reports\drift_report.html
# Linux/Mac:
open reports/drift_report.html
```


## 4️⃣ Exemplos de Chamadas à API

### 🔍 Health Check
```bash
curl http://localhost:8000/health
```
#### Resposta esperada (200 OK):
```bash
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "timestamp": "2024-03-06T10:30:00.123456"
}
```
### 🎯 Predição Individual
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "fase": "FASE 5",
        "iaa": 7,
        "ida": 7,
        "ieg": 7.5,
        "inde": 7.5,
        "instituicao_ensino": "Escola Pública",
        "ipp": 7,
        "ips": 7,
        "ipv": 7,
        "pedra": "Ametista",
        "ra": "RA-001",
        "turma": "A"
      }'
```

#### Resposta esperada (200 OK):
```bash
{
  "ra": "RA-001",
  "defasagem_estimada": 0.03,
  "risco": "BAIXO",
  "mensagem": "Estudante em nível de adequação ideal."
}
```

### 📊 Classificação de Risco

| Valor da Defasagem | Risco | Mensagem |
|-----------|------------|------------|
| `prediction <= -0.8` | ALTO | Intervenção pedagógica urgente necessária. |
| `-0.8 < prediction < 0` | MODERADO | Acompanhamento preventivo recomendado. |
| `prediction >= 0` | BAIXO | Estudante em nível de adequação ideal. |



## 5️⃣ Etapas do Pipeline de Machine Learning

### 🔹1. Pré-processamento dos Dados (`src/preprocessing.py`)

* **Padronização de colunas**: Conversão para lowercase e mapeamento de sufixos de ano (inde_2022 → inde)
* **Limpeza**: Remoção de valores especiais (ERROR, INCLUIR, #N/A) e conversão para numérico
* **Encoding**: 
    * `Pedra`: Encoding ordinal (Quartzo=1, Ágata=2, Ametista=3, Topázio=4)
    * `Fase`, `Turma`, `Instituição_Ensino`: LabelEncoder com fallback para categorias desconhecidas
* **Imputação**: Preenchimento de nulos com mediana (SimpleImputer)
* **Escalonamento**: Normalização com StandardScaler para features numéricas

### 🔹 2. Engenharia de Features (`notebook/eda_g.ipynb`)

* **Consolidação temporal**: União segura dos datasets de 2022, 2023 e 2024
* **Criação de `Pedra_Nivel`**: Feature ordinal baseada na classificação oficial da Passos Mágicos
* **Prevenção de leakage**: Exclusão de features derivadas da target (ex: IAN)
* **Análise de correlação**: Heatmap para identificar features mais relevantes para a target `Defasagem`

### 🔹 3. Treinamento e Validação (`src/train.py`)

* **Split estratificado**: 80% treino, 20% teste com `random_state=42` para reprodutibilidade
* **Modelo**: `RandomForestRegressor` com hiperparâmetros otimizados (`n_estimators=150`, `max_depth=12`)
* **Validação cruzada**: 5 folds para estimar performance em dados não vistos
* **Métricas salvas**: MAE, RMSE, R² e CV_RMSE em `models/metrics.json`
* **Serialização**: Artefato único com modelo, preprocessor e metadados em `models/model.joblib`

### 🔹 4. Seleção de Modelo

* Critério: Random Forest foi escolhido por:
    * Robustez a outliers e multicolinearidade
    * Interpretabilidade via feature importance
    * Performance estável em validação cruzada

### 🔹 5. Inferência via API (app/main.py)

* **Validação de entrada**: Schemas Pydantic garantem tipos e ranges corretos
* **Proteção de features**: Verificação de ordem e presença das colunas esperadas pelo modelo
* **Transformação consistente**: Reutilização do preprocessor treinado (`fit=False`)
* **Classificação de risco**: Threshold calibrado (`-0.8`) baseado na distribuição da defasagem no treino
* **Logs estruturados**: Monitoramento de requisições, erros e métricas de inferência

 ### 🔹 6. Monitoramento Contínuo

* **Detecção de drift**: Comparação estatística entre distribuição de treino e produção usando Evidently AI
* **Métricas monitoradas**:

    * `DatasetDriftMetric`: Visão geral do drift no dataset
    * `DataDriftPreset`: Conjunto de métricas pré-configuradas para drift
    * `ColumnDriftMetric`: Análise individual por feature

* **Relatório visual**: Geração de reports/drift_report.html com gráficos interativos
* **Threshold configurável**: Alerta quando % de colunas com drift > threshold (padrão: 30%)
* **Logs estruturados**: Auditoria completa em monitoring.log com timestamps
* **Execução agendável**: Script rodável via CRON (Linux) ou Task Scheduler (Windows)

### 🚀 Como Executar o Monitoramento
```bash
# Executar monitoramento manual
python src/monitoring.py

# Agendar execução diária (Linux/Mac - CRON)
0 2 * * * cd /caminho/do/projeto && python src/monitoring.py >> logs/monitor.log 2>&1

# Agendar execução diária (Windows - Task Scheduler)
# Configurar tarefa para executar: python.exe com argumento src/monitoring.py
```

### 📊 Interpretando o Relatório de Drift
| Seção | Descrição |
|-----------|------------|
| 📊 Dataset Summary| Visão geral: % de colunas com drift, métricas agregadas |
| 📈 Data Drift Plot| Gráficos de distribuição: referência vs. produção por feature |
| 📋 Data Drift Table| Tabela detalhada: p-value, teste estatístico, drift detectado |
| 🔍 Column Metrics| Detalhes por coluna: mean, std, drift score |

### ⚠️ Critério de Alerta
* ✅ SEM DRIFT: < 30% das colunas apresentam drift estatístico
* ⚠️ DRIFT DETECTADO: ≥ 30% das colunas apresentam drift → investigar causas


## 📬 Créditos

* **Autor**: Alex Soares da Silva
* **Instituição**: FIAP - Pós Tech em Machine Learning Engineering
* **Repositório**: github.com/alexsoares4a/5mlet_tc_05

## 📜 Licença
Este projeto é desenvolvido para fins educacionais no contexto do Datathon FIAP. O uso dos dados da Passos Mágicos deve respeitar os termos de confidencialidade e proteção de dados dos estudantes.