# =============================================================================
# src/monitoring.py - Monitoramento de Drift com Evidently AI (Versão Final)
# =============================================================================

import sys
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict

# Configurar raiz do projeto
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# IMPORTS DO EVIDENTLY (Corrigidos e Robustos)
# =============================================================================
try:
    from evidently.report import Report  # IMPORTAÇÃO ESSENCIAL
    from evidently import ColumnMapping
    
    # Tentar imports da versão mais recente primeiro
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
        USE_PRESET = True
    except ImportError:
        from evidently.metrics import DataDriftTable, DataDriftPlot, DatasetDriftMetric, ColumnDriftMetric
        USE_PRESET = False
    
    EVIDENTLY_AVAILABLE = True
    print("Evidently importado com sucesso!")
    
except ImportError as e:
    EVIDENTLY_AVAILABLE = False
    print(f"Evidently nao disponivel: {e}")

# Configurar logging (sem emojis para evitar erro de encoding no Windows)
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'monitoring.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DriftMonitor:
    """
    Classe para monitoramento de drift de dados com Evidently AI.
    """
    
    def __init__(self, model_path: str = None):
        if model_path is None:
            self.model_path = PROJECT_ROOT / "models" / "model.joblib"
        else:
            self.model_path = Path(model_path)
        
        self.model = None
        self.preprocessor = None
        self.reference_data = None
        self.feature_columns = None
        
    def load_model(self) -> bool:
        """Carrega o modelo e preprocessor treinados."""
        try:
            if not self.model_path.exists():
                logger.error(f"Arquivo do modelo nao encontrado: {self.model_path}")
                return False
            
            artifact = joblib.load(self.model_path)
            self.model = artifact['model']
            self.preprocessor = artifact['preprocessor']
            self.feature_columns = artifact.get('features', None)
            
            logger.info("Modelo e preprocessor carregados para monitoramento.")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            return False
    
    def load_reference_data(self, reference_path: str = None) -> bool:
        """Carrega os dados de referência (dataset de treino)."""
        if reference_path is None:
            reference_path = PROJECT_ROOT / "data" / "processed" / "dataset_consolidado_eda.csv"
        else:
            reference_path = Path(reference_path)
        
        try:
            if not reference_path.exists():
                logger.error(f"Dataset de referencia nao encontrado: {reference_path}")
                return False
            
            self.reference_data = pd.read_csv(reference_path)
            self.reference_data.columns = self.reference_data.columns.str.lower().str.strip()
            
            if self.feature_columns:
                cols_to_keep = [c for c in self.feature_columns if c in self.reference_data.columns]
                self.reference_data = self.reference_data[cols_to_keep]
            
            logger.info(f"Dados de referencia carregados: {self.reference_data.shape}")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar dados de referencia: {e}")
            return False
    
    def generate_drift_report(self, current_data: pd.DataFrame, output_path: Path) -> dict:
        """Gera relatório de drift com Evidently ou fallback básico."""
        
        # Se Evidently não estiver disponível, usa fallback
        if not EVIDENTLY_AVAILABLE:
            logger.info("Evidently nao disponivel. Usando fallback basico.")
            return self._generate_basic_report(current_data, output_path)
        
        try:
            # Padronizar colunas
            current_data = current_data.copy()
            current_data.columns = current_data.columns.str.lower().str.strip()
            
            # Filtrar colunas numéricas em comum
            common_cols = list(set(self.reference_data.columns) & set(current_data.columns))
            common_cols = [c for c in common_cols 
                          if pd.api.types.is_numeric_dtype(self.reference_data[c]) 
                          and pd.api.types.is_numeric_dtype(current_data[c])]
            
            if not common_cols:
                logger.error("Nenhuma coluna numerica em comum.")
                return self._generate_basic_report(current_data, output_path)
            
            # Configurar column mapping
            column_mapping = ColumnMapping()
            
            # Selecionar métricas baseadas na versão do Evidently
            if USE_PRESET:
                metrics_list = [DatasetDriftMetric(), DataDriftPreset()]
                logger.info("Usando DataDriftPreset (Evidently 0.5.x+)")
            else:
                metrics_list = [DatasetDriftMetric(), DataDriftTable(), DataDriftPlot()]
                logger.info("Usando DataDriftTable/Plot (Evidently 0.4.x)")
            
            # Criar e executar relatório
            report = Report(metrics=metrics_list)
            report.run(
                reference_data=self.reference_data[common_cols],
                current_data=current_data[common_cols],
                column_mapping=column_mapping
            )
            
            # Salvar relatório HTML
            output_path.parent.mkdir(parents=True, exist_ok=True)
            report.save_html(str(output_path))  # Converte Path para str
            logger.info(f"Relatorio de drift HTML salvo em: {output_path}")
            
            # Extrair resumo básico
            drift_summary = {
                "status": "success",
                "report_path": str(output_path),
                "columns_analyzed": len(common_cols),
                "timestamp": datetime.now().isoformat()
            }
            
            return drift_summary
            
        except Exception as e:
            logger.error(f"Erro no Evidently: {e}. Usando fallback basico.")
            return self._generate_basic_report(current_data, output_path)
    
    def _generate_basic_report(self, current_data: pd.DataFrame, output_path: Path) -> dict:
        """Gera relatório básico de drift sem Evidently (fallback)."""
        
        try:
            current_data = current_data.copy()
            current_data.columns = current_data.columns.str.lower().str.strip()
            
            common_cols = list(set(self.reference_data.columns) & set(current_data.columns))
            numeric_cols = [c for c in common_cols 
                           if pd.api.types.is_numeric_dtype(self.reference_data[c]) 
                           and pd.api.types.is_numeric_dtype(current_data[c])]
            
            drift_results = {
                "status": "basic_success",
                "timestamp": datetime.now().isoformat(),
                "total_columns": len(numeric_cols),
                "drifting_columns": 0,
                "drift_ratio": 0.0,
                "columns": {}
            }
            
            for col in numeric_cols:
                ref_mean = self.reference_data[col].mean()
                curr_mean = current_data[col].mean()
                ref_std = self.reference_data[col].std()
                
                if ref_std > 0 and not np.isnan(ref_std):
                    drift_score = abs(curr_mean - ref_mean) / ref_std
                    has_drift = drift_score > 0.5
                else:
                    drift_score = 0
                    has_drift = False
                
                drift_results["columns"][col] = {
                    "reference_mean": float(ref_mean) if not np.isnan(ref_mean) else None,
                    "current_mean": float(curr_mean) if not np.isnan(curr_mean) else None,
                    "drift_score": float(drift_score),
                    "drift_detected": has_drift
                }
                
                if has_drift:
                    drift_results["drifting_columns"] += 1
            
            drift_results["drift_ratio"] = drift_results["drifting_columns"] / max(drift_results["total_columns"], 1)
            
            # Salvar relatório JSON
            json_path = output_path.with_suffix('.json')
            json_path.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(drift_results, f, indent=2, default=str)
            
            logger.info(f"Relatorio basico JSON salvo em: {json_path}")
            
            drift_results["report_path"] = str(json_path)
            return drift_results
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatorio basico: {e}")
            return {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}
    
    def check_drift_threshold(self, drift_summary: dict, threshold: float = 0.3) -> Tuple[bool, str]:
        """Verifica se o drift ultrapassou um threshold."""
        drift_ratio = drift_summary.get("drift_ratio", 0.0)
        
        if drift_ratio > threshold:
            return True, f"DRIFT DETECTADO: {drift_ratio:.2%} das colunas (threshold: {threshold:.2%})"
        else:
            return False, f"SEM DRIFT SIGNIFICATIVO: {drift_ratio:.2%} das colunas (threshold: {threshold:.2%})"
    
    def monitor_batch(self, current_data_df: pd.DataFrame, output_path: Path = None, threshold: float = 0.3) -> dict:
        """Executa monitoramento completo."""
        
        logger.info("=" * 60)
        logger.info("INICIANDO MONITORAMENTO DE DRIFT")
        logger.info("=" * 60)
        
        if output_path is None:
            output_path = PROJECT_ROOT / "reports" / "drift_report.html"
        
        if not self.load_model():
            return {"error": "Falha ao carregar modelo"}
        
        if not self.load_reference_data():
            return {"error": "Falha ao carregar dados de referencia"}
        
        logger.info(f"Dados atuais: {current_data_df.shape}")
        
        drift_summary = self.generate_drift_report(current_data_df, output_path)
        
        if drift_summary.get("status") == "success":
            drift_detected, message = self.check_drift_threshold(drift_summary, threshold)
            logger.info(message)
            drift_summary["drift_detected"] = drift_detected
            drift_summary["threshold"] = threshold
        
        logger.info("=" * 60)
        logger.info("MONITORAMENTO CONCLUIDO")
        logger.info("=" * 60)
        
        return drift_summary


# =============================================================================
# Função Utilitária
# =============================================================================

def generate_sample_production_data(n_samples: int = 100) -> pd.DataFrame:
    """Gera dados sintéticos para simular produção."""
    np.random.seed(42)
    return pd.DataFrame({
        'inde': np.random.uniform(0, 10, n_samples),
        'iaa': np.random.uniform(0, 10, n_samples),
        'ieg': np.random.uniform(0, 10, n_samples),
        'ips': np.random.uniform(0, 10, n_samples),
        'ida': np.random.uniform(0, 10, n_samples),
        'ipp': np.random.uniform(0, 10, n_samples),
        'ipv': np.random.uniform(0, 10, n_samples),
    })


# =============================================================================
# Main para Teste
# =============================================================================

if __name__ == "__main__":
    print(f"Raiz do Projeto: {PROJECT_ROOT}")
    
    monitor = DriftMonitor()
    
    if not monitor.load_model():
        print("Falha ao carregar modelo")
        exit(1)
    
    if not monitor.load_reference_data():
        print("Falha ao carregar dados de referencia")
        exit(1)
    
    print("Gerando dados sintéticos para simulação...")
    production_data = generate_sample_production_data(n_samples=200)
    
    output_path = PROJECT_ROOT / "reports" / "drift_report.html"
    results = monitor.monitor_batch(production_data, output_path, threshold=0.3)
    
    print("\n" + "=" * 60)
    print("RESUMO DO MONITORAMENTO")
    print("=" * 60)
    
    if "error" in results:
        print(f"ERRO: {results['error']}")
    else:
        print(f"Status: {results.get('status', 'N/A')}")
        print(f"Timestamp: {results.get('timestamp', 'N/A')}")
        print(f"Colunas Analisadas: {results.get('columns_analyzed', results.get('total_columns', 'N/A'))}")
        print(f"Drift Detectado: {results.get('drift_detected', 'N/A')}")
        print(f"Drift Ratio: {results.get('drift_ratio', 0):.2%}")
        print(f"Relatório: {results.get('report_path', 'N/A')}")
    
    print("=" * 60)
    print(f"Verifique a pasta: {output_path.parent}")