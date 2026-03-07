# tests/test_train.py
"""
Testes unitários para o módulo de treinamento.
"""

import pandas as pd
import numpy as np
from src.train import TrainModel
from src.preprocessing import DataPreprocessor
from pathlib import Path

class TestTrainModel:
    """Classe de testes para TrainModel."""
    
    def test_init(self):
        """Testa a inicialização do TrainModel."""
        trainer = TrainModel()
        
        assert trainer.model is None
        assert trainer.preprocessor is not None
        assert trainer.metrics == {}
        assert isinstance(trainer.preprocessor, DataPreprocessor)
    
    def test_init_with_config(self):
        """Testa a inicialização com configuração customizada."""
        config = {
            'test_size': 0.3,
            'random_state': 123,
            'n_estimators': 200,
            'max_depth': 15
        }
        trainer = TrainModel(config=config)
        
        assert trainer.config['test_size'] == 0.3
        assert trainer.config['random_state'] == 123
        assert trainer.config['n_estimators'] == 200
    
    def test_init_default_config(self):
        """Testa a inicialização com configuração padrão."""
        trainer = TrainModel()
        
        assert trainer.config['test_size'] == 0.2
        assert trainer.config['random_state'] == 42
    
    def test_preprocessor_is_data_preprocessor(self):
        """Testa se o preprocessor é uma instância de DataPreprocessor."""
        trainer = TrainModel()
        
        assert isinstance(trainer.preprocessor, DataPreprocessor)
    
    def test_metrics_is_dict(self):
        """Testa se metrics é um dicionário."""
        trainer = TrainModel()
        
        assert isinstance(trainer.metrics, dict)
    
    def test_metrics_is_empty_on_init(self):
        """Testa se metrics está vazio na inicialização."""
        trainer = TrainModel()
        
        assert trainer.metrics == {}
    
    def test_model_is_none_on_init(self):
        """Testa se model é None na inicialização."""
        trainer = TrainModel()
        
        assert trainer.model is None
    
    def test_train_model_has_required_attributes(self):
        """Testa se TrainModel tem todos os atributos necessários."""
        trainer = TrainModel()
        
        assert hasattr(trainer, 'model')
        assert hasattr(trainer, 'preprocessor')
        assert hasattr(trainer, 'metrics')
        assert hasattr(trainer, 'config')
    
    def test_train_model_has_required_methods(self):
        """Testa se TrainModel tem todos os métodos necessários."""
        trainer = TrainModel()
        
        assert hasattr(trainer, 'run_pipeline')
        assert hasattr(trainer, 'save_artifacts')
    
    def test_config_has_required_keys(self):
        """Testa se config tem as chaves necessárias."""
        trainer = TrainModel()
        
        assert 'test_size' in trainer.config
        assert 'random_state' in trainer.config
        assert 'n_estimators' in trainer.config
        assert 'max_depth' in trainer.config
    
    def test_config_test_size_range(self):
        """Testa se test_size está no range correto."""
        trainer = TrainModel()
        
        assert 0 < trainer.config['test_size'] < 1
    
    def test_config_random_state_is_int(self):
        """Testa se random_state é um inteiro."""
        trainer = TrainModel()
        
        assert isinstance(trainer.config['random_state'], int)
    
    def test_config_n_estimators_is_int(self):
        """Testa se n_estimators é um inteiro."""
        trainer = TrainModel()
        
        assert isinstance(trainer.config['n_estimators'], int)
    
    def test_config_max_depth_is_int(self):
        """Testa se max_depth é um inteiro."""
        trainer = TrainModel()
        
        assert isinstance(trainer.config['max_depth'], int)
    
    def test_train_model_initialization_creates_preprocessor(self):
        """Testa se a inicialização cria um preprocessor."""
        trainer = TrainModel()
        
        assert trainer.preprocessor is not None
    
    def test_train_model_config_is_dict(self):
        """Testa se config é um dicionário."""
        trainer = TrainModel()
        
        assert isinstance(trainer.config, dict)
    
    def test_train_model_has_preprocessor_attribute(self):
        """Testa se train_model tem atributo preprocessor."""
        trainer = TrainModel()
        
        assert hasattr(trainer, 'preprocessor')
    
    def test_train_model_has_metrics_attribute(self):
        """Testa se train_model tem atributo metrics."""
        trainer = TrainModel()
        
        assert hasattr(trainer, 'metrics')
    
    def test_train_model_has_model_attribute(self):
        """Testa se train_model tem atributo model."""
        trainer = TrainModel()
        
        assert hasattr(trainer, 'model')
    
    def test_train_model_has_config_attribute(self):
        """Testa se train_model tem atributo config."""
        trainer = TrainModel()
        
        assert hasattr(trainer, 'config')
    
    def test_train_model_methods_are_callable(self):
        """Testa se os métodos são chamáveis."""
        trainer = TrainModel()
        
        assert callable(trainer.run_pipeline)
        assert callable(trainer.save_artifacts)
    
    def test_train_model_preprocessor_type(self):
        """Testa o tipo do preprocessor."""
        trainer = TrainModel()
        
        assert type(trainer.preprocessor).__name__ == 'DataPreprocessor'
    
    def test_train_model_config_default_values(self):
        """Testa os valores padrão da config."""
        trainer = TrainModel()
        
        assert trainer.config['test_size'] == 0.2
        assert trainer.config['random_state'] == 42
        assert trainer.config['n_estimators'] == 150
        assert trainer.config['max_depth'] == 12
    
    def test_train_model_config_custom_values(self):
        """Testa valores customizados da config."""
        config = {
            'test_size': 0.25,
            'random_state': 100,
            'n_estimators': 200,
            'max_depth': 15
        }
        trainer = TrainModel(config=config)
        
        assert trainer.config['test_size'] == 0.25
        assert trainer.config['random_state'] == 100
        assert trainer.config['n_estimators'] == 200
        assert trainer.config['max_depth'] == 15
    
    def test_train_model_init_sets_model_to_none(self):
        """Testa se model é None na inicialização."""
        trainer = TrainModel()
        
        assert trainer.model is None
    
    def test_train_model_init_sets_metrics_to_empty_dict(self):
        """Testa se metrics é um dict vazio na inicialização."""
        trainer = TrainModel()
        
        assert trainer.metrics == {}
    
    def test_train_model_has_run_pipeline_method(self):
        """Testa se tem método run_pipeline."""
        trainer = TrainModel()
        
        assert hasattr(trainer, 'run_pipeline')
    
    def test_train_model_has_save_artifacts_method(self):
        """Testa se tem método save_artifacts."""
        trainer = TrainModel()
        
        assert hasattr(trainer, 'save_artifacts')
    
    def test_train_model_config_is_not_none(self):
        """Testa se config não é None."""
        trainer = TrainModel()
        
        assert trainer.config is not None
    
    def test_train_model_preprocessor_is_not_none(self):
        """Testa se preprocessor não é None."""
        trainer = TrainModel()
        
        assert trainer.preprocessor is not None
    
    def test_train_model_metrics_type(self):
        """Testa o tipo de metrics."""
        trainer = TrainModel()
        
        assert type(trainer.metrics) == dict
    
    def test_train_model_config_type(self):
        """Testa o tipo de config."""
        trainer = TrainModel()
        
        assert type(trainer.config) == dict
    
    def test_train_model_preprocessor_instance(self):
        """Testa a instância do preprocessor."""
        trainer = TrainModel()
        
        from src.preprocessing import DataPreprocessor
        assert isinstance(trainer.preprocessor, DataPreprocessor)
    
    def test_train_model_model_initial_value(self):
        """Testa o valor inicial de model."""
        trainer = TrainModel()
        
        assert trainer.model is None
    
    def test_train_model_metrics_initial_value(self):
        """Testa o valor inicial de metrics."""
        trainer = TrainModel()
        
        assert trainer.metrics == {}
    
    def test_train_model_config_has_four_keys(self):
        """Testa se config tem 4 chaves."""
        trainer = TrainModel()
        
        assert len(trainer.config) == 4
    
    def test_train_model_config_keys(self):
        """Testa as chaves de config."""
        trainer = TrainModel()
        
        expected_keys = ['test_size', 'random_state', 'n_estimators', 'max_depth']
        for key in expected_keys:
            assert key in trainer.config
    
    def test_train_model_test_size_valid_range(self):
        """Testa se test_size está no range válido."""
        trainer = TrainModel()
        
        assert 0.0 < trainer.config['test_size'] < 1.0
    
    def test_train_model_random_state_positive(self):
        """Testa se random_state é positivo."""
        trainer = TrainModel()
        
        assert trainer.config['random_state'] >= 0
    
    def test_train_model_n_estimators_positive(self):
        """Testa se n_estimators é positivo."""
        trainer = TrainModel()
        
        assert trainer.config['n_estimators'] > 0
    
    def test_train_model_max_depth_positive(self):
        """Testa se max_depth é positivo."""
        trainer = TrainModel()
        
        assert trainer.config['max_depth'] > 0

def test_train_pipeline_coverage():
    # Criar um dataset sintético maior (20 linhas) para permitir cv=5
    data_size = 20
    data = {
        'ra': [f'RA-{i}' for i in range(data_size)],
        'inde': np.random.uniform(0, 10, data_size),
        'iaa': np.random.uniform(0, 10, data_size),
        'ieg': np.random.uniform(0, 10, data_size),
        'ips': np.random.uniform(0, 10, data_size),
        'ida': np.random.uniform(0, 10, data_size),
        'ipp': np.random.uniform(0, 10, data_size),
        'ipv': np.random.uniform(0, 10, data_size),
        'pedra': np.random.choice(['Quartzo', 'Ágata', 'Ametista', 'Topázio'], data_size),
        'fase': np.random.choice(['FASE 1', 'FASE 2', 'FASE 3'], data_size),
        'turma': np.random.choice(['A', 'B', 'C'], data_size),
        'instituicao_ensino': ['Escola A'] * data_size,
        'defasagem': np.random.uniform(-3, 3, data_size)
    }
    df_large = pd.DataFrame(data)
    
    # Garantir que o diretório existe
    temp_dir = Path('data/processed')
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_csv = temp_dir / "temp_coverage.csv"
    
    df_large.to_csv(temp_csv, index=False)
    
    try:
        trainer = TrainModel()
        trainer.run_pipeline(str(temp_csv))
        
        assert Path("models/model.joblib").exists()
    finally:
        # Limpeza
        if temp_csv.exists():
            temp_csv.unlink()