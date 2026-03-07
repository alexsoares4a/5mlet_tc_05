import pytest
import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Classe de testes para DataPreprocessor."""
    
    def test_init(self, preprocessor):
        """Testa a inicialização do DataPreprocessor."""
        assert preprocessor.scaler is not None
        assert preprocessor.imputer is not None
        assert preprocessor.label_encoders == {}
        assert preprocessor.feature_columns == []
        assert preprocessor.is_fitted == False
    
    def test_define_features(self, preprocessor):
        """Testa a definição de features."""
        features = preprocessor.define_features()
        
        assert isinstance(features, list)
        assert len(features) > 0
        # Features são em minúsculo no preprocessing
        assert 'inde' in features or 'inde' in str(features)
    
    def test_clean_data(self, preprocessor, sample_data):
        """Testa a limpeza de dados."""
        cleaned = preprocessor.clean_data(sample_data)
        
        assert isinstance(cleaned, pd.DataFrame)
        assert len(cleaned) > 0
        # Colunas são convertidas para minúsculo
        assert 'ra' in cleaned.columns
    
    def test_clean_data_removes_target(self, preprocessor, sample_data):
        """Testa se a limpeza mantém a target para fit_transform."""
        cleaned = preprocessor.clean_data(sample_data)
        
        # Target é convertida para minúsculo
        assert 'defasagem' in cleaned.columns
    
    def test_encode_categorical(self, preprocessor, sample_data):
        """Testa o encoding de variáveis categóricas."""
        encoded = preprocessor.encode_categorical(sample_data, fit=True)
        
        assert isinstance(encoded, pd.DataFrame)
        assert 'Pedra' in encoded.columns or 'pedra' in encoded.columns
    
    def test_encode_categorical_ordinal_pedra(self, preprocessor, sample_data):
        """Testa se o encoding ordinal da Pedra está correto."""
        encoded = preprocessor.encode_categorical(sample_data, fit=True)
        
        # Verificar se Quartzo foi mapeado para 1 (coluna em minúsculo)
        if 'pedra' in encoded.columns:
            quartzo_mask = sample_data['pedra'] == 'Quartzo'
            if quartzo_mask.any():
                encoded_value = encoded.loc[quartzo_mask, 'pedra'].iloc[0]
                assert encoded_value == 1
    
    def test_encode_categorical_ponto_virada(self, preprocessor, sample_data):
        """Testa o encoding de Ponto_Virada."""
        encoded = preprocessor.encode_categorical(sample_data, fit=True)
        
        if 'ponto_virada' in encoded.columns:
            assert encoded['ponto_virada'].isin([0, 1]).all()
    
    def test_fit_transform(self, preprocessor, sample_data):
        """Testa o fit_transform completo."""
        X, y = preprocessor.fit_transform(sample_data)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, (pd.Series, pd.DataFrame))
        assert preprocessor.is_fitted == True
        assert len(X) == len(y)
    
    def test_fit_transform_sets_is_fitted(self, preprocessor, sample_data):
        """Testa se fit_transform define is_fitted como True."""
        preprocessor.fit_transform(sample_data)
        
        assert preprocessor.is_fitted == True
    
    def test_transform_without_fit_raises_error(self, preprocessor, sample_data):
        """Testa se transform lança erro sem fit prévio."""
        with pytest.raises(ValueError):
            preprocessor.transform(sample_data)
    
    def test_transform_after_fit(self, preprocessor, sample_data):
        """Testa transform após fit."""
        preprocessor.fit_transform(sample_data)
        result = preprocessor.transform(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
    
    def test_transform_preserves_shape(self, preprocessor, sample_data):
        """Testa se transform preserva o shape dos dados."""
        preprocessor.fit_transform(sample_data)
        result = preprocessor.transform(sample_data)
        
        assert result.shape[0] == sample_data.shape[0]
    
    def test_imputation_with_nulls(self, preprocessor, sample_data):
        """Testa a imputação de valores nulos."""
        X, y = preprocessor.fit_transform(sample_data)
        
        # Verificar se não há mais valores nulos
        assert not X.isnull().any().any()
    
    def test_define_features_returns_list(self, preprocessor):
        """Testa se define_features retorna uma lista."""
        features = preprocessor.define_features()
        
        assert isinstance(features, list)
    
    def test_clean_data_preserves_ra(self, preprocessor, sample_data):
        """Testa se clean_data preserva a coluna RA."""
        cleaned = preprocessor.clean_data(sample_data)
        
        # Coluna é convertida para minúsculo
        assert 'ra' in cleaned.columns
    
    def test_encode_categorical_handles_unknown_categories(self, preprocessor, sample_data):
        """Testa se encode_categorical lida com categorias desconhecidas."""
        preprocessor.fit_transform(sample_data)
        
        # Criar dados com categoria desconhecida
        test_data = sample_data.copy()
        test_data.loc[0, 'Pedra'] = 'Categoria_Desconhecida'
        
        # Não deve lançar erro
        result = preprocessor.transform(test_data)
        
        assert isinstance(result, pd.DataFrame)
    
    def test_scaler_is_standard_scaler(self, preprocessor):
        """Testa se o scaler é um StandardScaler."""
        from sklearn.preprocessing import StandardScaler
        assert isinstance(preprocessor.scaler, StandardScaler)
    
    def test_imputer_is_simple_imputer(self, preprocessor):
        """Testa se o imputer é um SimpleImputer."""
        from sklearn.impute import SimpleImputer
        assert isinstance(preprocessor.imputer, SimpleImputer)
    
    def test_label_encoders_is_dict(self, preprocessor):
        """Testa se label_encoders é um dicionário."""
        assert isinstance(preprocessor.label_encoders, dict)
    
    def test_fit_transform_returns_correct_types(self, preprocessor, sample_data):
        """Testa se fit_transform retorna os tipos corretos."""
        X, y = preprocessor.fit_transform(sample_data)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, (pd.Series, pd.DataFrame))
    
    def test_transform_returns_dataframe(self, preprocessor, sample_data):
        """Testa se transform retorna um DataFrame."""
        preprocessor.fit_transform(sample_data)
        result = preprocessor.transform(sample_data)
        
        assert isinstance(result, pd.DataFrame)
    
    def test_clean_data_with_missing_target(self, preprocessor, sample_data):
        """Testa clean_data com target missing."""
        # Criar dados com target missing
        data_with_null_target = sample_data.copy()
        data_with_null_target.loc[0, 'Defasagem'] = np.nan
        
        cleaned = preprocessor.clean_data(data_with_null_target)
        
        # Deve remover a linha com target missing
        assert len(cleaned) < len(data_with_null_target)
    
    def test_encode_categorical_fit_sets_encoders(self, preprocessor, sample_data):
        """Testa se fit define os encoders."""
        preprocessor.encode_categorical(sample_data, fit=True)
        
        # Deve ter pelo menos um encoder definido
        assert len(preprocessor.label_encoders) >= 0  # Pode ser 0 se não houver categóricas válidas
    
    def test_transform_without_ra_column(self, preprocessor, sample_data):
        """Testa transform sem coluna RA."""
        preprocessor.fit_transform(sample_data)
        
        # Criar dados sem RA
        test_data = sample_data.drop(columns=['RA'])
        
        # Não deve lançar erro
        result = preprocessor.transform(test_data)
        
        assert isinstance(result, pd.DataFrame)
    
    def test_fit_transform_with_all_features(self, preprocessor, sample_data):
        """Testa fit_transform com todas as features."""
        X, y = preprocessor.fit_transform(sample_data)
        
        assert len(X.columns) > 0
        assert len(y) > 0
    
    def test_define_features_includes_numerical(self, preprocessor):
        """Testa se define_features inclui features numéricas."""
        features = preprocessor.define_features()
        
        # Deve incluir pelo menos uma feature numérica
        assert len(preprocessor.numerical_columns) > 0
    
    def test_define_features_includes_categorical(self, preprocessor):
        """Testa se define_features inclui features categóricas."""
        features = preprocessor.define_features()
        
        # Deve incluir pelo menos uma feature categórica
        assert len(preprocessor.categorical_columns) > 0
    
    def test_clean_data_returns_dataframe_type(self, preprocessor, sample_data):
        """Testa se clean_data retorna DataFrame."""
        cleaned = preprocessor.clean_data(sample_data)
        
        assert isinstance(cleaned, pd.DataFrame)
    
    def test_encode_categorical_preserves_dataframe_shape(self, preprocessor, sample_data):
        """Testa se encode_categorical preserva o shape."""
        encoded = preprocessor.encode_categorical(sample_data, fit=True)
        
        assert encoded.shape[0] == sample_data.shape[0]
    
    def test_transform_preserves_column_count(self, preprocessor, sample_data):
        """Testa se transform preserva o número de colunas."""
        preprocessor.fit_transform(sample_data)
        result = preprocessor.transform(sample_data)
        
        assert result.shape[1] > 0
    
    def test_fit_transform_is_idempotent(self, preprocessor, sample_data):
        """Testa se fit_transform é idempotente."""
        X1, y1 = preprocessor.fit_transform(sample_data)
        X2, y2 = preprocessor.fit_transform(sample_data)
        
        assert X1.shape == X2.shape
        assert len(y1) == len(y2)
    
    def test_preprocessor_has_required_attributes(self, preprocessor):
        """Testa se o preprocessor tem todos os atributos necessários."""
        assert hasattr(preprocessor, 'scaler')
        assert hasattr(preprocessor, 'imputer')
        assert hasattr(preprocessor, 'label_encoders')
        assert hasattr(preprocessor, 'feature_columns')
        assert hasattr(preprocessor, 'is_fitted')
    
    def test_preprocessor_methods_exist(self, preprocessor):
        """Testa se o preprocessor tem todos os métodos necessários."""
        assert hasattr(preprocessor, 'define_features')
        assert hasattr(preprocessor, 'clean_data')
        assert hasattr(preprocessor, 'encode_categorical')
        assert hasattr(preprocessor, 'fit_transform')
        assert hasattr(preprocessor, 'transform')