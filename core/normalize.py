import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_active_data(active_dict: dict) -> dict:
    """
    Recebe o dicionário de um ativo (com arrays 1D de treino e teste originais),
    ajusta o MinMaxScaler no treino, transforma ambos os conjuntos para a escala [0, 1]
    e armazena o objeto scaler sob a chave 'MinMaxScaler'.
    
    Parâmetros:
    - active_dict (dict): Dicionário contendo as chaves 'train' e 'test' com arrays NumPy 1D.
    
    Retorna:
    - dict: O dicionário do ativo atualizado com os dados normalizados e o scaler salvo.
    """
    # 1. Extrai os vetores unidimensionais brutos
    train_raw = active_dict["train"]
    test_raw = active_dict["test"]
    
    # 2. Instancia o scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 3. Ajusta (fit) e transforma o treino
    # O scikit-learn exige arrays 2D (n_samples, n_features) para o fit,
    # então usamos .reshape(-1, 1) temporariamente e depois .flatten() para voltar a 1D.
    train_reshaped = train_raw.reshape(-1, 1)
    train_scaled = scaler.fit_transform(train_reshaped).flatten()
    
    # 4. Transforma o teste usando os limites aprendidos no treino
    test_reshaped = test_raw.reshape(-1, 1)
    test_scaled = scaler.transform(test_reshaped).flatten()
    
    # 5. Atualiza o dicionário com os novos valores escalonados e salva o objeto scaler
    active_dict["train"] = train_scaled
    active_dict["test"] = test_scaled
    active_dict["MinMaxScaler"] = scaler
    
    return active_dict