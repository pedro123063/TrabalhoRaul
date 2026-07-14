import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

def unpack_active_data(active_dict: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Desempacota (depack) o dicionário de um único ativo, extraindo as matrizes
    e vetores de treino e teste, além do objeto MinMaxScaler correspondente.
    
    Parâmetros:
    - active_dict (dict): Dicionário contendo as chaves 'train', 'test' e 'MinMaxScaler'.
    
    Retorna:
    - tuple: (X_train, y_train, X_test, y_test, scaler)
    """
    X_train = active_dict["train"]["window_matrix"]
    y_train = active_dict["train"]["answers"]
    
    X_test = active_dict["test"]["window_matrix"]
    y_test = active_dict["test"]["answers"]
    
    scaler = active_dict["MinMaxScaler"]
    
    return X_train, y_train, X_test, y_test, scaler


def train_svr(X_train: np.ndarray, y_train: np.ndarray, kernel: str = "rbf", C: float = 1.0, epsilon: float = 0.1) -> SVR:
    """
    Instancia e treina o modelo Support Vector Regression (SVR) com os dados de treino.
    
    Parâmetros:
    - X_train (np.ndarray): Matriz de janelas de treino.
    - y_train (np.ndarray): Vetor de respostas de treino.
    - kernel (str): O tipo de kernel do SVR (padrão: 'rbf').
    - C (float): Parâmetro de regularização (padrão: 1.0).
    - epsilon (float): Margem de tolerância onde nenhuma penalidade é associada (padrão: 0.1).
    
    Retorna:
    - SVR: O modelo do scikit-learn já treinado.
    """
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(X_train, y_train)
    return model


def predict_svr(model: SVR, X_test: np.ndarray) -> np.ndarray:
    """
    Realiza as perguntas (predições) ao modelo treinado utilizando as janelas de teste.
    
    Parâmetros:
    - model (SVR): O modelo SVR previamente treinado.
    - X_test (np.ndarray): Matriz de janelas de teste.
    
    Retorna:
    - np.ndarray: Vetor com as previsões do modelo (ainda em escala normalizada [0, 1]).
    """
    predictions_scaled = model.predict(X_test)
    return predictions_scaled

def denormalize_data(scaled_array: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Desnormaliza (denorm) um vetor de dados, convertendo-os da escala [0, 1] 
    de volta para os valores reais originais.
    
    Parâmetros:
    - scaled_array (np.ndarray): Vetor 1D com os dados normalizados (previsões ou y_test).
    - scaler (MinMaxScaler): O objeto MinMaxScaler ajustado na fase de treino.
    
    Retorna:
    - np.ndarray: Vetor 1D contendo os preços reais (ex: em Reais).
    """
    # O scikit-learn exige arrays 2D para realizar a transformação inversa
    reshaped_array = scaled_array.reshape(-1, 1)
    
    # Aplica a reversão da escala e retorna para o formato 1D (flatten)
    denormalized_array = scaler.inverse_transform(reshaped_array).flatten()
    
    return denormalized_array

def run_svr_pipeline(active_dict: dict, kernel: str = "rbf", C: float = 1.0, epsilon: float = 0.1) -> dict:
    """
    Executa o pipeline completo do SVR para um único ativo: desempacota,
    treina, prevê e desnormaliza os resultados de volta para a escala original.
    
    Parâmetros:
    - active_dict (dict): Dicionário de um ativo contendo 'train', 'test' e 'MinMaxScaler'.
    - kernel (str): O kernel do SVR.
    - C (float): Parâmetro de regularização.
    - epsilon (float): Margem de erro do SVR.
    
    Retorna:
    - dict: Dicionário contendo os vetores de previsões e respostas reais desnormalizadas.
      Formato: {'predicted': np.ndarray, 'answers': np.ndarray}
    """
    # 1. Desempacota as partições e o scaler
    X_train, y_train, X_test, y_test, scaler = unpack_active_data(active_dict)
    
    # 2. Treina o modelo SVR com os dados escalonados
    model = train_svr(X_train, y_train, kernel=kernel, C=C, epsilon=epsilon)
    
    # 3. Gera as previsões (ainda na escala normalizada [0, 1])
    predictions_scaled = predict_svr(model, X_test)
    
    # 4. Desnormaliza as previsões e o gabarito original do teste
    predictions_real = denormalize_data(predictions_scaled, scaler)
    answers_real = denormalize_data(y_test, scaler)
    
    # 5. Retorna no formato solicitado
    return {
        "predicted": predictions_real,
        "answers": answers_real
    }