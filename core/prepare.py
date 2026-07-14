import pandas as pd
import numpy as np
from core import normalize

def convert_pd_to_numpy(df: pd.DataFrame, target_col: str = "Close") -> dict:
    """
    Separa um DataFrame multi-ações em séries temporais individuais para cada ticker
    e as converte em arrays unidimensionais do NumPy.
    
    Parâmetros:
    - df (pd.DataFrame): O DataFrame carregado e filtrado (contendo uma ou mais ações).
    - target_col (str): A coluna numérica que queremos isolar para o SVR (padrão: 'Close').
    
    Retorna:
    - dict: Um dicionário onde as chaves são os Tickers (str) e os valores são 
            arrays do NumPy (np.ndarray) com o histórico de preços daquela ação.
    """
    # Identifica todos os tickers únicos presentes no DataFrame
    unique_tickers = df['Ticker'].unique()
    
    ticker_matrices = {}
    
    print(f"Iniciando conversão para NumPy para os tickers: {list(unique_tickers)}")
    
    for ticker in unique_tickers:
        # 1. Isola apenas as linhas deste ticker específico
        df_single = df[df['Ticker'] == ticker].copy()
        
        # Como o df principal já foi ordenado por data na ingestão, 
        # a ordem cronológica aqui está 100% garantida.
        
        # 2. Extrai a coluna alvo como um array NumPy unidimensional
        numpy_array = df_single[target_col].to_numpy()
        
        # 3. Armazena no dicionário associado ao nome do ticker
        ticker_matrices[ticker] = numpy_array
        
        print(f"  -> Ticker '{ticker}': convertido com sucesso. Formato: {numpy_array.shape}")
        
    print("Conversão concluída para todas as ações!")
    return ticker_matrices

def temporal_split(data_array: np.ndarray, split_ratio: float = 0.8) -> dict:
    """
    Divide um array unidimensional do NumPy de forma cronológica (sem embaralhamento)
    em conjuntos de treino e teste.
    
    Parâmetros:
    - data_array (np.ndarray): O array do NumPy contendo a série temporal da ação.
    - split_ratio (float): Proporção de dados que irá para o treino (ex: 0.8 para 80%).
                           Deve estar estritamente entre 0.0 e 1.0.
    
    Retorna:
    - dict: Um dicionário com a estrutura {"train": np.ndarray, "test": np.ndarray}.
    """
    # Garante que o ratio de corte esteja dentro dos limites aceitáveis
    if not (0.0 < split_ratio < 1.0):
        raise ValueError("O parâmetro split_ratio deve estar entre 0.0 e 1.0 (exclusivo).")
        
    # Calcula o ponto exato de corte baseado no tamanho total do array
    total_size = len(data_array)
    split_index = int(total_size * split_ratio)
    
    # Divide cronologicamente as fatias de dados (slicing do NumPy)
    train_data = data_array[:split_index]
    test_data = data_array[split_index:]
    
    print(f"Divisão concluída (Ratio: {split_ratio:.1%}):")
    print(f"  -> Treino (0 até {split_index}): {train_data.shape}")
    print(f"  -> Teste ({split_index} até {total_size}): {test_data.shape}")
    
    return {
        "train": train_data,
        "test": test_data
    }
def create_sliding_windows(data_array: np.ndarray, k: int) -> dict:
    """
    Transforma um array unidimensional em uma matriz de janelas deslizantes (X)
    e um vetor de respostas correspondentes (y).
    
    Parâmetros:
    - data_array (np.ndarray): O array unidimensional do NumPy (tamanho n).
    - k (int): O tamanho da janela retrovisora (quantidade de colunas da matriz X).
    
    Retorna:
    - dict: Um dicionário com o formato:
            {
                'window_matrix': np.ndarray,  # Matriz de formato (n - k, k)
                'answers': np.ndarray         # Vetor de formato (n - k,)
            }
    """
    # 1. Verifica se a entrada é um array do NumPy
    if not isinstance(data_array, np.ndarray):
        raise TypeError("A entrada data_array deve ser estritamente um array do NumPy.")
        
    # 2. Verifica o tamanho n do array usando a propriedade shape
    n = data_array.shape[0]
    
    # 3. Validação lógica do tamanho da janela k
    if k <= 0:
        raise ValueError("O tamanho da janela k deve ser maior que zero.")
    if k >= n:
        raise ValueError(
            f"O tamanho da janela k ({k}) não pode ser maior ou igual "
            f"ao tamanho total do vetor n ({n})."
        )
        
    # 4. Aloca espaço ou constrói as matrizes baseadas em n - k passadas
    num_samples = n - k
    
    # Criamos a matriz X (window_matrix) preenchida com zeros para eficiência
    window_matrix = np.empty((num_samples, k))
    
    # Criamos o vetor y (answers) preenchido com zeros
    answers = np.empty(num_samples)
    
    # 5. Preenche as estruturas deslizando a janela
    for i in range(num_samples):
        # Janela de entrada vai do índice i até i + k (sem incluir o i + k)
        window_matrix[i] = data_array[i : i + k]
        # A resposta correspondente é exatamente o elemento no índice i + k
        answers[i] = data_array[i + k]
        
    print(f"Janelamento concluído (Janela k={k}):")
    print(f"  -> Matriz de janelas (X): {window_matrix.shape}")
    print(f"  -> Vetor de respostas (y): {answers.shape}")
    
    return {
        "window_matrix": window_matrix,
        "answers": answers
    }

def prepare_pipeline(
    df: pd.DataFrame, 
    split_ratio: float = 0.8, 
    window_ratio: float = 0.1, 
    target_col: str = "Close"
) -> dict:
    """
    Pipeline que encapsula a conversão do DataFrame filtrado para arrays NumPy,
    realiza a divisão temporal (split) para cada ticker, aplica a normalização 
    MinMaxScaler (ajustada no treino) e gera as matrizes de janelas deslizantes 
    (X) e respostas (y) para treino e teste.
    """
    # 1. Executa a conversão inicial (Gera dict de {ticker: array_completo})
    ticker_data = convert_pd_to_numpy(df, target_col=target_col)
    
    # 2. Varre o dicionário aplicando o split temporal em cada ativo
    for ticker, data_array in ticker_data.items():
        print(f"\n--- Processando divisões para {ticker} ---")
        
        # Realiza o split temporal bruto (Retorna dict com arrays 1D brutos de 'train' e 'test')
        split_result = temporal_split(data_array, split_ratio=split_ratio)
        
        # === NOVO PASSO: Normalização ===
        # Passamos o dicionário de divisões brutas para ser normalizado
        # A função altera split_result adicionando a chave 'MinMaxScaler' e normalizando 'train' e 'test'
        print(f"  -> Normalizando dados de {ticker}...")
        split_result = normalize.normalize_active_data(split_result)
        
        # Agora estes arrays já estão normalizados em escala [0, 1]
        train_array = split_result["train"]
        test_array = split_result["test"]
        n_test = len(test_array)
        
        # 3. Calcula o tamanho inteiro de k para a janela deslizante
        k = max(1, int(n_test * window_ratio))
        
        # Verificação de segurança para garantir que k seja válido
        n_train = len(train_array)
        if k >= n_train or k >= n_test:
            raise ValueError(
                f"O tamanho da janela calculado (k={k}) é inválido para as partições. "
                f"Garanta que o vetor de treino (tamanho {n_train}) e de teste (tamanho {n_test}) "
                f"sejam estritamente maiores que k."
            )
        print(f"  -> Tamanho de janela k calculado: {k} (usando {window_ratio:.1%} de {n_test} dias de teste)")
        
        # 4. Aplica o janelamento em ambos os conjuntos de forma simétrica (usando os dados já normalizados)
        print("  -> Aplicando janelamento no Treino...")
        split_result["train"] = create_sliding_windows(train_array, k=k)
        
        print("  -> Aplicando janelamento no Teste...")
        split_result["test"] = create_sliding_windows(test_array, k=k)
        
        # Atualiza o dicionário global com a nova estrutura simétrica contendo:
        # - "train": {"window_matrix": ..., "answers": ...}
        # - "test": {"window_matrix": ..., "answers": ...}
        # - "MinMaxScaler": objeto_scaler
        ticker_data[ticker] = split_result
        
    print("\nPipeline de preparação com janelamento simétrico concluído com sucesso!")
    return ticker_data