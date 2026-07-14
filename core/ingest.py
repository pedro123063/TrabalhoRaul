import pandas as pd
import os

def load_stock_data(tickers: list = None, file_path: str = "stock_prices_daily.csv") -> pd.DataFrame:
    """
    Carrega o dataset de ações do formato CSV para um DataFrame do Pandas,
    aplicando a filtragem por tickers selecionados.
    
    Parâmetros:
    - tickers (list): Lista de tickers para filtrar (ex: ['AAPL']). Se None, carrega todos.
    - file_path (str): Caminho para o arquivo .csv.
    
    Retorna:
    - pd.DataFrame: DataFrame filtrado e ordenado cronologicamente por data.
    """
    # Se o arquivo não estiver no diretório atual, tenta buscar uma pasta acima (..)
    if not os.path.exists(file_path):
        alternative_path = os.path.join("..", file_path)
        if os.path.exists(alternative_path):
            file_path = alternative_path
        else:
            raise FileNotFoundError(
                f"Erro: O arquivo '{file_path}' não foi encontrado no diretório atual "
                f"nem em '{alternative_path}'."
            )
        
    print(f"Lendo dados de: {file_path}...")
    
    # 1. Leitura inicial convertendo a data
    df = pd.read_csv(file_path, parse_dates=['Date'])
    
    # 2. Filtra os dados primeiro (reduz drasticamente o uso de memória antes da ordenação)
    if tickers is not None:
        df = filter_by_tickers(df, tickers)
    
    # 3. Ordenação cronológica definitiva (essencial para as séries temporais)
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    print(f"Carga final concluída! Total de registros em memória: {len(df)}")
    return df

def filter_by_tickers(df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Filtra o DataFrame de ações para manter apenas os registros dos tickers especificados.
    """
    if not tickers:
        print("Aviso: Lista de tickers vazia. Retornando o DataFrame original.")
        return df
        
    print(f"Filtrando dados para os tickers: {tickers}...")
    
    # Filtra mantendo apenas as linhas dos tickers desejados
    filtered_df = df[df['Ticker'].isin(tickers)].copy()
    
    print(f"Filtragem concluída! Registros correspondentes: {len(filtered_df)}")
    return filtered_df