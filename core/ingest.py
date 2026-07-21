import pandas as pd
import os

def load_stock_data(tickers: list = None, file_path: str = "stock_prices_daily.csv") -> pd.DataFrame:
    
    if not os.path.exists(file_path):
        alternative_path = os.path.join("..", file_path)
        if os.path.exists(alternative_path):
            file_path = alternative_path
        else:
            raise FileNotFoundError(f"Erro: O arquivo '{file_path}' não foi encontrado no diretório atual "+f"nem em '{alternative_path}'.")

    df = pd.read_csv(file_path, parse_dates=['Date'])
    
    if tickers is not None:
        df = filter_by_tickers(df, tickers)
    
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    return df

def filter_by_tickers(df: pd.DataFrame, tickers: list) -> pd.DataFrame:

    if not tickers:
        print("Aviso: Lista de tickers vazia. Retornando o DataFrame original.")
        return df

    filtered_df = df[df['Ticker'].isin(tickers)].copy()
    return filtered_df