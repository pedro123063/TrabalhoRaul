import pandas as pd
import numpy as np
from core import normalize
from core import promptBuilder

def convert_pd_to_numpy(df: pd.DataFrame, target_col: str = "Close") -> dict:   

    unique_tickers = df['Ticker'].unique()
    
    ticker_matrices = {}
    
    
    for ticker in unique_tickers:
        df_single = df[df['Ticker'] == ticker].copy()
       
        numpy_array = df_single[target_col].to_numpy()
        
        ticker_matrices[ticker] = numpy_array
        
    return ticker_matrices

def temporal_split(data_array: np.ndarray, split_ratio: float = 0.8) -> dict:

   
    if not (0.0 < split_ratio < 1.0):
        raise ValueError("O parâmetro split_ratio deve estar entre 0.0 e 1.0 (exclusivo).")
        
    
    total_size = len(data_array)
    split_index = int(total_size * split_ratio)
    
    train_data = data_array[:split_index]
    test_data = data_array[split_index:]
    return {"train": train_data,"test": test_data}

def create_sliding_windows(data_array: np.ndarray, k: int) -> dict:

    
    if not isinstance(data_array, np.ndarray):
        raise TypeError("A entrada data_array deve ser estritamente um array do NumPy.")
        
    
    n = data_array.shape[0]
    
    if k <= 0:
        raise ValueError("O tamanho da janela k deve ser maior que zero.")
    if k >= n:
        raise ValueError(
            f"O tamanho da janela k ({k}) não pode ser maior ou igual "
            f"ao tamanho total do vetor n ({n})."
        )
        
    num_samples = n - k

    window_matrix = np.empty((num_samples, k))

    answers = np.empty(num_samples)

    for i in range(num_samples):

        window_matrix[i] = data_array[i : i + k]
        answers[i] = data_array[i + k]
        
   
    return {
        "window_matrix": window_matrix,
        "answers": answers
    }

def separate_llm(split_result, llm_days) -> np.ndarray:

    dados_1d = np.asarray(split_result).ravel()

    tamanho_necessario = max(llm_days + 1, 31) 
    
    return dados_1d[-tamanho_necessario:].copy()
def prepare_pipeline(
    df: pd.DataFrame, 
    split_ratio: float = 0.8, 
    window_ratio: float = 0.1, 
    target_col: str = "Close",
    llm_days=29 ) -> dict:
    
    ticker_data = convert_pd_to_numpy(df, target_col=target_col)

    for ticker, data_array in ticker_data.items():
        split_result = temporal_split(data_array, split_ratio=split_ratio)

        llm_data={}
        llm_data["data"]=separate_llm(split_result=split_result["test"],llm_days=llm_days)        
        llm_data["prompts"]={}
        llm_data["prompts"]["few-shot"]=promptBuilder.assembleFewShot(llm_data["data"],5,5)
        llm_data["prompts"]["zero-shot"]=promptBuilder.assembleZeroShot(llm_data["data"],5)

        split_result = normalize.normalize_active_data(split_result)

        train_array = split_result["train"]
        test_array = split_result["test"]
        n_test = len(test_array)

        k = max(1, int(n_test * window_ratio))
        

        n_train = len(train_array)
        if k >= n_train or k >= n_test:
            raise ValueError(
                f"O tamanho da janela calculado (k={k}) é inválido para as partições. "
                f"Garanta que o vetor de treino (tamanho {n_train}) e de teste (tamanho {n_test}) "
                f"sejam estritamente maiores que k."
            )
        
        split_result["train"] = create_sliding_windows(train_array, k=k)
        
        split_result["test"] = create_sliding_windows(test_array, k=k)
        split_result["llm"]=llm_data

        ticker_data[ticker] = split_result
        

    return ticker_data