import numpy as np

# =====================================================================
# 1. MÉTRICAS ACUMULADAS (Para a Tabela do Seminário)
# =====================================================================


def calculate_rmse(results_dict: dict) -> float :
    """
    Calcula a Raiz do Erro Quadrático Médio (RMSE) geral (um único número).
    """
    predicted = results_dict["predicted"]
    answers = results_dict["answers"]
    errors = predicted - answers
    rmse = np.sqrt(np.mean(errors ** 2))
    return float(rmse)


def calculate_mape(results_dict: dict) -> float:
    """
    Calcula o Erro Percentual Absoluto Médio (MAPE) geral (um único número).
    """
    predicted = results_dict["predicted"]
    answers = results_dict["answers"]
    absolute_percentage_errors = np.abs((answers - predicted) / answers)
    mape = np.mean(absolute_percentage_errors) * 100
    return float(mape)


# =====================================================================
# 2. MÉTRICAS PONTO A PONTO (Para gerar as curvas dos Gráficos)
# =====================================================================

def calculate_daily_absolute_errors(results_dict: dict) -> np.ndarray:
    """
    Calcula o erro absoluto ponto a ponto (em Reais) para cada dia do teste.
    Equivalente diário que compõe a lógica do RMSE.
    
    Parâmetros:
    - results_dict (dict): Dicionário contendo 'predicted' e 'answers'.
    
    Retorna:
    - np.ndarray: Array unidimensional com a série temporal de erros em Reais.
    """
    predicted = results_dict["predicted"]
    answers = results_dict["answers"]
    
    # Diferença absoluta dia a dia: |Real - Previsto|
    daily_errors = np.abs(answers - predicted)
    
    return daily_errors


def calculate_daily_percentage_errors(results_dict: dict) -> np.ndarray:
    """
    Calcula o erro percentual ponto a ponto (em %) para cada dia do teste.
    Equivalente diário que compõe a lógica do MAPE.
    
    Parâmetros:
    - results_dict (dict): Dicionário contendo 'predicted' e 'answers'.
    
    Retorna:
    - np.ndarray: Array unidimensional com a série temporal de erros em %.
    """
    predicted = results_dict["predicted"]
    answers = results_dict["answers"]
    
    # Erro percentual absoluto dia a dia: (|Real - Previsto| / Real) * 100
    daily_percentage_errors = (np.abs(answers - predicted) / answers) * 100
    
    return daily_percentage_errors

def run_analysis_pipeline(resultados_finais: dict) -> dict:
    """
    Recebe os resultados brutos do SVR e encapsula todas as análises matemáticas
    (métricas acumuladas e séries temporais de erros) em uma estrutura unificada.
    
    Parâmetros:
    - resultados_finais (dict): Dicionário no formato:
                                {ticker: {"predicted": np.ndarray, "answers": np.ndarray}}
                                
    Retorna:
    - dict: Estrutura completa contendo métricas e séries de erros por ativo,
            pronta para alimentar o gerador de gráficos e tabelas.
    """
    analise_consolidada = {}
    
    print("\n=== INICIANDO PROCESSAMENTO DAS MÉTRICAS DE ANÁLISE ===")
    
    for ticker, resultado_ativo in resultados_finais.items():
        print(f"-> Analisando desvios e métricas para: {ticker}")
        
        # 1. Cálculos de performance geral (para as tabelas)
        rmse_geral = calculate_rmse(resultado_ativo)
        mape_geral = calculate_mape(resultado_ativo)
        
        # 2. Cálculos ponto a ponto (para as curvas dos gráficos)
        erros_absolutos = calculate_daily_absolute_errors(resultado_ativo)
        erros_percentuais = calculate_daily_percentage_errors(resultado_ativo)
        
        # 3. Montagem da estrutura organizada por ativo
        analise_consolidada[ticker] = {
            "metrics": {
                "rmse": rmse_geral,
                "mape": mape_geral
            },
            "time_series": {
                "predicted": resultado_ativo["predicted"],
                "answers": resultado_ativo["answers"],
                "daily_absolute_errors": erros_absolutos,
                "daily_percentage_errors": erros_percentuais
            }
        }
        
    print("=== ANÁLISE CONCLUÍDA E ESTRUTURADA COM SUCESSO! ===")
    return analise_consolidada