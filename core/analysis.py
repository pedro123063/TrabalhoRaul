import numpy as np

def calculate_rmse(predicted: np.ndarray, answers: np.ndarray) -> float:
    errors = predicted - answers
    return float(np.sqrt(np.mean(errors ** 2)))

def calculate_mape(predicted: np.ndarray, answers: np.ndarray) -> float:
    absolute_percentage_errors = np.abs((answers - predicted) / answers)
    return float(np.mean(absolute_percentage_errors) * 100)

def calculate_daily_errors(predicted: np.ndarray, answers: np.ndarray) -> dict:
    return {
        "absolute": np.abs(answers - predicted),
        "percentage": (np.abs(answers - predicted) / answers) * 100
    }

def run_analysis_pipeline(resultados_finais: dict) -> dict:
    print("\n=== INICIANDO PROCESSAMENTO DAS MÉTRICAS DE ANÁLISE ===")
    
    analise_completa = {
        "todos_cenarios": {},
        "campeoes_rmse": {},
        "campeoes_mape": {}
    }
    
    for ticker, kernels in resultados_finais.items():
        print(f"-> Analisando múltiplos eixos (incluindo janela) e identificando campeões para: {ticker}")
        
        analise_completa["todos_cenarios"][ticker] = {}
        
        melhor_rmse = float("inf")
        melhor_mape = float("inf")
        
        campeao_rmse_data = {}
        campeao_mape_data = {}
        
        # Varredura dos 5 eixos
        for kernel, splits in kernels.items():
            analise_completa["todos_cenarios"][ticker][kernel] = {}
            for split, cs in splits.items():
                analise_completa["todos_cenarios"][ticker][kernel][split] = {}
                for c_value, wrs in cs.items():
                    analise_completa["todos_cenarios"][ticker][kernel][split][c_value] = {}
                    for w_ratio, dados in wrs.items():
                        pred = dados["predicted"]
                        real = dados["answers"]
                        
                        rmse_val = calculate_rmse(pred, real)
                        mape_val = calculate_mape(pred, real)
                        erros_diarios = calculate_daily_errors(pred, real)
                        
                        cenario_res = {
                            "metrics": {"rmse": rmse_val, "mape": mape_val},
                            "time_series": {
                                "predicted": pred,
                                "answers": real,
                                "daily_absolute_errors": erros_diarios["absolute"],
                                "daily_percentage_errors": erros_diarios["percentage"]
                            }
                        }
                        
                        # Salva na árvore de 5 níveis
                        analise_completa["todos_cenarios"][ticker][kernel][split][c_value][w_ratio] = cenario_res
                        
                        # Identifica campeão por RMSE
                        if rmse_val < melhor_rmse:
                            melhor_rmse = rmse_val
                            campeao_rmse_data = {
                                "config": {"kernel": kernel, "split": split, "C": c_value, "window_ratio": w_ratio},
                                "metrics": {"rmse": rmse_val, "mape": mape_val},
                                "time_series": cenario_res["time_series"]
                            }
                            
                        # Identifica campeão por MAPE
                        if mape_val < melhor_mape:
                            melhor_mape = mape_val
                            campeao_mape_data = {
                                "config": {"kernel": kernel, "split": split, "C": c_value, "window_ratio": w_ratio},
                                "metrics": {"rmse": rmse_val, "mape": mape_val},
                                "time_series": cenario_res["time_series"]
                            }
        
        analise_completa["campeoes_rmse"][ticker] = campeao_rmse_data
        analise_completa["campeoes_mape"][ticker] = campeao_mape_data
        
    print("=== ANÁLISE CONCLUÍDA E MODELOS SELECIONADOS COM SUCESSO! ===")
    return analise_completa