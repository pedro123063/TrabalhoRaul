import numpy as np

def calculate_rmse(predicted: np.ndarray, answers: np.ndarray) -> float:
    errors = predicted - answers
    return float(np.sqrt(np.mean(errors ** 2)))

def calculate_mape(predicted: np.ndarray, answers: np.ndarray) -> float:
    absolute_percentage_errors = np.abs((answers - predicted) / answers)
    return float(np.mean(absolute_percentage_errors) * 100)

def run_analysis_pipeline(resultados_finais: dict) -> dict:

    campeoes_por_ticker = {}

    for ticker, kernels in resultados_finais.items():
        campeoes_por_ticker[ticker] = {
            "campeao_rmse": None,
            "campeao_mape": None,
            "campeao_tempo": None
        }
        
        melhor_rmse = float('inf')
        melhor_mape = float('inf')
        melhor_tempo = float('inf')
        

        for kernel, splits in kernels.items():
            for split, c_values in splits.items():
                for c_value, windows in c_values.items():
                    for w_ratio, epsilons in windows.items():
                        for epsilon_val, dados in epsilons.items():
                            
                            predicted = dados["predicted"]
                            answers = dados["answers"]

                            tempo_treino = dados.get("tempo_treino", None)
                            tempo_predicao = dados.get("tempo_predicao", None)
                            tempo_total = dados.get("tempo_total", tempo_treino if tempo_treino else 0.0)
                            

                            rmse = calculate_rmse(predicted, answers)
                            mape = calculate_mape(predicted, answers)
                            
                            cenario_summary = {
                                "config": {
                                    "ticker": ticker,
                                    "kernel": kernel,
                                    "split": split,
                                    "c": c_value,
                                    "window_ratio": w_ratio,
                                    "epsilon": epsilon_val
                                },
                                "metrics": {
                                    "rmse": rmse,
                                    "mape": mape,
                                    "tempo_treino": tempo_treino,
                                    "tempo_predicao": tempo_predicao,
                                    "tempo_total": tempo_total
                                }
                            }
                            

                            if rmse < melhor_rmse:
                                melhor_rmse = rmse
                                campeoes_por_ticker[ticker]["campeao_rmse"] = cenario_summary

                            if mape < melhor_mape:
                                melhor_mape = mape
                                campeoes_por_ticker[ticker]["campeao_mape"] = cenario_summary


                            if tempo_total < melhor_tempo:
                                melhor_tempo = tempo_total
                                campeoes_por_ticker[ticker]["campeao_tempo"] = cenario_summary


    return campeoes_por_ticker