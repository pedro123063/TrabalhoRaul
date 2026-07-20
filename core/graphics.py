import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def _garantir_diretorio(diretorio: str):
    if not os.path.exists(diretorio):
        os.makedirs(diretorio)

# =====================================================================
# 1. GERADOR DE TABELA CONSOLIDADA (Salva como imagem e printa no console)
# =====================================================================
def gerar_tabela_consolida(analise_completa: dict, output_dir: str):
    _garantir_diretorio(output_dir)
    linhas = []
    
    # Nova varredura com 5 níveis
    for ticker, kernels in analise_completa["todos_cenarios"].items():
        for kernel, splits in kernels.items():
            for split, cs in splits.items():
                for c_value, wrs in cs.items():
                    for w_ratio, dados in wrs.items():
                        rmse = dados["metrics"]["rmse"]
                        mape = dados["metrics"]["mape"]
                        linhas.append({
                            "Ticker": ticker,
                            "Kernel": kernel,
                            "Split": split,
                            "C": c_value,
                            "Window": f"{w_ratio * 100:.0f}%", # Novo campo visual
                            "RMSE (R$)": f"R$ {rmse:.4f}",
                            "MAPE (%)": f"{mape:.2f}%"
                        })
    
    df_tabela = pd.DataFrame(linhas)
    
    print("\n" + "="*80)
    print("           TABELA CONSOLIDADA DE MÉTRICAS MULTIVARIADAS")
    print("="*80)
    print(df_tabela.to_string(index=False))
    print("="*80)
    
    fig, ax = plt.subplots(figsize=(10, len(df_tabela) * 0.35 + 1.5))
    ax.axis("tight")
    ax.axis("off")
    
    tabela_plot = ax.table(
        cellText=df_tabela.values,
        colLabels=df_tabela.columns,
        cellLoc="center",
        loc="center"
    )
    tabela_plot.auto_set_font_size(False)
    tabela_plot.set_fontsize(9)
    tabela_plot.scale(1.2, 1.2)
    
    plt.title("Métricas Consolidadas por Cenário", fontsize=12, weight="bold", pad=20)
    plt.savefig(os.path.join(output_dir, "tabela_consolidada.png"), bbox_inches="tight", dpi=150)
    plt.close()

# =====================================================================
# 2. GRÁFICOS DE SENSIBILIDADE POR TICKER (Absoluto e Relativo)
# =====================================================================
def plotar_sensibilidade_por_ticker(analise_completa: dict, output_dir: str):
    _garantir_diretorio(output_dir)
    
    for ticker, kernels in analise_completa["todos_cenarios"].items():
        # --- 2.1 Erros Absolutos (R$) ---
        plt.figure(figsize=(12, 6))
        for kernel, splits in kernels.items():
            for split, cs in splits.items():
                for c_value, wrs in cs.items():
                    for w_ratio, dados in wrs.items():
                        erros_absolutos = dados["time_series"]["daily_absolute_errors"]
                        passos_tempo = np.arange(len(erros_absolutos))
                        label_curva = f"K: {kernel} | S: {split} | C: {c_value} | W: {w_ratio:.2f}"
                        plt.plot(passos_tempo, erros_absolutos, label=label_curva, alpha=0.8)
        
        plt.title(f"Sensibilidade de Erro Diário Absoluto (R$) - {ticker}", fontsize=14, weight="bold")
        plt.xlabel("Dias de Teste", fontsize=11)
        plt.ylabel("Erro Absoluto (R$)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
        plt.savefig(os.path.join(output_dir, f"sensibilidade_absoluta_{ticker}.png"), bbox_inches="tight", dpi=150)
        plt.close()

        # --- 2.2 Erros Relativos (%) ---
        plt.figure(figsize=(12, 6))
        for kernel, splits in kernels.items():
            for split, cs in splits.items():
                for c_value, wrs in cs.items():
                    for w_ratio, dados in wrs.items():
                        erros_percentuais = dados["time_series"]["daily_percentage_errors"]
                        passos_tempo = np.arange(len(erros_percentuais))
                        label_curva = f"K: {kernel} | S: {split} | C: {c_value} | W: {w_ratio:.2f}"
                        plt.plot(passos_tempo, erros_percentuais, label=label_curva, alpha=0.8)
        
        plt.title(f"Sensibilidade de Erro Diário Percentual (%) - {ticker}", fontsize=14, weight="bold")
        plt.xlabel("Dias de Teste", fontsize=11)
        plt.ylabel("Erro Percentual (%)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
        plt.savefig(os.path.join(output_dir, f"sensibilidade_relativa_{ticker}.png"), bbox_inches="tight", dpi=150)
        plt.close()

# =====================================================================
# 3. GRÁFICOS DOS CAMPEÕES (Melhor Absoluto vs Melhor Relativo)
# =====================================================================
def plotar_campeoes_comparativos(analise_completa: dict, output_dir: str):
    _garantir_diretorio(output_dir)
    
    # 3.1 Campeões por RMSE (Métrica Absoluta - Erro em R$)
    plt.figure(figsize=(10, 5))
    for ticker, campeao in analise_completa["campeoes_rmse"].items():
        erros_absolutos = campeao["time_series"]["daily_absolute_errors"]
        cfg = campeao["config"]
        plt.plot(
            erros_absolutos, 
            label=f"{ticker} ({cfg['kernel']}, S:{cfg['split']}, C:{cfg['C']}, W:{cfg['window_ratio']:.2f}) | RMSE: R$ {campeao['metrics']['rmse']:.2f}"
        )
    plt.title("Performance dos Modelos Campeões (Menor RMSE Absoluto)", fontsize=12, weight="bold")
    plt.xlabel("Dias de Teste")
    plt.ylabel("Erro Absoluto (R$)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9)
    plt.savefig(os.path.join(output_dir, "campeoes_melhor_rmse.png"), bbox_inches="tight", dpi=150)
    plt.close()

    # 3.2 Campeões por MAPE (Métrica Relativa - Erro em %)
    plt.figure(figsize=(10, 5))
    for ticker, campeao in analise_completa["campeoes_mape"].items():
        erros_percentuais = campeao["time_series"]["daily_percentage_errors"]
        cfg = campeao["config"]
        plt.plot(
            erros_percentuais, 
            label=f"{ticker} ({cfg['kernel']}, S:{cfg['split']}, C:{cfg['C']}, W:{cfg['window_ratio']:.2f}) | MAPE: {campeao['metrics']['mape']:.2f}%"
        )
    plt.title("Performance dos Modelos Campeões (Menor MAPE Relativo)", fontsize=12, weight="bold")
    plt.xlabel("Dias de Teste")
    plt.ylabel("Erro Percentual (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9)
    plt.savefig(os.path.join(output_dir, "campeoes_melhor_mape.png"), bbox_inches="tight", dpi=150)
    plt.close()

# =====================================================================
# PIPELINE INTEGRADO DE GRÁFICOS
# =====================================================================
def run_graphics_pipeline(analise_completa: dict, output_dir: str = "output"):
    print("\n=== INICIANDO EXPORTAÇÃO DE GRÁFICOS E TABELAS ===")
    
    gerar_tabela_consolida(analise_completa, output_dir)
    print("-> Tabela consolidada criada com sucesso!")
    
    plotar_sensibilidade_por_ticker(analise_completa, output_dir)
    print("-> Gráficos de sensibilidade (Absoluta e Relativa) salvos por ticker!")
    
    plotar_campeoes_comparativos(analise_completa, output_dir)
    print("-> Gráficos comparativos de campeões (RMSE/MAPE) salvos!")
    
    print("=== PIPELINE DE GRÁFICOS CONCLUÍDO COM SUCESSO! ===")