import os
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_graphics_pipeline(analise_consolidada: Dict[str, dict], output_dir: str = "output") -> None:
    """
    Consome a estrutura de dados consolidados da análise para gerar
    as visualizações e relatórios do seminário:
    
    1. Imprime a tabela de métricas (RMSE e MAPE) no terminal e a salva como CSV.
    2. Renderiza e salva a tabela de métricas como uma imagem PNG polida.
    3. Gera o gráfico de evolução temporal de erros absolutos diários (R$).
    4. Gera o gráfico de evolução temporal de erros percentuais diários (%).
    
    Parâmetros:
    - analise_consolidada (dict): Dicionário contendo a estrutura gerada pelo 
                                  pipeline do módulo de análise.
    - output_dir (str): Nome do diretório onde as imagens serão salvas.
    """
    # 1. Garante a existência do diretório de saída
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # =========================================================================
    # 2. CONSTRUÇÃO DA TABELA DE MÉTRICAS GERAIS
    # =========================================================================
    resumo_metricas = []
    
    for ticker, info in analise_consolidada.items():
        rmse_geral = info["metrics"]["rmse"]
        mape_geral = info["metrics"]["mape"]
        
        resumo_metricas.append({
            "Ticker": ticker,
            "RMSE (R$)": f"R$ {rmse_geral:.4f}",
            "MAPE (%)": f"{mape_geral:.2f}%"
        })
        
    df_metricas = pd.DataFrame(resumo_metricas)
    
    # 2.1. Exibição clássica no terminal
    print("\n" + "="*60)
    print("           TABELA CONSOLIDADA DE MÉTRICAS DO MODELO SVR")
    print("="*60)
    print(df_metricas.to_string(index=False))
    print("="*60 + "\n")
    
    # 2.2. Salva como arquivo de dados CSV
    path_csv = os.path.join(output_dir, "metricas_consolidadas.csv")
    df_metricas.to_csv(path_csv, index=False)
    
    # Define um estilo estético limpo para as plotagens
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # =========================================================================
    # 3. GERAR IMAGEM DA TABELA (PNG)
    # =========================================================================
    # Criamos uma figura compacta para abrigar a tabela de forma elegante
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')  # Esconde os eixos do gráfico para focar apenas na tabela
    
    # Renderiza a tabela do Matplotlib
    tabela_plot = ax.table(
        cellText=df_metricas.values,
        colLabels=df_metricas.columns,
        cellLoc='center',
        loc='center'
    )
    
    # Estilização visual profissional (Header azul escuro, texto branco e linhas suaves)
    tabela_plot.auto_set_font_size(False)
    tabela_plot.set_fontsize(11)
    tabela_plot.scale(1.2, 1.6)  # Dá um espaçamento (padding) interno agradável para as células
    
    # Aplica cores ao cabeçalho (linha 0) e bordas
    for (row, col), cell in tabela_plot.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#1f4e79')  # Azul corporativo clássico para o header
        else:
            cell.set_facecolor('#f2f2f2' if row % 2 == 0 else 'white')  # Efeito zebrado suave
        cell.set_edgecolor('#d3d3d3')
        
    path_tabela_img = os.path.join(output_dir, "tabela_metricas.png")
    plt.savefig(path_tabela_img, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"-> Imagem da Tabela salva com sucesso em: '{path_tabela_img}'")
    
    # =========================================================================
    # 4. GRÁFICO 1: EVOLUÇÃO TEMPORAL DOS ERROS ABSOLUTOS (EM REAIS)
    # =========================================================================
    plt.figure(figsize=(11, 5.5))
    
    for ticker, info in analise_consolidada.items():
        erros_absolutos = info["time_series"]["daily_absolute_errors"]
        dias = np.arange(1, len(erros_absolutos) + 1)
        
        plt.plot(
            dias, 
            erros_absolutos, 
            label=f"Erro {ticker}", 
            linewidth=2,
            alpha=0.85
        )
        
    plt.title("Evolução Temporal do Erro Absoluto Diário (R$)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Dia de Teste", fontsize=11, labelpad=10)
    plt.ylabel("Erro Absoluto (R$)", fontsize=11, labelpad=10)
    plt.legend(frameon=True, facecolor='white', edgecolor='lightgrey', loc='upper right')
    plt.tight_layout()
    
    path_erros_absolutos = os.path.join(output_dir, "grafico_erros_absolutos.png")
    plt.savefig(path_erros_absolutos, dpi=300)
    plt.close()
    print(f"-> Gráfico de Erros Absolutos salva com sucesso em: '{path_erros_absolutos}'")
    
    # =========================================================================
    # 5. GRÁFICO 2: EVOLUÇÃO TEMPORAL DOS ERROS PERCENTUAIS (%)
    # =========================================================================
    plt.figure(figsize=(11, 5.5))
    
    for ticker, info in analise_consolidada.items():
        erros_percentuais = info["time_series"]["daily_percentage_errors"]
        dias = np.arange(1, len(erros_percentuais) + 1)
        
        plt.plot(
            dias, 
            erros_percentuais, 
            label=f"Erro % {ticker}", 
            linewidth=2,
            alpha=0.85
        )
        
    plt.title("Evolução Temporal do Erro Percentual Diário (%)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Dia de Teste", fontsize=11, labelpad=10)
    plt.ylabel("Erro Percentual (%)", fontsize=11, labelpad=10)
    plt.legend(frameon=True, facecolor='white', edgecolor='lightgrey', loc='upper right')
    plt.tight_layout()
    
    path_erros_percentuais = os.path.join(output_dir, "grafico_erros_percentuais.png")
    plt.savefig(path_erros_percentuais, dpi=300)
    plt.close()
    print(f"-> Gráfico de Erros Percentuais salvo com sucesso em: '{path_erros_percentuais}'")