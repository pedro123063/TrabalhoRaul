import os
import pandas as pd
import matplotlib.pyplot as plt


def _build_dataframe_from_metric(analise_campeoes: dict, dict_key: str) -> pd.DataFrame:
    rows = []
    for ticker, data in analise_campeoes.items():
        champion_data = data.get(dict_key)
        if not champion_data:
            continue
        
        cfg = champion_data["config"]
        met = champion_data["metrics"]
        
        rows.append({
            "Ticker": cfg["ticker"],
            "Kernel": cfg["kernel"],
            "Split": cfg["split"],
            "C": cfg["c"],
            "Window Ratio": cfg["window_ratio"],
            "Epsilon": cfg["epsilon"],
            "RMSE": met["rmse"],
            "MAPE (%)": met["mape"],
            "Tempo Treino (s)": met["tempo_treino"] if met["tempo_treino"] is not None else 0.0,
            "Tempo Predição (s)": met["tempo_predicao"] if met["tempo_predicao"] is not None else 0.0,
            "Tempo Total (s)": met["tempo_total"]
        })
    
    if not rows:
        return pd.DataFrame()
    
    return pd.DataFrame(rows)


def _render_table_png(df: pd.DataFrame, title: str, output_path: str) -> str:
    df_display = df.copy()
    df_display["RMSE"] = df_display["RMSE"].map("{:.4f}".format)
    df_display["MAPE (%)"] = df_display["MAPE (%)"].map("{:.2f}%".format)
    df_display["Tempo Treino (s)"] = df_display["Tempo Treino (s)"].map("{:.4f}".format)
    df_display["Tempo Predição (s)"] = df_display["Tempo Predição (s)"].map("{:.4f}".format)
    df_display["Tempo Total (s)"] = df_display["Tempo Total (s)"].map("{:.4f}".format)

    fig, ax = plt.subplots(figsize=(14, max(2.5, len(df_display) * 0.7 + 1.2)))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        cellLoc='center',
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1f4e78')
            cell.set_text_props(color='white', weight='bold')
        else:
            if r % 2 == 0:
                cell.set_facecolor('#f2f2f2')
            else:
                cell.set_facecolor('#ffffff')

    plt.title(title, fontsize=14, weight='bold', pad=15)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return output_path


def generate_rmse_table(analise_campeoes: dict, output_dir: str = ".") -> str:
    os.makedirs(output_dir, exist_ok=True)
    df = _build_dataframe_from_metric(analise_campeoes, "campeao_rmse")
    if df.empty:
        return ""
    
    df = df.sort_values(by="RMSE", ascending=True)
    output_path = os.path.join(output_dir, "tabela_campeoes_rmse.png")
    return _render_table_png(df, "Modelos Campeões por RMSE", output_path)


def generate_mape_table(analise_campeoes: dict, output_dir: str = ".") -> str:
    os.makedirs(output_dir, exist_ok=True)
    df = _build_dataframe_from_metric(analise_campeoes, "campeao_mape")
    if df.empty:
        return ""
    
    df = df.sort_values(by="MAPE (%)", ascending=True)
    output_path = os.path.join(output_dir, "tabela_campeoes_mape.png")
    return _render_table_png(df, "Modelos Campeões por MAPE", output_path)


def generate_tempo_table(analise_campeoes: dict, output_dir: str = ".") -> str:
    os.makedirs(output_dir, exist_ok=True)
    df = _build_dataframe_from_metric(analise_campeoes, "campeao_tempo")
    if df.empty:
        return ""
    
    df = df.sort_values(by="Tempo Total (s)", ascending=True)
    output_path = os.path.join(output_dir, "tabela_campeoes_tempo.png")
    return _render_table_png(df, "Modelos Campeões por Tempo Total", output_path)


def run_graphics_pipeline(analise_campeoes: dict, output_dir: str = ".") -> list:
    files = [
        generate_rmse_table(analise_campeoes, output_dir),
        generate_mape_table(analise_campeoes, output_dir),
        generate_tempo_table(analise_campeoes, output_dir)
    ]
    return [f for f in files if f]