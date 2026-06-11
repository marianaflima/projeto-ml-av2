import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib

sns.set_theme(style="whitegrid")

OUTPUT_DIR = 'src/visualization/'

def plot_correlation_heatmap():
    df_treatment = pd.read_parquet('data/processed/df_treatment.parquet')
    numerical_df = df_treatment.select_dtypes(include=['number', 'bool']).drop(
        columns=['raceId', 'driverId', 'constructorId', 'circuitId']
    )
    numerical_df['FreshTyre'] = numerical_df['FreshTyre'].astype(int)

    plt.figure(figsize=(15, 12))
    sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap of Numerical Variables')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}heatmap_correlacao.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo: heatmap_correlacao.png")


def plot_histogram_pitstop():
    df_stops = pd.read_parquet('data/processed/df_stops.parquet')
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df_stops, x='pitstop_turn',
        kde=True, bins=30, color='royalblue', edgecolor='black'
    )
    plt.title('Distribuição Física das Paradas nos Boxes (pitstop_turn)', fontsize=14, fontweight='bold')
    plt.xlabel('Volta da Parada (Pit Stop)', fontsize=12)
    plt.ylabel('Frequência (Contagem)', fontsize=12)
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='Outliers: Paradas na Volta 1')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}histograma_pitstop.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo: histograma_pitstop.png")


def plot_scatter_tracktemp():
    df_stops = pd.read_parquet('data/processed/df_stops.parquet')
    plt.figure(figsize=(10, 6))
    sns.regplot(
        x='TrackTemp', y='pitstop_turn',
        data=df_stops,
        scatter_kws={'alpha': 0.3},
        line_kws={'color': 'red'}
    )
    plt.title('Gráfico de Dispersão: Temperatura da Pista vs. Volta do Pit Stop')
    plt.xlabel('Temperatura da Pista (°C)')
    plt.ylabel('Volta do Pit Stop (pitstop_turn)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}dispersao_tracktemp_pitstop.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo: dispersao_tracktemp_pitstop.png")


def plot_tyrelife_line():
    df_treatment = pd.read_parquet('data/processed/df_treatment.parquet')
    media_pneu = (
        df_treatment.groupby('TyreLife')['milliseconds']
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=media_pneu, x='TyreLife', y='milliseconds')
    plt.title('Tempo Médio de Volta por Idade do Pneu')
    plt.xlabel('Idade do Pneu (voltas)')
    plt.ylabel('Tempo Médio de Volta (ms)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}linha_tyrelife_milliseconds.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo: linha_tyrelife_milliseconds.png")


def plot_boxplot_compound():
    df_stops = pd.read_parquet('data/processed/df_stops.parquet')
    df_stops_plot = df_stops[df_stops['Compound'] != 'UNKNOWN'].copy()
    ordem_compostos = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
    palette = {
        'SOFT': '#e8002d', 'MEDIUM': '#ffd600', 'HARD': '#c8c8c8',
        'INTERMEDIATE': '#39b54a', 'WET': '#0067ff'
    }

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x='Compound', y='pitstop_turn',
        data=df_stops_plot,
        order=ordem_compostos, palette=palette,
        hue='Compound', legend=False
    )
    plt.title('Distribuição da Volta do Pit Stop por Composto do Pneu')
    plt.xlabel('Composto do Pneu')
    plt.ylabel('Volta do Pit Stop (pitstop_turn)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}boxplot_compound_pitstop.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo: boxplot_compound_pitstop.png")


def plot_feature_importance():
    pipeline_final = joblib.load('experiments/linear_regression_model.joblib')

    feature_names = pipeline_final.named_steps['preprocessor'].get_feature_names_out()
    coeficientes = pipeline_final.named_steps['model'].coef_

    coef_df = (
        pd.DataFrame({'feature': feature_names, 'coeficiente': coeficientes})
        .reindex(pd.Series(coeficientes).abs().sort_values(ascending=False).index)
        .reset_index(drop=True)
    )

    plt.figure(figsize=(10, 5))
    cores = ['#e8002d' if c > 0 else '#0067ff' for c in coef_df['coeficiente']]
    plt.barh(coef_df['feature'], coef_df['coeficiente'], color=cores)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title('Importância das Features — Coeficientes da Regressão Linear')
    plt.xlabel('Coeficiente')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}importancia_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico salvo: importancia_features.png")


def plot_residual_diagnostics():
    pipeline_final = joblib.load('experiments/linear_regression_model.joblib')
    split_data = joblib.load('experiments/split_data.joblib')
    X_test, y_test = split_data['X_test'], split_data['y_test']

    y_pred_final = pipeline_final.predict(X_test)
    residuos = y_test - y_pred_final

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred_final, y_test, alpha=0.3, color='steelblue')
    axes[0].plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()], 'r--', linewidth=1
    )
    axes[0].set_xlabel('Valor Predito')
    axes[0].set_ylabel('Valor Real')
    axes[0].set_title('Predito vs Real')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    sns.histplot(residuos, kde=True, color='steelblue', ax=axes[1])
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Resíduo (Real − Predito)')
    axes[1].set_ylabel('Frequência')
    axes[1].set_title('Distribuição dos Resíduos')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}diagnostico_residuos.png', dpi=300, bbox_inches='tight')
    plt.close()

    with open('experiments/residuos_summary.txt', 'w') as f:
        f.write(f"Resíduo médio:    {residuos.mean():.3f} voltas\n")
        f.write(f"Desvio-padrão:    {residuos.std():.3f} voltas\n")
        f.write(f"Resíduo máximo:   {residuos.max():.1f} voltas\n")
        f.write(f"Resíduo mínimo:   {residuos.min():.1f} voltas\n")

    print("Gráfico salvo: diagnostico_residuos.png")
    print("Summary salvo: residuos_summary.txt")


if __name__ == '__main__':
    plot_correlation_heatmap()
    plot_histogram_pitstop()
    plot_scatter_tracktemp()
    plot_tyrelife_line()
    plot_boxplot_compound()
    plot_feature_importance()
    plot_residual_diagnostics()
    print("\nTodos os gráficos foram salvos em src/visualization/")
