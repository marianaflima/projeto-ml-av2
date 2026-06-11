from src.data.build_dataset import build_dataset
from src.features.build_features import build_features
from src.models.train_model import train_models
from src.evaluation.evaluate_model import evaluate_models
from src.visualization.visualize import (
    plot_correlation_heatmap,
    plot_histogram_pitstop,
    plot_scatter_tracktemp,
    plot_tyrelife_line,
    plot_boxplot_compound,
    plot_feature_importance,
    plot_residual_diagnostics,
)


def main():
    print("=" * 60)
    print("ETAPA 1/5 — Build Dataset (carregar, limpar, criar target)")
    print("=" * 60)
    build_dataset()

    print("\n" + "=" * 60)
    print("ETAPA 2/5 — Build Features (split + pré-processamento)")
    print("=" * 60)
    build_features()

    print("\n" + "=" * 60)
    print("ETAPA 3/5 — Train Models")
    print("=" * 60)
    train_models()

    print("\n" + "=" * 60)
    print("ETAPA 4/5 — Evaluate Models (CV + Wilcoxon)")
    print("=" * 60)
    evaluate_models()

    print("\n" + "=" * 60)
    print("ETAPA 5/5 — Visualização (gráficos e diagnósticos)")
    print("=" * 60)
    plot_correlation_heatmap()
    plot_histogram_pitstop()
    plot_scatter_tracktemp()
    plot_tyrelife_line()
    plot_boxplot_compound()
    plot_feature_importance()
    plot_residual_diagnostics()

    print("\n" + "=" * 60)
    print("Pipeline completo executado com sucesso!")
    print("=" * 60)


if __name__ == '__main__':
    main()
