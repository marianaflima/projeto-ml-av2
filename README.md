# Pit Stop Predictor

Modelo de regressão para prever a volta ideal de troca de pneus (pit stop) na Fórmula 1.

## Problema

Definir em qual volta um piloto deve realizar seu pit stop com base em informações disponíveis durante a corrida: número do stint, composto do pneu e temperatura da pista.

**Variável-alvo:** `pitstop_turn` — volta em que o pit stop ocorreu.

## Dataset

- **Fonte:** https://www.kaggle.com/datasets/navenkumar1998/formula-1-dataset-with-weather-and-tyre-features
- **Formato bruto:** 69.230 linhas × 29 colunas (uma linha por volta de cada piloto)
- **Formato modelagem:** Agregado por stint (~9.600 registros de pit stop)

### Download

O arquivo `data/raw/f1_all.parquet` não está versionado no Git. Para obtê-lo, é necessário criar uma conta no Kaggle e fazer download:

[f1_all.parquet](https://www.kaggle.com/datasets/navenkumar1998/formula-1-dataset-with-weather-and-tyre-features)

>Deve-se criar os diretórios `/data/raw` e `/data/processed` e inserir o arquivo no diretório `/data/raw`, para evitar possíveis problemas de execução no pipeline, pois ele precisa da existência de ambos para funcionar.

### Features utilizadas

| Feature | Tipo | Descrição |
|---|---|---|
| `Stint` | Numérico | Número do stint (período entre trocas de pneus) |
| `TrackTemp` | Numérico | Temperatura da pista (°C) |
| `Compound` | Categórico | Composto do pneu (SOFT, MEDIUM, HARD, INTERMEDIATE, WET) |

## Estrutura do Repositório

```
projeto-ml-av2/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── data/
│   ├── raw/          # Dados brutos
│   └── processed/    # Dados tratados
├── notebooks/
├── src/
│   ├── data/         # Carga e limpeza
│   ├── features/     # Split e pré-processamento
│   ├── models/       # Treinamento
│   ├── evaluation/   # Validação cruzada e teste estatístico
│   └── visualization/# Gráficos
├── experiments/      # Modelos serializados e resultados
├── docs/             # Documentação técnica
└── article/          # Artigo científico
```

## Instalação

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Execução

```bash
uv run main.py
```

O pipeline executa 5 etapas: build dataset → build features → treinamento → avaliação → visualização.

## Resultados

### Desempenho nos modelos (conjunto de teste)

| Modelo | RMSE Teste | MAE Teste | R² Teste |
|---|---|---|---|
| Linear Regression | 12.512 | 9.980 | 0.413 |
| Ridge (α=1.0) | 12.514 | 9.978 | 0.412 |
| Lasso (α=0.1) | 12.600 | 10.009 | 0.404 |

### Validação Cruzada (10-fold)

| Modelo | RMSE Médio | Desvio-Padrão |
|---|---|---|
| Linear Regression | 12.326 | 0.601 |
| Ridge | 12.326 | 0.600 |
| Lasso | 12.391 | 0.585 |

### Teste Estatístico (Wilcoxon, α=0.05)

| Comparação | p-value | Conclusão |
|---|---|---|
| Linear vs Ridge | 1.0000 | Equivalentes |
| Linear vs Lasso | 0.0273 | Linear superior |
| Ridge vs Lasso | 0.0195 | Ridge superior |

**Modelo final selecionado:** Regressão Linear (equivalente à Ridge, com menor complexidade).

### Importância das Features (coeficientes)

| Feature | Coeficiente |
|---|---|
| Compound_WET | -20.87 |
| Compound_INTERMEDIATE | -12.08 |
| Compound_SOFT | -10.43 |
| Stint | +9.79 |
| Compound_MEDIUM | -7.65 |
| TrackTemp | -1.84 |

## Limitações

- **R² de ~0,41:** apenas 41% da variabilidade é explicada pelas features atuais
- **Fatores táticos não capturados:** safety car, posição na corrida, undercut, bandeira vermelha
- **Número total de voltas da corrida:** não disponível como feature, mas impacta a janela de parada
- **Generalização:** modelo treinado com dados agregados de múltiplos circuitos, sem distinguir características específicas de cada um

## Tecnologias

Python 3.14, scikit-learn, pandas, matplotlib, seaborn, scipy.
