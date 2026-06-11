# Decisões Técnicas

## Objetivo do Projeto

Fazer um modelo preditivo que determine a melhor volta para se realizar uma troca de pneus (pit stop) na Fórmula 1, utilizando dados históricos de voltas, condições climáticas e características dos pneus.

## Perguntas de Pesquisa

1. O número do stint (Stint) influencia a volta em que o pit stop ocorre?
2. O composto do pneu (Compound) resulta em voltas de parada significativamente diferentes?
3. A temperatura da pista (TrackTemp) antecipa ou posterga o momento do pit stop?

## Hipóteses

1. Stints mais avançados (maior Stint) estão associados a pit stops em voltas mais tardias, refletindo a progressão natural da corrida.
2. Compostos macios (SOFT) antecipam a parada; duros (HARD) permitem stints mais longos.
3. Temperaturas de pista mais altas aceleram a degradação dos pneus, antecipando a volta de parada.

## Dimensões do Dataset

O dataset bruto (`f1_all.parquet`) possui **69.230 linhas × 29 colunas**. Cada linha representa uma volta de um piloto em uma corrida. Para a modelagem, o dataset foi agregado por stint — um registro por pit stop registrado.

## Tratamento de Dados

### Substituição de 'None' no Compound

O valor `'None'` na coluna `Compound` foi substituído por `'UNKNOWN'`, pois `'None'` é a representação Python para valor nulo e, neste contexto, equivalente a composto desconhecido. A verificação mostrou que, em certos casos (ex: Tsunoda no stint 3), os dados de composto simplesmente não foram registrados, tornando `'UNKNOWN'` a classificação mais adequada.

### Remoção de Colunas

As seguintes colunas foram removidas por não agregarem valor preditivo ou causarem vazamento de informação:

| Coluna | Motivo |
|---|---|
| `position_x`, `position_y` | Não influenciam a estratégia de pit stop |
| `time` | Representação textual do tempo; `milliseconds` já cobre essa informação |
| `grid`, `name` | Não mudam durante a corrida; independentes da dinâmica da prova |
| `LapNumber` | Cópia da coluna `lap` |
| `Driver` | Cópia da coluna `code` |
| `statusId`, `status` | Só são conhecidos após o fim da corrida (vazamento de informação) |

### Tratamento de Nulos em TyreLife

A coluna `TyreLife` possuía 553 valores nulos. Após análise exploratória, constatou-se que esses nulos ocorriam majoritariamente nos primeiros registros de um novo stint quando `FreshTyre=False`, indicando falha no registro. Optou-se por **remover essas linhas** (dropna) por serem uma fração pequena do dataset total (~0,8%).

### Criação da Variável Alvo (pitstop_turn)

A variável `pitstop_turn` foi criada através dos seguintes passos:

1. **Agrupamento**: Os dados foram agrupados por `raceId`, `driverId` e `Stint` (período entre trocas de pneus).
2. **Identificação da última volta**: A última volta (`lap`) de cada stint foi extraída como candidata a pit stop.
3. **Filtragem do último stint**: O último stint de cada piloto em cada corrida foi removido, pois representa o fim da corrida, não uma parada real.
4. **Enriquecimento**: Foram adicionadas a temperatura média da pista (`TrackTemp`) e o composto (`Compound`) de cada stint.

## Análise Exploratória de Dados

### Correlação entre Variáveis

Observou-se uma correlação negativa fraca (aproximadamente -0,23) entre `TyreLife` e `milliseconds`. Embora pareça contraintuitivo (pneus mais velhos → voltas mais rápidas), isso ocorre porque pneus velhos aparecem mais em corridas secas com compostos duros, enquanto pneus novos aparecem em relargadas após safety car com voltas atipicamente lentas. A correlação captura esse padrão agregado, não a curva real de degradação de um pneu específico. Correlações lineares podem não capturar a totalidade de uma relação não linear complexa.

### Distribuição das Paradas

A distribuição das voltas de pit stop mostrou maior frequência em determinadas faixas, com outliers na volta 1 (indicando incidentes ou toques).

### Temperatura da Pista vs Pit Stop

O gráfico de dispersão indicou que pistas mais quentes tendem a antecipar a parada (degradação acelerada), mas a dispersão considerável mostra que a temperatura é apenas um dos fatores — o composto e o circuito também exercem influência significativa.

### Idade do Pneu vs Tempo de Volta

Observou-se que conforme a idade do pneu aumenta, o tempo médio de volta tende a diminuir em um primeiro momento (atingindo ponto ideal de temperatura/aderência) e depois se estabiliza. Após certo número de voltas, a degradação do pneu provavelmente fará com que os tempos voltem a aumentar.

### Composto vs Volta do Pit Stop

O boxplot confirmou a hipótese central: compostos mais duros (HARD) apresentam stints mais longos, SOFT resulta em paradas mais precoces e com maior variabilidade, e INTERMEDIATE/WET dependem das condições climáticas, sendo outliers estratégicos naturais.

## Preparação de Dados

### Divisão Treino-Teste

Os dados foram divididos em treino (80%) e teste (20%) utilizando `train_test_split` com `random_state=42`. A separação foi realizada antes de qualquer pré-processamento para evitar vazamento de dados.

### Features Utilizadas

- **`Stint`** (numérica): Número do stint do piloto.
- **`TrackTemp`** (numérica): Temperatura média da pista no stint.
- **`Compound`** (categórica): Tipo de composto do pneu (SOFT, MEDIUM, HARD, INTERMEDIATE, WET, UNKNOWN).

Foram excluídos `raceId` e `driverId` das features para que o modelo generalize para corridas e pilotos não vistos.

### Pipeline de Pré-processamento

- **Imputação**: `SimpleImputer(strategy='median')` para numéricas e `SimpleImputer(strategy='most_frequent')` para categóricas (mantida para robustez futura, embora não houvesse nulos).
- **Codificação**: `OneHotEncoder(drop='first')` para `Compound`, evitando a dummy variable trap.
- **Padronização**: `StandardScaler` para variáveis numéricas, fundamental para modelos lineares regularizados.

## Modelagem

Três modelos de regressão foram treinados e comparados:

1. **Regressão Linear**: Modelo baseline, sem regularização.
2. **Regressão Ridge** (alpha=1.0): Regularização L2 para reduzir multicolinearidade.
3. **Regressão Lasso** (alpha=0.1): Regularização L1 com seleção automática de variáveis.

### Resultados

| Modelo | RMSE Treino | RMSE Teste | MAE Teste | R² Teste |
|---|---|---|---|---|
| Linear Regression | 12,286 | 12,512 | 9,980 | 0,413 |
| Ridge | 12,286 | 12,514 | 9,978 | 0,412 |
| Lasso | 12,349 | 12,600 | 10,009 | 0,404 |

### Interpretação

Os três modelos apresentaram desempenhos semelhantes. A Regressão Linear obteve o menor RMSE no teste (12,512) e o maior R² (0,413), indicando que ~41% da variabilidade da volta de parada é explicada pelas variáveis utilizadas. Fatores adicionais não presentes no dataset provavelmente influenciam a estratégia, como posição na corrida, safety car, número total de voltas e diferenciação entre stint 1 e stint 2.

### Validação Cruzada (10-Fold)

| Modelo | RMSE Médio | Desvio-Padrão |
|---|---|---|
| Linear Regression | 12,326 | 0,601 |
| Ridge | 12,326 | 0,600 |
| Lasso | 12,391 | 0,585 |

### Teste Estatístico de Wilcoxon

| Comparação | p-value | Significância (α=0,05) |
|---|---|---|
| Linear vs Ridge | 1,0000 | Não significativa |
| Linear vs Lasso | 0,0273 | Significativa |
| Ridge vs Lasso | 0,0195 | Significativa |

Regressão Linear e Ridge são estatisticamente equivalentes (p=1,0); ambas são superiores ao Lasso.

### Modelo Final

A **Regressão Linear** foi selecionada como modelo final por:
- Apresentar o menor RMSE no conjunto de teste (12,512).
- Ser estatisticamente equivalente à Ridge (p=1,0) e superior ao Lasso (p=0,027).
- Ter menor complexidade computacional e melhor interpretabilidade.

### Importância das Features (Coeficientes)

- `Compound_WET` (-20,87): Maior impacto negativo — pneus de chuva extrema associados a paradas muito mais precoces (~20,9 voltas antes que HARD).
- `Compound_INTERMEDIATE` (-12,08) e `Compound_SOFT` (-10,43): Antecipam a parada em relação ao HARD.
- `Stint` (+9,79): A cada incremento no stint, a parada ocorre ~9,8 voltas mais tarde.
- `Compound_MEDIUM` (-7,65): Antecipa a parada em menor grau que SOFT.
- `TrackTemp` (-1,84): A cada °C adicional, a parada é antecipada em ~1,84 voltas.

### Análise dos Resíduos

- **Resíduo médio: 0,424 voltas** — próximo de zero, indicando ausência de viés sistemático.
- **Desvio-padrão: 12,518 voltas** — grande variabilidade nos erros de previsão.
- **Resíduo máximo: 39,4 voltas** (subestimação) e **mínimo: -35,4 voltas** (superestimação).

A alta variabilidade reforça que fatores táticos não capturados pelas features atuais desempenham papel crucial na determinação do momento do pit stop.
