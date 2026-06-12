# Previsão da Volta Ideal de Pit Stop na Fórmula 1 com Modelos de Regressão Linear

**Resumo** - A estratégia de pit stop é um dos fatores mais determinantes no resultado de uma corrida de Fórmula 1. Este trabalho propõe modelos de aprendizado de máquina baseados em regressão linear para prever a volta ideal de troca de pneus, utilizando dados históricos de telemetria disponibilizados publicamente no Kaggle. O conjunto de dados original contém 69.230 registros de voltas; após agregação por stint, foram obtidos aproximadamente 9.600 registros de parada. Três modelos foram treinados e comparados: Regressão Linear (baseline), Ridge (regularização L2) e Lasso (regularização L1). A Regressão Linear obteve RMSE de 12,51 voltas e R² de 0,413 no conjunto de teste, sendo selecionada como modelo final por ser estatisticamente equivalente à Ridge (p=1,00, teste de Wilcoxon) e superior ao Lasso (p=0,027). O composto do pneu e o número do stint foram as variáveis de maior influência. Os resultados evidenciam tanto a viabilidade de modelos lineares para o problema quanto as limitações impostas pela ausência de variáveis táticas, como safety car e posição em pista.

**Palavras-chave:** aprendizado de máquina, regressão linear, Fórmula 1, pit stop, estratégia de corrida.

## 1. Introdução

A Fórmula 1 é amplamente reconhecida como um dos esportes mais intensivos em dados do mundo. Cada carro moderno gera volumes expressivos de telemetria por corrida, abrangendo sensores de temperatura, desgaste de pneus, consumo de combustível, comportamento aerodinâmico e condições climáticas. Nesse contexto, decisões estratégicas tomadas em frações de segundo podem definir vitórias ou eliminações, e o pit stop é o momento central dessa equação.

O pit stop é a interrupção planejada durante a corrida para troca de pneus e eventuais ajustes no carro. A escolha da volta correta equilibra o desgaste do composto atual, o ganho de tempo em pista limpa com pneus novos e o risco de perder posição para adversários que ainda não pararam. Como aponta Maciel (2019), ao tratar da modelagem de resultados esportivos com regressão linear, o uso de técnicas estatísticas adequadas torna-se uma ferramenta valiosa para o planejamento de ações que possibilitem a obtenção de melhores resultados. Esse princípio aplica-se diretamente à Fórmula 1, onde a análise de dados históricos pode revelar padrões de comportamento que orientam decisões estratégicas.

A intersecção entre aprendizado de máquina e esportes motorizados tem crescido de forma consistente na última década. Heilmeier et al. (2020) propuseram um Engenheiro de Estratégia Virtual que combina uma rede neural de alimentação direta e uma rede LSTM para tomar decisões automatizadas de estratégia de corrida, com foco na otimização do momento do pit stop e na seleção do composto correto. O modelo foi simulado com dados do Grande Prêmio da China de 2019 e obteve uma posição média de chegada de 9,51 ao longo de 1.000 simulações, demonstrando a viabilidade da abordagem orientada a dados. Mais recentemente, Sasikumar, Leema e Balakrishnan (2025) apresentaram um framework baseado em arquiteturas profundas como Bi-LSTM, TCN-GRU e CNN-BiLSTM para prever janelas ideais de pit stop com dados de 2020 a 2024, demonstrando que a abordagem orientada a dados supera métodos baseados em julgamento humano em condições de corrida que mudam rapidamente.

No domínio dos fundamentos teóricos, Faceli et al. (2021) discutem em profundidade os modelos preditivos e descritivos no contexto do aprendizado de máquina, abordando regressão, sobreajuste e técnicas de validação. Complementando essa base, Passos (2022) analisa especificamente o comportamento dos métodos de regularização Ridge e Lasso, demonstrando que a regressão Ridge reduz os coeficientes sem anulá-los, enquanto o Lasso pode zerá-los e realizar seleção automática de variáveis. Para a validação estatística dos modelos, Demsar (2006) recomenda o teste de Wilcoxon para amostras pareadas como o método não paramétrico mais adequado para comparar algoritmos de aprendizado sem a premissa de normalidade dos erros.

Apesar da relevância prática do problema, a literatura acadêmica sobre previsão de pit stop com modelos lineares e dados históricos abertos ainda é escassa. Este trabalho tem como objetivo construir e avaliar modelos de regressão linear, Ridge e Lasso para prever a variável alvo denominada pitstop_turn, isto é, a volta em que o pit stop ocorreu.

### 1.1 Perguntas de Pesquisa

1. O número do stint influencia a volta em que o pit stop ocorre?
2. O composto do pneu resulta em voltas de parada significativamente diferentes?
3. A temperatura da pista antecipa ou posterga o momento do pit stop?

### 1.2 Hipóteses

1. Stints mais avançados estão associados a pit stops em voltas mais tardias, refletindo a progressão natural da corrida.
2. Compostos macios (SOFT) antecipam a parada; compostos duros (HARD) permitem stints mais longos.
3. Temperaturas de pista mais altas aceleram a degradação dos pneus, antecipando a volta de parada.

## 2. Metodologia

### 2.1 Conjunto de Dados

O conjunto de dados utilizado é o "Formula 1 Dataset with Weather and Tyre Features", disponibilizado publicamente por Naven Kumar no repositório Kaggle (Kumar, 2024). O arquivo principal, no formato Parquet, contém 69.230 registros em que cada linha representa uma volta de um piloto em uma corrida, com 29 colunas cobrindo informações de corrida, piloto, pneus e clima.

Para a modelagem, os dados foram agregados por stint, resultando em um conjunto com um registro por pit stop registrado, com aproximadamente 9.600 observações. A variável alvo foi construída como a última volta de cada stint, excluindo o stint final de cada piloto em cada corrida, pois este representa o fim da prova e não uma parada planejada.

### 2.2 Pré-processamento

**Remoção de colunas:** Colunas que não contribuem para a previsão ou que introduzem vazamento de informação foram eliminadas. As colunas de status são conhecidas apenas após o fim da corrida; as de posição cartesiana não influenciam a estratégia; o tempo de volta textual é redundante com os milissegundos; e a posição no grid e o nome do circuito são constantes durante a corrida. Colunas duplicatas de identificadores de volta e piloto também foram removidas.

**Tratamento de valores ausentes:** A coluna TyreLife possuía 553 valores nulos, representando cerca de 0,8% do total, ocorrendo majoritariamente no início de novos stints com pneus não frescos, indicando falha de registro. Essas linhas foram removidas. A string "None" na coluna Compound foi substituída pelo rótulo "UNKNOWN", já existente no dataset, usada para representar um composto não identificado ou detectado. 

**Pipeline de pré-processamento:** O pipeline foi estruturado em dois ramos paralelos aplicados via transformador de colunas. Para as variáveis numéricas Stint e TrackTemp, aplicou-se imputação por mediana seguida de padronização. Para a variável categórica Compound, aplicou-se imputação pela moda seguida de codificação com exclusão da primeira categoria, evitando a armadilha de variável dummy descrita por Faceli et al. (2021). As etapas de imputação foram mantidas no pipeline para garantir robustez e reprodutibilidade caso novos dados sejam incorporados futuramente. Os identificadores de corrida e piloto foram explicitamente excluídos das features para que o modelo generalize para corridas e pilotos não vistos durante o treinamento. A divisão treino-teste foi realizada antes de qualquer pré-processamento para evitar vazamento de dados, com 80% para treino e 20% para teste.

### 2.3 Features Utilizadas

Foram selecionadas três features para o modelo: Stint, que representa o número do stint do piloto na corrida; TrackTemp, que representa a temperatura média da pista durante o stint em graus Celsius; e Compound, que representa o tipo de composto do pneu, podendo assumir os valores SOFT, MEDIUM, HARD, INTERMEDIATE, WET e UNKNOWN. As demais variáveis climáticas disponíveis foram excluídas por não apresentarem ganho relevante de desempenho durante a fase de experimentação.

### 2.4 Modelos e Avaliação

Três modelos de regressão foram treinados com a biblioteca scikit-learn (Pedregosa et al., 2011), cada um encapsulado em um pipeline junto às etapas de pré-processamento: Regressão Linear sem regularização como baseline, Ridge com parâmetro alpha igual a 1,0 aplicando regularização L2, e Lasso com alpha igual a 0,1 aplicando regularização L1 com seleção automática de variáveis, conforme descrito por Passos (2022).

Os modelos foram avaliados com RMSE (raiz do erro quadrático médio), MAE (erro absoluto médio) e R² no conjunto de teste. A estabilidade foi verificada por validação cruzada com 10 folds, calculando-se a média e o desvio-padrão do RMSE obtido em cada partição. Para a comparação estatística, aplicou-se o teste de Wilcoxon para amostras pareadas (Demsar, 2006) com nível de significância alpha igual a 0,05, comparando diretamente os vetores de RMSE por fold de cada par de modelos.

## 3. Resultados

### 3.1 Análise Exploratória

A análise exploratória revelou padrões relevantes para a modelagem. A distribuição da variável alvo pitstop_turn, apresentada na Figura 1, mostra assimetria à direita com maior concentração entre as voltas 15 e 40, e outliers na volta 1, indicativos de incidentes no início da prova.

![Figura 1](/src/visualization/histograma_pitstop.png)
**Figura 1** - Distribuição da frequência de pit stops por volta da corrida. A linha vermelha tracejada marca os outliers da volta 1.

O mapa de calor das correlações entre variáveis numéricas, apresentado na Figura 2, revela que Stint e TyreLife são as variáveis com maior correlação positiva com o número da volta (0,63 e 0,51, respectivamente), enquanto as variáveis climáticas entre si apresentam correlações moderadas, como AirTemp e TrackTemp (0,64).

![Figura 2](/src/visualization/heatmap_correlacao.png)
**Figura 2** - Mapa de calor das correlações de Pearson entre as variáveis numéricas do conjunto de dados. Valores próximos de 1 (vermelho) indicam correlação positiva forte; próximos de -1 (azul), correlação negativa forte.

A correlação entre TyreLife e milliseconds foi de aproximadamente -0,23. Embora contraintuitiva, esse resultado reflete um efeito de confundimento: pneus velhos aparecem majoritariamente em corridas secas com compostos duros, enquanto pneus novos surgem após relargadas em safety car com tempos de volta atipicamente lentos. A correlação captura esse padrão agregado, não a curva real de degradação de um pneu específico. Conforme apontado por Faceli et al. (2021), correlações lineares podem não capturar a totalidade de relações não lineares complexas. Essa dinâmica é visível na Figura 3, que mostra o tempo médio de volta por idade do pneu: o tempo cai acentuadamente nos primeiros stints, quando o pneu aquece e atinge aderência ideal, estabiliza-se na faixa intermediária e apresenta picos acentuados em idades avançadas, que correspondem a situações atípicas de poucos registros.

![Figura 3](/src/visualization/linha_tyrelife_milliseconds.png)
**Figura 3** - Evolução do tempo médio de volta (em milissegundos) conforme o pneu envelhece. Os picos nas voltas 47 e 53 refletem situações atípicas com poucos registros.

O gráfico de dispersão da Figura 4 confirma a tendência de paradas mais precoces em pistas mais quentes, coerente com a Hipótese 3 de degradação acelerada pelo calor. A linha de regressão vermelha evidencia a inclinação negativa, mas a dispersão considerável dos pontos sinaliza que a temperatura é apenas um dos fatores em jogo.

![Figura 4](/src/visualization/dispersao_tracktemp_pitstop.png)
**Figura 4** - Gráfico de dispersão entre temperatura da pista (eixo x) e volta em que o pit stop ocorreu (eixo y). A linha vermelha representa a tendência linear ajustada.

O boxplot da Figura 5 confirma a Hipótese 2: compostos HARD resultam em stints mais longos, com mediana próxima de 37 voltas, SOFT apresenta paradas mais precoces, com mediana em torno de 17 voltas e maior variabilidade, e os compostos INTERMEDIATE e WET são outliers estratégicos dependentes de condições climáticas, não do desgaste convencional.

![Figura 5](/src/visualization/boxplot_compound_pitstop.png)
**Figura 5** - Boxplot da volta do pit stop segmentado por composto do pneu. O composto UNKNOWN foi excluído para maior clareza visual.

### 3.2 Desempenho dos Modelos

A Tabela 1 apresenta os resultados dos três modelos no conjunto de teste.

**Tabela 1 - Desempenho no conjunto de teste**

| Modelo            | RMSE Treino | RMSE Teste | MAE Teste | R² Teste |
| ----------------- | ----------- | ---------- | --------- | -------- |
| Regressão Linear  | 12,286      | 12,512     | 9,980     | 0,413    |
| Ridge (alpha=1,0) | 12,286      | 12,514     | 9,978     | 0,412    |
| Lasso (alpha=0,1) | 12,349      | 12,600     | 10,009    | 0,404    |

Os três modelos apresentaram desempenhos próximos. A Regressão Linear obteve o menor RMSE no teste, de 12,512 voltas, e o maior R², de 0,413, indicando que aproximadamente 41% da variabilidade da volta de parada é explicada pelas três features utilizadas. Em relação ao MAE, todos os modelos apresentaram erro médio próximo de 10 voltas, indicando que, em média, a previsão difere aproximadamente dez voltas do valor real. A diferença entre RMSE de treino e de teste é pequena para todos os modelos, sugerindo ausência de sobreajuste relevante.

### 3.3 Validação Cruzada

A Tabela 2 apresenta os resultados da validação cruzada com 10 folds.

**Tabela 2 - Validação cruzada (10 folds)**

| Modelo           | RMSE Médio | Desvio-Padrão |
| ---------------- | ---------- | ------------- |
| Regressão Linear | 12,326     | 0,601         |
| Ridge            | 12,326     | 0,600         |
| Lasso            | 12,391     | 0,585         |

Os resultados são consistentes com os do conjunto de teste, confirmando a estabilidade dos modelos. A diferença entre Regressão Linear e Ridge no RMSE médio foi inferior a 0,001 volta. O baixo desvio-padrão, inferior a 0,65 para todos, indica boa capacidade de generalização entre os folds.

### 3.4 Comparação Estatística

A Tabela 3 apresenta os resultados do teste de Wilcoxon entre os pares de modelos, aplicado sobre os vetores de RMSE por fold da validação cruzada, seguindo a metodologia recomendada por Demsar (2006).

**Tabela 3 - Teste de Wilcoxon (alpha=0,05)**

| Comparação      | p-valor | Conclusão         |
| --------------- | ------- | ----------------- |
| Linear vs Ridge | 1,0000  | Não significativa |
| Linear vs Lasso | 0,0273  | Linear superior   |
| Ridge vs Lasso  | 0,0195  | Ridge superior    |

Regressão Linear e Ridge são estatisticamente equivalentes, enquanto ambas superam o Lasso de forma significativa.

### 3.5 Importância das Features

A Figura 6 apresenta os coeficientes do modelo final após padronização das variáveis numéricas, e a Tabela 4 lista os valores correspondentes.

![Figura 6](/src/visualization/importancia_features.png)
**Figura 6** - Coeficientes padronizados do modelo de Regressão Linear. Barras azuis indicam efeito negativo sobre a volta de parada (parada antecipada); a barra vermelha indica efeito positivo (parada postergada).

**Tabela 4 - Coeficientes do modelo final (Regressão Linear)**

|Feature|Coeficiente|
|---|---|
|Compound_WET|-20,87|
|Compound_INTERMEDIATE|-12,08|
|Compound_SOFT|-10,43|
|Stint|+9,79|
|Compound_MEDIUM|-7,65|
|TrackTemp|-1,84|

Os coeficientes do modelo confirmam as três hipóteses de pesquisa. O composto WET apresenta o maior impacto: pneus de chuva extrema estão associados a paradas aproximadamente 20,9 voltas mais cedo que o composto de referência HARD, coerente com a estratégia de corrida em que pneus WET são trocados rapidamente assim que a pista começa a secar, corroborando a Hipótese 2. Os compostos INTERMEDIATE (-12,08) e SOFT (-10,43) também antecipam a parada em relação ao HARD, refletindo sua menor durabilidade. O número do stint tem efeito positivo relevante: a cada incremento, a parada tende a ocorrer cerca de 9,8 voltas mais tarde, confirmando a Hipótese 1 de que a progressão natural da corrida é capturada pelo modelo. A temperatura da pista contribui negativamente, com cada grau Celsius adicional antecipando a parada em aproximadamente 1,84 volta, validando a Hipótese 3.

### 3.6 Análise dos Resíduos

A análise dos resíduos, apresentada na Figura 7, revelou resíduo médio de 0,424 voltas, próximo de zero e indicativo de ausência de viés sistemático. O desvio-padrão foi de 12,518 voltas, o resíduo máximo chegou a +39,4 voltas e o mínimo a -35,4 voltas.

![Figura 7](/src/visualization/diagnostico_residuos.png)
**Figura 7** - À esquerda, dispersão entre valores preditos e reais; a linha tracejada representa a previsão perfeita. À direita, distribuição dos resíduos com a curva de densidade sobreposta; a linha vermelha tracejada marca o resíduo zero.

O gráfico predito versus real evidencia que o modelo captura a tendência geral, com os pontos se agrupando em torno da linha diagonal ideal, mas com dispersão notável para valores mais elevados de pitstop_turn. A distribuição dos resíduos apresenta forma aproximadamente gaussiana centrada em zero, o que é consistente com as premissas do modelo linear, embora a cauda direita mais pesada revele situações em que o modelo subestima consideravelmente a volta de parada. A alta variabilidade dos resíduos indica que fatores não capturados pelas features atuais exercem influência considerável sobre a volta de parada.


## 4. Discussão

Os resultados demonstram que modelos lineares simples conseguem capturar aproximadamente 41% da variabilidade da volta de pit stop a partir de apenas três features. Esse valor, embora modesto em termos absolutos, é expressivo considerando a natureza altamente tática e contextual das decisões de pit stop. O R² encontrado é coerente com o observado por Maciel (2019) ao modelar resultados esportivos com regressão linear múltipla, onde a explicação parcial da variância pela abordagem linear é esperada diante da complexidade inerente às competições.

As perguntas de pesquisa formuladas foram respondidas afirmativamente. A influência do número do stint (Pergunta 1) foi confirmada pelo coeficiente positivo de +9,79, indicando que a progressão natural da corrida é capturada pelo modelo. A diferença entre compostos (Pergunta 2) foi a dimensão de maior impacto, com variação de até 20,9 voltas entre os extremos WET e HARD. O efeito da temperatura da pista (Pergunta 3), embora de menor magnitude, mostrou-se estatisticamente consistente com a hipótese de degradação acelerada pelo calor.

A equivalência estatística entre Regressão Linear e Ridge, com p igual a 1,00, sugere que a multicolinearidade entre as preditoras não é um problema severo no subconjunto de features utilizado. Seguindo o princípio da parcimônia discutido por Faceli et al. (2021), entre modelos equivalentes prefere-se o de menor complexidade e maior interpretabilidade, o que justifica a seleção da Regressão Linear como modelo final. O desempenho inferior do Lasso pode ser atribuído à eliminação de coeficientes que, neste caso, possuem relevância preditiva, confirmando a observação de Passos (2022) de que o Lasso é mais vantajoso em cenários de alta dimensionalidade com variáveis irrelevantes.

A principal limitação identificada é a ausência de variáveis táticas fundamentais. Heilmeier et al. (2020) demonstram que o desempenho de modelos de estratégia melhora substancialmente com a inclusão de features que representam a dinâmica de posicionamento entre os pilotos, e Sasikumar, Leema e Balakrishnan (2025) reforçam que a incorporação de features temporais e contextuais é necessária para capturar plenamente a complexidade estratégica do problema. Fatores como posição na corrida e gap para o carro à frente ou atrás, safety car e bandeira vermelha, número total de voltas da corrida e a distinção entre primeiro e segundo stint não estão presentes no conjunto de dados utilizado, e sua ausência contribui diretamente para a variabilidade residual observada.

No que diz respeito às ameaças à validade, identificam-se quatro riscos principais. O primeiro é o viés de seleção na construção da variável alvo: o modelo aprende quando as paradas ocorreram, não quando deveriam ter ocorrido sob a ótica da otimização estratégica, pois paradas reativas e planejadas não são distinguidas. O segundo é a homogeneização de circuitos: o conjunto de dados agrega múltiplos circuitos e temporadas sem distinção das características de cada pista, como número total de voltas ou perfil de degradação de pneus, o que pode distorcer os coeficientes estimados. O terceiro é o risco de sobreajuste temporal: embora a diferença entre RMSE de treino e teste seja pequena, a generalização para temporadas com mudanças regulatórias radicais, como a temporada de 2022, não é garantida. O quarto é a validade externa: a capacidade de prever a volta de pit stop em tempo real durante uma corrida futura não foi verificada, e a distribuição das variáveis pode diferir significativamente entre temporadas, especialmente com a introdução de novos compostos pela Pirelli.


## 5. Conclusão

Este trabalho apresentou três modelos de regressão linear para previsão da volta ideal de pit stop na Fórmula 1, utilizando três features derivadas de um conjunto de dados históricos de telemetria público. O modelo final de Regressão Linear obteve RMSE de 12,51 voltas e R² de 0,413 no conjunto de teste, sendo estatisticamente equivalente à Ridge e superior ao Lasso pelo teste de Wilcoxon aplicado sobre os RMSE da validação cruzada de 10 folds.

As três hipóteses de pesquisa foram confirmadas pelos resultados: compostos mais duros permitem stints mais longos (Hipótese 2), temperaturas mais altas antecipam a parada (Hipótese 3) e o número do stint possui forte influência positiva sobre a volta de parada (Hipótese 1). O composto WET apresentou o maior coeficiente de impacto, com paradas aproximadamente 20,9 voltas mais cedo que o composto HARD de referência. A alta variabilidade dos resíduos evidencia que fatores táticos não capturados pelo modelo, como safety car, undercut, posição em pista e número total de voltas da prova, exercem papel relevante na determinação da volta de parada.


## Referências

DEMSAR, J. Statistical comparisons of classifiers over multiple data sets. **Journal of Machine Learning Research**, v. 7, p. 1-30, 2006. Disponível em: https://jmlr.org/papers/v7/demsar06a.html. Acesso em: 4 mai. 2026.

FACELI, K.; LORENA, A. C.; GAMA, J.; ALMEIDA, T. A.; CARVALHO, A. C. P. L. F. **Inteligência artificial: uma abordagem de aprendizado de máquina**. 2. ed. Rio de Janeiro: LTC, 2021. Disponível em: https://repositorio.usp.br/item/003128493. Acesso em: 1 jun. 2026.

HEILMEIER, A.; THOMASER, A.; GRAF, M.; BETZ, J. Virtual Strategy Engineer: using artificial neural networks for making race strategy decisions in circuit motorsport. **Applied Sciences**, v. 10, n. 21, p. 7805, 2020. DOI: https://doi.org/10.3390/app10217805. Acesso em 5 abr. 2026.

KUMAR, N. **Formula 1 Dataset with Weather and Tyre Features**. Kaggle, 2024. Disponível em: https://www.kaggle.com/datasets/navenkumar1998/formula-1-dataset-with-weather-and-tyre-features. Acesso em: 4 abr. 2026.

MACIEL, L. F. V. **Regressão linear múltipla na modelagem de resultados na National Basketball Association (NBA)**. Trabalho de Conclusão de Curso (Bacharelado em Estatística) - Universidade Federal de Uberlândia, Uberlândia, 2019. Disponível em: https://repositorio.ufu.br/handle/123456789/28341. Acesso em: 4 jun. 2026.

PASSOS, L. F. C. **Métodos de regularização no aprendizado de máquinas: Ridge e LASSO**. Trabalho de Conclusão de Curso (Graduação em Estatística) - Universidade Federal Fluminense, Niterói, 2022. Disponível em: https://app.uff.br/riuff/handle/1/28391. Acesso em: 4 jun. 2026.

PEDREGOSA, F. et al. Scikit-learn: machine learning in Python. **Journal of Machine Learning Research**, v. 12, p. 2825-2830, 2011. Disponível em: https://jmlr.org/papers/v12/pedregosa11a.html. Acesso em: 1 jun. 2026.

SASIKUMAR, A.; LEEMA, A. A.; BALAKRISHNAN, P. Data-driven pit stop decision support for Formula 1 using deep learning models. **Frontiers in Artificial Intelligence**, v. 8, artigo 1673148, 2025. DOI: https://doi.org/10.3389/frai.2025.1673148. Acesso em 1 jun. 2026.

VASCONCELOS, B. F. B. **Poder preditivo de métodos de machine learning com processos de seleção de variáveis: uma aplicação às projeções de produto de países**. Tese (Doutorado em Economia) - Universidade de Brasília, Brasília, 2017. Disponível em: https://repositorio.unb.br/bitstream/10482/23995/1/2017_BrunoFreitasBoynarddeVasconcelos.pdf. Acesso em: 25 mai. 2026.