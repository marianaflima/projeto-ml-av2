# Dicionário de Dados

## Race Info (Informações da Corrida)

| Indicador | Descrição |
|---|---|
| raceId | ID da corrida. |
| year | O ano da corrida. |
| round | O número da rodada da corrida dentro da temporada. |
| circuitId | ID do circuito. |
| name | Nome do circuito. |

---

## Driver & Lap Info (Informações do Piloto e da Volta)

| Indicador | Descrição |
|---|---|
| driverId | ID do piloto. |
| constructorId | ID da equipe (construtora) do piloto. |
| lap / LapNumber | O número da volta. |
| code / Driver | Código de três letras do piloto. |
| time | O tempo registrado para a volta convertido em número legível. |
| milliseconds | O tempo da volta em milissegundos. |
| status | Informa se o piloto terminou a corrida ou não. |
| statusId | ID do status do piloto. |
| position_x | Posição do piloto no eixo X, considerando um plano cartesiano. |
| position_y | Posição do piloto no eixo Y, considerando um plano cartesiano. |
| grid | A posição do piloto no grid de largada. |
| Stint | O número do stint (período entre trocas de pneus) do piloto. |

---

## Tyre Features (Características dos Pneus)

| Indicador | Descrição |
|---|---|
| Compound | O tipo de composto do pneu (SOFT, MEDIUM, HARD, INTERMEDIATE, WET, UNKNOWN). |
| TyreLife | Quantas voltas o pneu já rodou naquele stint. |
| FreshTyre | Indica se o pneu é novo (True) ou usado (False). |

---

## Weather Features (Características do Clima)

| Indicador | Descrição |
|---|---|
| TrackTemp | A temperatura da pista (°C). |
| AirTemp | A temperatura do ar (°C). |
| Humidity | A umidade relativa do ar (%). |
| Pressure | A pressão atmosférica (mbar). |
| Rainfall | Indica se havia chuva (booleano). |
| WindSpeed | A velocidade do vento (km/h). |
| WindDirection | A direção do vento (graus). |

---

## Target (Variável Alvo)

| Indicador | Descrição |
|---|---|
| pitstop_turn | Número da volta em que o piloto realizou o pit stop. Derivada como a última volta de cada stint, excluindo o stint final da corrida. |

---

## Colunas Criadas Durante o Processamento

| Indicador | Descrição |
|---|---|
| df_stops | Dataset de trabalho contendo um registro por pit stop, com colunas: raceId, driverId, Stint, pitstop_turn, is_last_turn, TrackTemp (média do stint), Compound (primeiro do stint). |
| is_last_turn | Flag booleana indicando se o stint é o último da corrida (descartado na criação do target). |
| df_treatment | Dataset intermediário após limpeza (remoção de colunas, substituição de 'None' por 'UNKNOWN' em Compound, remoção de nulos em TyreLife). |
