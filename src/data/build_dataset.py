import pandas as pd
from pathlib import Path

PROCESSED_PATH = Path('data/processed/')

def build_dataset():
    df = pd.read_parquet('data/raw/f1_all.parquet')

    df_treatment = df.copy()
    
    df_treatment['Compound'] = df_treatment['Compound'].replace({'None': 'UNKNOWN'})

    df_treatment = df_treatment.drop(columns=[
        'position_x', 'position_y', 'time', 'grid', 'name',
        'LapNumber', 'Driver', 'statusId', 'status'
    ])

    df_treatment = df_treatment.dropna(subset='TyreLife')

    paradas = df_treatment.groupby(['raceId', 'driverId', 'Stint'])['lap'].max().reset_index()
    paradas.rename(columns={'lap': 'pitstop_turn'}, inplace=True)

    ultimo_stint = df_treatment.groupby(['raceId', 'driverId'])['Stint'].max().reset_index()
    ultimo_stint['is_last_turn'] = True

    paradas = paradas.merge(ultimo_stint, on=['raceId', 'driverId', 'Stint'], how='left')
    df_stops = paradas[paradas['is_last_turn'].isnull()].copy()

    temp_por_stint = (
        df_treatment
        .groupby(['raceId', 'driverId', 'Stint'])['TrackTemp']
        .mean()
        .reset_index()
    )

    composto_por_stint = (
        df_treatment
        .groupby(['raceId', 'driverId', 'Stint'])['Compound']
        .first()
        .reset_index()
    )

    df_stops = df_stops.merge(temp_por_stint, on=['raceId', 'driverId', 'Stint'], how='left')
    df_stops = df_stops.merge(composto_por_stint, on=['raceId', 'driverId', 'Stint'], how='left')

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    df_stops.to_parquet('data/processed/df_stops.parquet', index=False)
    df_treatment.to_parquet('data/processed/df_treatment.parquet', index=False)

    print(f'df_treatment: {df_treatment.shape}')
    print(f'df_stops: {df_stops.shape[0]} pit stops | colunas: {df_stops.columns.tolist()}')

    return df_treatment, df_stops


if __name__ == '__main__':
    build_dataset()
