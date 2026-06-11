import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib

PROCESSED_PATH = 'data/processed/df_stops.parquet'
PREPROCESSOR_PATH = 'experiments/preprocessor.joblib'
SPLIT_PATH = 'experiments/split_data.joblib'


def build_features():
    df_stops = pd.read_parquet(PROCESSED_PATH)

    y = df_stops['pitstop_turn']
    X = df_stops.drop(columns=['pitstop_turn', 'is_last_turn', 'raceId', 'driverId'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    num_features = ['Stint', 'TrackTemp']
    cat_features = ['Compound']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump({
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test
    }, SPLIT_PATH)

    print("Pipeline de pré-processamento criado com sucesso.")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

    return preprocessor, X_train, X_test, y_train, y_test


if __name__ == '__main__':
    build_features()
