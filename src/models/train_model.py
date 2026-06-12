import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

EXPERIMENTS_PATH = Path('experiments/')
EXPERIMENTS_PATH.mkdir(parents=True, exist_ok=True)

PREPROCESSOR_PATH = 'experiments/preprocessor.joblib'
SPLIT_PATH = 'experiments/split_data.joblib'
MODEL_PATH = 'experiments/linear_regression_model.joblib'
RESULTS_PATH = 'experiments/training_results.csv'


def train_models():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    split_data = joblib.load(SPLIT_PATH)
    X_train, X_test, y_train, y_test = (
        split_data['X_train'], split_data['X_test'],
        split_data['y_train'], split_data['y_test']
    )

    models = {
        'Linear Regression': {'model': LinearRegression(), 'params': {'alpha': 'N/A'}},
        'Ridge': {'model': Ridge(alpha=1.0), 'params': {'alpha': 1.0}},
        'Lasso': {'model': Lasso(alpha=0.1, random_state=42), 'params': {'alpha': 0.1}}
    }

    results = []
    pipeline_final = None
    data_exec = datetime.now().strftime('%Y-%m-%d %H:%M')
    obs = {
        'Linear Regression': 'Modelo baseline sem regularização',
        'Ridge': 'Regularização L2 (alpha=1.0)',
        'Lasso': 'Regularização L1 (alpha=0.1) com seleção automática'
    }

    for name, config in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', config['model'])
        ])

        pipeline.fit(X_train, y_train)

        if name == 'Linear Regression':
            pipeline_final = pipeline

        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)

        results.append({
            'Modelo': name,
            'alpha': config['params']['alpha'],
            'RMSE Treino': round(np.sqrt(mean_squared_error(y_train, y_pred_train)), 3),
            'RMSE Teste': round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 3),
            'MAE Teste': round(mean_absolute_error(y_test, y_pred_test), 3),
            'R² Teste': round(r2_score(y_test, y_pred_test), 3),
            'data': data_exec,
            'observações': obs[name]
        })

    results_df = pd.DataFrame(results).sort_values('RMSE Teste')
    results_df.to_csv(RESULTS_PATH, index=False)
    joblib.dump(pipeline_final, MODEL_PATH)

    print("Resultados do treinamento:")
    print(results_df.to_string(index=False))
    print(f"\nModelo final salvo em: {MODEL_PATH}")
    print(f"Resultados salvos em: {RESULTS_PATH}")

    return pipeline_final, results_df


if __name__ == '__main__':
    train_models()
