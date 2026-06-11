import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from scipy.stats import wilcoxon

PREPROCESSOR_PATH = 'experiments/preprocessor.joblib'
SPLIT_PATH = 'experiments/split_data.joblib'
CV_RESULTS_PATH = 'experiments/cv_results.csv'
WILCOXON_RESULTS_PATH = 'experiments/wilcoxon_results.csv'


def evaluate_models():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    split_data = joblib.load(SPLIT_PATH)
    X_train, y_train = split_data['X_train'], split_data['y_train']
    X_test, y_test = split_data['X_test'], split_data['y_test']

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1, random_state=42)
    }

    results_cv = []

    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=kf, scoring='neg_root_mean_squared_error'
        )

        rmse_scores = -scores

        results_cv.append([
            name,
            round(rmse_scores.mean(), 6),
            round(rmse_scores.std(), 6)
        ])

    cv_results_df = pd.DataFrame(
        results_cv,
        columns=['Modelo', 'RMSE Médio', 'Desvio-Padrão']
    ).sort_values('RMSE Médio')

    cv_results_df.to_csv(CV_RESULTS_PATH, index=False)

    pipeline_linear = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])
    pipeline_ridge = Pipeline([
        ('preprocessor', preprocessor),
        ('model', Ridge(alpha=1.0))
    ])
    pipeline_lasso = Pipeline([
        ('preprocessor', preprocessor),
        ('model', Lasso(alpha=0.1, random_state=42))
    ])

    linear_scores = -cross_val_score(
        pipeline_linear, X_train, y_train, cv=kf,
        scoring='neg_root_mean_squared_error'
    )
    ridge_scores = -cross_val_score(
        pipeline_ridge, X_train, y_train, cv=kf,
        scoring='neg_root_mean_squared_error'
    )
    lasso_scores = -cross_val_score(
        pipeline_lasso, X_train, y_train, cv=kf,
        scoring='neg_root_mean_squared_error'
    )

    wilcoxon_results = pd.DataFrame([
        {
            'Comparação': 'Linear vs Ridge',
            'p-value': round(wilcoxon(linear_scores, ridge_scores).pvalue, 6)
        },
        {
            'Comparação': 'Linear vs Lasso',
            'p-value': round(wilcoxon(linear_scores, lasso_scores).pvalue, 6)
        },
        {
            'Comparação': 'Ridge vs Lasso',
            'p-value': round(wilcoxon(ridge_scores, lasso_scores).pvalue, 6)
        }
    ])

    wilcoxon_results.to_csv(WILCOXON_RESULTS_PATH, index=False)

    print("=== Validação Cruzada (10-Fold) ===")
    print(cv_results_df.to_string(index=False))
    print("\n=== Teste de Wilcoxon ===")
    print(wilcoxon_results.to_string(index=False))

    return cv_results_df, wilcoxon_results


if __name__ == '__main__':
    evaluate_models()
