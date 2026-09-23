"""
Módulo: XGBoost.py

Descripción:
Este script entrena y evalúa un modelo de regresión XGBoost sobre series temporales multivariantes,
utilizando la variación porcentual a 30 días (`pct_cambio_30d`) como variable objetivo. Realiza una
búsqueda de hiperparámetros mediante validación cruzada temporal (TimeSeriesSplit) y registra todo el
proceso en MLflow.

Características principales:
- Carga y preprocesamiento de los datos multivariantes.
- División del conjunto de datos en entrenamiento y prueba respetando la estructura temporal.
- Búsqueda de hiperparámetros con `ParameterGrid` y validación cruzada en series temporales.
- Entrenamiento del mejor modelo XGBoost encontrado.
- Evaluación de métricas de error estándar y generación de gráficos de predicción.
- Registro completo del experimento con parámetros, métricas y visualizaciones en MLflow.

Dependencias:
- pandas
- numpy
- mlflow
- xgboost
- sklearn (model_selection, metrics, preprocessing)
- utils personalizados:
    - utils.modelos_sklearn.* (carga, preprocesamiento, evaluación, gráficos)
    - utils.mlflow_utils (configuración de experimentos MLflow)
    - utils.paths (manejo de rutas)

Uso:
Ejecutar este script para entrenar un modelo XGBoost sobre una serie temporal multivariante con el fin
de predecir la variable `pct_cambio_30d`. Ideal para análisis financieros con múltiples indicadores
predictivos.

"""

import pandas as pd
import numpy as np
import mlflow
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from utils.modelos_sklearn.A_cargar_datos import cargar_datos_30d
from utils.modelos_sklearn.B_preprocesamiento import preprocesar_datos_30d
from utils.modelos_sklearn.D_division_temporal import dividir_datos_temporalmente_30d
from utils.modelos_sklearn.F_metricas import evaluar_modelo_30d
from utils.modelos_sklearn.G_graficar_predicciones import plot_predictions
from utils.mlflow_utils import setup_mlflow
from utils.paths import get_path


def validar_modelo_xgboost(X, y, param_grid, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_mse = np.inf
    best_params = None
    best_model = None

    for params in ParameterGrid(param_grid):
        fold_mse = []
        for train_idx, test_idx in tscv.split(X):
            X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(X_train_fold, y_train_fold)
            y_pred_fold = model.predict(X_test_fold)
            mse = mean_squared_error(y_test_fold, y_pred_fold)
            fold_mse.append(mse)

        avg_mse = np.mean(fold_mse)
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_params = params
            best_model = xgb.XGBRegressor(**params)
            best_model.fit(X, y)

    print(f"Mejores hiperparámetros encontrados: {best_params}")
    return best_model, best_params


def main():
    mlflow.set_tracking_uri("mlruns")
    experiment_name = "XGBoost"
    setup_mlflow(experiment_name)

    with mlflow.start_run(
        run_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        # 1. Cargar y preprocesar datos
        df = cargar_datos_30d(get_path("data", "parquet", "data_30d.parquet"))
        target_col = "pct_cambio_30d"
        df = preprocesar_datos_30d(df, target_col)

        # 2. División en train/test (multivariado=True para XGBoost)
        train_df, test_df = dividir_datos_temporalmente_30d(
            df, target_col, multivariado=True
        )
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        # 3. Definir búsqueda de hiperparámetros
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_alpha": [0, 0.1, 1, 10],  # Regularización L1 (Lasso)
            "reg_lambda": [1, 5, 10, 100],  # Regularización L2 (Ridge)
        }

        # 4. Validación y entrenamiento
        best_model, best_params = validar_modelo_xgboost(
            X_train, y_train, param_grid, n_splits=3
        )

        # 5. Predicción y evaluación
        y_pred = best_model.predict(X_test)
        mse, mae, rmse, r2, acc = evaluar_modelo_30d(y_test.values, y_pred)

        # 6. Registro en MLflow
        mlflow.log_params({"order": best_params})
        mlflow.log_metrics({"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "acc": acc})

        fig = plot_predictions(y_test, pred=y_pred)
        mlflow.log_figure(fig, "prediction_plot.png")


if __name__ == "__main__":
    main()
