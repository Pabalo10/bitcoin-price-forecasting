"""
Módulo: SARIMA.py

Descripción:
Este script entrena y evalúa 4 modelos SARIMA (Seasonal ARIMA) sobre series temporales que representan
la variación porcentual a 30 días (`pct_cambio_30d`). Se realiza una búsqueda de hiperparámetros tanto
para el componente no estacional (p, d, q) como para el estacional (P, D, Q, s), considerando patrones
como estacionalidad semanal o mensual.

Características principales:
- Carga y preprocesamiento de los datos de entrada.
- División temporal del conjunto de datos en entrenamiento y prueba.
- Validación cruzada de combinaciones de parámetros SARIMA mediante un wrapper genérico.
- Entrenamiento del mejor modelo SARIMA encontrado.
- Predicción y evaluación del modelo usando métricas de error estándar.
- Visualización de resultados y registro completo con MLflow.

Dependencias:
- pandas
- statsmodels
- mlflow
- matplotlib
- utils personalizados:
    - utils.modelos_sklearn.* (carga, preprocesamiento, evaluación, gráficos, validación de 4 modelos)
    - utils.paths (manejo de rutas)
    - utils.mlflow_utils (configuración de experimentos MLflow)

Uso:
Ejecutar este script para evaluar el comportamiento de 4 modelos SARIMA sobre la serie temporal de
interés (`pct_cambio_30d`), ideal para detectar patrones recurrentes en los datos financieros o similares.

"""

import mlflow
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utils.modelos_sklearn.A_cargar_datos import cargar_datos_30d
from utils.modelos_sklearn.B_preprocesamiento import preprocesar_datos_30d
from utils.modelos_sklearn.D_division_temporal import dividir_datos_temporalmente_30d
from utils.modelos_sklearn.E_ajuste_hiperparametros import validar_modelo_timeseries
from utils.modelos_sklearn.F_metricas import evaluar_modelo_30d
from utils.modelos_sklearn.G_graficar_predicciones import plot_predictions
from utils.mlflow_utils import setup_mlflow
from utils.paths import get_path


def main(parquet_path: str, target_col: str = "pct_cambio_30d"):
    df = cargar_datos_30d(parquet_path)
    df = preprocesar_datos_30d(df, target_col)

    # Dividir en entrenamiento y test real
    y_train, y_test = dividir_datos_temporalmente_30d(df, target_col, test_ratio=0.2)

    param_grid = [(1, 1, 1), (2, 1, 2), (1, 0, 1), (2, 0, 2)]
    seasonal_param_grid = [
        (1, 0, 1, 7),  # estacionalidad semanal
        (0, 1, 1, 7),
        (1, 1, 1, 7),
        (1, 0, 1, 30),  # estacionalidad mensual (por si acaso)
    ]

    setup_mlflow("SARIMA Modular")

    hora_actual = datetime.now().strftime("%H-%M-%S")
    with mlflow.start_run(run_name=f"SARIMA-{hora_actual}"):
        model, (best_order, best_seasonal_order) = validar_modelo_timeseries(
            y_train,
            param_grid,
            SARIMAX,
            seasonal=True,
            seasonal_param_grid=seasonal_param_grid,
        )

        pred = model.predict(start=y_test.index[0], end=y_test.index[-1])
        mse, mae, rmse, r2, acc = evaluar_modelo_30d(y_test, pred)

        fig = plot_predictions(y_test, pred=pred)

        mlflow.log_params({"order": best_order, "seasonal_order": best_seasonal_order})
        mlflow.log_metrics({"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "acc": acc})
        mlflow.log_figure(fig, "prediction_plot.png")


if __name__ == "__main__":
    main(get_path("data", "parquet", "data_30d.parquet"))
