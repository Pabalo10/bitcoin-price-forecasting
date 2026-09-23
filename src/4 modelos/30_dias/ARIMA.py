"""
Módulo: ARIMA.py

Descripción:
Este script implementa el entrenamiento y evaluación de 4 modelos ARIMA aplicados a series
temporales de variación porcentual a 30 días (`pct_cambio_30d`). Está diseñado para encontrar
el mejor modelo ARIMA mediante validación cruzada y evaluar su desempeño en datos de prueba.

Características principales:
- Carga y preprocesamiento de los datos de entrada.
- División temporal de la serie en conjuntos de entrenamiento y prueba.
- Validación de combinaciones de parámetros ARIMA (p, d, q) sobre los datos de entrenamiento.
- Entrenamiento del modelo ARIMA con la mejor combinación de parámetros.
- Generación de predicciones y evaluación del modelo con múltiples métricas.
- Visualización de resultados y logging completo del experimento usando MLflow.

Dependencias:
- pandas
- statsmodels
- scikit-learn
- mlflow
- matplotlib
- utils personalizados:
    - utils.modelos_sklearn.* (funciones de preprocesamiento, validación, evaluación, visualización)
    - utils.paths (gestión de rutas)
    - utils.mlflow_utils (configuración y logging en MLflow)

Uso:
Ejecutar este script para entrenar y evaluar un modelo ARIMA sobre series temporales
mensuales, registrando automáticamente resultados y gráficos en MLflow.

"""

from utils.modelos_sklearn.B_preprocesamiento import preprocesar_datos_30d
from utils.modelos_sklearn.E_ajuste_hiperparametros import validar_modelo_timeseries
from utils.modelos_sklearn.F_metricas import evaluar_modelo_30d
from utils.modelos_sklearn.A_cargar_datos import cargar_datos_30d
from utils.modelos_sklearn.D_division_temporal import dividir_datos_temporalmente_30d
from utils.modelos_sklearn.G_graficar_predicciones import plot_predictions
from utils.mlflow_utils import setup_mlflow
from utils.paths import get_path
import mlflow
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA


def main(parquet_path: str, target_col: str = "pct_cambio_30d"):
    df = cargar_datos_30d(parquet_path)
    df = preprocesar_datos_30d(df, target_col)

    # Dividir datos en entrenamiento y test antes de ajustar
    y_train, y_test = dividir_datos_temporalmente_30d(df, target_col, test_ratio=0.2)

    param_grid = [(1, 1, 1), (2, 1, 2), (1, 0, 1), (2, 0, 2)]

    setup_mlflow("ARIMA")

    hora_actual = datetime.now().strftime("%H-%M-%S")
    with mlflow.start_run(run_name=f"ARIMA-{hora_actual}"):

        model, best_order = validar_modelo_timeseries(y_train, param_grid, ARIMA)
        pred = model.predict(start=y_test.index[0], end=y_test.index[-1])
        mse, mae, rmse, r2, acc = evaluar_modelo_30d(y_test, pred)

        fig = plot_predictions(y_test, pred=pred)

        mlflow.log_params({"order": best_order})
        mlflow.log_metrics({"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "acc": acc})
        mlflow.log_figure(fig, "prediction_plot.png")


if __name__ == "__main__":
    main(get_path("data", "parquet", "data_30d.parquet"))
