"""
Módulo: ARBOLES_DE:DECISION.py

Descripción:
Este script entrena y evalúa un modelo de regresión basado en un Árbol de Decisión
para predecir el cambio porcentual a 7 días (`pct_cambio_7d`) en un conjunto de datos de series temporales.
El flujo incluye la carga y preprocesamiento de datos, la división temporal en conjuntos de entrenamiento y prueba,
ajuste de hiperparámetros, evaluación del modelo, visualización de predicciones y análisis de importancia de variables.

También se realiza un reentrenamiento final utilizando únicamente las variables más importantes,
y todo el proceso es registrado mediante MLflow.

Dependencias:

- pandas
- scikit-learn (sklearn)
- mlflow
- datetime
- Funciones auxiliares personalizadas desde:
    - utils.modelos_sklearn.A_cargar_datos
    - utils.modelos_sklearn.B_preprocesamiento
    - utils.modelos_sklearn.D_division_temporal
    - utils.modelos_sklearn.E_ajuste_hiperparametros
    - utils.modelos_sklearn.F_metricas
    - utils.modelos_sklearn.G_graficar_predicciones
    - utils.mlflow_utils
    - utils.paths

Uso:

Ejecutar directamente este script para entrenar y registrar un modelo de árbol de decisión con MLflow.

"""

import mlflow
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from utils.modelos_sklearn.A_cargar_datos import cargar_datos_7d
from utils.modelos_sklearn.B_preprocesamiento import preprocesar_datos_7d
from utils.modelos_sklearn.D_division_temporal import dividir_datos_temporalmente_7d
from utils.modelos_sklearn.E_ajuste_hiperparametros import ajustar_modelo
from utils.modelos_sklearn.F_metricas import evaluar_modelo_7d
from utils.modelos_sklearn.G_graficar_predicciones import plot_predictions
from utils.mlflow_utils import setup_mlflow
from utils.paths import get_path
import pandas as pd


def analizar_importancias(modelo, X, umbral_importancia=0.01, top_n=20):
    """Muestra, guarda y retorna las variables más importantes según el árbol."""
    importancias = pd.Series(modelo.feature_importances_, index=X.columns)
    importancias = importancias.sort_values(ascending=False)

    # Mostrar top n en consola
    print("\nTop variables por importancia:")
    print(importancias.head(top_n))
    print()

    # Seleccionar las variables importantes
    variables_importantes = importancias[
        importancias > umbral_importancia
    ].index.tolist()

    return variables_importantes


def main():
    setup_mlflow("Modelo Arbol")

    df = cargar_datos_7d(get_path("data", "parquet", "data_7d.parquet"))
    X, y, _ = preprocesar_datos_7d(df, tecnica="arbol", target_col="pct_cambio_7d")

    # Cambiado: ahora separamos solo train_val y test
    X_train_val, y_train_val, X_test, y_test = dividir_datos_temporalmente_7d(X, y)

    modelo = DecisionTreeRegressor(random_state=42)

    param_grid = {
        "max_depth": [3, 5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [
            "sqrt",
            "log2",
            None,
        ],  # Para controlar el número de variables evaluadas en cada split
    }

    try:
        hora_actual = datetime.now().strftime("%H-%M-%S")
        with mlflow.start_run(run_name=f"Arbol-{hora_actual}"):
            best_model, best_params = ajustar_modelo(
                modelo, X_train_val, y_train_val, param_grid
            )
            mse, mae, rmse, r2, acc = evaluar_modelo_7d(best_model, X_test, y_test)

            fig = plot_predictions(y_test, modelo=best_model, X=X_test)

            mlflow.log_params(best_params)
            mlflow.log_metrics(
                {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "acc": acc}
            )
            mlflow.log_figure(fig, "prediction_plot.png")

            # Seleccionar las variables importantes
            variables_importantes = analizar_importancias(best_model, X_train_val)

            # Quedarse solo variables importantes
            X_train_val_filtrado = X_train_val[variables_importantes]
            X_test_filtrado = X_test[variables_importantes]

            # Reentrenar el modelo con las variables importantes y train_val completo
            modelo_reentrenado = DecisionTreeRegressor(random_state=42, **best_params)
            modelo_reentrenado.fit(X_train_val_filtrado, y_train_val)

            # Evaluar el modelo reducido
            mse2, mae2, rmse2, r2_2, acc2 = evaluar_modelo_7d(
                modelo_reentrenado, X_test_filtrado, y_test
            )
            fig2 = plot_predictions(y_test, modelo=best_model, X=X_test)

            mlflow.log_metrics(
                {"mse": mse2, "mae": mae2, "rmse": rmse2, "r2": r2_2, "acc": acc2}
            )
            mlflow.log_figure(fig2, "prediction_plot.png")

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
