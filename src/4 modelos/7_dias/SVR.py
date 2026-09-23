"""
Módulo: SVR.py

Descripción:
Este script implementa una serie de experimentos para entrenar 4 modelos de regresión por
vectores de soporte (SVR) aplicados a series temporales financieras. Se exploran diferentes
estrategias de selección de variables y se prueban hiperparámetros para abordar el sobreajuste.

Características principales:
- Carga y preprocesamiento de datos.
- Selección de características mediante técnicas como `f_regression` o `mutual_info_regression`.
- Escalado de datos con `StandardScaler`.
- Entrenamiento y validación de 4 modelos SVR con búsqueda aleatoria (`RandomizedSearchCV`).
- Evaluación del modelo en datos fuera de muestra (test set).
- Reentrenamiento del mejor modelo con todos los datos de entrenamiento y validación.
- Registro automático de métricas, parámetros y visualizaciones usando MLflow.

Dependencias:
- pandas
- numpy
- scikit-learn
- mlflow
- matplotlib
- utils personalizados:
    - utils.modelos_sklearn.* (funciones de carga, preprocesamiento, evaluación, etc.)
    - utils.paths (gestión de rutas)
    - utils.mlflow_utils (configuración y logging en MLflow)

Uso:
Ejecutar este script para entrenar múltiples 4 modelos SVR con distintas combinaciones de
características seleccionadas y configuraciones de hiperparámetros, y registrar todo el
proceso con MLflow para su posterior análisis.

"""

from utils.modelos_sklearn.A_cargar_datos import cargar_datos_7d
from utils.modelos_sklearn.D_division_temporal import dividir_datos_temporalmente_7d
from utils.modelos_sklearn.E_ajuste_hiperparametros import ajustar_modelo
from utils.modelos_sklearn.C_seleccion_variables import seleccionar_variables
from utils.modelos_sklearn.F_metricas import evaluar_modelo_7d
from utils.modelos_sklearn.G_graficar_predicciones import plot_predictions
from utils.mlflow_utils import setup_mlflow
from utils.paths import get_path

import mlflow
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from datetime import datetime


def main():
    setup_mlflow("Modelo SVR")

    # Cargar y preparar datos
    df = cargar_datos_7d(get_path("data", "parquet", "data_7d.parquet"))
    df.dropna(inplace=True)
    y = df["pct_cambio_7d"]
    X_original = df.drop(columns=["pct_cambio_7d"])

    # Dividir en train/val y test
    X_train_val, y_train_val, X_test, y_test = dividir_datos_temporalmente_7d(
        X_original, y
    )

    # Escalado solo sobre los datos de entrenamiento
    scaler = StandardScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)

    # Convertir los datos escalados de vuelta a un DataFrame de pandas con las mismas columnas
    X_train_val_scaled = pd.DataFrame(
        X_train_val_scaled, columns=X_train_val.columns, index=X_train_val.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled, columns=X_test.columns, index=X_test.index
    )

    # Parámetros para explorar
    k_vars_a_probar = [5, 10, min(15, X_original.shape[1])]
    c_values_a_probar = [0.01, 0.1, 1, 10]
    epsilon_values_a_probar = [0.1, 0.2, 0.5]
    kernels_a_probar = ["rbf", "linear"]

    for k_vars in k_vars_a_probar:
        # Selección de variables SOLO en train
        conjuntos_X_train = seleccionar_variables(
            X_train_val_scaled, y_train_val, tecnica="svr", k_vars=k_vars
        )

        for nombre_conjunto, X_train_sel in conjuntos_X_train.items():
            print(
                f"\nEntrenando y evaluando con el conjunto de variables: {nombre_conjunto} (k={k_vars})"
            )

            # Aplicar misma selección al test
            X_test_sel = X_test_scaled[X_train_sel.columns]

            svr_model = SVR()
            param_grid_svr = {
                "C": c_values_a_probar,
                "epsilon": epsilon_values_a_probar,
                "kernel": kernels_a_probar,
            }

            hora_actual = datetime.now().strftime("%H-%M-%S")
            with mlflow.start_run(run_name=f"SVR-f_regression-k{k_vars}-{hora_actual}"):

                # Ajuste de hiperparámetros
                best_model_svr, best_params_svr = ajustar_modelo(
                    svr_model,
                    X_train_sel,
                    y_train_val,
                    param_grid_svr,
                    strategy="random",
                    scoring="r2",
                    n_iter=50,
                )

                # Evaluación en test
                mse_svr, mae_svr, rmse_svr, r2_svr, acc_svr = evaluar_modelo_7d(
                    best_model_svr, X_test_sel, y_test
                )
                fig_svr = plot_predictions(y_test, modelo=best_model_svr, X=X_test_sel)

                # Log
                mlflow.log_params(best_params_svr)
                mlflow.log_metrics(
                    {
                        "mse": mse_svr,
                        "mae": mae_svr,
                        "rmse": rmse_svr,
                        "r2": r2_svr,
                        "acc": acc_svr,
                        "k_vars": k_vars,
                    }
                )
                mlflow.log_figure(
                    fig_svr, f"prediction_plot_{nombre_conjunto}_k{k_vars}.png"
                )


if __name__ == "__main__":
    main()
