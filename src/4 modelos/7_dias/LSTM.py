"""
Módulo: LSTM.py

Descripción:
Este script implementa un flujo completo de experimentación con 4 modelos LSTM
(Long Short-Term Memory) para series temporales. El código entrena múltiples 4 modelos con distintas ventanas temporales
y realiza la selección automática del mejor modelo basándose en el error cuadrático medio (MSE).
Incluye visualizaciones de predicción y análisis de atención.

Características principales:
- Preparación automática de datos (carga, preprocesamiento, selección de variables, ventanas temporales).
- Entrenamiento y evaluación de 4 modelos LSTM usando una interfaz estilo `sklearn`.
- Ajuste de hiperparámetros vía búsqueda aleatoria (`RandomizedSearchCV`).
- Registro de experimentos y resultados usando MLflow.
- Visualización de predicciones y de los pesos de atención.

Dependencias:
- pandas
- numpy
- matplotlib
- mlflow
- sklearn
- utils personalizados:
    - utils.modelos_sklearn.* (para carga, preprocesamiento, selección, métricas, etc.)
    - utils.paths (gestión de rutas)
    - utils.arquitecturas.lstm_model (modelo LSTM con mecanismo de atención y API tipo sklearn)

Uso:
Ejecutar este script directamente para realizar múltiples experimentos con diferentes tamaños de ventana,
seleccionar el mejor modelo y registrar resultados visuales y métricos con MLflow.

"""

import mlflow
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from utils.modelos_sklearn.A_cargar_datos import cargar_datos_7d
from utils.modelos_sklearn.B_preprocesamiento import preprocesar_datos_7d
from utils.modelos_sklearn.C_seleccion_variables import seleccionar_variables
from utils.modelos_sklearn.D_division_temporal import dividir_datos_temporalmente_7d
from utils.modelos_sklearn.E_ajuste_hiperparametros import ajustar_modelo
from utils.modelos_sklearn.F_metricas import evaluar_modelo_7d
from utils.modelos_sklearn.G_graficar_predicciones import plot_predictions
from utils.mlflow_utils import setup_mlflow
from utils.paths import get_path

from utils.arquitecturas.lstm_model import SklearnLikeLSTM  # Clase adaptada


def crear_ventanas(X, y, window_size=10):
    """
    Crea ventanas temporales para el modelo LSTM.

    :param X: Datos de características
    :param y: Datos objetivo
    :param window_size: Tamaño de la ventana temporal

    :returns: Secuencias X, valores y correspondientes, y fechas asociadas
    """
    Xs, ys, fechas = [], [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i : i + window_size])
        ys.append(y.iloc[i + window_size])
        fechas.append(y.index[i + window_size])
    return np.array(Xs), np.array(ys), pd.to_datetime(fechas)


def preparar_datos(window_size=10, test_size=0.2):
    """
    Prepara los datos para el entrenamiento, validación y prueba.

    :param window_size: Tamaño de la ventana temporal
    :param test_size: Proporción de datos para prueba

    :returns: Datos preparados para el modelo
    """
    # Cargar y preprocesar datos
    df = cargar_datos_7d(get_path("data", "parquet", "data_7d.parquet"))
    X, y, scaler = preprocesar_datos_7d(df, tecnica="lstm", target_col="pct_cambio_7d")

    # Seleccionar variables importantes
    X, selector = seleccionar_variables(X, y, tecnica="lstm")

    # Crear ventanas temporales
    X_seq, y_seq, fechas = crear_ventanas(X, y, window_size)

    # División temporal en entrenamiento y prueba
    X_train, y_train, X_test, y_test = dividir_datos_temporalmente_7d(
        X_seq, y_seq, test_size
    )

    # Obtener fechas para conjuntos de entrenamiento y prueba
    n = len(fechas)
    test_start = int(n * (1 - test_size))
    fechas_train, fechas_test = fechas[:test_start], fechas[test_start:]

    # Devolver todos los conjuntos de datos
    return {
        "train": (X_train, y_train, fechas_train),
        "test": (X_test, y_test, fechas_test),
        "scaler": scaler,
        "selector": selector,
        "input_shape": X_train.shape[1:],
    }


def visualizar_atencion(model, X_test, fechas_test, window_size, show=True):
    """
    Visualiza los pesos de atención para muestras del conjunto de prueba.

    :param model: Modelo entrenado
    :param X_test: Datos de prueba
    :param fechas_test: Fechas correspondientes
    :param window_size: Tamaño de la ventana temporal
    :param show: Mostrar la gráfica

    :returns: Figura con visualización de atención
    """
    if not model.attention:
        return None

    # Seleccionar algunas muestras para visualizar
    indices = np.linspace(0, len(X_test) - 1, 3, dtype=int)

    fig, axes = plt.subplots(len(indices), 1, figsize=(12, 4 * len(indices)))
    if len(indices) == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        # Obtener pesos de atención para esta muestra
        attn_weights = model.get_attention_weights(X_test[idx : idx + 1])
        if attn_weights is None:
            continue

        # Fecha de predicción
        fecha_pred = fechas_test[idx]

        # Visualizar pesos de atención
        axes[i].bar(range(window_size), attn_weights[0], alpha=0.7)
        axes[i].set_title(f"Pesos de atención para predicción en {fecha_pred.date()}")
        axes[i].set_xlabel("Posición temporal en la ventana")
        axes[i].set_ylabel("Peso de atención")
        axes[i].grid(alpha=0.3)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def main():
    """Función principal para la experimentación con 4 modelos LSTM"""
    setup_mlflow("LSTM-PyTorch")

    # Configuración de la experimentación
    windows_size = [1, 7, 14, 21, 28]  # Ventanas temporales a probar
    test_size = 0.1  # Proporción de datos para prueba

    # Variables para el seguimiento del mejor modelo
    best_results = {
        "model": None,
        "params": None,
        "window_size": None,
        "metrics": {"mse": float("inf")},
        "data": None,
    }

    # Experimentar con diferentes tamaños de ventana
    for window_size in windows_size:
        print(f"\n{'=' * 50}")
        print(f"Experimentando con window_size = {window_size}")
        print(f"{'=' * 50}")

        # Preparar datos
        data_dict = preparar_datos(window_size, test_size)
        X_train, y_train, fechas_train = data_dict["train"]
        X_test, y_test, fechas_test = data_dict["test"]
        input_shape = data_dict["input_shape"]

        print(f"Dimensiones: X_train: {X_train.shape}, X_test: {X_test.shape}")

        # Crear modelo base
        lstm_modelo = SklearnLikeLSTM(input_shape=input_shape)

        # Definir grid de hiperparámetros
        param_grid = {
            "hidden_size": [64, 128],
            "num_layers": [1, 2],
            "dropout": [0.0, 0.2],
            "bidirectional": [False, True],
            "attention": [True],  # Usar mecanismo de atención
            "epochs": [50],
            "batch_size": [32, 64],
            "lr": [0.001, 0.0005],
            "weight_decay": [0.0, 0.0001],
            "early_stopping_patience": [5],  # Usar early stopping
            "input_shape": [
                input_shape
            ],  # Necesario para instanciar el modelo correctamente
        }

        # Iniciar run de MLflow
        hora_actual = datetime.now().strftime("%H-%M-%S")
        run_name = f"LSTM-window_size-{window_size}-{hora_actual}"

        with mlflow.start_run(run_name=run_name):
            # Ajustar el modelo con validación cruzada
            best_model, best_params = ajustar_modelo(
                lstm_modelo,
                X_train,
                y_train,
                param_grid,
                strategy="random",  # Para usar RandomizedSearchCV
                verbose=1,
            )

            # Evaluar modelo en conjunto de prueba
            mse, mae, rmse, r2, acc = evaluar_modelo_7d(best_model, X_test, y_test)

            # Visualizar predicciones
            pred_fig = plot_predictions(
                y_test, modelo=best_model, X=X_test, fechas=fechas_test, show=False
            )

            # Visualizar pesos de atención si el modelo los utiliza
            attn_fig = visualizar_atencion(
                best_model, X_test, fechas_test, window_size, show=False
            )

            # Registrar métricas, parámetros y figuras en MLflow
            mlflow.log_params(best_params)
            mlflow.log_metrics(
                {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "acc": acc}
            )
            mlflow.log_figure(pred_fig, "prediction_plot.png")

            if attn_fig:
                mlflow.log_figure(attn_fig, "attention_weights.png")

            # Imprimir métricas
            print(f"\nMétricas en conjunto de prueba:")
            print(f"MSE: {mse:.6f}")
            print(f"MAE: {mae:.6f}")
            print(f"RMSE: {rmse:.6f}")
            print(f"R²: {r2:.6f}")
            print(f"Accuracy: {acc:.2%}")

            # Actualizar mejor modelo si es necesario
            if mse < best_results["metrics"]["mse"]:
                print(
                    f"\n>>> Nuevo mejor modelo encontrado (window_size={window_size})!"
                )
                best_results["model"] = best_model
                best_results["params"] = best_params
                best_results["window_size"] = window_size
                best_results["metrics"] = {
                    "mse": mse,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "acc": acc,
                }
                best_results["data"] = {
                    "X_test": X_test,
                    "y_test": y_test,
                    "fechas_test": fechas_test,
                }

    # Registrar el mejor modelo global en MLflow
    print("\n" + "=" * 50)
    print(
        f"Registrando el mejor modelo global (window_size={best_results['window_size']})"
    )
    print("=" * 50)

    hora_actual = datetime.now().strftime("%H-%M-%S")
    run_name = f"BEST-LSTM-window_size-{best_results['window_size']}-{hora_actual}"

    with mlflow.start_run(run_name=run_name):
        # Métricas del mejor modelo
        metrics = best_results["metrics"]

        # Visualizar predicciones del mejor modelo
        best_fig = plot_predictions(
            best_results["data"]["y_test"],
            modelo=best_results["model"],
            X=best_results["data"]["X_test"],
            fechas=best_results["data"]["fechas_test"],
            show=True,
        )

        # Visualizar pesos de atención del mejor modelo
        best_attn_fig = visualizar_atencion(
            best_results["model"],
            best_results["data"]["X_test"],
            best_results["data"]["fechas_test"],
            best_results["window_size"],
            show=True,
        )

        # Registrar métricas, parámetros y figuras
        mlflow.log_params(best_results["params"])
        mlflow.log_metrics(metrics)
        mlflow.log_figure(best_fig, "best_prediction_plot.png")

        if best_attn_fig:
            mlflow.log_figure(best_attn_fig, "best_attention_weights.png")

        # Imprimir métricas finales
        print(f"\nMétricas del mejor modelo:")
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"R²: {metrics['r2']:.6f}")
        print(f"Accuracy: {metrics['acc']:.2%}")


if __name__ == "__main__":
    main()
