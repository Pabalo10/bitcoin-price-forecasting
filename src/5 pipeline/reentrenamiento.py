"""
entrenamiento_modelos.py

Este script entrena, evalúa y reentrena modelos de predicción de series temporales
(SVR y XGBoost) aplicados a la predicción del cambio porcentual en datos financieros
(u otra serie temporal). Utiliza estrategias como la selección de variables mediante
información mutua, escalado de características, validación cruzada temporal y gestión
robusta de datos nuevos.

Dependencias:
-------------
- os: Para operaciones del sistema de archivos.
- joblib: Para guardar y cargar modelos entrenados.
- numpy: Operaciones numéricas y vectorización.
- pandas: Manipulación de estructuras tabulares.
- sklearn:
    - SVR: Modelo de regresión basado en máquinas de soporte vectorial.
    - preprocessing.StandardScaler: Para estandarizar características.
    - impute.SimpleImputer: Para imputar valores faltantes.
    - model_selection.TimeSeriesSplit: Para validación cruzada temporal.
    - feature_selection.mutual_info_regression: Selección de variables.
    - metrics.r2_score: Métrica de evaluación R^2.
- xgboost.XGBRegressor: Regresor basado en boosting de gradiente.
- utils.modelos_sklearn:
    - A_cargar_datos: Funciones personalizadas para cargar datos parquet de 7d o 30d.
    - F_metricas: Funciones para evaluar modelos.
- utils.paths.get_path: Función auxiliar para construir rutas de archivos.

Funciones:
----------
- seleccionar_variables_mutual_info_k15(X, y, k_vars=15):
    Aplica `mutual_info_regression` para seleccionar las k_vars variables más
    informativas con respecto al target `y`. Devuelve el subconjunto de X con
    esas columnas y la lista de nombres seleccionados.

- accuracy_direccion(y_true, y_pred):
    Calcula la "accuracy direccional", es decir, cuántas veces la dirección
    del cambio predicho coincide con la dirección real. Útil para tareas de
    predicción de tendencias.

- entrenar_evaluar_y_reentrenar_modelo(tecnica, test_size=0.2):
    Función principal del script. Realiza el flujo completo de entrenamiento y
    evaluación para una técnica específica ('svr' o 'xgboost'):

    Para SVR:
        - Carga datos históricos y nuevos de horizonte 7 días.
        - Selecciona variables mediante mutual information.
        - Aplica imputación, escalado y entrena modelo con SVR.
        - Evalúa en conjunto nuevo y reentrena con todos los datos.
        - Guarda modelo y escalador final.

    Para XGBoost:
        - Carga datos de horizonte 30 días.
        - Aplica imputación, escalado y entrenamiento con validación cruzada
          (TimeSeriesSplit) para evitar data leakage.
        - Evalúa en datos nuevos y reentrena con todo el histórico.
        - Guarda modelo y escalador final.
"""

import os
import joblib
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score
import numpy as np
from utils.modelos_sklearn.A_cargar_datos import cargar_datos_7d, cargar_datos_30d
from utils.modelos_sklearn.F_metricas import evaluar_modelo_7d, evaluar_modelo_30d
from utils.paths import get_path


def seleccionar_variables_mutual_info_k15(X, y, k_vars=15):
    mi = mutual_info_regression(X, y)
    mi_series = pd.Series(mi, index=X.columns)
    columnas_seleccionadas = (
        mi_series.sort_values(ascending=False).head(k_vars).index.tolist()
    )
    return X[columnas_seleccionadas], columnas_seleccionadas


def accuracy_direccion(y_true, y_pred):
    """Calcula la accuracy direccional."""
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    return np.mean(true_direction == pred_direction)


def entrenar_evaluar_y_reentrenar_modelo(tecnica, test_size=0.2):
    """
    Entrena, evalúa y reentrena el modelo SVR o ARIMA.

    Args:
        tecnica (str): 'svr' o 'xgboost' para especificar el modelo.
        test_size (float): Proporción de datos a usar como conjunto de prueba
                           durante la evaluación inicial.
    Returns:
        None
    """
    os.makedirs("modelos", exist_ok=True)

    if tecnica == "svr":
        # Cargar y preparar datos de entrenamiento
        parquet_file_path_train = get_path("data", "parquet", "data_7d.parquet")
        df_train = cargar_datos_7d(parquet_file_path_train)
        df_train_sin_nan = df_train.dropna(subset=["pct_cambio_7d"])
        y_train = df_train_sin_nan["pct_cambio_7d"]
        X_train = df_train_sin_nan.drop(columns=["pct_cambio_7d"])

        # Selección de columnas con mutual_info_regression
        X_train_sel, columnas_seleccionadas = seleccionar_variables_mutual_info_k15(
            X_train, y_train, k_vars=15
        )

        # Imputación y escalado
        imputador = SimpleImputer(strategy="mean")
        X_train_sel = imputador.fit_transform(X_train_sel)
        X_train_sel = pd.DataFrame(X_train_sel, columns=columnas_seleccionadas)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)

        # Entrenar SVR con mejores parámetros
        best_params_svr = {"kernel": "linear", "epsilon": 0.5, "C": 0.1}
        modelo_svr = SVR(**best_params_svr)
        modelo_svr.fit(X_train_scaled, y_train)

        # Ver resultados
        print("Reentrenamiento SVR:")
        metricas_svr_nuevos = evaluar_modelo_7d(modelo_svr, X_train_scaled, y_train)

        # ------------------------
        # Evaluación en nuevos datos
        # ------------------------
        parquet_file_path_nuevos = get_path(
            "data", "parquet nuevos", "data_7d_nuevos.parquet"
        )
        df_nuevos = cargar_datos_7d(parquet_file_path_nuevos)
        df_nuevos_sin_nan = df_nuevos.dropna(subset=["pct_cambio_7d"])
        y_nuevos = df_nuevos_sin_nan["pct_cambio_7d"]
        X_nuevos = df_nuevos_sin_nan.drop(columns=["pct_cambio_7d"])

        # Asegurar columnas consistentes
        for col in columnas_seleccionadas:
            if col not in X_nuevos.columns:
                X_nuevos[col] = np.nan
        X_nuevos_sel = X_nuevos[columnas_seleccionadas]

        # Imputar y escalar nuevos
        X_nuevos_imputado = imputador.transform(X_nuevos_sel)
        X_nuevos_escalado = scaler.transform(X_nuevos_imputado)

        print("Métricas SVR en datos nuevos:")
        metricas_svr_nuevos = evaluar_modelo_7d(modelo_svr, X_nuevos_escalado, y_nuevos)

        # ------------------------
        # Reentrenamiento con todos los datos
        # ------------------------
        df_final = pd.concat([df_train_sin_nan, df_nuevos_sin_nan])
        y_final = df_final["pct_cambio_7d"]
        X_final = df_final.drop(columns=["pct_cambio_7d"])

        # Usar las mismas columnas seleccionadas
        for col in columnas_seleccionadas:
            if col not in X_final.columns:
                X_final[col] = np.nan
        X_final_sel = X_final[columnas_seleccionadas]

        # Imputar y escalar
        X_final_imputado = imputador.fit_transform(X_final_sel)
        X_final_escalado = scaler.fit_transform(X_final_imputado)

        # Reentrenar modelo final
        modelo_svr_final = SVR(**best_params_svr)
        modelo_svr_final.fit(X_final_escalado, y_final)

        # Guardar modelo final
        ruta_modelo_final = "modelos/SVR.joblib"
        ruta_scaler_final = "modelos/SVR_scaler.joblib"
        joblib.dump(modelo_svr_final, ruta_modelo_final)
        joblib.dump(scaler, ruta_scaler_final)

        print(f"Modelo SVR final guardado en: {ruta_modelo_final}")
        print(f"Scaler final guardado en: {ruta_scaler_final}")

        # Evaluación final
        pred_final = modelo_svr_final.predict(X_final_escalado)
        print("SVR (Final):")
        evaluar_modelo_30d(y_final, pred_final)

    elif tecnica == "xgboost":

        print(f"\n--- Entrenando, evaluando y reentrenando modelo XGBoost ---")

        # Cargar y preparar datos de entrenamiento
        parquet_file_path_train = get_path("data", "parquet", "data_30d.parquet")
        df_train = cargar_datos_30d(parquet_file_path_train)
        df_train_sin_nan = df_train.dropna(subset=["pct_cambio_30d"])

        y_train = df_train_sin_nan["pct_cambio_30d"]
        X_train = df_train_sin_nan.drop(columns=["pct_cambio_30d"])

        # Imputación y escalado (sin selección de variables)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Hiperparámetros más conservadores para evitar overfitting
        best_params_xgb = {
            'colsample_bytree': 0.6,
            'learning_rate': 0.1,
            'max_depth': 1,
            'n_estimators': 50,
            'reg_alpha': 10,
            'reg_lambda': 10,
            'subsample': 0.6
        }

        # -----------------------------
        # Validación cruzada temporal sin data leakage
        # -----------------------------
        print("Validación cruzada con TimeSeriesSplit:")
        tscv = TimeSeriesSplit(n_splits=5)
        r2_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Imputación y escalado dentro del split (evita leakage)
            imputador_cv = SimpleImputer(strategy="mean")
            X_train_cv_imp = imputador_cv.fit_transform(X_train_cv)
            X_val_cv_imp = imputador_cv.transform(X_val_cv)

            scaler_cv = StandardScaler()
            X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv_imp)
            X_val_cv_scaled = scaler_cv.transform(X_val_cv_imp)

            modelo_cv = XGBRegressor(**best_params_xgb)
            modelo_cv.fit(X_train_cv_scaled, y_train_cv)

            pred_cv = modelo_cv.predict(X_val_cv_scaled)
            r2 = r2_score(y_val_cv, pred_cv)
            r2_scores.append(r2)

        
        # Entrenamiento con todos los datos
        modelo_xgb = XGBRegressor(**best_params_xgb)
        modelo_xgb.fit(X_train_scaled, y_train)
        evaluar_modelo_7d(modelo_xgb, X_train_scaled, y_train)

        # Evaluación en nuevos datos
        parquet_file_path_nuevos = get_path(
            "data", "parquet nuevos", "data_30d_nuevos.parquet"
        )
        df_nuevos = cargar_datos_30d(parquet_file_path_nuevos)
        df_nuevos_sin_nan = df_nuevos.dropna(subset=["pct_cambio_30d"])
        y_nuevos = df_nuevos_sin_nan["pct_cambio_30d"]
        X_nuevos = df_nuevos_sin_nan.drop(columns=["pct_cambio_30d"])

        for col in X_train.columns:
            if col not in X_nuevos.columns:
                X_nuevos[col] = np.nan

        X_nuevos = X_nuevos[X_train.columns]
        X_nuevos_escalado = scaler.transform(X_nuevos)
        print("Métricas XGBoost en datos nuevos:")
        evaluar_modelo_7d(modelo_xgb, X_nuevos_escalado, y_nuevos)

        # Reentrenamiento con todos los datos
        df_final = pd.concat([df_train_sin_nan, df_nuevos_sin_nan])
        y_final = df_final["pct_cambio_30d"]
        X_final = df_final.drop(columns=["pct_cambio_30d"])

        for col in X_train.columns:
            if col not in X_final.columns:
                X_final[col] = np.nan

        X_final = X_final[X_train.columns]
        X_scaled_final = scaler.fit_transform(X_final)

        modelo_xgb_final = XGBRegressor(**best_params_xgb)
        modelo_xgb_final.fit(X_scaled_final, y_final)

        os.makedirs("modelos", exist_ok=True)
        joblib.dump(modelo_xgb_final, "modelos/XGB.joblib")
        joblib.dump(scaler, "modelos/XGB_scaler.joblib")
        pred_final_total = modelo_xgb_final.predict(X_scaled_final)

        print("XGBoost (Final):")
        evaluar_modelo_30d(y_final, pred_final_total)


if __name__ == "__main__":
    # Entrenar, evaluar y reentrenar SVR
    entrenar_evaluar_y_reentrenar_modelo(tecnica="svr", test_size=0.2)

    # Entrenar, evaluar y reentrenar ARIMA
    entrenar_evaluar_y_reentrenar_modelo(tecnica="xgboost", test_size=0.2)