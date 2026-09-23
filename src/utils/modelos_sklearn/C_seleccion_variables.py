from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


def seleccionar_variables(X, y, tecnica, k_vars=0):
    """
    Selecciona variables en función de la técnica especificada.
    Para 'var', se seleccionan aquellas con alta correlación con la variable objetivo.

    :param X: DataFrame de características
    :param y: Serie objetivo
    :param tecnica: Nombre del modelo (e.g. 'svr', 'var', etc.)
    :return: X reducido y el selector (o None)
    """
    if tecnica == "svr":
        resultados = {}

        # Guardar los nombres de las columnas originales
        original_columns = X.columns

        # 1. Usando f_regression (relación lineal)
        selector_f = SelectKBest(score_func=f_regression, k=min(k_vars, X.shape[1]))
        X_kbest_f = selector_f.fit_transform(X, y)
        selected_columns_indices_f = selector_f.get_support()
        selected_columns_f = original_columns[selected_columns_indices_f]
        resultados["f_regression"] = pd.DataFrame(
            X_kbest_f, columns=selected_columns_f, index=X.index
        )

        # 2. Usando mutual_info_regression (relación no lineal)
        selector_mi = SelectKBest(
            score_func=mutual_info_regression, k=min(k_vars, X.shape[1])
        )
        X_kbest_mi = selector_mi.fit_transform(X, y)
        selected_columns_indices_mi = selector_mi.get_support()
        selected_columns_mi = original_columns[selected_columns_indices_mi]
        resultados["mutual_info"] = pd.DataFrame(
            X_kbest_mi, columns=selected_columns_mi, index=X.index
        )

        # 3. Todas las variables
        resultados["all"] = X.copy()

        return resultados

    elif tecnica == "lstm":
        """
        Selecciona variables usando Random Forest para la selección de características.
        Solo se seleccionan aquellas características cuya importancia sea mayor al umbral dado.
        Esta versión no realiza imputación de los valores faltantes.

        :param X: ndarray o DataFrame de características
        :param y: Serie objetivo (pandas Series)
        :param tecnica: Parámetro ignorado (solo para compatibilidad de firma)
        :param threshold: Umbral de importancia para la selección de características

        :return: (X_selected, selected_features)
        """
        threshold = 0.01
        # 1) Convertir X a DataFrame si viene como ndarray
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"var_{i}" for i in range(X.shape[1])])
        elif not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 2) Entrenar RandomForestRegressor sobre los datos “tal cual”
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # 3) Extraer importancias y filtrar según el umbral
        importances = rf.feature_importances_
        selected_features = X.columns[importances > threshold].tolist()

        # 4) Imprimir recuento y nombres
        print(f"\nNúmero de variables seleccionadas: {len(selected_features)}")
        print(f"Variables seleccionadas: {selected_features}")

        # 5) Devolver X reducido y la lista de nombres
        X_selected = X[selected_features]
        return X_selected, selected_features

    else:
        return X, None