from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
import numpy as np


def ajustar_modelo(
    modelo,
    X,
    y,
    param_grid,
    strategy="grid",
    scoring="r2",
    n_splits=5,
    random_state=42,
    n_iter=20,
    verbose=1,
):
    """
    Ajusta un modelo sklearn usando validación cruzada temporal.

    :param modelo: Instancia del modelo sin entrenar
    :param X: Matriz de características (estandarizadas)
    :param y: Vector objetivo
    :param param_grid: Diccionario de hiperparámetros
    :param strategy: 'grid' o 'random'
    :param scoring: Métrica de evaluación
    :param n_splits: Número de divisiones temporales
    :param random_state: Para reproducibilidad en RandomizedSearchCV
    :param verbose: Nivel de detalle

    :return: Mejor modelo ajustado, mejores parámetros
    """
    cv = TimeSeriesSplit(n_splits=n_splits)

    if strategy == "grid":
        search = GridSearchCV(
            estimator=modelo,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=verbose,
        )
    else:
        search = RandomizedSearchCV(
            estimator=modelo,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=verbose,
            random_state=random_state,
        )

    search.fit(X, y)

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_

    if verbose:
        print(f"\nMejor puntuación CV ({scoring}): {best_score:.5f}")
        print(f"Mejores parámetros encontrados: {best_params}")

    return best_model, best_params


def validar_modelo_timeseries(
    y, param_grid, model_class, seasonal=False, seasonal_param_grid=None, n_splits=3
):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_mse = np.inf
    best_params = None
    best_model = None

    # Si es un modelo estacional y hay una grilla de parámetros estacionales
    if seasonal and seasonal_param_grid:
        for order in param_grid:
            for seasonal_order in seasonal_param_grid:
                fold_mse = []
                for train_idx, test_idx in tscv.split(y):
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    try:
                        model = model_class(
                            y_train, order=order, seasonal_order=seasonal_order
                        )
                        model_fit = model.fit(disp=False)
                        pred = model_fit.predict(
                            start=y_test.index[0], end=y_test.index[-1]
                        )
                        mse = np.mean((y_test - pred[: len(y_test)]) ** 2)
                        fold_mse.append(mse)
                    except Exception:
                        # Si ocurre un error, asigna infinito al error de ese split
                        fold_mse.append(np.inf)

                avg_mse = np.mean(fold_mse)
                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_params = (order, seasonal_order)
                    best_model = model_class(
                        y, order=order, seasonal_order=seasonal_order
                    ).fit(disp=False)
    else:
        # Sin parámetros estacionales
        for order in param_grid:
            fold_mse = []
            for train_idx, test_idx in tscv.split(y):
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                try:
                    model = model_class(y_train, order=order)
                    model_fit = model.fit()
                    pred = model_fit.predict(
                        start=y_test.index[0], end=y_test.index[-1]
                    )
                    mse = np.mean((y_test - pred[: len(y_test)]) ** 2)
                    fold_mse.append(mse)
                except Exception:
                    fold_mse.append(np.inf)

            avg_mse = np.mean(fold_mse)
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_params = order
                best_model = model_class(y, order=order).fit()

    print(best_params)
    return best_model, best_params
