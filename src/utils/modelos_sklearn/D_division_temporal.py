import pandas as pd


def dividir_datos_temporalmente_7d(X, y, test_size=0.1):
    """
    Divide los datos en train+val (para validación cruzada temporal) y test (últimos datos).

    :param X: Matriz de características
    :param y: Variable objetivo
    :param test_size: Proporción del conjunto test

    :return: X_train_val, y_train_val, X_test, y_test
    """
    n = len(X)
    test_start = int(n * (1 - test_size))

    X_train_val = X[:test_start]
    y_train_val = y[:test_start]
    X_test = X[test_start:]
    y_test = y[test_start:]

    return X_train_val, y_train_val, X_test, y_test


def dividir_datos_temporalmente_30d(df, target_col, test_ratio=0.2, multivariado=False):
    n_test = int(len(df) * test_ratio)
    train = df.iloc[:-n_test]
    test = df.iloc[-n_test:]

    if multivariado:
        return train, test
    else:
        y_train = train[target_col]
        y_test = test[target_col]
        return y_train, y_test
