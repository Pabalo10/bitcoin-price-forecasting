from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
)
import numpy as np


def evaluar_modelo_7d(modelo, X_test, y_test):
    predictions = modelo.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mse**0.5
    r2 = r2_score(y_test, predictions)

    # Accuracy direccional (signo)
    pred_sign = np.sign(predictions)
    real_sign = np.sign(y_test)
    directional_acc = accuracy_score(real_sign, pred_sign)

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Accuracy Direccional: {directional_acc:.4f}")

    return mse, mae, rmse, r2, directional_acc


def evaluar_modelo_30d(y_test, pred):
    mse = np.mean((y_test - pred[: len(y_test)]) ** 2)
    mae = np.mean(np.abs(y_test - pred[: len(y_test)]))
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred[: len(y_test)])
    acc = np.mean(np.sign(y_test) == np.sign(pred[: len(y_test)]))

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Accuracy Direccional: {acc:.4f}")

    return mse, mae, rmse, r2, acc
