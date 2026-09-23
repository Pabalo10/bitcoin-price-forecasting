"""
Este script limpia y organiza los datos del índice de "Miedo y Codicia" (Fear & Greed Index)
relacionado con Bitcoin.

Se carga el archivo CSV, se convierte la columna `timestamp` a formato de fecha y se ordenan
los registros cronológicamente. No se detectan valores nulos, por lo que no se requiere imputación
ni eliminación.

Finalmente, los datos se guardan en un nuevo archivo CSV para su uso en análisis posteriores.

"""

import os
import pandas as pd
from utils.paths import get_path

if __name__ == "__main__":
    # Cargar datos desde un CSV (ajusta la ruta a tu archivo)
    df = pd.read_csv(get_path("data", "extraccion", "datos_bitcoin_fear_greed.csv"))

    # Ver las primeras filas
    print(df.head())
    # Comprobar el tipo de las variables
    print(df.dtypes)

    # Convertir timestamp a fecha
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # Convertir a formato fecha
    df.sort_values("timestamp", inplace=True)  # Ordenar por fecha

    # Ver las primeras filas
    print(df.head())
    # Comprobar el tipo de las variables
    print(df.dtypes)

    print(df.isnull().sum())  # no hay nulos

    output_path = get_path("data", "limpieza", "limpios_fear_greed.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
