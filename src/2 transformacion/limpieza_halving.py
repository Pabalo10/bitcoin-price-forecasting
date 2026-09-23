"""
Este script limpia los datos relacionados con los eventos de halving de Bitcoin.
Se convierte la columna de fechas al formato datetime y se renombra a 'timestamp'
para mantener la consistencia con el resto de datasets del proyecto.
El archivo resultante se guarda en formato CSV para facilitar su uso en análisis
temporales y la integración con otros conjuntos de datos.

"""

from utils.paths import get_path
import pandas as pd
import os


if __name__ == "__main__":
    df = pd.read_csv(get_path("data", "extraccion", "halving_data.csv"))

    print(df.info())  # Ver tipos de datos y valores nulos
    print(df.describe())  # Estadísticas básicas
    print(df.head())  # Primeras filas

    # convertimos la columna timestamp a datetime
    df["Fecha"] = pd.to_datetime(df["Fecha"]).dt.date
    df = df.rename(columns={"Fecha": "timestamp"})  # cambio el nombre para que coincida

    output_path = get_path("data", "limpieza", "limpios_halving.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, encoding="utf-8-sig", index=False)
    print(df)
