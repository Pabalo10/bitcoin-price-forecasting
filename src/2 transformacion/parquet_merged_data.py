"""
Script para convertir archivos CSV procesados a formato Parquet.

Pasos:
1. Carga archivos CSV ya limpios (cleaned_data.csv, dataset_7d.csv, dataset_30d.csv).
2. Convierte cada uno a Parquet.
3. Guarda los archivos optimizados en el directorio correspondiente.
4. Muestra las primeras filas como verificación.

IMPORTANTE: Ejecutar primero los notebooks de análisis (eliminación de variables y estudio de correlaciones) antes de correr este script.
"""

import os
import pandas as pd
from utils.paths import get_path


def convertir_csv_a_parquet(nombre_csv, nombre_parquet):
    """
    Convierte un archivo CSV a Parquet y muestra sus primeras filas.

    :param nombre_csv: Ruta relativa al archivo CSV desde la raíz del proyecto.
    :param nombre_parquet: Ruta relativa donde guardar el archivo Parquet.
    """
    input_path = get_path(*nombre_csv)
    output_path = get_path(*nombre_parquet)

    # Crear carpeta de salida si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Cargar CSV, guardar como Parquet y mostrar contenido
    df = pd.read_csv(input_path)
    df.to_parquet(output_path, engine="pyarrow")
    df_parquet = pd.read_parquet(output_path)
    print(f"\n Guardado: {output_path}")
    print(df_parquet.head())


if __name__ == "__main__":
    convertir_csv_a_parquet(
        ["data", "limpieza", "cleaned_data.csv"],
        ["data", "parquet", "data.parquet"],
    )

    convertir_csv_a_parquet(
        ["data", "limpieza", "dataset_7d.csv"],
        ["data", "parquet", "data_7d.parquet"],
    )

    convertir_csv_a_parquet(
        ["data", "limpieza", "dataset_30d.csv"],
        ["data", "parquet", "data_30d.parquet"],
    )
