"""
Script para combinar datasets CSV y exportar los datos nuevos en formato Parquet.

Este script compara archivos CSV antiguos con archivos nuevos para extraer únicamente
las filas con fechas posteriores a las ya existentes. Posteriormente, guarda los datos
filtrados en formato Parquet para su uso eficiente en posteriores análisis o procesos.

Dependencias:
- os: para la manipulación de rutas y creación de directorios.
- pandas: para la lectura, filtrado y escritura de datos.
- utils.paths.get_path: función personalizada para construir rutas relativas
  a carpetas específicas del proyecto (por ejemplo, 'data/limpieza', 'data/nuevos', etc.).
- pyarrow: requerido por pandas para guardar archivos Parquet.

Funciones:
----------

combinar_csvs(csv_antiguo, csv_nuevo, columna_fecha):
    Combina dos archivos CSV, manteniendo únicamente las filas nuevas (con fechas posteriores).

    Parámetros:
    - csv_antiguo (str): Nombre del archivo CSV antiguo ubicado en 'data/limpieza/'.
    - csv_nuevo (str): Nombre del archivo CSV nuevo ubicado en 'data/nuevos/'.
    - columna_fecha (str): Nombre de la columna que contiene las fechas (formato datetime).

    Comportamiento:
    - Lee ambos archivos CSV.
    - Filtra del nuevo CSV solo las filas con fecha mayor que la última del CSV antiguo.
    - Sobrescribe el archivo CSV nuevo con esas filas nuevas.

    Retorna:
    - pd.DataFrame: DataFrame con los datos filtrados del CSV nuevo.

guardar_dataframe_como_parquet(df, salida_parquet):
    Guarda un DataFrame como archivo Parquet en la carpeta 'data/parquet nuevos/'.

    Parámetros:
    - df (pd.DataFrame): El DataFrame a guardar.
    - salida_parquet (str): Nombre del archivo Parquet de salida.

    Comportamiento:
    - Crea la carpeta de destino si no existe.
    - Guarda el DataFrame como archivo Parquet usando el engine 'pyarrow'.

Bloque principal:
-----------------
Si el script se ejecuta directamente, combina y guarda en Parquet tres pares de datasets:
1. dataset_30d.csv + dataset_30d_nuevos.csv → data_30d_nuevos.parquet
2. cleaned_data.csv + cleaned_data_nuevos.csv → cleaned_data_nuevos.parquet
3. dataset_7d.csv + dataset_7d_nuevos.csv → data_7d_nuevos.parquet
"""

import os
import pandas as pd
from utils.paths import get_path


def combinar_csvs(csv_antiguo, csv_nuevo, columna_fecha):
    """
    Combina dos archivos CSV, filtrando las fechas del nuevo CSV.

    :param csv_antiguo: Ruta al CSV antiguo.
    :param csv_nuevo: Ruta al CSV nuevo.
    :param columna_fecha: Nombre de la columna de fechas.
    :param offset_dias: Número de días para ajustar las fechas del nuevo CSV.
                        Si es positivo, las fechas se desplazan hacia el futuro.
                        Si es negativo, las fechas se desplazan hacia el pasado.
                        Por defecto es 0 (sin desplazamiento).
    :return: DataFrame combinado.
    """
    # Leer ambos CSV
    df_antiguo = pd.read_csv(
        get_path("data", "limpieza", csv_antiguo), parse_dates=[columna_fecha]
    )
    df_nuevo = pd.read_csv(
        get_path("data", "nuevos", csv_nuevo), parse_dates=[columna_fecha]
    )

    # Obtener la fecha máxima del DataFrame original
    ultima_fecha = df_antiguo[columna_fecha].max()

    df_nuevo_filtrado = df_nuevo[df_nuevo[columna_fecha] > ultima_fecha]

    csv_path = get_path("data", "nuevos", csv_nuevo)
    os.makedirs(
        os.path.dirname(csv_path), exist_ok=True
    )  # Crear el directorio si no existe
    df_nuevo_filtrado.to_csv(csv_path, index=False)

    return df_nuevo_filtrado


def guardar_dataframe_como_parquet(df, salida_parquet):
    """
    Guarda un DataFrame como archivo Parquet.

    :param df: DataFrame a guardar.
    :param salida_parquet: Ruta de salida del archivo Parquet.
    """
    parquet_path = get_path("data", "parquet nuevos", salida_parquet)  # Cambio de ruta
    os.makedirs(
        os.path.dirname(parquet_path), exist_ok=True
    )  # Crear el directorio si no existe
    df.to_parquet(parquet_path, engine="pyarrow", index=False)  # Se agrega index=False
    print(f"DataFrame guardado en: {parquet_path}")


if __name__ == "__main__":

    df_combinado_30d = combinar_csvs(
        csv_antiguo="dataset_30d.csv",
        csv_nuevo="dataset_30d_nuevos.csv",
        columna_fecha="timestamp",
    )
    guardar_dataframe_como_parquet(
        df=df_combinado_30d, salida_parquet="data_30d_nuevos.parquet"
    )

    df_combinado_cleaned = combinar_csvs(
        csv_antiguo="cleaned_data.csv",
        csv_nuevo="cleaned_data_nuevos.csv",
        columna_fecha="timestamp",
    )

    guardar_dataframe_como_parquet(
        df=df_combinado_cleaned, salida_parquet="cleaned_data_nuevos.parquet"
    )

    df_combinado_7d = combinar_csvs(
        csv_antiguo="dataset_7d.csv",
        csv_nuevo="dataset_7d_nuevos.csv",
        columna_fecha="timestamp",
    )
    guardar_dataframe_como_parquet(
        df=df_combinado_7d, salida_parquet="data_7d_nuevos.parquet"
    )
