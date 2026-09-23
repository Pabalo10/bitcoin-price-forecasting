"""
Módulo: extracción del índice de Miedo y Codicia (Fear & Greed Index) de Bitcoin.

Este script se conecta a la API pública de alternative.me para obtener datos históricos
del índice de Miedo y Codicia relacionado con el mercado de criptomonedas (principalmente Bitcoin).
Permite configurar el número de registros a recuperar, el formato de fechas y la ruta de salida
para guardar los datos en formato CSV.

Este módulo forma parte de un conjunto de scripts dedicados a la extracción de datos para análisis financiero y de criptomonedas.
"""

import requests
import pandas as pd
import argparse
import os
from utils.paths import get_path
from typing import Optional


def get_fear_greed_index(
    limit: int = 0, date_format: str = "world", output_path: Optional[str] = None
):
    """
    Obtiene el índice de Miedo y Codicia desde la API de alternative.me.

    Parámetros:
    - limit (int): Número de registros a obtener (0 para todos).
    - date_format (str): Formato de fecha ('world' o 'us').
    - output_path (str, opcional): Ruta donde guardar el CSV.

    Retorna:
    - pd.DataFrame con los datos obtenidos.
    """

    url = f"https://api.alternative.me/fng/?limit={limit}&format=json&date_format={date_format}"

    response = requests.get(url)
    response.raise_for_status()  # Verifica si la solicitud fue exitosa
    data = response.json().get("data", [])

    df = pd.DataFrame(data)

    # Convertir 'timestamp' a formato de fecha, asegurando que se interprete correctamente
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], dayfirst=(date_format == "world"), errors="coerce"
    )

    # Seleccionar y renombrar columnas
    df = df[["timestamp", "value"]]
    df.rename(columns={"value": "fear_greed_index"}, inplace=True)

    # Asegurar que el índice de miedo y codicia sea numérico
    df["fear_greed_index"] = pd.to_numeric(df["fear_greed_index"], errors="coerce")

    # Guardar CSV si se proporciona una ruta
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Datos guardados en: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Obtener el índice de Miedo y Codicia de Bitcoin"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Número de registros a obtener (0 para todos)",
    )
    parser.add_argument(
        "--date_format",
        type=str,
        choices=["world", "us"],
        default="world",
        help="Formato de fecha",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Ruta donde guardar el CSV",
    )

    # Parsear los argumentos
    args = parser.parse_args()

    if args.output_path is None:
        output_path = get_path("data", "extraccion", "datos_bitcoin_fear_greed.csv")
    else:
        output_path = args.output_path

    # Llamar a la función con los parámetros proporcionados
    fear_greed_df = get_fear_greed_index(
        limit=args.limit, date_format=args.date_format, output_path=output_path
    )

    print(fear_greed_df.shape)
    print(fear_greed_df.head())


if __name__ == "__main__":
    main()
