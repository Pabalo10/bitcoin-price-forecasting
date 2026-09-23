"""
Este script obtiene los datos de las tendencias de Google para un conjunto de palabras clave especificadas. Utiliza la librería `pytrends` para interactuar con la API de Google Trends y obtiene información
sobre la popularidad de búsqueda de las palabras clave a lo largo del tiempo.

Parámetros:
    - keywords: Lista de palabras clave para buscar en Google Trends (por defecto, "bitcoin").
    - timeframe: Intervalo de tiempo para los datos de tendencias (por defecto, "today 5-y").
    - geo: Código geográfico para restringir los resultados (por defecto, vacío para todos los países).
    - output_path: Ruta donde guardar los datos como archivo CSV (opcional).

Funciones principales:
    - get_google_trends: Solicita los datos de Google Trends para las palabras clave y devuelve un DataFrame con los resultados.
    - main: Función principal que maneja la interacción con el usuario a través de la línea de comandos, solicitando los parámetros de entrada y llamando a `get_google_trends` para obtener los datos.
"""

import time
from utils.paths import get_path
from pytrends.request import TrendReq
from typing import List, Optional
import argparse
import requests
import os


def get_google_trends(
    keywords: List[str] = ["bitcoin"],
    timeframe: str = "today 5-y",
    geo: str = "",
    output_path: Optional[str] = None,
):
    """
    Obtiene los datos de tendencias de Google para las palabras clave especificadas.

    :param keywords: Un conjunto de palabras clave a buscar (por defecto, {'bitcoin'}).
    :param timeframe: El intervalo de tiempo para los datos de tendencias (por defecto, "today 5-y").
    :param geo: El código geográfico para restringir los resultados (por defecto, vacío para todos los países).
    :param output_path: Ruta donde guardar los datos como un archivo CSV (opcional).

    :return: Un DataFrame con los resultados de las tendencias de Google.
    """
    pytrends = TrendReq()

    try:
        # Solicitar datos de Google Trends
        time.sleep(5)
        pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
        df = pytrends.interest_over_time().reset_index()

        if df.empty:
            raise ValueError(
                "No se recibieron datos. Puede que Google esté bloqueando la solicitud."
            )

        # Renombrar y guardar
        df.rename(columns={"date": "timestamp"}, inplace=True)
        df = df[["timestamp"] + keywords]

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"Datos guardados en: {output_path}")

        return df

    except (requests.exceptions.RequestException, ValueError) as e:
        # Captura errores de solicitud y de falta de datos
        raise e


def main():
    parser = argparse.ArgumentParser(
        description="Obtener datos de tendencias de Google para las palabras clave especificadas."
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default="bitcoin",
        help="Palabras clave a buscar, separadas por coma (por defecto: 'bitcoin').",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="today 5-y",
        help="Intervalo de tiempo para los datos de tendencias (por defecto: 'today 5-y').",
    )
    parser.add_argument(
        "--geo",
        type=str,
        default="",
        help="Código geográfico para restringir los resultados (por defecto: todos los países).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Ruta donde guardar los datos como CSV (opcional).",
    )

    # Parsear los argumentos
    args = parser.parse_args()

    if args.output_path is None:
        output_path = get_path("data", "extraccion", "datos_google_trends.csv")
    else:
        output_path = args.output_path

    # Convertir la cadena de keywords en un list
    keywords = list(args.keywords.split(","))

    # Llamar a la función con los parámetros proporcionados
    google_trends_df = get_google_trends(
        keywords=keywords,
        timeframe=args.timeframe,
        geo=args.geo,
        output_path=output_path,
    )

    print(google_trends_df.iloc[0, 0])
    print(google_trends_df.iloc[-1, 0])
    print(google_trends_df.shape)
    print(google_trends_df.head())


if __name__ == "__main__":
    main()
