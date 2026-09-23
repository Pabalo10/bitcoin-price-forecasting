"""
coingecko.py

Este script obtiene datos históricos de precios, capitalización de mercado y volumen
de una criptomoneda específica (por defecto Bitcoin) desde la API pública de CoinGecko.

Permite especificar la criptomoneda, la moneda base (usd, eur), y la cantidad de días
de historial a recuperar (hasta 365 días en la versión gratuita). Los datos se guardan
en un archivo CSV si se proporciona una ruta de salida.

"""

import requests
import argparse
import pandas as pd
import os
from utils.paths import get_path
from typing import Optional


def get_coingecko_data(
    symbol: str = "bitcoin",
    vs_currency: str = "usd",
    days: int = 365,
    output_path: Optional[str] = None,
):
    """
    Obtiene datos históricos de criptomonedas desde CoinGecko.

    :param symbol: (str) criptomoneda a consultar (ej. ['bitcoin', 'ethereum']).
    :param vs_currency: (str): Moneda base ('usd', 'eur', etc.).
    :param days: (int) Número de días de datos a obtener.
    :param output_path: (str, opcional): Ruta donde guardar el CSV.

    :return: pd.DataFrame con los datos obtenidos.
    """

    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}

    response = requests.get(url, params=params)
    response.raise_for_status()  # Lanza un error si la solicitud falla
    data = response.json()

    # Convertir datos a DataFrame
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["market_cap"] = [x[1] for x in data["market_caps"]]
    df["volumen"] = [x[1] for x in data["total_volumes"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Guardar CSV si se proporciona una ruta
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Datos guardados en: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Obtener datos históricos de criptomonedas desde CoinGecko"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="bitcoin",
        help="criptomoneda a consultar (ej. ['bitcoin', 'ethereum'])",
    )
    parser.add_argument(
        "--vs_currency",
        type=str,
        choices=["usd", "eur"],
        default="usd",
        help="Moneda base ('usd', 'eur')",
    )
    parser.add_argument(
        "--days", type=int, default=365, help="Número de días de datos a obtener"
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
        output_path = get_path("data", "extraccion", "datos_coingecko.csv")
    else:
        output_path = args.output_path

    # Llamar a la funcion con los datos parseados
    coingecko_df = get_coingecko_data(
        symbol=args.symbol,
        vs_currency=args.vs_currency,
        days=args.days,
        output_path=output_path,
    )

    print(coingecko_df.shape)
    print(coingecko_df.head())


if __name__ == "__main__":
    main()
