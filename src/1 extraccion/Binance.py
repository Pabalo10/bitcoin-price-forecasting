"""
Módulo: extracción de datos históricos de Bitcoin desde Binance.

Este script se conecta a la API pública de Binance para obtener datos históricos
de precios en formato OHLCV (Open, High, Low, Close, Volume), junto con el número de transacciones.
Se puede especificar el par de trading (por defecto BTCUSDT), el intervalo de tiempo,
el número de registros a recuperar, y opcionalmente un rango temporal específico.

Los datos se pueden guardar automáticamente en un archivo CSV para su posterior análisis
financiero, modelado o visualización.

Este script forma parte del conjunto de herramientas de extracción de datos para análisis del mercado de criptomonedas.
"""

import requests
import pandas as pd
import argparse
import os
from utils.paths import get_path
from typing import Optional


def get_binance_data(
    symbol: str = "BTCUSDT",
    interval: str = "1d",
    limit: int = 1000,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    output_path: Optional[str] = None,
):
    """
    Obtiene datos históricos de Binance.

    :param symbol: (str) Par de trading (ej. 'BTCUSDT').
    :param interval: (str) Intervalo de tiempo ('1m', '1h', '1d', etc.).
    :param limit: (int) Número de registros a obtener.
    :param start_time: (int, opcional) Tiempo de inicio en milisegundos.
    :param end_time: (int, opcional) Tiempo de fin en milisegundos.
    :param output_path: (str, opcional) Ruta donde guardar el CSV.
    :return: pd.DataFrame con los datos obtenidos.
    """

    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
    }
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    response = requests.get(url, params=params)
    response.raise_for_status()  # Lanza un error si la solicitud falla
    data = response.json()

    df = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    # Cambiar los tipos de datos
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Seleccionar solo las columnas necesarias
    df = df[["timestamp", "open", "high", "low", "close", "volume", "trades"]]

    # Guardar CSV si se proporciona una ruta
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Datos guardados en: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Obtener datos históricos de Binance")
    parser.add_argument(
        "--symbol", type=str, default="BTCUSDT", help="Par de trading (ej. 'BTCUSDT')"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Intervalo de tiempo ('1m', '1h', '1d', etc.)",
    )
    parser.add_argument(
        "--limit", type=int, default=1000, help="Número de registros a obtener"
    )
    parser.add_argument(
        "--start_time", type=int, default=None, help="Tiempo de inicio en milisegundos"
    )
    parser.add_argument(
        "--end_time", type=int, default=None, help="Tiempo de fin en milisegundos"
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
        output_path = get_path("data", "extraccion", "datos_bitcoin_binance.csv")
    else:
        output_path = args.output_path

    # Llamar a la funcion con los datos parseados
    binance_df = get_binance_data(
        symbol=args.symbol,
        interval=args.interval,
        limit=args.limit,
        start_time=args.start_time,
        end_time=args.end_time,
        output_path=output_path,
    )

    print(binance_df.shape)
    print(binance_df.head())


if __name__ == "__main__":
    main()
