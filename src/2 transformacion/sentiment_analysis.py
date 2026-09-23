"""
Este script realiza un análisis de sentimiento sobre un conjunto de tweets relacionados con Bitcoin.

1. Carga un archivo CSV que contiene los tweets.

2. Utiliza la librería TextBlob para analizar el sentimiento de cada tweet, clasificándolos como 'Positivo',
 'Negativo' o 'Neutral'.

3. Muestra las estadísticas del análisis de sentimiento (conteo de cada tipo de sentimiento).

4. Guarda el conjunto de datos con la columna adicional de sentimiento en un nuevo archivo CSV para su
 posterior uso.

"""

import pandas as pd
from textblob import TextBlob
from utils.paths import get_path
import os


def get_sentiment(text):
    """Analiza el sentimiento de un texto y lo clasifica."""
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positivo"
    elif polarity < 0:
        return "Negativo"
    else:
        return "Neutral"


def analyze_tweets(file_path):
    """Carga el archivo CSV, analiza el sentimiento de los tweets y muestra estadísticas."""
    # Cargar datos
    df = pd.read_csv(file_path)

    # Asegurar que la columna de texto es string
    df["Texto"] = df["Texto"].astype(str)

    # Aplicar análisis de sentimiento
    df["Sentimiento"] = df["Texto"].apply(get_sentiment)

    # Mostrar resultados
    print(df["Sentimiento"].value_counts())
    return df


if __name__ == "__main__":
    # Ruta del archivo (modificar según sea necesario)
    file_path = get_path("data", "extraccion", "tweets_bitcoin.csv")
    df_resultado = analyze_tweets(file_path)
    print(df_resultado.head())

    # Guardar el conjunto de datos
    output_path = get_path("data", "limpieza", "sentiment_analysis.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_resultado.to_csv(output_path, index=False)
