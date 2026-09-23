"""
Módulo: extracción de datos históricos de Bitcoin desde la web de Bitget mediante scraping.

Este script automatiza la navegación del sitio web de Bitget para recopilar datos históricos de precios
de Bitcoin. Utiliza Selenium con Chrome en modo headless para recorrer las distintas páginas de la tabla
de datos, capturar el contenido y convertirlo en un DataFrame.

Se puede configurar:
- la URL de origen (por defecto, Bitget Bitcoin historical data),
- el número máximo de páginas a recorrer,
- el tiempo de espera entre acciones (para asegurar que los elementos carguen correctamente),
- y la ruta de salida para guardar los datos en formato CSV.


Este script forma parte del módulo de extracción de datos para análisis de criptomonedas.
"""

import time
import argparse
import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from utils.paths import get_path
from typing import Optional


def scrape_bitget_data(
    url: str,
    max_pages: Optional[int] = None,
    wait_time: int = 3,
    output_path: Optional[str] = None,
):
    """
    Extrae datos históricos de Bitcoin desde Bitget y los guarda en un archivo CSV.

    :param url: URL de la página de datos históricos de Bitget.
    :param max_pages: Número máximo de páginas a recorrer (None para todas).
    :param wait_time: Tiempo de espera entre interacciones para cargar los datos.
    :param output_path: Ruta del archivo CSV de salida.

    :return: pd.DataFrame con los datos históricos.
    """
    # Configurar Selenium con Chrome
    options = Options()
    options.add_argument("--headless")  # Modo sin interfaz gráfica
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    # Iniciar WebDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)

    # Esperar a que la tabla cargue completamente
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))

    # Listas para almacenar datos
    all_rows = []
    page_count = 0

    while True:
        time.sleep(wait_time)  # Esperar para que cargue la página

        try:
            table = driver.find_element(By.TAG_NAME, "table")
            rows = table.find_elements(By.TAG_NAME, "tr")
        except:
            print("Error al encontrar la tabla, reintentando...")
            continue

        # Obtener encabezados si es la primera página
        if not all_rows:
            headers = [
                th.text.strip() for th in rows[0].find_elements(By.TAG_NAME, "th")
            ]

        # Obtener datos de la tabla
        page_data = []
        for row in rows[1:]:  # Omitir encabezados
            try:
                cols = [td.text.strip() for td in row.find_elements(By.TAG_NAME, "td")]
                if cols:
                    page_data.append(cols)
            except:
                print("Fila descartada por error de referencia.")
                continue

        # Agregar datos de la página actual
        all_rows.extend(page_data)
        page_count += 1
        print(f"{page_count} páginas leidas")

        # Verificar si se alcanzó el límite de páginas
        if max_pages and page_count >= max_pages:
            print(f"Se alcanzó el límite de {max_pages} páginas. Finalizando...")
            break

        # Buscar botón "Next Page"
        try:
            next_button = driver.find_element(
                By.XPATH, "//li[contains(@class, 'ant-pagination-next')]"
            )

            # Si está deshabilitado, salir del bucle
            if "ant-pagination-disabled" in next_button.get_attribute("class"):
                print("Última página alcanzada. Finalizando...")
                break

            # Hacer clic en la flecha "Siguiente"
            driver.execute_script("arguments[0].click();", next_button)
            time.sleep(wait_time)
        except:
            print(
                "No se encontró la flecha de 'Siguiente' o ya estamos en la última página."
            )
            break

    # Cerrar el navegador
    driver.quit()

    # Convertir datos en DataFrame
    df_bitget = pd.DataFrame(all_rows, columns=headers)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_bitget.to_csv(output_path, encoding="utf-8-sig", index=False)
        print(f"Datos guardados en {output_path}")

    return df_bitget


def main():
    parser = argparse.ArgumentParser(
        description="Extraer datos históricos de Bitcoin desde Bitget"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://www.bitget.com/es/price/bitcoin/historical-data",
        help="URL de la página de datos históricos de Bitget",
    )
    parser.add_argument(
        "--max_pages",
        type=int,
        default=None,
        help="Número máximo de páginas a recorrer (None para todas)",
    )
    parser.add_argument(
        "--wait_time",
        type=int,
        default=3,
        help="Tiempo de espera entre interacciones para cargar los datos",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Ruta del archivo CSV de salida",
    )

    # Parsear los argumentos
    args = parser.parse_args()

    if args.output_path is None:
        output_path = get_path("data", "extraccion", "bitget_scrapping.csv")
    else:
        output_path = args.output_path

    # Llamar a la funcion con los datos parseados
    df_bitget = scrape_bitget_data(
        url=args.url,
        max_pages=args.max_pages,
        wait_time=args.wait_time,
        output_path=output_path,
    )

    print(df_bitget.shape)
    print(df_bitget.head())


if __name__ == "__main__":
    main()
