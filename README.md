# CryptoVision Insights
![LOGO](https://img.bitgetimg.com/multiLang/web/ebb606b3d63bb1f7d48e7ccdfbb984c1.png)

# Descripción de los objetivos.
El proyecto consiste en un sistema de **predicción de tendencias de Bitcoin a través de su porcentaje de cambio sobre el precio de cierre**, que permite analizar los movimientos del mercado y prever posibles fluctuaciones en el precio de la criptomoneda. Para ello, se utilizan datos en tiempo real, junto con modelos de análisis de datos.

El objetivo principal es predecir la tendencia de Bitcoin en base a los datos históricos, patrones y a la evolución de la moneda utilizando técnicas de aprendizaje automático. (Proyecto Universitario)
# Estructura del repositorio

```
CryptoVision-Insights/
├── src/                         # Código fuente principal organizado por etapas del flujo de datos
│   ├── 1 extraccion/            # Scripts para extraer datos desde múltiples fuentes
│   │   ├── Twitter/             # Extracción específica desde la API de Twitter
│   │   │   ├── config.py        # Configuración y autenticación para Twitter API
│   │   │   └── Twitter.py       # Script de conexión y extracción de tweets
│   │   ├── Alternative.py       # Fuente alternativa de métricas de mercado (miedo y codicia del bitcoin)
│   │   ├── Binance.py           # Datos históricos de precios de Binance de Bitcoin
│   │   ├── Bitget.py            # Datos de exchange de la web de Bitget sobre Bitcoin
│   │   ├── CoinGecko.py         # API pública de criptomonedas para extraer datos de Bitcoin
│   │   ├── Google_Trends.py     # Tendencias de búsqueda sobre cripto y, en especial, Bitcoin
│   │   ├── Halving.py           # Información sobre eventos de halving de Bitcoin
│   │   ├── Hashrate.py          # Métricas de poder de minado de Bitcoin
│   │   ├── mercadoDerivados_binance.py  # Datos de derivados desde Binance sobre Bitcoin
│   │   ├── mercadoDerivados_deribit.py  # Datos de derivados desde Deribit sobre Bitcoin
│   │   └── Yahoo_Finance.py     # Datos financieros tradicionales (Nasdaq, SP500...) desde Yahoo Finance
│
│   ├── 2 transformacion/        # Procesamiento y limpieza de los datos crudos
│   │   ├── creacion_variables.py                # Creación de nuevas variables derivadas para su posterior uso
│   │   ├── limpieza_*.py                        # Scripts de limpieza por fuente de datos
│   │   ├── merge.py                             # Fusión de todas las fuentes en un dataset único
│   │   ├── parquet_merged_data.py               # Exportación a formato parquet
│   │   └── sentiment_analysis.py                # Análisis de sentimiento de los tweets relacionados con Bitcoin
│
│   ├── 3 analisis/              # Análisis exploratorio y visualización
│   │   ├── visualizacion merged datos.ipynb     # Visualización de datos ya fusionados
│   │   ├── visualización creacion variables.ipynb  # Análisis de variables generadas
│   │   └── visualización variable respuesta.ipynb  # Análisis de la variable objetivo
│
│   ├── 4 modelos/               # Modelado predictivo con diferentes algoritmos
│   │   ├── 7_dias/              # Modelos con ventana de predicción de 7 días
│   │   │   ├── ARBOLES.py       # Modelos tipo árbol de regresión
│   │   │   ├── LSTM.py          # Red neuronal (LSTM)
│   │   │   └── SVR.py           # Support Vector Regression
│   │   ├── 30_dias/             # Modelos con ventana de 30 días
│   │   │   ├── ARIMA.py         # Modelo Arima
│   │   │   ├── SARIMA.py        # Modelo Sarima
│   │   │   └── XGBoost.py       # Modelo XGBoost
│   │   ├── baseline.py          # Comparación con modelos base
│   │   ├── eliminación de variables.ipynb  # Feature selection (se ha de ejecutar antes de los modelos)
│   │   └── estudio de correlaciones.ipynb  # Análisis de correlación entre variables
│
│   ├── 5 pipeline/              # Automatización de tareas: entrenamiento y predicción
│   │   ├── modelos/             # Modelos entrenados y sus escaladores
│   │   │   ├── SVR.joblib
│   │   │   ├── SVR_scaler.joblib
│   │   │   ├── XGB.joblib
│   │   │   └── XGB_scaler.joblib
│   │   ├── actualizar_datos.py  # Script para actualizar predicciones automáticamente y obtener los nuevos datos
│   │   └── reentrenamiento.py   # Reentrenamiento automático del modelo
│
│   └── 6 nuevos datos/          # Preparación de nuevos datos para predicciones en tiempo real
│       ├── creacion_nuevos_datos.py                        # Creacion de nuevos datos (ejecutar como tercer script del apartado 6)
│       ├── eliminación de variables nuevos datos.ipynb     # Eliminación de variables (ejecutar como primer script del apartado 6)
│       └── merge_nuevos_datos.py                           # Unión de los nuevos datos (ejecutar como segundo script del apartado 6)
│
├── utils/                       # Funciones utilitarias y herramientas de ML
│   ├── arquitecturas/
│   │   ├── __init__.py
│   │   └── lstm_model.py        # Definición de arquitectura LSTM personalizada
│   ├── modelos_sklearn/         # Pipeline modular para modelos sklearn
│   │   ├── __init__.py
│   │   ├── A_cargar_datos.py             # Carga de datos
│   │   ├── B_preprocesamiento.py         # Preprocesamiento general
│   │   ├── C_seleccion_variables.py      # Selección de variables
│   │   ├── D_division_temporal.py        # División de datos en sets
│   │   ├── E_ajuste_hiperparametros.py   # Optimización de modelos
│   │   ├── F_metricas.py                 # Métricas de evaluación
│   │   └── G_graficar_predicciones.py    # Visualización de predicciones
│   ├── mlflow_utils.py          # Integración con MLflow para tracking
│   └── paths.py                 # Rutas comunes para acceso a datos y modelos
│
├── pyproject.toml               # Configuración del proyecto para herramientas modernas
├── uv.lock                      # Archivo de bloqueo de versiones con uv
├── .gitignore                   # Archivos ignorados por Git
└── README.md                    # Documentación del proyecto (este archivo)
 ```
# Instrucciones para iniciar el entorno de desarrollo con sus dependencias.

El proyecto está desarrollado en **Python 3.10**, utilizando Selenium para la automatización de la extracción de datos. Para ejecutar o modificar el código, se recomienda utilizar **Jupyter Notebook** o un entorno de desarrollo como **VS Code** o **PyCharm**.
La carpeta de datos del [drive](https://drive.google.com/drive/folders/1rx4XvYLFwyn86TofruAenmslBwZ_eMg2?usp=sharing).

Para la correcta ejecución del proyecto, es necesario instalar las siguientes dependencias:

1. **Google Chrome**: Navegador web necesario para la automatización.
   - Instalación: [https://www.google.com/chrome/](https://www.google.com/chrome/)
2. **ChromeDriver**: Controlador que permite a Selenium interactuar con Chrome.
   - Instalación: [https://chromedriver.chromium.org/downloads](https://chromedriver.chromium.org/downloads)
3. **Instalación del Kernel de Jupyter en el entorno virtual**:  
   Para el pipeline correctamente asegurando la ejecución de los notebooks ejecutar el siguiente comando en la terminal con el entorno virtual activado
   ```bash
   python -m ipykernel install --user --name=c2425-r6 --display-name "Python (c2425-R6)"
  
Se debe ejecutar este código en la terminal para la configuración entorno virtual:
```bash
uv sync
```
# Instrucciones para ejecutar los scripts del proyecto.
Para la correcta ejecución de los scrpits del proyecto se ha de seguir el siguiente orden:
1. **Extracción:**  
   En esta primera parte del proyecto no hay ningún script que tenga que ser ejecutado antes que otro, pero debes ejecutar todos ellos antes de pasar a la siguiente fase.
   Los ficheros de Google Trends, Twitter y Yahoo Finance pueden dar error de Too Many Requests si se ejecutan más de un número de veces en el mismo día.

2. **Transformación:**  
   Para esta segunda, se han debido ejecutar todos los scripts de la fase anterior.  
   Y además, dentro de esta fase sí que se deben ejecutar antes todos los scripts de limpieza junto a _sentiment_analysis.py_ antes de ejecutar cualquier otro script.  

   Posteriormente se debe ejecutar el script _merge.py_.  
   Es indiferente cuándo se ejecuta el script _creacion_variables.py_ debido a que solo contiene funciones que llaman los scripts de limpieza.  
   Y no se debe ejecutar en ningún caso _parquet_merged_data.py_; se ejecutará más adelante cuando se indique.


3. **Análisis:**  
   Para esta tercera fase, se deben haber ejecutado las dos fases anteriores.  
   Dentro de esta es indiferente el orden de ejecución de los scripts.


4. **Modelos:**  
   Para esta cuarta fase, se deben haber ejecutado las tres fases anteriores.  
   Y dentro de esta, se debe seguir el siguiente orden:  
   Primero se debe ejecutar el notebook _eliminacion de variables.ipynb_,  
   y a continuación se debe ejecutar el script del apartado de transformación que nos saltamos anteriormente (_parquet_merged_data.py_),  
   y ya por último se puede ejecutar cualquier script de los modelos (ya sean de 7 o 30 días) o el _baseline.py_ indiferentemente.  

   El notebook _estudio de correlaciones.ipynb_ se puede ejecutar en cualquier momento ya que no depende de ningún script.


5. **Pipeline:**  
   Para esta quinta fase, se debe haber ejecutado todo lo mencionado anteriormente.  
   Además, dentro de esta primero se debe ejecutar el script _actualizar_datos.py_  , si da error la compilación inicial, es por la función ejecutar_scripts que hay implementada en el script. Para arreglarlo, ejecutar en consola el siguiente comando: python -m ipykernel install --user --name=c2425-r6 --display-name "Python (c2425-R6)" (explicado en el comentario inicial del script).
   Después se podría ejecutar _reentrenamiento.py_  
   (aunque simplemente con ejecutar _actualizar_datos.py_ ya ejecutará todos los scripts de la fase 5 y 6 en el orden óptimo).
   

6. **Nuevos datos:**  
   Para esta última fase se debe haber ejecutado todo lo dicho anteriormente.  
   Y dentro de esta se ha de seguir el siguiente orden:  
   primero se debe ejecutar el _merge_nuevos_datos.py_,  
   después el notebook _eliminacion de variables nuevos datos.ipynb_,  
   y por último el script _creacion_nuevos_datos.py_  
   (aunque solo con ejecutar el script _actualizar_datos.py_ de la fase 5 ya te ejecuta todos estos scripts).
# Resumen de los resultados de los mejores modelos construidos.

### Modelos para Predicción a 7 Días

| Modelo         | MSE     | RMSE   | MAE    | R²      | Accuracy |
|----------------|---------|--------|--------|---------|----------|
| **Baseline**   | 51.6097 | 7.1839 | 5.5585 | -0.6235 | 0.5564   |
| Árbol Decisión | 13.7469 | 3.7077 | 3.1287 | -0.1264 | 0.5200   |
| LSTM           | 14.2768 | 3.7784 | 3.0281 | -0.1270 | 0.4722   |
| **SVR**        | **8.7346** | **2.9554** | **2.4299** | **0.2843** | **0.7200** |

**Comentario**:  
Para la predicción a 7 días, el modelo **SVR (Support Vector Regression)** obtiene claramente los mejores resultados. Es el único con un **R² positivo**, y logra la menor **MAE** y mayor **Accuracy** (72%).

---

### Modelos para Predicción a 30 Días

| Modelo       | MSE      | RMSE    | MAE     | R²      | Accuracy |
|--------------|----------|---------|---------|---------|----------|
| **Baseline** | 249.4163 | 15.7929 | 13.0654 | -0.6583 | 0.5337   |
| ARIMA        | 176.7573 | 13.2950 | 10.1990 | -0.1218 | 0.8108   |
| SARIMA       | 163.7651 | 12.7971 | 10.3525 | -0.0393 | 0.8108   |
| **XGBoost**  | **110.4928** | **10.5116** | **7.7751** | **0.2988** | **0.8108** |

**Comentario**:  
En la predicción a 30 días, **XGBoost** destaca con el **mejor rendimiento general**, siendo el único modelo con **R² positivo** y errores más bajos. Aunque comparte la misma precisión que ARIMA y SARIMA (81.08%), XGBoost es más robusto y preciso.

---

### Conclusión
- A partir de los modelos evaluados, se han seleccionado **SVR** para el horizonte de **7 días** y **XGBoost** para el de **30 días**, 
al ser los que han ofrecido los mejores resultados en sus respectivos contextos. La comparación se ha realizado usando 
como métrica principal el error cuadrático medio (MSE), junto con métricas complementarias como RMSE, MAE, R² y
una medida de accuracy que refleja la capacidad del modelo para anticipar correctamente la dirección del cambio.


- Ambos modelos superan claramente al baseline, lo que valida el uso de técnicas avanzadas de regresión y series temporales para predecir el comportamiento del mercado de Bitcoin.
#  Equipo de desarrollo.
Rodrigo Jesús-Portanet    
Pablo Alonso    
Rosa Gómez-Gil    
Ignacio Ramírez    
Daniel Higueras    
Vega García


